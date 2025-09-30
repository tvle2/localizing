#!/usr/bin/env python3
from typing import Literal, List, Optional, Tuple
import jax.numpy as jnp
from tensorflow import cast, ones, \
    gather, concat, reshape, norm, expand_dims, \
        Variable, Tensor,broadcast_to,shape,float32,\
            __version__,config,switch_case,constant

from tensorflow.math import exp, cos, abs, round

from tensorflow.random import uniform
from tensorflow.keras.optimizers import Adam

from numpy import ceil, sqrt, zeros, savetxt, loadtxt

from argparse import ArgumentParser

from os import makedirs

from src.qsensoropt import  InverseSqrtDecay,ParticleFilter, Parameter, \
        SimulationParameters
from src.qsensoropt.utils import train,standard_model

from nv_center_dc import NVCenter, Magnetometry





# class NVCenterDCMagnPhase(NVCenter):

#     def __init__(
#         self, batchsize: int, params: List[Parameter],
#         prec: Literal["float64", "float32"] = "float64",
#         res: Literal["meas", "time"] = "meas",
#         invT2: Optional[float] = None,
#         ):

#         super().__init__(
#             batchsize, params, prec=prec, res=res,
#             control_phase=True,
#         )

#         self.invT2 = invT2

#     def model(
#         self, idxN:int,outcomes: Tensor, controls: Tensor,
#         parameters: Tensor, meas_step: Tensor,
#         num_systems: int = 1,
#         ) -> Tensor:

#         evolution_time = controls[:, :, 0]
#         phase = controls[:, :, 1]
#         omega = parameters[:, :, 0]
#         if self.invT2 is not None:
#             invT2 = self.invT2
#         else:
#             invT2 = parameters[:, :, 1]
#         exp_decay = exp(-evolution_time*invT2)#noisedc*
#         ramsey_out = outcomes[:, :, 0]
#         noise_less = (1.0-ramsey_out*\
#             cos(omega*evolution_time+phase))/2.0
#         return exp_decay*noise_less + (1.0-exp_decay)/2.0  # Equation (1) in quantum sensing paper

# Withh fidelity
class NVCenterDCMagnPhase(NVCenter):
    def __init__(self, batchsize, params,
                 prec: Literal["float64","float32"] = "float32",
                 res:  Literal["meas","time"] = "time",
                 invT2: Optional[float] = None,
                 F0: float = 1.0, F1: float = 1.0):        # ← 与原码不同：支持读出保真度
        super().__init__(batchsize, params, prec=prec, res=res, control_phase=True)
        self.invT2 = invT2
        self.F0, self.F1 = float(F0), float(F1)

    def model(self, idxN, outcomes, controls, parameters, meas_step, num_systems=1):
        tau  = controls[:, :, 0]
        phi  = controls[:, :, 1]
        omega = parameters[:, :, 0]
        invT2_loc = self.invT2 if self.invT2 is not None else parameters[:, :, 1]
        decay = tf.exp(-tau * invT2_loc)

        r = outcomes[:, :, 0]  # ±1，**与原码一致：+1 表示测到 μ=1，-1 表示 μ=0**

        # p0 with fidelity (论文公式，F0=F1=1 时退化为 0.5 + 0.5*decay*cos)
        bias     = 0.5 * (1.0 + self.F0 - self.F1)
        contrast = 0.5 * (self.F0 + self.F1 - 1.0)
        p0 = bias + contrast * decay * tf.cos(omega * tau + phi)

        # 概率应返回“观测到 r 的概率”：
        # r>0 ↔ μ=1 → p1 = 1 - p0；r<0 ↔ μ=0 → p0
        p_obs = tf.where(r > 0.0, 1.0 - p0, p0)         # ← 与你当前代码不同（修正了映射）
        return p_obs

def parse_args():

    parser = ArgumentParser()
    parser.add_argument("--scratch-dir", type=str, required=False)
    parser.add_argument("--trained-models-dir", type=str,
                        default="./EEnv_center_dc_phase10/trained_models")
    parser.add_argument("--data-dir", type=str,
                        default="./EEnv_center_dc_phase10/data")
    parser.add_argument("--prec", type=str, default="float32")
    parser.add_argument('--n', type=int, default=64)
    parser.add_argument("--num-particles", type=int, default=int(240))#int(240*2)
    parser.add_argument("--iterations", type=int, default=2048)#256#4096#
    parser.add_argument("--scatter-points", type=int, default=32)

    return parser.parse_args()

def static_field_estimation(
    args,idxn, batchsize: int, max_res: float,
    learning_rate: float = 1e-2,
    gradient_accumulation: int = 1,
    cumulative_loss: bool = False,
    log_loss: bool = False,
    res: Literal["meas", "time"] ="time",
    invT2: Optional[float] = None,
    invT2_bound: Optional[Tuple] = None,
    cov_weight_matrix: Optional[List] = None,
    omega_bounds: Tuple[float, float] = (0.0,1.0),#(0.0, 2*3.14*25),#(0.0,2)
    ):

    if invT2 is None and invT2_bound is None:
        raise ValueError("At least one between invT2 and" \
            "invT2_bound must be passed to static_field_estimation")

    d = 2 if invT2 is None else 1
    input_size_original = d**2+2*d+2
    input_size = input_size_original
    control_size = 2

    name = "nv_center_" + res
    if invT2:
        name += f"_invT2_{invT2:.4f}"
    if invT2_bound:
        name += f"_invT2_min_{invT2_bound[0]:.4f}_" \
            f"invT2_max_{invT2_bound[1]:.4f}"
    
    name += f"_lr_{learning_rate}"


    print(torch.version.cuda)  # Should output 11.2
    print(torch.cuda.is_available())  # Should return True

    print(__version__)
    print(config.list_physical_devices('GPU'))
    torch.cuda.set_per_process_memory_fraction(0.5, device=0)
    network = standard_model(
        input_size=input_size,
        controls_size=control_size,
        neurons_per_layer=args.n,
        prec=args.prec,
        )

    # Preconditioning of the network
    input_tensor = 2*uniform((16384, input_size),
                             dtype=args.prec)-1
    output_tensor_tau = (input_tensor[:, -1:]+1)/2
    output_tensor_phase = 2*uniform((16384, 1), 
                             dtype=args.prec)-1
    output_tensor = concat(
        [output_tensor_tau, output_tensor_phase], 1,
        )
    network.compile(loss='mean_squared_error',
                    optimizer='adam')
    network.fit(
        input_tensor, output_tensor,
        epochs=12, batch_size=1024, verbose=0,
        )

    nv_center = NVCenterDCMagnPhase(
        batchsize=batchsize,
        params=[Parameter(bounds=omega_bounds, \
                          name="omega"),] if invT2 is not None else
            [Parameter(bounds=omega_bounds, name="omega"),
            Parameter(bounds=invT2_bound, name="invT2"),],
        prec=args.prec,
        res=res, invT2=invT2,
        )
        
    pf = ParticleFilter(
        num_particles=args.num_particles,
        phys_model=nv_center,
        prec=args.prec,
    )
    
    simpars = SimulationParameters(
        sim_name=name,
        num_steps=max_res,
        max_resources=max_res,
        prec=args.prec,
        cumulative_loss=cumulative_loss,
        log_loss=log_loss,
    )

    # Computation of the prefactor
    prefact = max_res/20 if res=="time" \
        else ceil(2**(sqrt(max_res)))
    invT2_min = invT2 if invT2 is not None \
        else invT2_bound[0]
    if invT2_min > 0.0:
        prefact = min(prefact, 1.0/invT2_min)

    numN = 10
    networks = [network for _ in range(numN)]




    def control_nn(idxN:int,input_tensor: Tensor):
        #def select_network(input_tensor, idxN):
            pi = constant(3.141592653589793, dtype=float32)

            def get_network_output(i):
                tau = prefact * abs(networks[i](input_tensor)[:, 0:1]) + 1.0
                phase = pi * networks[i](input_tensor)[:, 1:2]
                return lambda: concat([tau, phase], 1)

            ctrl = switch_case(idxN, branch_fns={i: get_network_output(i) for i in range(len(networks))})

            return ctrl





    sim_nn = Magnetometry(
        particle_filter=pf,
        simpars=simpars,
        phys_model=nv_center,
        control_strategy=control_nn,
        cov_weight_matrix=cov_weight_matrix,
        eta_flag=res=="time",
    )

    decaying_learning_rate = InverseSqrtDecay(
        learning_rate, args.prec
    )



    # print(sim_nn)
    # #
    train(
        sim_nn, Adam(learning_rate=decaying_learning_rate),
        args.iterations, args.trained_models_dir,
        networks=networks,
        xla_compile=False,
        gradient_accumulation=gradient_accumulation,
    )



def main():
    import pandas as pd
    # ##########----Training-------------------------------------#####################
    args = parse_args()
    dir_name = args.data_dir+str(0)
    makedirs(dir_name, exist_ok=True)
    args.data_dir = dir_name
    static_field_estimation(
        args,idxn=int(0), batchsize=1024, max_res=int(10000), learning_rate=1e-3, res='time',#max_res=int(2796)
        log_loss=True, cumulative_loss=True,
        invT2=1/1500,
        )



if __name__ == "__main__":
    main()
