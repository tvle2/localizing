#!/usr/bin/env python3
from typing import Literal, List, Optional, Tuple
import jax.numpy as jnp
from tensorflow import cast, ones, \
    gather, concat, reshape, norm, expand_dims, \
        Variable, Tensor,broadcast_to,shape,float32,\
            __version__,config,switch_case,constant

from tensorflow.math import exp, cos, abs, round
from tensorflow.linalg import trace
from tensorflow.random import uniform
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
from tensorflow.config.experimental import get_memory_info
from numpy import ceil, sqrt, zeros, savetxt, loadtxt
from numpy.random import rand
from argparse import ArgumentParser
from os.path import join
from math import pi
from os import makedirs
from src.qsensoropt import InverseSqrtDecay, \
    ParticleFilter, Parameter, \
        SimulationParameters
from src.qsensoropt.utils import performance_evaluation, store_input_control, \
        standard_model, denormalize

from nv_center_dc import NVCenter, Magnetometry

# No fidelity version
# class NVCenterDCMagnPhase(NVCenter):
#     r"""Model describing the estimation of a
#     static magnetic field with an NV center
#     used as magnetometer.
#     The transversal relaxation time :math:`T_2^{-1}` can
#     be either known, or be a parameter to
#     estimate. The estimation will be
#     formulated in terms of the precession
#     frequency :math:`\omega` of the NV center, which
#     is proportional to the magnetic
#     field :math:`B`.
#     With respect to the :py:mod:`nv_center_dc` module
#     we add here the possibility
#     of imprinting an arbitrary phase on the NV-center
#     state before the photon counting measurement.
#     """
#     def __init__(
#         self, batchsize: int, params: List[Parameter],
#         prec: Literal["float64", "float32"] = "float64",
#         res: Literal["meas", "time"] = "meas",
#         invT2: Optional[float] = None,
#         self.F0 = F0
#         self.F1 = F1
#         ):
#         r"""Constructor
#         of the :py:obj:`~.NVCenterDCMagnPhase` class.

#         Parameters
#         ----------
#         batchsize: int
#             Batchsize of the simulation, i.e. number of estimations
#             executed simultaneously.
#         params: List[:py:obj:`~.Parameter`]
#             List of unknown parameters to estimate in
#             the NV center experiment, with their
#             respective bounds. This contains either
#             the precession frequency only or
#             the frequency and the inverse coherence time.
#         prec : {"float64", "float32"}
#             Precision of the floating point operations in the 
#             simulation.
#         res: {"meas", "time"}
#             Resource type for the present metrological task, 
#             can be either the total evolution time, i.e. `time`,
#             or the total number of measurements on
#             the NV center, i.e. `meas`.
#         invT2: float, optional
#             If this parameter is specified only the precession
#             frequency :math:`\omega` is considered as an unknown
#             parameter, while the inverse of the
#             transverse relaxation time :math:`T_2^{-1}` is fixed
#             to the value `invT2`. In this case the list `params`
#             must contain a single parameter, i.e. `omega`.
#             If no `invT2` is specified,
#             it is assumed that it is an unknown parameter that
#             will be estimated along the frequency in the Bayesian
#             procedure. In this case `params` must contain two
#             objects, the second of them should be the inverse of the
#             transversal relaxation time.
#         """
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
#         r"""Model for the outcome of a measurement
#         on a NV center that has been precessing in a static
#         magnetic field.  The probability of getting the
#         outcome :math:`+1` is

#         .. math::

#             p(+1|\omega, T_2, \tau) := e^{-\frac{\tau}{T_2}}
#             \cos^2 \left( \frac{\omega}{2} \tau + \phi \right) +
#             \frac{1-e^{-\frac{\tau}{T_2}}}{2} \; .

#         The evolution time :math:`\tau` and
#         the phase :math:`\phi` are controlled
#         by the trainable agent, and :math:`\omega`
#         is the unknown precession frequency, which is
#         proportional to the magnetic field.
#         The parameter :math:`T_2`
#         may or may not be an unknown in the estimation,
#         according to the value of the attribute `invT2
#         of the :py:obj:`NVCenterDCMagnPhase` class.
#         """
#         #print('SOrt0--', parameters[:, :, 0])
#         #noisedc =Nois[idxN]
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
#         return exp_decay*noise_less + (1.0-exp_decay)/2.0

# Withh fidelity
class NVCenterDCMagnPhase(NVCenter):
    r"""NV-center DC magnetometry with optional readout fidelity.
    Adds the ability to imprint a control phase before readout.

    This variant supports fixed T2 (via invT2) and readout fidelity (F0, F1).
    F1 = P(observe +1 | true +1);  F0 = P(observe -1 | true -1).
    """

    def __init__(
        self,
        batchsize: int,
        params: List[Parameter],
        prec: Literal["float64", "float32"] = "float64",
        res: Literal["meas", "time"] = "meas",
        invT2: Optional[float] = None,
        F0: float = 0.88,
        F1: float = 0.95,
    ):
        """Constructor of the NVCenterDCMagnPhase class."""
        super().__init__(batchsize, params, prec=prec, res=res, control_phase=True)
        self.invT2 = invT2
        self.F0 = F0
        self.F1 = F1

    def model(
        self,
        idxN: int,
        outcomes: Tensor,
        controls: Tensor,
        parameters: Tensor,
        meas_step: Tensor,
        num_systems: int = 1,
    ) -> Tensor:
        r"""Return p(observed outcome | parameters, controls) with fidelity.

        Ideal (no readout error) model under dephasing:
            p_true(+1)  = 0.5 * (1 - e^{-τ/T2} * cos(ωτ + φ))
            p_true(-1)  = 1 - p_true(+1)

        With readout fidelity:
            p_obs(+1)   = F1 * p_true(+1) + (1 - F0) * p_true(-1)
            p_obs(-1)   = F0 * p_true(-1) + (1 - F1) * p_true(+1)
        """
        tau   = controls[:, :, 0]
        phi   = controls[:, :, 1]
        omega = parameters[:, :, 0]
        invT2 = self.invT2 if self.invT2 is not None else parameters[:, :, 1]

        E = exp(-tau * invT2)                       # dephasing factor
        c = cos(omega * tau + phi)

        # Ideal probabilities for μ=+1 and μ=-1 (shape: [bs, 1])
        p_true_plus  = 0.5 * (1.0 - E * c)
        p_true_minus = 1.0 - p_true_plus

        mu = outcomes[:, :, 0]                      # observed μ ∈ {−1, +1}
        mask_plus  = (mu + 1.0) * 0.5               # 1 for μ=+1, 0 otherwise
        mask_minus = 1.0 - mask_plus

        # Readout confusion matrix
        p_obs_plus  = self.F1 * p_true_plus  + (1.0 - self.F0) * p_true_minus
        p_obs_minus = self.F0 * p_true_minus + (1.0 - self.F1) * p_true_plus

        # Return p(μ_observed)
        return mask_plus * p_obs_plus + mask_minus * p_obs_minus

def parse_args():
    """Arguments
    ---------
    scratch_dir: str
        Directory in which the intermediate models should
        be saved alongside the loss history.
    trained_models_dir: str = "./nv_center_dc_phase/trained_models"
        Directory in which the finalized trained model
        should be saved.
    data_dir: str = "./nv_center_dc_phase/data"
        Directory containing the csv files
        produced by the :py:func:`~.utils.performance_evaluation`
        and the :py:func:`~.utils.store_input_control` functions.
    prec: str = "float32"
        Floating point precision of the
        whole simulation.
    n: int = 64
        Number of neurons per layer in the neural network.
    num_particles: int = 480
        Number of particles in the ensemble representing
        the posterior.
    iterations: int = 32768
        Number of training steps.
    scatter_points: int = 32
        Number of points in the Resources/Precision
        csv produced by
        :py:func:`~.utils.performance_evaluation`.
    """
    parser = ArgumentParser()
    parser.add_argument("--scratch-dir", type=str, required=False)
    parser.add_argument("--trained-models-dir", type=str,
                        default="./nv_center_dc_phase20/trained_models")
    parser.add_argument("--data-dir", type=str,
                        default="./EnTnv_center_dc_phaseNN100/data")
    parser.add_argument("--prec", type=str, default="float32")
    parser.add_argument('--n', type=int, default=64)
    parser.add_argument("--num-particles", type=int, default=int(240*2))
    parser.add_argument("--iterations", type=int, default=2048)#256#4096#2048
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
    omega_bounds: Tuple[float, float] = (0.0,2),#(0.0, 2*3.14*25),
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
        def select_network(input_tensor, idxN):
            pi = 3.14

            tau = prefact * abs(networks[idxN](input_tensor)[:, 0:1]) + 1.0
            phase = pi * networks[idxN](input_tensor)[:, 1:2]
            ctrl = concat([tau, phase], 1)
            return ctrl
        #idxN = int(1)
        ctrl = select_network(input_tensor, idxN)
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





    save_path = args.trained_models_dir

    file_path = save_path + \
                "_history_weights/"
    file_name = file_path + str(128)

    for i in range(numN):
        networks[i].load_weights(file_name + str(i))
        networks[i].save(
            join(save_path, str(i)),
        )


        networks[i]=load_model(
          join(args.trained_models_dir, str(i)),
            )

    sim_nn.eta_flag = False

    if res=="time":
        precision_fit = None
        delta_resources = 500
    else:
        precision_fit = None if res=="meas" else \
            {'num_points': args.scatter_points,
                'batchsize': 16384,
                'epochs': 8,
                'direct_func': lambda res, prec: res*prec,
                'inverse_func': lambda res, c: c/res}
        delta_resources = 1.0 if res=="meas" else None

    performance_evaluation(
        sim_nn,
        idxn,
        gradient_accumulation*200,#4096,
        args.data_dir,
        xla_compile=False,
        precision_fit=precision_fit,
        delta_resources=delta_resources,
        y_label='MSE',
    )

    store_input_control(
        sim_nn,idxn, args.data_dir, 1,
        xla_compile=False,
    )

def main():
    import pandas as pd
    for i in range(0,10):
    #i = 0
        args = parse_args()
        dir_name = args.data_dir+str(i)
        makedirs(dir_name, exist_ok=True)
        args.data_dir = dir_name
        static_field_estimation(
            args,idxn=int(i), batchsize=1024, max_res=int(10000), learning_rate=1e-3, res='time',
            log_loss=True, cumulative_loss=True,
            invT2=1/1500,
            )





if __name__ == "__main__":
    main()
