#!/usr/bin/env python3
from typing import Literal, List, Optional, Tuple
import jax.numpy as jnp
from tensorflow import (cast, ones, \
    gather, concat, reshape, norm, expand_dims, \
        Variable, Tensor,broadcast_to,shape,float32,
                        distribute,config,test,__version__,device)
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
import torch
from src.qsensoropt import InverseSqrtDecay, \
    ParticleFilter, Parameter, \
        SimulationParameters
from src.qsensoropt.utils import train, \
    performance_evaluation, store_input_control, \
        standard_model, denormalize

from nv_center_dc import NVCenter, Magnetometry

class NVCenterDCMagnPhase(NVCenter):
    r"""Model describing the estimation of a
    static magnetic field with an NV center
    used as magnetometer.
    The transversal relaxation time :math:`T_2^{-1}` can
    be either known, or be a parameter to
    estimate. The estimation will be
    formulated in terms of the precession
    frequency :math:`\omega` of the NV center, which
    is proportional to the magnetic
    field :math:`B`.
    With respect to the :py:mod:`nv_center_dc` module
    we add here the possibility
    of imprinting an arbitrary phase on the NV-center
    state before the photon counting measurement.
    """
    def __init__(
        self, batchsize: int, params: List[Parameter],
        prec: Literal["float64", "float32"] = "float64",
        res: Literal["meas", "time"] = "meas",
        invT2: Optional[float] = None,
        ):
        r"""Constructor
        of the :py:obj:`~.NVCenterDCMagnPhase` class.

        Parameters
        ----------
        batchsize: int
            Batchsize of the simulation, i.e. number of estimations
            executed simultaneously.
        params: List[:py:obj:`~.Parameter`]
            List of unknown parameters to estimate in
            the NV center experiment, with their
            respective bounds. This contains either
            the precession frequency only or
            the frequency and the inverse coherence time.
        prec : {"float64", "float32"}
            Precision of the floating point operations in the 
            simulation.
        res: {"meas", "time"}
            Resource type for the present metrological task, 
            can be either the total evolution time, i.e. `time`,
            or the total number of measurements on
            the NV center, i.e. `meas`.
        invT2: float, optional
            If this parameter is specified only the precession
            frequency :math:`\omega` is considered as an unknown
            parameter, while the inverse of the
            transverse relaxation time :math:`T_2^{-1}` is fixed
            to the value `invT2`. In this case the list `params`
            must contain a single parameter, i.e. `omega`.
            If no `invT2` is specified,
            it is assumed that it is an unknown parameter that
            will be estimated along the frequency in the Bayesian
            procedure. In this case `params` must contain two
            objects, the second of them should be the inverse of the
            transversal relaxation time.
        """
        super().__init__(
            batchsize, params, prec=prec, res=res,
            control_phase=True,
        )

        self.invT2 = invT2

    def model(
        self, outcomes: Tensor, controls: Tensor,
        parameters: Tensor, meas_step: Tensor,
        num_systems: int = 1,
        ) -> Tensor:
        r"""Model for the outcome of a measurement
        on a NV center that has been precessing in a static
        magnetic field.  The probability of getting the
        outcome :math:`+1` is

        .. math::

            p(+1|\omega, T_2, \tau) := e^{-\frac{\tau}{T_2}}
            \cos^2 \left( \frac{\omega}{2} \tau + \phi \right) +
            \frac{1-e^{-\frac{\tau}{T_2}}}{2} \; .

        The evolution time :math:`\tau` and
        the phase :math:`\phi` are controlled
        by the trainable agent, and :math:`\omega`
        is the unknown precession frequency, which is
        proportional to the magnetic field.
        The parameter :math:`T_2`
        may or may not be an unknown in the estimation,
        according to the value of the attribute `invT2`
        of the :py:obj:`NVCenterDCMagnPhase` class.
        """
        evolution_time = controls[:, :, 0]
        phase = controls[:, :, 1]
        omega = parameters[:, :, 0]
        if self.invT2 is not None:
            invT2 = self.invT2
        else:
            invT2 = parameters[:, :, 1]
        exp_decay = exp(-evolution_time*invT2)
        ramsey_out = outcomes[:, :, 0]
        noise_less = (1.0-ramsey_out*\
            cos(omega*evolution_time+phase))/2.0
        return exp_decay*noise_less + (1.0-exp_decay)/2.0

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
                        default="./nv_center_dc_phase245/trained_models")
    parser.add_argument("--data-dir", type=str,
                        default="./nv_center_dc_phase245/data")
    parser.add_argument("--prec", type=str, default="float32")
    parser.add_argument('--n', type=int, default=64)
    parser.add_argument("--num-particles", type=int, default=int(240*4))
    parser.add_argument("--iterations", type=int, default=int(1024*16))#
    parser.add_argument("--scatter-points", type=int, default=32)

    return parser.parse_args()

def static_field_estimation(
    args, batchsize: int, max_res: float,
    learning_rate: float = 1e-2,
    gradient_accumulation: int = 1,
    cumulative_loss: bool = False,
    log_loss: bool = False,
    res: Literal["meas", "time"] ="time",
    invT2: Optional[float] = None,
    invT2_bound: Optional[Tuple] = None,
    cov_weight_matrix: Optional[List] = None,
    omega_bounds: Tuple[float, float] = (0.0, 10),
    ):

    if invT2 is None and invT2_bound is None:
        raise ValueError("At least one between invT2 and" \
            "invT2_bound must be passed to static_field_estimation")

    d = 2 if invT2 is None else 1
    input_size_original = d**2+2*d+2
    input_size = input_size_original
    control_size = 1

    name = "nv_center_" + res
    if invT2:
        name += f"_invT2_{invT2:.4f}"
    if invT2_bound:
        name += f"_invT2_min_{invT2_bound[0]:.4f}_" \
            f"invT2_max_{invT2_bound[1]:.4f}"
    
    name += f"_lr_{learning_rate}"
    # gpus = config.list_physical_devices('GPU')
    # if gpus:
    #     try:
    #         for gpu in gpus:
    #             config.experimental.set_memory_growth(gpu, True)
    #         print("GPU memory growth enabled.")
    #     except RuntimeError as e:
    #         print(e)
    #
    # with device('/GPU:0'):
    #     print("Num GPUs Available:", len(config.experimental.list_physical_devices('GPU')))
    #     print("Available GPUs:",config.list_physical_devices('GPU'))
    # print(torch.version.cuda)  # Should output 11.2
    # print(torch.cuda.is_available())  # Should return True
    #
    # print(__version__)
    # print(config.list_physical_devices('GPU'))
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    print("Devices:", config.list_physical_devices())

    print("Current directory:", os.getcwd())

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



    def control_nn(input_tensor: Tensor):
        phase = pi*network(input_tensor)[:, 0:1]
        return phase

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

    # train_nn_graph(
    #     sim_nn,
    #     Adam(learning_rate=decaying_learning_rate),
    #     network,
    # )

    # train_nn_profiler(
    #     sim_nn, Adam(learning_rate=decaying_learning_rate),
    #     network, xla_compile=False,
    # )

    print(sim_nn)
  
    train(
        sim_nn, Adam(learning_rate=decaying_learning_rate),
        args.iterations, args.trained_models_dir,
        network=network,
        xla_compile=False,
        gradient_accumulation=gradient_accumulation,
    )

    # save_path = args.trained_models_dir
    #
    # file_path = save_path + \
    #             "_history_weights/"
    # file_name = file_path + str(691)
    #
    # network.load_weights(file_name)

    #print(get_memory_info('GPU:0')['peak']/1024**3)

    network.save(
        join(args.trained_models_dir, str(2)),
        )
    network=load_model(
      join(args.trained_models_dir, str(2)),
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
        gradient_accumulation*200,
        args.data_dir,
        xla_compile=False,
        precision_fit=precision_fit,
        delta_resources=delta_resources,
        y_label='MSE',
    )

    store_input_control(
        sim_nn, args.data_dir, 5,
        xla_compile=False,
    )

def main():

    args = parse_args()

    static_field_estimation(
        args, 1024, max_res=int(22000), learning_rate=1e-3, res='time',#max_res=int(2796)
        log_loss=True, cumulative_loss=True,
        invT2=1/96,
        )


if __name__ == "__main__":
    main()
