#!/usr/bin/env python3
from typing import Tuple, List

from tensorflow import cast, ones, stop_gradient, ones, \
    constant, concat, expand_dims, print, Tensor
from tensorflow.math import log, cos, abs, pow
from tensorflow.random import stateless_uniform, Generator
from tensorflow.keras.optimizers import Adam
from argparse import ArgumentParser
from math import pi

from qsensoropt import StatefulPhysicalModel, \
    StatelessMetrology, InverseSqrtDecay, \
    ParticleFilter, Parameter, Control, \
    SimulationParameters, StatefulMetrology, \
    StatelessPhysicalModel
from qsensoropt.utils import train, \
    performance_evaluation, get_seed, \
    store_input_control, standard_model

# Interferometric experiment with a damage
# model for the measured sample
class DamageInterferometer(StatefulPhysicalModel):
    """Single photon interferometer with
    non-unitary visibility
    with damage model.

    For each measurement the photon passing
    through the sample
    (which could be biological in origin)
    releases some energy
    and damages it stochastically.
    This model will be used by the simulation
    to extract the
    measurement outcome.
    """

    def __init__(
            self, batchsize: int, params: List[Parameter],
            prec: str = "float64", max_damage: float = 0.95,
            importance_sampling: bool = False,
            ):
        
        if importance_sampling:
            self.imp_coeff = 0.8
        else:
            self.imp_coeff = 1.0

        controls = [Control(name="ControlPhase")]

        super().__init__(
            batchsize, controls, params,
            state_specifics={'size': 1, 'type': "float64"},
            prec=prec,
        )

        self.max_damage = max_damage

    def initialize_state(
            self, parameters: Tensor,
            num_systems: int) -> Tensor:
        return ones(
            (self.bs, num_systems, 1),
            dtype=self.state_specifics['type'],
        )

    def perform_measurement(
        self, controls: Tensor, parameters: Tensor,
        true_state: Tensor,
        meas_step: Tensor, rangen: Generator,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        r"""Measurement with outcomes -1 and +1.

        With a stochastic damage acting on
        the sample which transforms
        the phase to be sampled to
        $\theta_{t} = D_{t} \theta$, with
        $D_{t} = \alpha_{t} D_{t-1}$,
        being $\alpha_{t}$ the damage
        caused by the measurement at step t.
        """
        list_plus = ones(
            (self.bs, 1, 1), dtype=self.prec,
            )
        parameters_mod = concat(
            [parameters[:, :, 0:1],
             self.imp_coeff*parameters[:, :, 1:2]], 2,
        )
        prob_plus, _ = self.model(
            list_plus, controls, parameters_mod,
            true_state, meas_step,
        )
        # Extraction of the actual outcomes
        seed = get_seed(rangen)
        outcomes = 2*cast((stateless_uniform(
            (self.bs, 1), seed, dtype=self.prec) <
            stop_gradient(abs(prob_plus))), dtype="int8")-1
        outcomes = expand_dims(
            cast(outcomes, dtype=self.prec), axis=1,
            )
        prob_outcomes, _ = self.model(
            outcomes, controls, parameters_mod,
            true_state, meas_step,
        )
        log_prob = cast(
            log(prob_outcomes), dtype=self.prec,
            )
        # Perturbation of the state which
        # happens after the measurement
        true_state *= (1.0-self.max_damage)*\
            stateless_uniform(
            (self.bs, 1, 1), seed,
            dtype=self.state_specifics['type'],
        )+self.max_damage
        return outcomes, log_prob, true_state

    def model(
        self, outcomes: Tensor, controls: Tensor,
        parameters: Tensor, state: Tensor,
        meas_step: Tensor,
        num_systems: int = 1,
    ) -> Tensor:
        """Probability for the photon
        two be detected
        in the first or the second port.

        This function will be used only
        for extracting the observed
        outcomes in the case the damage
        value is not observed
        """
        phase = parameters[:, :, 0]
        visibility = parameters[:, :, 1]
        control_phase = controls[:, :, 0]
        damage = cast(state[:, :, 0], dtype=self.prec)
        phot_measure = outcomes[:, :, 0]
        prob = (1.0+phot_measure*visibility *
                cos(damage*phase-control_phase))/2.0
        return prob, state

    def count_resources(
        self, resources: Tensor, outcomes: Tensor,
        controls: Tensor, true_values: Tensor, state: Tensor,
        meas_step: Tensor,
    ):
        """The resource is the number of
        measurements.
        """
        return resources+1.0

# Interferometric experiment with a
# damage model for the measured sample.
class UnknownDamageInterferometer(StatelessPhysicalModel):
    """Single photon interferometer with non-unitary visibility
    with damage model.

    For each measurement the photon passing through the sample
    (which could be biological in origin) releases some energy
    and damages it stochastically. This is the model used by
    the particle filter to perform the Bayesian inference.
    We implement only the function model, as this is the only
    one used by the PF.
    """
    def __init__(
            self, batchsize: int, params,
            prec: str = "float64",
            max_damage: float = 0.95,
            ):

        controls = [Control(name="ControlPhase")]

        super().__init__(
            batchsize, controls, params,
            prec=prec,
        )

        self.max_damage = max_damage

    def model(
        self, outcomes: Tensor, controls: Tensor,
        parameters: Tensor, meas_step: Tensor,
        num_systems: int = 1,
    ) -> Tensor:
        """Probability for the photon two be detected
        in the first or the second port.

        This function is use only in the
        particle filter to estimate
        the probabilities of observing a certain outcome.
        Since the damage is unknown and is
        only manifest in a stochastic
        perturbation of the true state of the system we
        cannot reference it in the model.
        We introduce instead
        an expected damage, which is computed
        observing that the individual
        damages are independent.
        """
        phase = parameters[:, :, 0]
        visibility = parameters[:, :, 1]
        control_phase = controls[:, :, 0]
        average_photon_damage = constant(
            (1.0+self.max_damage)/2, dtype=self.prec,
        )
        estimated_damage = pow(
            average_photon_damage,
            cast(meas_step[:, :, 0], dtype=self.prec),
        )
        phot_measure = outcomes[:, :, 0]
        prob = (1.0+phot_measure*visibility*\
                cos(estimated_damage*phase-control_phase))/2.0
        return prob

# Interferometric experiment with a
# damage model for the measured sample
class ObservedDamageInterferometer(StatefulPhysicalModel):
    """Single photon interferometer
    with non-unitary visibility
    with damage model.

    For each measurement the photon passing through the sample
    (which could be biological in origin) releases some energy
    and damages it stochastically. In this scenario we suppose
    that the damage degree of each photon can be measured
    with some independent observation on the sample.
    """

    def __init__(
            self, batchsize: int, params,
            prec: str = "float64",
            max_damage: float = 0.95,
            ):

        controls = [Control(name="ControlPhase")]

        super().__init__(
            batchsize, controls, params,
            state_specifics={'size': 1, 'type': "float64",},
            prec=prec,
            outcomes_size = 2
        )

        self.max_damage = max_damage

    def initialize_state(
            self, parameters: Tensor,
            num_systems: int) -> Tensor:
        return ones(
            (self.bs, num_systems, 1),
            dtype=self.state_specifics['type'],
        )

    def perform_measurement(
        self, controls: Tensor, parameters: Tensor,
        true_state: Tensor,
        meas_step: Tensor, rangen: Generator,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Measurement with outcomes -1 and +1.
        """
        list_plus = ones(
            (self.bs, 1, 2), dtype=self.prec,
        )
        prob_plus, _ = self.model(
            list_plus, controls, parameters,
            true_state, meas_step,
        )
        # Extraction of the actual outcomes
        seed = get_seed(rangen)
        # Observed interferometer port outcome
        outcomes = 2*cast((stateless_uniform((self.bs, 1),
            seed, dtype=self.prec)<stop_gradient(
            abs(prob_plus))), dtype="int8")-1
        outcomes = cast(outcomes, dtype=self.prec)
        # Observed state damage
        seed = get_seed(rangen)
        step_damage = (1.0-self.max_damage)*stateless_uniform(
            (self.bs, 1), seed, dtype=self.prec,
        )+self.max_damage
        joint_outcomes = expand_dims(
            concat([outcomes, step_damage], 1),
            axis=1,
        )
        # Prob of the observed photodetection
        # outcome and state update
        prob_outcomes, true_state = self.model(
            joint_outcomes, controls,
            parameters, true_state, meas_step,
        )
        log_prob = cast(
            log(prob_outcomes), dtype=self.prec,
            )
        return joint_outcomes, log_prob, true_state

    def model(
        self, outcomes: Tensor, controls: Tensor,
        parameters: Tensor, state: Tensor,
        meas_step: Tensor,
        num_systems: int = 1,
    ) -> Tuple[Tensor, Tensor]:
        """Probability for the photon two be detected
        in the first or the second port.
        This function will be used only
        for extracting the observed
        outcomes in the case the damage
        value is not observed
        """
        phase = parameters[:, :, 0]
        visibility = parameters[:, :, 1]
        control_phase = controls[:, :, 0]
        damage = cast(state[:, :, 0], dtype=self.prec)
        pd_outcome = outcomes[:, :, 0]
        prob = (1.0+pd_outcome*visibility*\
                cos(damage*phase-control_phase))/2.0
        # Perturbation of the state
        # which happens after the measurement
        state *= cast(
            expand_dims(outcomes[:, :, 1], axis=2),
            dtype=self.state_specifics['type'],
        )
        return prob, state

    def count_resources(
        self, resources: Tensor, outcomes: Tensor,
        controls: Tensor, true_values: Tensor, state: Tensor,
        meas_step: Tensor,
    ):
        """The resource is the number
        of measurements.
        """
        return resources+1.0


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--scratch-dir", type=str, required=True)
    parser.add_argument("--trained-models-dir", type=str,
                        default="./interferometer/trained_models")
    parser.add_argument("--data-dir", type=str,
                        default="./interferometer/data")
    parser.add_argument("--prec", type=str, default="float32")
    parser.add_argument("--n", type=int, default=64)
    parser.add_argument("--batchsize", type=int, default=256)
    parser.add_argument("--num-particles", type=int, default=256)
    parser.add_argument("--num-steps", type=int, default=128)
    parser.add_argument("--iterations", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--scatter-points", type=int, default=32)

    return parser.parse_args()

def main():
    args = parse_args()

    network = standard_model(
        input_size=10,
        controls_size=1,
        neurons_per_layer=args.n,
        prec=args.prec,
    )
    network.compile()

    interferometer = DamageInterferometer(
        batchsize=args.batchsize,
        params=[
            Parameter(bounds=(0.0, 2*pi), name="phase"),
            Parameter(bounds=(0.95, 1.00), name="visibility"),
        ],
        prec=args.prec,
    )
    
    pf = ParticleFilter(
        num_particles=args.num_particles,
        phys_model=UnknownDamageInterferometer(
            batchsize=args.batchsize,
            params=[
                Parameter(bounds=(0.0, 2*pi), name="phase"),
                Parameter(bounds=(0.95, 1.00), name="visibility"),
            ],
            prec=args.prec,
        ),
        prec=args.prec,
    )

    simpars = SimulationParameters(
        sim_name="inter_damage",
        num_steps=args.num_steps,
        max_resources=args.num_steps,
        cumulative_loss=True,
        prec=args.prec,
    )

    sim_nn = StatelessMetrology(
        particle_filter=pf,
        simpars=simpars,
        phys_model=interferometer,
        control_strategy=network,
    )

    decaying_learning_rate = InverseSqrtDecay(
        args.learning_rate, prec=args.prec,
    )

    train(
        sim_nn,
        Adam(learning_rate=decaying_learning_rate),
        args.iterations,
        args.scratch_dir, network=network,
        xla_compile=True,
    )

    performance_evaluation(
        sim_nn, args.iterations, args.data_dir,
        xla_compile=True,
        delta_resources=1,
        y_label='MSE',
    )

    store_input_control(
        sim_nn, args.data_dir, 5,
        xla_compile=True,
    )

    # --------------------------------------------------

    network = standard_model(
        input_size=11,
        controls_size=1,
        neurons_per_layer=args.n,
        prec=args.prec,
    )
    network.compile()

    interferometer = ObservedDamageInterferometer(
        batchsize=args.batchsize,
        params=[
            Parameter(bounds=(0.0, 2*pi), name="phase"),
            Parameter(bounds=(0.95, 1.00), name="visibility"),
        ],
        prec=args.prec,
    )

    pf = ParticleFilter(
        num_particles=args.num_particles,
        phys_model=interferometer,
        prec=args.prec,
    )

    simpars = SimulationParameters(
        sim_name="inter_damage_stateful",
        num_steps=args.num_steps,
        max_resources=args.num_steps,
        cumulative_loss=True,
        prec=args.prec,
    )

    sim_nn = StatefulMetrology(
        particle_filter=pf,
        simpars=simpars,
        phys_model=interferometer,
        control_strategy=network,
    )

    decaying_learning_rate = InverseSqrtDecay(
        args.learning_rate, prec=args.prec,
    )

    train(
        sim_nn,
        Adam(learning_rate=decaying_learning_rate),
        args.iterations,
        args.scratch_dir, network=network,
        xla_compile=False,
    )

    performance_evaluation(
        sim_nn, args.iterations, args.data_dir,
        xla_compile=True,
        delta_resources=1,
        y_label='MSE',
    )

    store_input_control(
        sim_nn, args.data_dir, 5,
        xla_compile=True,
    )

    # print("Memory:")
    # print(get_memory_info('GPU:0')['peak']/1024**3)

if __name__ == "__main__":
    main()
