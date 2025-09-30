#!/usr/bin/env python3
from typing import Tuple, List

from tensorflow import cast, ones, stop_gradient, ones, \
    expand_dims, gather, Tensor, Variable
from tensorflow.math import log, cos, abs, floormod
from tensorflow.random import stateless_uniform
from tensorflow.random import Generator
from tensorflow.keras.optimizers import Adam
from numpy import zeros as npzeros
from argparse import ArgumentParser
from math import pi

from qsensoropt import StatelessMetrology, \
    InverseSqrtDecay, ParticleFilter, Parameter, \
        Control, SimulationParameters, \
            StatelessPhysicalModel

from qsensoropt.utils import train, \
    performance_evaluation, get_seed, \
        store_input_control, standard_model

# We test the particle filter with a
# simple model of an interferometer,
# being use to estimate a phase and
# having non-unitary visibility.
class Interferometer(StatelessPhysicalModel):
    """Single photon interferometer
    with non-unitary visibility.
    """
    def __init__(
            self, batchsize: int,
            params: List[Parameter],
            prec: str = "float64",
        ):

        controls = [Control(name="ControlPhase")]

        super().__init__(
            batchsize, controls, params, prec=prec,
            )

    def perform_measurement(
        self, controls: Tensor, parameters: Tensor,
        meas_step: Tensor,
        rangen: Generator,
        ) -> Tuple[Tensor, Tensor]:
        """Measurement with outcomes -1 and +1.
        """
        list_plus = ones((self.bs, 1, 1), dtype=self.prec)
        prob_plus = self.model(
            list_plus, controls, parameters, meas_step,
            )
        # Extraction of the actual outcomes
        seed = get_seed(rangen)
        outcomes = 2*cast(
            (stateless_uniform((self.bs, 1), seed, dtype=self.prec) <
                stop_gradient(abs(prob_plus))), dtype="int8")-1
        outcomes = expand_dims(
            cast(outcomes, dtype=self.prec), axis=1,
            ) #(self.bs, 1, 1)
        prob_outcomes = self.model(
            outcomes, controls, parameters, meas_step,
            )
        log_prob = cast(
            log(prob_outcomes), dtype=self.prec,
            )
        return outcomes, log_prob

    def model(
        self, outcomes: Tensor, controls: Tensor,
        parameters: Tensor, meas_step: Tensor,
        num_systems: int = 1,
        ) -> Tensor:
        """Probability for the photon two be detected
        in the first or the second port.
        """
        phase = parameters[:, :, 0]
        visibility = parameters[:, :, 1]
        pd_outcomes = outcomes[:, :, 0]
        control_phase = controls[:, :, 0]
        return (1.0+pd_outcomes*visibility*\
                cos(phase-control_phase))/2.0

    def count_resources(
        self, resources: Tensor, outcomes: Tensor,
        controls: Tensor, true_values: Tensor,
        meas_step: Tensor,
    ) -> Tensor:
        """The resource is the number
        of measurements.
        """
        # return resources+abs(control)
        return resources+1.0

# We test the particle filter with a
# simple model of an interferometer,
# being use to estimate a phase and
# having non-unitary visibility.
class PerfectInterferometer(Interferometer):
    """Single photon interferometer
    with unitary visibility.
    """
    def model(
        self, outcomes: Tensor, control: Tensor,
        parameters: Tensor, meas_step: Tensor,
        num_systems: int = 1,
        ) -> Tensor:
        """Probability for the photon
        two be detected in the first or
        the second port.
        """
        phase = parameters[:, :, 0]
        pd_outcomes = outcomes[:, :, 0]
        control_phase = control[:, :, 0]
        return (1.0+pd_outcomes*\
                cos(phase-control_phase))/2.0


class StatelessMetrologyModified(StatelessMetrology):

    def generate_input(
        self, weights: Tensor, particles: Tensor,
        meas_step: Tensor, used_resources: Tensor,
        rangen,
        ) -> Tuple[Tensor, Tensor]:

        pars = self.simpars
        return ones((self.bs, 10), dtype=pars.prec)


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--scratch-dir", type=str, required=True)
    parser.add_argument("--trained-models-dir", type=str,
                        default="./interferometer/trained_models")
    parser.add_argument("--data-dir", type=str,
                        default="./interferometer/data")
    parser.add_argument("--prec", type=str, default="float32")
    parser.add_argument("--n", type=int, default=64)
    parser.add_argument("--batchsize", type=int, default=2048)
    parser.add_argument("--num-particles", type=int, default=512)
    parser.add_argument("--num-steps", type=int, default=128)
    parser.add_argument("--iterations", type=int, default=256)
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

    interferometer = Interferometer(
        batchsize=args.batchsize,
        params=[
            Parameter(bounds=(0.0, pi), name="phase"),
            Parameter(bounds=(0.999, 1.00), name="visibility"),
        ],
        prec=args.prec
    )

    pf = ParticleFilter(
        num_particles=args.num_particles,
        phys_model=interferometer,
        prec=args.prec
    )

    simpars = SimulationParameters(
        sim_name="interferometer",
        num_steps=args.num_steps,
        max_resources=args.num_steps,
        cumulative_loss=True,
        prec=args.prec
        )

    sim_nn = StatelessMetrology(
        particle_filter=pf,
        simpars=simpars,
        phys_model=interferometer,
        control_strategy=lambda input: \
            floormod(network(input), 2*pi)
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
        sim_nn, 2048, args.data_dir,
        xla_compile=True,
        delta_resources=1,
        y_label='MSE',
    )

    store_input_control(
        sim_nn, args.data_dir, 5,
        xla_compile=True,
    )

    # Static strategy
    # ----------------------------------------

    #Initial value of the controls in
    # the static optimization
    initial_state = npzeros((int(args.num_steps), 1))
    for i in range(int(args.num_steps)):
        initial_state[i, :] = 1.0

    static_variables = Variable(
        initial_state, dtype=args.prec,
        )

    simpars = SimulationParameters(
        sim_name="interferometer_static",
        num_steps=args.num_steps,
        max_resources=args.num_steps,
        cumulative_loss=True,
        prec=args.prec
        )

    sim_static = StatelessMetrology(
        particle_filter=pf,
        simpars=simpars,
        phys_model=interferometer,
        control_strategy=lambda input_tensor:
            gather(
                params=static_variables,
                indices=cast(args.num_steps*\
                    input_tensor[:, -1], dtype="int32"),
            ),
    )

    decaying_learning_rate = InverseSqrtDecay(
        0.1, prec=args.prec,
    )

    train(
        sim_static,
        Adam(learning_rate=decaying_learning_rate),
        args.iterations,
        args.scratch_dir,
        custom_controls=static_variables,
        xla_compile=True,
    )

    performance_evaluation(
        sim_static, 2048, args.data_dir,
        xla_compile=True,
        delta_resources=1,
        y_label='MSE',
    )

    store_input_control(
        sim_static, args.data_dir, 5,
        xla_compile=True,
    )


if __name__ == "__main__":
    main()
