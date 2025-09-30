#!/usr/bin/env python3
from typing import Callable

from tensorflow import cast, ones, ones, \
    concat, zeros, gather, where, \
        Tensor, Variable
from tensorflow.math import floormod, equal, greater
from tensorflow.random import Generator
from tensorflow.keras.optimizers import Adam
from numpy import zeros as npzeros
from argparse import ArgumentParser
from math import pi

from qsensoropt import StatelessSimulation, \
    InverseSqrtDecay, ParticleFilter, Parameter, \
        SimulationParameters, PhysicalModel

from qsensoropt.utils import train, \
    performance_evaluation, store_input_control, \
        standard_model

from interferometer import PerfectInterferometer
from qsensoropt.utils import normalize

class StatelessTwoHypothesisTesting(StatelessSimulation):
    """Distinguished between two hypothesis in a stateless
    """
    def __init__(
            self, particle_filter: ParticleFilter,
            phys_model: PhysicalModel, 
            control_strategy: Callable, simpars: SimulationParameters,
            ):
        """Set the hyperparameters,
        """
        input_size = 4
        # Set the name of each column of the input
        input_name = ["Prob1", "Prob2"] + \
            ["StepOverMaxStep", "ResOverMaxRes", ]
        
        super().__init__(
            particle_filter, phys_model, 
            control_strategy, 
            input_size, input_name,
            simpars,
            )

        self.true_tensor = cast(
            ones((self.bs, 1), dtype=simpars.prec),
            dtype="bool",
            )
        self.false_tensor = cast(
            zeros((self.bs, 1), dtype=simpars.prec),
            dtype="bool",
            )
        self.ones_tensor = ones(
            (self.bs, 1), dtype=simpars.prec,
            )
        self.zeros_tensor = zeros(
            (self.bs, 1), dtype=simpars.prec,
            )
        
    def generate_input(
            self, weights: Tensor, particles: Tensor, 
            meas_step: Tensor, used_resources: Tensor, 
            rangen: Generator,
            ) -> Tensor:
        """Feed the probabilities of the
        two hypothesis and the
        number of resources and steps
        to the NN.
        """

        scaled_step = normalize(
            meas_step, [0, self.simpars.num_steps],
            )
        scaled_time = normalize(
            used_resources,
            [0, self.simpars.max_resources],
            )
        
        return concat(
            [weights, scaled_step, scaled_time,], 1,
            name="input_tensor",
        )


    def loss_function(
            self, weights: Tensor, particles: Tensor,
            true_values: Tensor, 
            used_resources: Tensor, meas_step: Tensor,
            ) -> Tensor:
        """Gives a penality if the guess is wrong.
        """
        is_first = equal(
            true_values[:, 0, :], particles[:, 0, :],
            )

        guess_first = where(
            greater(weights[:, 0:1], 0.5*self.ones_tensor),
            self.true_tensor, self.false_tensor,
            )
        guessed_right = where(
            equal(guess_first, is_first),
            self.true_tensor, self.false_tensor
        )
        return where(
            guessed_right, self.zeros_tensor,
            self.ones_tensor,
        )
        
        
def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--scratch-dir", type=str, required=True)
    parser.add_argument("--trained-models-dir",
                        type=str, default="./interferometer/trained_models")
    parser.add_argument("--data-dir", type=str,
                        default="./interferometer/data")
    parser.add_argument("--prec", type=str, default="float32")
    parser.add_argument("--n", type=int, default=64)
    parser.add_argument("--batchsize", type=int, default=2048)
    parser.add_argument("--num-steps", type=int, default=128)
    parser.add_argument("--iterations", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-2)
    parser.add_argument("--scatter-points", type=int, default=32)

    return parser.parse_args()


def main():
    args = parse_args()

    network = standard_model(
        input_size=4,
        controls_size=1,
        neurons_per_layer=args.n,
        prec=args.prec,
    )
    network.compile()

    interferometer = PerfectInterferometer(
        batchsize=args.batchsize,
        params=[
            Parameter(values=(pi/6, pi/3),
                      randomize="deterministic",
                      name="phase",
                      ),
        ],
        prec=args.prec
    )

    pf = ParticleFilter(
        num_particles=2,
        phys_model=interferometer,
        prec=args.prec
    )

    simpars = SimulationParameters(
        sim_name="interferometer_hypothesis",
        num_steps=args.num_steps,
        max_resources=args.num_steps,
        cumulative_loss=True,
        prec=args.prec
        )

    sim_nn = StatelessTwoHypothesisTesting(
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
        sim_nn, args.iterations, args.data_dir,
        xla_compile=True,
        delta_resources=1,
        y_label='ProbError',
    ) 

    store_input_control(
        sim_nn, args.data_dir, 5,
        xla_compile=True,
    )

    # Static strategy
    # ---------------------------------------------------------

    #Initial value of the controls
    # in the static optimization
    initial_state = npzeros(
        (int(args.num_steps), 1),
        )
    for i in range(int(args.num_steps)):
        initial_state[i, :] = 1.0

    static_variables = Variable(
        initial_state, dtype=args.prec,
        )

    simpars = SimulationParameters(
        sim_name="interferometer_hypothesis_static",
        num_steps=args.num_steps,
        max_resources=args.num_steps,
        cumulative_loss=True,
        prec=args.prec
        )

    sim_static = StatelessTwoHypothesisTesting(
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
        sim_static, args.iterations,
        args.data_dir,
        xla_compile=True,
        delta_resources=1,
        y_label='ProbError',
    )

    store_input_control(
        sim_static, args.data_dir, 5,
        xla_compile=True,
    )

if __name__ == "__main__":
    main()
