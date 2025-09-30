#!/usr/bin/env python3
from tensorflow.keras.optimizers import Adam
from argparse import ArgumentParser

from qsensoropt import InverseSqrtDecay, \
    Parameter, SimulationParameters, BoundSimulation
from qsensoropt.utils import train, \
    standard_model

from stateful_interferometer import DamageInterferometer

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--scratch-dir", type=str, required=True)
    parser.add_argument("--trained-models-dir", type=str,
                        default="./interferometer/trained_models")
    parser.add_argument("--data-dir", type=str,
                        default="./interferometer/data")
    parser.add_argument("--prec", type=str, default="float64")
    parser.add_argument("--n", type=int, default=64)
    parser.add_argument("--batchsize", type=int, default=512)
    parser.add_argument("--num-steps", type=int, default=12)
    parser.add_argument("--iterations", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--scatter-points", type=int, default=32)

    return parser.parse_args()

def main():
    args = parse_args()

    network = standard_model(
        input_size=6,
        controls_size=1,
        neurons_per_layer=args.n,
        prec=args.prec,
    )
    network.compile()

    interferometer = DamageInterferometer(
        batchsize=args.batchsize,
        params=[
            Parameter(values=(0.001, ), name="phase"),
            Parameter(values=(1.0, ), name="visibility"),
        ],
        prec=args.prec,
        max_damage=1.00,
        importance_sampling=True,
    )

    simpars = SimulationParameters(
        sim_name="bound_inter_damage",
        num_steps=args.num_steps,
        max_resources=args.num_steps,
        prec=args.prec,
    )

    sim_nn = BoundSimulation(
        simpars=simpars,
        phys_model=interferometer,
        control_strategy=network,
        cov_weight_matrix=[[1, 0], [0, 0]],
        importance_sampling=True,
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

    # print("Memory:")
    # print(get_memory_info('GPU:0')['peak']/1024**3)


if __name__ == "__main__":
    main()
