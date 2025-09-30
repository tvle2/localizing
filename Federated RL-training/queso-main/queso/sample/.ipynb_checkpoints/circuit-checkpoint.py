
from tensorflow import cast, stop_gradient, ones, expand_dims, Tensor, dtypes
from tensorflow.math import exp, log, cos, abs
from tensorflow.python.ops.ragged.ragged_array_ops import zeros
from tensorflow.random import stateless_uniform, Generator



import time
import os

import h5py
import argparse

import jax
import jax.numpy as jnp

from queso.io import IO
from queso.configs import Configuration
import numpy as np

# %%
def sample_circuit(
):

    jax.config.update("jax_default_device", jax.devices(os.getenv("DEFAULT_DEVICE_SAMPLE_CIRC", "cpu"))[0])


    phi_range = [0,3.14]
    n_phis = 100
    n_shots = 5000
    bs = 1024
    invT2 = 1/96
    tau = 10**(-3)*20


    def model(n_shots,outcomes, evolution_time,omega):

        exp_decay = exp(-evolution_time*invT2)#1#
        ot =  np.zeros((len(omega),n_shots))

        for j in range(len(omega)):
            ot[j,:] = cos(omega[j])#*evolution_time-3.14/2

        ramsey_out = outcomes
        print((ramsey_out*ot).shape)
        noise_less = (1.0-ramsey_out*ot)/2.0

        return exp_decay*noise_less + (1.0-exp_decay)/2.0

    def get_seed(
            random_generator: Generator,
    ) -> Tensor:

        return random_generator.uniform(
            [2, ], minval=0, maxval=dtypes.int32.max,
            dtype="int32", name="seed",
        )

    def perform_measurement(n_shots,
            controls, parameters,
           rangen: Generator = Generator.from_seed(0xdeadd0d0)
    ):

        list_plus = ones((len( parameters),n_shots), dtype="float32")
        prob_plus = model(n_shots,
            list_plus, controls, parameters
        )
        # Extraction of the actual outcomes
        seed = get_seed(rangen)
        outcomes = 2 * cast((stateless_uniform((len( parameters),n_shots),
                                               seed, dtype="float32") <
                             stop_gradient(abs(prob_plus))),
                            dtype="int8") - 1
        outcomes = cast(
            outcomes,
            dtype="float32", name="outcomes",
        )
        prob_outcomes = model(n_shots,outcomes, controls, parameters)
        log_prob = cast(
            log(prob_outcomes),
            dtype="float32", name="log_prob",
        )
        return outcomes, log_prob



    # %% training data set
    print(
        f"Sampling {n_shots} shots for {n_phis} phase value between {phi_range[0]} and {phi_range[1]}."
    )
    phis = (phi_range[1] - phi_range[0]) * jnp.arange(n_phis) / (
        n_phis - 1
    ) + phi_range[0]
    t0 = time.time()
    evolution_time = 3.14/25#(2**5) * tau#** jnp.arange(10)#1#3.14/20

    shots, probs = perform_measurement(n_shots,evolution_time,phis)
    t1 = time.time()
    print(f"Sampling took {t1 - t0} seconds.")

    # %%
    outcomes = shots




    # # %%
    hf = h5py.File(io.path.joinpath("train_samples.h5"), "w")
    hf.create_dataset("probs", data=probs)
    hf.create_dataset("shots", data=shots)
    hf.create_dataset("phis", data=phis)
    hf.close()

    print(f"Finished sampling the circuits.")
    return


# %%
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", type=str, default="tmp")
    args = parser.parse_args()
    folder = args.folder

    io = IO(folder=f"{folder}")
    print(io)

    print(f"Sampling circuit: {folder} | Devices {jax.devices()} | Full path {io.path}")
    sample_circuit()
