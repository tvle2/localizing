
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
# def sample_circuit(
# ):

#     jax.config.update("jax_default_device", jax.devices(os.getenv("DEFAULT_DEVICE_SAMPLE_CIRC", "cpu"))[0])


#     phi_range = [0,3.14]
#     n_phis = 100
#     n_shots = 5000
#     bs = 1024
#     invT2 = 1/96
#     tau = 10**(-3)*20


#     def model(n_shots,outcomes, evolution_time,omega):

#         exp_decay = exp(-evolution_time*invT2)#1#
#         ot =  np.zeros((len(omega),n_shots))

#         for j in range(len(omega)):
#             ot[j,:] = cos(omega[j])#*evolution_time-3.14/2

#         ramsey_out = outcomes
#         print((ramsey_out*ot).shape)
#         noise_less = (1.0-ramsey_out*ot)/2.0

#         return exp_decay*noise_less + (1.0-exp_decay)/2.0

#     def get_seed(
#             random_generator: Generator,
#     ) -> Tensor:

#         return random_generator.uniform(
#             [2, ], minval=0, maxval=dtypes.int32.max,
#             dtype="int32", name="seed",
#         )

#     def perform_measurement(n_shots,
#             controls, parameters,
#            rangen: Generator = Generator.from_seed(0xdeadd0d0)
#     ):

#         list_plus = ones((len( parameters),n_shots), dtype="float32")
#         prob_plus = model(n_shots,
#             list_plus, controls, parameters
#         )
#         # Extraction of the actual outcomes
#         seed = get_seed(rangen)
#         outcomes = 2 * cast((stateless_uniform((len( parameters),n_shots),
#                                                seed, dtype="float32") <
#                              stop_gradient(abs(prob_plus))),
#                             dtype="int8") - 1
#         outcomes = cast(
#             outcomes,
#             dtype="float32", name="outcomes",
#         )
#         prob_outcomes = model(n_shots,outcomes, controls, parameters)
#         log_prob = cast(
#             log(prob_outcomes),
#             dtype="float32", name="log_prob",
#         )
#         return outcomes, log_prob



#     # %% training data set
#     print(
#         f"Sampling {n_shots} shots for {n_phis} phase value between {phi_range[0]} and {phi_range[1]}."
#     )
#     phis = (phi_range[1] - phi_range[0]) * jnp.arange(n_phis) / (
#         n_phis - 1
#     ) + phi_range[0]
#     t0 = time.time()
#     evolution_time = 3.14/25#(2**5) * tau#** jnp.arange(10)#1#3.14/20

#     shots, probs = perform_measurement(n_shots,evolution_time,phis)
#     t1 = time.time()
#     print(f"Sampling took {t1 - t0} seconds.")

#     # %%
#     outcomes = shots




#     # # %%
#     hf = h5py.File(io.path.joinpath("train_samples.h5"), "w")
#     hf.create_dataset("probs", data=probs)
#     hf.create_dataset("shots", data=shots)
#     hf.create_dataset("phis", data=phis)
#     hf.close()

#     print(f"Finished sampling the circuits.")
#     return

def sample_circuit(
    io,                           # your IO helper
    omega_max: float = 10.0,      # MHz
    T2: float = 1500.0,           # same unit as τ (here φ is unitless and τ is a scalar)
    F0: float = 0.88,             # readout fidelity for |0>
    F1: float = 0.95,             # readout fidelity for |1>
    n_phis: int = 100,            # NOTE: in the original code this is actually the number of ω bins
    n_shots: int = 5000,
    seed: int = 1234,
):
    """
    Paper-aligned generator for BNN stage-1 training data.

    - Discretize ω into `n_phis` bins over [0, omega_max] (endpoint=True).
    - Use fixed τ = π / ω_max and φ = 0.
    - Single-shot probability with readout fidelity and T2:
        p(0 | ω) = a + b * exp(-τ/T2) * cos(ω * τ + 0),
      where a=(1+F0-F1)/2, b=(F0+F1-1)/2.
    - Save:
        shots: (P, S) in {-1, +1}
        grid : (P,)  == ω grid (MHz)
        phis : (P,)  (kept for backward-compat; same as grid)
        probs: (P,1)  theoretical p(0 | ω) for reference (not used by training)
      and HDF5 attrs: tau, F0, F1, T2 for eval/training consistency.
    """
    import jax, jax.numpy as jnp, numpy as np, h5py

    # Discrete ω grid (P bins)
    omega_grid = jnp.linspace(0.0, omega_max, n_phis, endpoint=True, dtype=jnp.float32)  # (P,)
    tau = jnp.pi / float(omega_max)   # fixed τ
    phi0 = 0.0

    # Fidelity parameters
    a = 0.5 * (1.0 + F0 - F1)
    b = 0.5 * (F0 + F1 - 1.0)
    decay = jnp.exp(-tau / float(T2))

    # Theoretical single-shot probability for measuring "0" given ω
    p0_grid = a + b * decay * jnp.cos(omega_grid * tau + phi0)   # (P,)
    p0_grid = jnp.clip(p0_grid, 1e-6, 1.0 - 1e-6)                # numeric safety

    # Sample S shots per ω bin; map {0,1} -> {-1,+1}
    key = jax.random.PRNGKey(int(seed))
    key, kdraw = jax.random.split(key)
    u = jax.random.uniform(kdraw, (n_phis, n_shots), dtype=jnp.float32)  # (P,S)
    outcomes01 = (u < p0_grid[:, None]).astype(jnp.int8)                 # (P,S)
    shots_pm = (2 * outcomes01 - 1).astype(jnp.float32)                  # (P,S)

    # Write H5 (names chosen to keep backward-compat with original code)
    with h5py.File(io.path.joinpath("train_samples.h5"), "w") as hf:
        hf.create_dataset("shots", data=np.array(shots_pm))              # (P,S)
        hf.create_dataset("grid",  data=np.array(omega_grid))            # (P,)
        hf.create_dataset("phis",  data=np.array(omega_grid))            # compat alias
        hf.create_dataset("probs", data=np.array(p0_grid[:, None]))      # (P,1) for reference
        hf.attrs["tau"] = float(tau)
        hf.attrs["F0"]  = float(F0)
        hf.attrs["F1"]  = float(F1)
        hf.attrs["T2"]  = float(T2)

    print(f"[BNN data] Saved {io.path/'train_samples.h5'} "
          f"with ω∈[0,{omega_max}] (P={n_phis}), τ=π/{omega_max}, T2={T2}, F0={F0}, F1={F1}")


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
