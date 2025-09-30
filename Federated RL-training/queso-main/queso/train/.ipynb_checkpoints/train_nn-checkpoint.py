# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Copyright (c) 2022-2024 Benjamin MacLellan

import time
import os
import tqdm
import matplotlib.pyplot as plt

import pandas as pd
import h5py
import argparse
import warnings

import jax
import jax.numpy as jnp
import optax

from flax.training import train_state, orbax_utils
from orbax.checkpoint import (
    Checkpointer,

    PyTreeCheckpointHandler,
)

from queso.estimators.flax.dnn import BayesianDNNEstimator
from queso.io import IO
from queso.configs import Configuration
from queso.utils import get_machine_info


# def train_nn(
#     io: IO,
#     config: Configuration,
#     key: jax.random.PRNGKey,
#     plot: bool = False,
#     progress: bool = True,
# ):
#     """
#     Trains a neural network based on the provided configuration.

#     This function initializes a BayesianDNNEstimator, sets up the optimizer and loss function, and then performs the training steps.
#     It also saves the training results and metadata, and optionally plots the training progress.

#     Args:
#         io (IO): An instance of the IO class for handling input/output operations.
#         config (Configuration): An instance of the Configuration class containing the settings for the training.
#         key (jax.random.PRNGKey): A random number generator key from JAX.
#         plot (bool, optional): If True, plots of the training progress are generated and saved. Defaults to False.
#         progress (bool, optional): If True, a progress bar is displayed during training. Defaults to True.

#     Returns:
#         None

#     Raises:
#         Warning: If the grid and training data do not match.
#     """
#     jax.config.update("jax_default_device", jax.devices(os.getenv("DEFAULT_DEVICE_TRAIN_NN", "cpu"))[0])

#     # %%
#     nn_dims = config.nn_dims + [config.n_grid]
#     n_grid = config.n_grid
#     lr = config.lr_nn
#     l2_regularization = config.l2_regularization
#     n_epochs = config.n_epochs
#     batch_size = config.batch_size
#     from_checkpoint = config.from_checkpoint
#     logit_norm = False

#     # %% extract data from H5 file
#     t0 = time.time()

#     hf = h5py.File(io.path.joinpath("train_samples.h5"), "r")
#     shots = jnp.array(hf.get("shots"))

#     probs = jnp.array(hf.get("probs"))
#     phis = jnp.array(hf.get("phis"))
#     hf.close()

#     # %%
#     n_shots = shots.shape[1]
#     n_phis = shots.shape[0]

#     # %%
#     assert n_shots % batch_size == 0
#     n_batches = n_shots // batch_size
#     n_steps = n_epochs * n_batches



#     # %%
#     dphi = phis[1] - phis[0]
#     phi_range = (jnp.min(phis), jnp.max(phis))

#     grid = (phi_range[1] - phi_range[0]) * jnp.arange(n_grid) / (
#         n_grid - 1
#     ) + phi_range[0]
#     index = jnp.stack([jnp.argmin(jnp.abs(grid - phi)) for phi in phis])



#     labels = jax.nn.one_hot(index, num_classes=n_grid)

#     print(index)
#     print(labels.sum(axis=0))

#     # %%
#     model = BayesianDNNEstimator(nn_dims)
    
#     # Changed
#     xi = shots
#     m  = xi.mean(axis=1, keepdims=True)      # (n_phis, 1)  ~  E[±1] = 2p-1
#     xi = jnp.tile(m, (1, xi.shape[1]))       # (n_phis, n_shots)
#     x  = jnp.expand_dims(xi, axis=-1)        # (n_phis, n_shots, 1)  ← shapes unchanged
    
#     y = labels

#     # mu, sig = 0.0, 0.05
#     # g = (1/sig/jnp.sqrt(2 * jnp.pi)) * jnp.exp(- (jnp.linspace(-1, 1, n_grid) - mu) ** 2 / 2 / sig**2)
#     # yg = jnp.fft.ifft(jnp.fft.fft(y, axis=1) * jnp.fft.fft(jnp.fft.fftshift(g)), axis=1).real
#     # fig, ax = plt.subplots()
#     # sns.heatmap(yg, ax=ax)
#     # plt.show()

#     # %%

#     print("nndim",x.shape)
#     x_init = x[1:10, 1:10,:]
#     print(model.tabulate(jax.random.PRNGKey(0), x_init))

#     # %%
#     def l2_loss(w, alpha):
#         return alpha * (w**2).mean()

#     @jax.jit
#     def train_step(state, batch):
#         x_batch, y_batch = batch

#         def loss_fn(params):
#             logits = state.apply_fn({"params": params}, x_batch)
#             # loss = optax.softmax_cross_entropy(
#             #     logits,
#             #     y_batch
#             # ).mean(axis=(0, 1))

#             if logit_norm:
#                 eps = 1e-10
#                 tau = 10.0
#                 logits = (
#                     (logits + eps)
#                     / (jnp.sqrt((logits**2 + eps).sum(axis=-1, keepdims=True)))
#                     / tau
#                 )

#             # standard cross-entropy
#             print("softmax", jax.nn.log_softmax(logits, axis=-1).shape)
#             loss = -jnp.sum(
#                 y_batch[:, None, :] * jax.nn.log_softmax(logits, axis=-1), axis=-1
#             ).mean(axis=(0, 1))

#             # cross-entropy with ReLUmax instead of softmax
#             # log_relumax = jnp.log(jax.nn.relu(logits) / jnp.sum(jax.nn.relu(logits), axis=-1, keepdims=True))
#             # print('relumax', log_relumax)
#             # loss = -jnp.sum(y_batch[:, None, :] * log_relumax, axis=-1).mean(axis=(0, 1))

#             # MSE loss
#             # loss = jnp.sum((y_batch[:, None, :] - jax.nn.softmax(logits, axis=-1))**2, axis=-1).mean(axis=(0, 1))

#             # CE loss + convolution w/ Gaussian
#             # loss = -jnp.sum(yg[:, None, :] * jax.nn.log_softmax(logits, axis=-1), axis=-1).mean(axis=(0, 1))

#             loss += sum(
#                 l2_loss(w, alpha=l2_regularization) for w in jax.tree_leaves(params)
#             )
#             return loss

#         loss_val_grad_fn = jax.value_and_grad(loss_fn)
#         loss, grads = loss_val_grad_fn(state.params)
#         print('Grads--',grads,'------0end')
#         state = state.apply_gradients(grads=grads)
#         return state, loss

#     # %%
#     def create_train_state(model, init_key, x, learning_rate):

#         if from_checkpoint:
#             ckpt_dir = io.path.joinpath("ckpts")
#             ckptr = Checkpointer(
#                 PyTreeCheckpointHandler()
#             )  # A stateless object, can be created on the fly.
#             restored = ckptr.restore(ckpt_dir, item=None)
#             params = restored["params"]
#             print(f"Loading parameters from checkpoint: {ckpt_dir}")
#         else:
#             params = model.init(init_key, x)["params"]
#             print(f"Random initialization of parameters")

#         # print("Initial parameters", params)
#         # schedule = optax.constant_schedule(lr)
#         schedule = optax.polynomial_schedule(
#             init_value=lr,
#             end_value=lr**2,
#             power=1,
#             transition_steps=n_steps // 4,
#             transition_begin=3 * n_steps // 2,
#         )
#         tx = optax.adam(learning_rate=schedule)
#         # tx = optax.adamw(learning_rate=learning_rate, weight_decay=1e-5)

#         state = train_state.TrainState.create(
#             apply_fn=model.apply, params=params, tx=tx
#         )
#         return state

#     init_key = jax.random.PRNGKey(time.time_ns())
#     state = create_train_state(model, init_key, x_init, learning_rate=lr)
#     # del init_key

#     # %%
#     x_batch = x[:, 0:batch_size,:]
#     y_batch = y
#     batch = (x_batch, y_batch)

#     state, loss = train_step(state, batch)

#     # %%
#     keys = jax.random.split(key, (n_epochs))
#     metrics = []
#     pbar = tqdm.tqdm(
#         total=n_epochs, disable=(not progress), mininterval=0.333
#     )
#     for i in range(n_epochs):
#         # shuffle shots
#         # subkeys = jax.random.split(keys[i], n_phis)
#         # x = jnp.stack([jax.random.permutation(subkey, x[k, :, :]) for k, subkey in enumerate(subkeys)])

#         for j in range(n_batches):
#             x_batch = x[:, j * batch_size : (j + 1) * batch_size,:]
#             y_batch = y  # use all phases each batch, but not all shots per phase
#             batch = (x_batch, y_batch)
#             state, loss = train_step(state, batch)
#             if progress:
#                 pbar.update()
#                 pbar.set_description(
#                     f"Epoch {i} | Batch {j:04d} | Loss: {loss:.10f}", refresh=False
#                 )
#             metrics.append(dict(step=i * n_batches + j, loss=loss))

#     pbar.close()
#     metrics = pd.DataFrame(metrics)

#     # %%
#     hf = h5py.File(io.path.joinpath("nn.h5"), "w")
#     hf.create_dataset("grid", data=grid)
#     hf.close()



#     # %% save to disk
#     metadata = dict(nn_dims=nn_dims, lr=lr, time=time.time() - t0)
#     io.save_json(metadata, filename="nn-metadata.json")
#     io.save_csv(metrics, filename="metrics")

#     # %%
#     info = get_machine_info()
#     io.save_json(info, filename="machine-info.json")


#     # %%
#     ckpt = {"params": state.params, "nn_dims": nn_dims}
#     ckpt_dir = io.path.joinpath("ckpts")

#     ckptr = Checkpointer(
#         PyTreeCheckpointHandler()
#     )  # A stateless object, can be created on the fly.
#     ckptr.save(
#         ckpt_dir, ckpt, save_args=orbax_utils.save_args_from_target(ckpt), force=True
#     )
#     restored = ckptr.restore(ckpt_dir, item=None)

#     params = restored["params"]

#     # Print or inspect the parameters to verify
#     #print(params)

#     print(f"Finished training the estimator.")

#     # %%
#     if plot:
#         # %% plot prior
#         # fig, ax = plt.subplots()
#         # ax.stem(prior)
#         # fig.show()
#         # io.save_figure(fig, filename="prior.png")

#         # %% plot NN loss minimization
#         fig, ax = plt.subplots()
#         ax.plot(metrics.step, metrics.loss)
#         ax.set(xlabel="Optimization step", ylabel="Loss")
#         fig.show()
#         io.save_figure(fig, filename="nn-loss.png")


#         plt.show()

def train_nn(
    io,
    config,                   # your Configuration dataclass
    key,                      # jax.random.PRNGKey
    plot: bool = False,
    progress: bool = True,
):
    """
    Paper-aligned BNN training:
    - Treat 'grid' (or fallback 'phis') as the ω grid (classes).
    - Keep per-shot inputs x: shape (P, S, 1) with values in {-1,+1}.
    - Build soft labels q(r|ω) from the physics model with F0/F1/T2 and fixed τ.
    - Optimize soft cross-entropy + L2.
    - Save nn.h5 'grid' and a checkpoint with params and nn_dims, fully compatible with eval().
    """
    import os, time, tqdm, pandas as pd, h5py
    import jax, jax.numpy as jnp
    import optax
    from flax.training import train_state
    from orbax.checkpoint import Checkpointer, PyTreeCheckpointHandler, utils as orbax_utils

    # Device selection (same as original)
    jax.config.update("jax_default_device", jax.devices(os.getenv("DEFAULT_DEVICE_TRAIN_NN", "cpu"))[0])

    # NN dimensions and hyper-params
    nn_dims = config.nn_dims + [config.n_grid]
    n_grid = int(config.n_grid)
    lr = float(config.lr_nn)
    l2_regularization = float(config.l2_regularization)
    n_epochs = int(config.n_epochs)
    batch_size = int(config.batch_size)
    from_checkpoint = getattr(config, "from_checkpoint", False)
    logit_norm = False

    t0 = time.time()

    # -------- Load training data --------
    hf = h5py.File(io.path.joinpath("train_samples.h5"), "r")
    shots = jnp.array(hf["shots"])                                   # (P,S) in {-1,+1}
    if "grid" in hf.keys():
        omega_grid = jnp.array(hf["grid"])                           # (P,)
    else:
        omega_grid = jnp.array(hf["phis"])                           # (P,)  (compat)
    tau_attr = hf.attrs.get("tau", None)
    F0 = float(hf.attrs.get("F0", 0.88))
    F1 = float(hf.attrs.get("F1", 0.95))
    T2 = float(hf.attrs.get("T2", 1500.0))                           # your requested default
    hf.close()

    P = int(omega_grid.shape[0])         # number of ω bins (classes)
    S = int(shots.shape[1])              # number of shots per bin

    assert S % batch_size == 0, "batch_size must divide n_shots"
    n_batches = S // batch_size
    n_steps   = n_epochs * n_batches

    # DO NOT rebuild or re-center the grid; use the file's ω grid as-is
    grid = omega_grid  # (B==P,) class centers

    # -------- Build inputs --------
    # Keep per-shot inputs; expand to (P, S, 1)
    x = jnp.expand_dims(shots, axis=-1)   # (P, S, 1)

    # -------- Physics-based soft labels q(r|ω) --------
    # τ from H5 attrs if present, else τ=π/ω_max (ω_max≈grid[-1])
    tau_bnn = float(tau_attr) if (tau_attr is not None) else float(jnp.pi / float(grid[-1]))
    phi0 = 0.0

    a = 0.5 * (1.0 + F0 - F1)
    b = 0.5 * (F0 + F1 - 1.0)
    decay = jnp.exp(-tau_bnn / T2)                                   # scalar

    # p(0 | ω) for each class in the grid
    p0_grid = a + b * decay * jnp.cos(grid * tau_bnn + phi0)         # (B,)
    p0_grid = jnp.clip(p0_grid, 1e-6, 1.0 - 1e-6)

    # Map r∈{-1,+1} to likelihood over classes: q ∝ P(r | ω)
    # If r=+1 (we observed "0"), use p0_grid; else use (1 - p0_grid).
    r_all = jnp.squeeze(x, axis=-1)                                  # (P,S)
    P_lik = jnp.where(r_all[..., None] > 0,                          # broadcast over classes
                      p0_grid[None, None, :],
                      1.0 - p0_grid[None, None, :])                  # (P,S,B)
    q_all = P_lik / jnp.clip(P_lik.sum(axis=-1, keepdims=True), 1e-12, None)  # normalize over classes

    # -------- Model --------
    model = BayesianDNNEstimator(tuple(nn_dims))

    # small dummy for tabulate / shape check
    x_init = x[1:10, 1:10, :]  # (≈9, ≈9, 1)
    try:
        print(model.tabulate(jax.random.PRNGKey(0), x_init))
    except Exception as _:
        pass

    def l2_loss(w, alpha):
        return alpha * (w**2).mean()

    @jax.jit
    def train_step(state, batch):
        x_batch, q_batch = batch  # x: (P,batch,1), q: (P,batch,B)

        def loss_fn(params):
            logits = state.apply_fn({"params": params}, x_batch)     # (P,batch,B)

            if logit_norm:
                eps = 1e-10
                tau = 10.0
                logits = ((logits + eps) /
                          (jnp.sqrt((logits**2 + eps).sum(axis=-1, keepdims=True))) / tau)

            logp = jax.nn.log_softmax(logits, axis=-1)              # (P,batch,B)
            ce  = -(q_batch * logp).sum(axis=-1).mean(axis=(0, 1))  # soft CE over classes
            reg = sum(l2_loss(w, alpha=l2_regularization) for w in jax.tree_leaves(params))
            return ce + reg

        loss_val_grad_fn = jax.value_and_grad(loss_fn)
        loss, grads = loss_val_grad_fn(state.params)
        state = state.apply_gradients(grads=grads)
        return state, loss

    # -------- Create train state --------
    def create_train_state(model, init_key, x, learning_rate):
        if from_checkpoint:
            ckpt_dir = io.path.joinpath("ckpts")
            ckptr = Checkpointer(PyTreeCheckpointHandler())
            restored = ckptr.restore(ckpt_dir, item=None)
            params = restored["params"]
            print(f"Loading parameters from checkpoint: {ckpt_dir}")
        else:
            params = model.init(init_key, x)["params"]
            print("Random initialization of parameters")

        schedule = optax.polynomial_schedule(
            init_value=lr,
            end_value=lr**2,
            power=1,
            transition_steps=max(1, n_steps // 4),
            transition_begin=3 * n_steps // 2,
        )
        tx = optax.adam(learning_rate=schedule)
        return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)

    init_key = jax.random.PRNGKey(time.time_ns())
    state = create_train_state(model, init_key, x_init, learning_rate=lr)

    # -------- Warm-up one step --------
    x_batch = x[:, 0:batch_size, :]     # (P, batch, 1)
    q_batch = q_all[:, 0:batch_size, :] # (P, batch, B)
    state, loss = train_step(state, (x_batch, q_batch))

    # -------- Training loop --------
    keys = jax.random.split(key, (n_epochs,))
    metrics = []
    pbar = tqdm.tqdm(total=n_epochs, disable=(not progress), mininterval=0.333)
    for i in range(n_epochs):
        for j in range(n_batches):
            sl = slice(j * batch_size, (j + 1) * batch_size)
            x_batch = x[:, sl, :]
            q_batch = q_all[:, sl, :]
            state, loss = train_step(state, (x_batch, q_batch))
        if progress:
            pbar.update()
            pbar.set_description(f"Epoch {i} | Loss: {float(loss):.6f}", refresh=False)
        metrics.append(dict(step=i, loss=float(loss)))
    pbar.close()
    metrics = pd.DataFrame(metrics)

    # -------- Save ω grid for eval decoding --------
    with h5py.File(io.path.joinpath("nn.h5"), "w") as hf:
        hf.create_dataset("grid", data=np.array(grid))

    # -------- Save training artifacts --------
    metadata = dict(nn_dims=list(nn_dims), lr=lr, time=time.time() - t0)
    io.save_json(metadata, filename="nn-metadata.json")
    io.save_csv(metrics, filename="metrics")
    from platform import node, system, processor
    io.save_json({"host": node(), "os": system(), "cpu": processor()}, filename="machine-info.json")

    ckpt = {"params": state.params, "nn_dims": list(nn_dims)}
    ckpt_dir = io.path.joinpath("ckpts")
    ckptr = Checkpointer(PyTreeCheckpointHandler())
    ckptr.save(ckpt_dir, ckpt, save_args=orbax_utils.save_args_from_target(ckpt), force=True)
    _ = ckptr.restore(ckpt_dir, item=None)

    print("Finished training the estimator.")

    if plot:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.plot(metrics.step, metrics.loss)
        ax.set(xlabel="Epoch", ylabel="Loss")
        io.save_figure(fig, filename="nn-loss.png")
        plt.show()


# %%
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", type=str, default="tmp")
    args = parser.parse_args()
    folder = args.folder

    io = IO(folder=f"{folder}")
    print(io)
    config = Configuration.from_yaml(io.path.joinpath("config.yaml"))
    key = jax.random.PRNGKey(config.seed)
    print(f"Training NN: {folder} | Devices {jax.devices()} | Full path {io.path}")
    print(f"Config: {config}")
    train_nn(io, config, key, progress=True, plot=True)
