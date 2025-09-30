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

from __future__ import annotations

import time
from dataclasses import dataclass, field, fields, asdict
import networkx as nx
import yaml
import pathlib
import jax.numpy as jnp

# %%
@dataclass
class Configuration:
    # folder: str = "tmp"
    seed: int = None
    train_circuit: bool = False
    sample_circuit: bool = False
    sample_circuit_training_data: bool = False
    sample_circuit_testing_data: bool = False
    train_nn: bool = False
    benchmark_estimator: bool = False

    # circuit args
    n: int = 1
    k: int = 2

    preparation: str = "hardware_efficient_ansatz"
    interaction: str = "local_rx"
    detection: str = "local_r"
    loss_fi: str = "loss_cfi"
    backend: str = "ket"

    # optional circuit args
    gamma_dephasing: float = None
    n_ancilla: int = None

    # training circuit args
    n_phis: int = 100
    n_steps: int = 20000
    lr_circ: float = 1e-3
    metrics: list[str] = field(default_factory=lambda: ['entropy_vn', 'qfi'])
    phi_fi: float = 0.0
    phi_center: float = 0.0
    phi_range: list[float] = field(default_factory=lambda: [0, 3.14])

    # sample circuit args
    n_shots: int = 5000
    n_shots_test: int = 1000
    phis_test: list = field(default_factory=lambda: [0.1,0.9,1.02,2.147, 3.0])

    # train estimator args
    n_epochs: int = 100
    batch_size: int = 50
    n_grid: int = (
        100  # todo: make more general - not requiring matching training phis and grid
    )
    nn_dims: list[int] = field(default_factory=lambda: [32, 32])
    lr_nn: float = 1e-3
    l2_regularization: float = 0.0  # L2 regularization for NN estimator
    from_checkpoint: bool = False

    # benchmark estimator args
    n_trials: int = 100
    phis_inds: list[int] = field(default_factory=lambda: [50])
    n_sequences: list[int] = field(default_factory=lambda: [1, 10, 100, 1000])

    @classmethod
    def from_yaml(cls, file):

        with open(file, "r") as fid:
            data = yaml.safe_load(fid)
        return cls(**data)

    def __post_init__(self):
        if self.seed is None:
            self.seed = time.time_ns()

        # if self.n_grid != self.n_phis:
        # raise Warning("should be the same")

        # convert all lists to jax.numpy arrays
        # for field in fields(self.__class__):
        #     val = getattr(self, field.name)
        #     if isinstance(val, list):
        #         val = jnp.array(val)
        #         setattr(self, field.name, val)


