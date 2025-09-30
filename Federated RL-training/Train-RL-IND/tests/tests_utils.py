#!/usr/bin/env python3
from tensorflow import constant, identity
from tensorflow.test import TestCase, main
from tensorflow.math import is_nan, reduce_any
from tensorflow.keras.optimizers import Adam
from numpy import array
from os.path import join
from math import pi

from qsensoropt import InverseSqrtDecay, \
    ParticleFilter, Parameter, SimulationParameters
from qsensoropt.utils import train, \
    performance_evaluation, store_input_control, \
        sqrt_hmatrix, train_nn_graph, \
            train_nn_profiler, store_input_control, \
                standard_model, normalize, denormalize

from interferometer import Interferometer, \
    StatelessMetrologyModified

class UtilsTest(TestCase):
    """Tests for the utils
    """
    def setUp(self):
        self.interferometer = Interferometer(
            batchsize=3,
            params=[
                Parameter(bounds=(0.0, 2*pi),
                          name="phase"),
                Parameter(bounds=(0.95, 1.00),
                          name="visibility"),
            ],
        )

        self.pf = ParticleFilter(
            num_particles=5,
            phys_model=self.interferometer,
            resample_fraction=0.7,
            alpha=0.5,
            beta=0.9,
            gamma=0.8,
        )

        self.network = standard_model(
            input_size=10, controls_size=1,
            neurons_per_layer=4,
            )
        
        self.network.compile()

        simpars = SimulationParameters(
            sim_name="interferometer",
            num_steps=10,
            max_resources=10,
            cumulative_loss=True,
        )

        self.sim_nn = StatelessMetrologyModified(
            particle_filter=self.pf,
            simpars=simpars,
            phys_model=self.interferometer,
            control_strategy=self.network,
        )

    def test_train(self):
        """Test of the NN training.

        This is an integration test.
        We check that the NN weights are updated
        after 10 steps of the simulation.
        That is we check
        that the final loss is connected
        to the NN variables.
        """
        scratch_dir = self.get_temp_dir()
        trained_models_dir = self.get_temp_dir()
        data_dir = self.get_temp_dir()

        old_weights = self.network.trainable_variables

        old_weights_constant = [identity(layer_old)
                                for layer_old in old_weights]

        decaying_learning_rate = InverseSqrtDecay(
            1.0e-4, prec="float64",
        )

        train(
            self.sim_nn,
            Adam(learning_rate=decaying_learning_rate), 10,
            scratch_dir, network=self.network,
            gradient_accumulation=3,
            interval_save=1,
            xla_compile=True,
        )

        self.network.save(
            join(trained_models_dir, str(self.sim_nn)),
            )

        performance_evaluation(
            self.sim_nn, 10, data_dir,
            xla_compile=True,
            precision_fit={'num_points': 20,
                           'batchsize': 4,
                           'epochs': 1,
                           'direct_func': lambda _, prec: prec,
                           'inverse_func': lambda _, c: c},
        )

        new_weights = self.network.trainable_variables

        new_weights_constant = [identity(layer_new)
                                for layer_new in new_weights]

        for old_layer, new_layer in \
            zip(old_weights_constant, new_weights_constant):

            self.assertNotAllEqual(
                old_layer,
                new_layer,
                msg="Simulation: the NN is not training properly",
            )

            self.assertEqual(
                reduce_any(is_nan(new_layer)), constant(False),
                msg="Simulation: the NN training produces nan values"
            )

    def test_train_nn_graph(self):

        data_dir = self.get_temp_dir()

        train_nn_graph(
            self.sim_nn,
            Adam(learning_rate=1e-4),
            data_dir, network=self.network,
        )

    def test_train_nn_profiler(self):

        data_dir = self.get_temp_dir()

        train_nn_profiler(
            self.sim_nn,
            Adam(learning_rate=1e-4),
            data_dir, network=self.network,
        )

    def test_store_input_control(self):

        data_dir = self.get_temp_dir()

        store_input_control(
            self.sim_nn, data_dir, 5,
        )

    def test_sqrt_hmatrix(self):

        A = constant(array(
            [[0.8, 0.2], [0.2, 0.8]]),
            dtype="float64",
        )

        B = sqrt_hmatrix(A)

        expected_B = constant(array(
            [[0.88729833, 0.11270167],
             [0.11270167, 0.88729833]]),
            dtype="float64",
        )

        self.assertAllClose(
            B,
            expected_B,
            atol=1e-8,
            rtol=0.0,
            msg="Utils: sqrt_hmatrix gives the wrong result",
        )

    def test_normalize(self):
        
        to_normalize = constant(array(
            [[10, 5], [2, 3]]),
            dtype="float64",
        )

        normalized_expected = constant(array(
            [[1.0, 0.0], [-0.6, -0.4]]),
            dtype="float64",
        )

        normalized = normalize(to_normalize, (0, 10.0))

        self.assertAllClose(
            normalized_expected,
            normalized,
            atol=1e-10,
            rtol=0.0,
            msg="Utils: normalize gives the wrong result",
        )

    def test_denormalize(self):

        normalized = constant(array(
            [[1.0, 0.0], [-0.6, -0.4]]),
            dtype="float64",
        )

        denormalized_expected = constant(array(
            [[10, 5], [2, 3]]),
            dtype="float64",
        )

        denormalized = denormalize(normalized, (0, 10.0))

        self.assertAllClose(
            denormalized_expected,
            denormalized,
            atol=1e-10,
            rtol=0.0,
            msg="Utils: normalize gives the wrong result",
        )

    def test_standard_model(self):

        network = standard_model(
            input_size=4,
            controls_size=3,
            neurons_per_layer=48,
            num_mid_layers=8,
            prec="float32",
        )

        self.assertEqual(
            len(network.layers), 9,
            msg="Utils: standard_model creates the wrong number of layers"
        )

        self.assertEqual(
            network.layers[0].input_shape[1], 4,
            msg="Utils: standard_model creates the wrong input shape"
        )

        self.assertEqual(
            network.layers[0].units, 48,
            msg="Utils: standard_model creates the wrong number of neurons"
        )

        self.assertEqual(
            network.layers[1].units, 48,
            msg="Utils: standard_model creates the wrong number of neurons"
        )

        self.assertEqual(
            network.layers[-1].units, 3,
            msg="Utils: standard_model creates the wrong number of neurons"
        )

        self.assertEqual(
            network.layers[-1].dtype, "float32",
            msg="Utils: standard_model creates model with wrong type"
        )

        self.assertEqual(
            network.layers[0].dtype, "float32",
            msg="Utils: standard_model creates model with wrong type"
        )

        self.assertEqual(
            network.layers[1].dtype, "float32",
            msg="Utils: standard_model creates model with wrong type"
        )


if __name__ == "__main__":
    main()
