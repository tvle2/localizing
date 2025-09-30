#!/usr/bin/env python3
from tensorflow.test import TestCase, main
from tensorflow import constant, zeros, ones, reduce_sum, \
    concat, TensorShape
from tensorflow.random import Generator
from tensorflow.math import is_nan, reduce_any
from numpy import array
from math import sqrt, pi

from qsensoropt import ParticleFilter, \
    Control, Parameter

from interferometer import Interferometer

class ParticleFilterTest(TestCase):

    def setUp(self):

        self.interferometer = Interferometer(
            batchsize=3,
            params=[
                Parameter(bounds=(0.0, 2*pi), name="phase"),
                Parameter(bounds=(0.95, 1.00), name="visibility"),
            ],
        )

        self.pf = ParticleFilter(
            num_particles=5,
            phys_model=self.interferometer,
            resample_fraction=0.6,
            alpha=0.5,
            beta=0.9,
            gamma=0.8,
        )

    def test_true_values_uniform(self):
        """Test of the _true_values_uniform function in PhysicalModel.
        """
        rangen = Generator.from_seed(0xdeadbeef)

        true_values = self.interferometer.true_values(
            rangen,
        )

        self.assertEqual(
            true_values.shape, TensorShape([3, 1, 2]),
            msg="PhysicalModel: wrong shape _true_values_uniform"
        )

    def test_reset_pf(self):
        """Tests the shape and the normalization
            of the particles and weights produced by
            pf.reset()
        """
        rangen = Generator.from_seed(0xdeadbeef)

        weights, particles = self.pf.reset(rangen)

        self.assertEqual(weights.shape, TensorShape([3, 5]))
        self.assertEqual(particles.shape, TensorShape([3, 5, 2]))

        sum_weights = reduce_sum(weights, axis=1)

        self.assertAllClose(
            sum_weights, ones((3, ), dtype="float64"),
            atol=1e-10,
            rtol=0.0,
            msg="ParticleFIlter: initial weights not normalized"
        )
        self.assertAllInRange(
            particles[..., 0],
            zeros((3, 5), dtype="float64"),
            2*pi*ones((3, 5), dtype="float64"),
        )
        self.assertAllInRange(
            particles[..., 1],
            0.95*ones((3, 5), dtype="float64"),
            ones((3, 5), dtype="float64"),
        )

    def test_apply_measurement(self):
        """Test of the bayesian update

        Tests the normalization of the weights after the update.
        """

        # Testing the normalization of the weights after
        # the Bayesian update.
        rangen = Generator.from_seed(0xdeadbeef)

        weights, particles = self.pf.reset(rangen)

        state_filter = self.pf.phys_model.wrapper_initialize_state(
            particles, self.pf.np,
        )

        self.assertEqual(
            state_filter.shape, TensorShape([3, 5, 0]),
            msg="Physical Model: external_initialize state gives the wrong shape."
        )

        meas_step = zeros((3, 1), dtype="float64")
        control = zeros((3, 1), dtype="float64")
        outcomes = ones((3, 1), dtype="float64")

        weights, post_state_filter = self.pf.apply_measurement(
            weights, particles, state_filter, outcomes, control, meas_step,
        )

        # The state should be untouched from this operation
        self.assertAllClose(
            post_state_filter, state_filter,
            atol=1e-10,
            rtol=0.0,
            msg="ParticleFilter: apply_measurement changes the state when it shouldn't"
        )

        sum_weights = reduce_sum(weights, axis=1)

        self.assertAllClose(
            sum_weights, ones((3, ), dtype="float64"),
            atol=1e-10,
            rtol=0.0,
            msg="ParticleFilter: apply_measurement does not normalize the weights"
        )

        # Manual test of the application of the Bayes rule.
        weights, particles = self.pf.reset(rangen)

        outcomes = constant(
            [[1.0], [1.0], [-1.0]], dtype="float64",
        )

        particles_phase = constant(
            array(
                [[[pi/6], [pi/6], [pi/3], [pi/3], [pi/3]],
                 [[pi/6], [pi/6], [pi/3], [pi/3], [pi/3]],
                 [[pi/2], [pi/2], [pi/3], [pi/3], [pi/3]]]
            ), dtype="float64")

        particles_visibility = ones((3, 5, 1), dtype="float64")

        particles = concat([particles_phase, particles_visibility], 2)

        target_weights_unnormalized = constant(
            array(
                [[1/2+sqrt(3)/4, 1/2+sqrt(3)/4, 3/4, 3/4, 3/4],
                 [1/2+sqrt(3)/4, 1/2+sqrt(3)/4, 3/4, 3/4, 3/4],
                 [1/2, 1/2, 1/4, 1/4, 1/4]]
            ), dtype="float64")

        normalizations = constant(
            array(
                [[(13+2*sqrt(3))/4],
                 [(13+2*sqrt(3))/4],
                 [7/4]]
            ), dtype="float64")

        target_weights = target_weights_unnormalized/normalizations

        weights, post_state_filter = self.pf.apply_measurement(
            weights, particles, state_filter, outcomes, control, meas_step,
        )

        self.assertAllClose(
            weights,
            target_weights,
            atol=1e-10,
            rtol=0.0,
            msg="ParticleFilter: apply_measurement gets the Bayes rule wrong"
        )

        # The state should be untouched from this operation
        self.assertAllClose(
            post_state_filter, state_filter,
            atol=1e-10,
            rtol=0.0,
            msg="ParticleFilter: apply_measurement changes the state when it shouldn't"
        )

        # Test of the safety mechanism, that preserves the validity
        # of the weights when an observed outcome is not compatibile
        # with the observed outcomes

        rangen = Generator.from_seed(0xdeadbeef)

        weights, _, = self.pf.reset(rangen)

        particles_phase = constant(
            array(
                [[[pi/6], [pi/6], [pi/3], [pi/3], [pi/3]],
                 [[pi], [pi], [pi], [pi], [pi]],
                 [[pi/2], [pi/2], [pi/3], [pi/3], [pi/3]]]
            ), dtype="float64")

        particles_visibility = ones((3, 5, 1), dtype="float64")

        particles = concat([particles_phase, particles_visibility], 2)

        target_weights_unnormalized = constant(
            array(
                [[1/2+sqrt(3)/4, 1/2+sqrt(3)/4, 3/4, 3/4, 3/4],
                 [0.2, 0.2, 0.2, 0.2, 0.2],
                 [1/2, 1/2, 1/4, 1/4, 1/4]]
            ), dtype="float64")

        normalizations = constant(
            array(
                [[(13+2*sqrt(3))/4],
                 [1],
                 [7/4]]
            ), dtype="float64")

        target_weights = target_weights_unnormalized/normalizations

        weights, post_state_filter = self.pf.apply_measurement(
            weights, particles, state_filter, outcomes, control, meas_step,
        )

        self.assertAllClose(
            weights,
            target_weights,
            atol=1e-10,
            rtol=0.0,
            msg="ParticleFilter: apply_measurement does't enforce the sanity of the weights"
        )

        # The state should be untouched from this operation
        self.assertAllClose(
            post_state_filter, state_filter,
            atol=1e-10,
            rtol=0.0,
            msg="ParticleFilter: apply_measurement changes the state when it shouldn't"
        )

    def test_outlier_removal(self):
        """Test of the outlier remotion code in tf.apply_measurement
        """
        rangen = Generator.from_seed(0xdeadbeef)
        # Manual test of the application of the Bayes rule.
        weights, particles = self.pf.reset(rangen)

        state_filter = self.pf.phys_model.wrapper_initialize_state(
            particles, self.pf.np,
        )

        self.assertEqual(
            state_filter.shape, TensorShape([3, 5, 0]),
            msg="Physical Model: external_intialize_state produces the wrong shape."
        )

        control = zeros((3, 1), dtype="float64")
        outcomes = constant(
            [[1.0], [1.0], [-1.0]], dtype="float64",
        )

        particles_phase = constant(
            array(
                [[[pi/6], [pi/6], [pi/3], [pi/3], [pi/3]],
                 [[pi], [pi], [pi], [pi], [pi]],
                 [[pi/2], [pi/2], [pi/3], [pi/3], [pi/3]]]
            ), dtype="float64")

        particles_visibility = ones((3, 5, 1), dtype="float64")

        particles = concat([particles_phase, particles_visibility], 2)

        target_weights_unnormalized = constant(
            array(
                [[1/2+sqrt(3)/4, 1/2+sqrt(3)/4, 3/4, 3/4, 3/4],
                 [1.0, 1.0, 1.0, 1.0, 1.0],
                 [1/2, 1/2, 1/4, 1/4, 1/4]]
            ), dtype="float64")

        normalizations = constant(
            array(
                [[(13+2*sqrt(3))/4],
                 [5.0],
                 [7/4]]
            ), dtype="float64")

        target_weights = target_weights_unnormalized/normalizations

        meas_step = zeros((3, 1), dtype="float64")

        weights, post_state_filter = self.pf.apply_measurement(
            weights, particles, state_filter, outcomes, control, meas_step,
        )

        # The state should be untouched from this operation
        self.assertAllClose(
            post_state_filter, state_filter,
            atol=1e-10,
            rtol=0.0,
            msg="ParticleFilter: apply_measurement changes the state when it shouldn't"
        )

        self.assertAllClose(
            weights,
            target_weights,
            atol=1e-10,
            rtol=0.0,
            msg="ParticleFilter: apply_measurement doesn't remove the outliers correctly"
        )
        
    def test_check_resampling(self):
        """Test of the check_resampling function.
        """
        
        weights = constant(
            array(
                [[0.0, 0.0, 0.5, 0.5, 0.0],
                 [0.2, 0.2, 0.2, 0.2, 0.2],
                 [0.2, 0.2, 0.2, 0.2, 0.2]]
            ), dtype="float64")

        count_for_resampling = ones((3, ), dtype="bool")

        trigger = self.pf.check_resampling(
            weights, count_for_resampling,
        )
        
        expected_trigger = constant(
            array([True, False, False]),
            dtype="bool",
            )

        print(trigger)
        print(expected_trigger)
        
        self.assertAllEqual(
            trigger, expected_trigger,
            msg="ParticleFilter: check_resampling is not working"
        )

        # The first and the last need resampling
        weights = constant(
            array(
                [[0.0, 0.0, 0.5, 0.5, 0.0],
                 [0.2, 0.2, 0.2, 0.2, 0.2],
                 [0.9, 0.0, 0.1, 0.0, 0.0]]
            ), dtype="float64")

        # Only the first two batch elements count for the resampling
        count_for_resampling = constant(
            [[True], [True], [False]],
            dtype="bool",
        )

        trigger = self.pf.check_resampling(
            weights, count_for_resampling,
        )

        self.assertAllEqual(
            trigger, expected_trigger,
            msg="ParticleFilter: check_resampling is not working"
        )

    def test_full_resampling(self):
        """Test of the full_resampling function.
        """
        rangen = Generator.from_seed(0xdeadbeef)

        _, particles = self.pf.reset(rangen)

        weights = constant(
            array(
                [[0.0, 0.0, 0.5, 0.5, 0.0],
                 [0.2, 0.2, 0.2, 0.2, 0.2],
                 [0.2, 0.2, 0.2, 0.2, 0.2]]
            ), dtype="float64")

        count_for_resampling = ones((3, ), dtype="bool")

        rangen = Generator.from_seed(0xdeadbeef)

        _, _, resampled = self.pf.full_resampling(
            weights, particles, count_for_resampling, rangen,
        )

        self.assertEqual(
            resampled, constant(False),
            msg="ParticleFilter: check_resampling resampled when not needed"
        )

        weights = constant(
            array(
                [[0.0, 0, 0.5, 0.5, 0],
                 [0.2, 0.2, 0.2, 0.2, 0.2],
                 [1.0, 0.0, 0.0, 0.0, 0.0]]
            ), dtype="float64")

        rangen = Generator.from_seed(0xdeadbeef)

        post_weights, _, resampled = self.pf.full_resampling(
            weights, particles, count_for_resampling, rangen
        )

        self.assertEqual(
            resampled, constant(True),
            msg="ParticleFilter: check_resampling has not resampled when needed"
        )

        self.assertNotAllClose(
            post_weights,
            weights,
            atol=1e-10,
            rtol=0.0,
            msg="ParticleFilter: check_resampling has not resampled the weights",
        )

        self.assertEqual(
            reduce_any(is_nan(particles)), constant(False),
            msg="ParticleFilter: the resampling produce nan values"
        )

        self.assertEqual(
            reduce_any(is_nan(particles)), constant(False),
            msg="ParticleFilter: the resampling produce nan values"
        )

        # The first and the last need resampling
        weights = constant(
            array(
                [[0.0, 0.0, 0.5, 0.5, 0.0],
                 [0.2, 0.2, 0.2, 0.2, 0.2],
                 [0.9, 0.0, 0.1, 0.0, 0.0]]
            ), dtype="float64")

        # Only the first two batch elements count for the resampling
        count_for_resampling = constant(
            [[True], [True], [False]], dtype="bool",
        )

        # With resampling_fraction = 0.6, no resampling
        # should be triggered.

        _, _, resampled = self.pf.full_resampling(
            weights, particles, count_for_resampling, rangen
        )

        self.assertEqual(
            resampled, constant(False),
            msg="ParticleFilter: check_resampling has resampled when not needed"
        )

        # If we set resampling_fraction to 0.2 a resampling should be triggered

        self.pf.res_frac = 0.2

        _, _, resampled = self.pf.full_resampling(
            weights, particles, count_for_resampling, rangen,
        )

        self.assertEqual(
            resampled, constant(True),
            msg="ParticleFilter: check_resampling has not resampled when needed"
        )
        
    def test_partial_resampling(self):
        """Test of the partial_resampling function.
        """
        rangen = Generator.from_seed(0xdeadbeef)

        _, particles = self.pf.reset(rangen)

        weights = constant(
            array(
                [[0.0, 0, 0.5, 0.5, 0],
                 [0.2, 0.2, 0.2, 0.1, 0.3],
                 [1.0, 0.0, 0.0, 0.0, 0.0]]
            ), dtype="float64")
        
        count_for_resampling = ones((3, ), dtype="bool")

        rangen = Generator.from_seed(0xdeadbeef)

        post_weights, _ = self.pf.partial_resampling(
            weights, particles, count_for_resampling, rangen
        )

        # The middle element should not be resampled.
        self.assertAllClose(
            post_weights[1, :],
            weights[1, :],
            atol=1e-10,
            rtol=0.0,
            msg="ParticleFilter: check_resampling has not resampled the weights",
        )
        
        # The first element should be resampled.
        self.assertNotAllClose(
            post_weights[0, :],
            weights[0, :],
            atol=1e-10,
            rtol=0.0,
            msg="ParticleFilter: check_resampling has not resampled the weights",
        )
        
        # The last element should be resampled.
        self.assertNotAllClose(
            post_weights[2, :],
            weights[2, :],
            atol=1e-10,
            rtol=0.0,
            msg="ParticleFilter: check_resampling has not resampled the weights",
        )

        self.assertEqual(
            reduce_any(is_nan(particles)), constant(False),
            msg="ParticleFilter: the resampling produce nan values"
        )

        self.assertEqual(
            reduce_any(is_nan(particles)), constant(False),
            msg="ParticleFilter: the resampling produce nan values"
        )
        
    def test_trim_particles(self):
        """Test of pf.trim_particles
        """

        particles_phase = 6.0*constant(
            array(
                [[[0.10], [0.11], [6.12], [0.13], [34.14]],
                 [[0.15], [4.16], [0.17], [0.18], [0.19]],
                 [[0.20], [0.21], [0.22], [0.23], [10.24]]]
            ), dtype="float64")

        particles_visibility = 2.0*ones((3, 5, 1), dtype="float64")

        particles = concat([particles_phase, particles_visibility], 2)

        particles = self.pf._trim_particles(
            particles,
            )

        self.assertAllInRange(
            particles[..., 0],
            zeros((3, 5), dtype="float64"),
            2*pi*ones((3, 5), dtype="float64"),
        )
        self.assertAllInRange(
            particles[..., 1],
            0.95*ones((3, 5), dtype="float64"),
            ones((3, 5), dtype="float64"),
        )

    def test_compute_mean(self):
        """Test of pf.compute_mean
        """
        weights = constant(
            array(
                [[0.1, 0.2, 0.2, 0.25, 0.25],
                 [0.05, 0.05, 0.9, 0.0, 0.0],
                 [0.0, 0.05, 0.05, 0.8, 0.1]]
            ), dtype="float64")

        particles_phase = constant(
            array(
                [[[0.10], [0.9], [0.12], [0.8], [0.4]],
                 [[0.15], [0.6], [0.7], [0.2], [0.3]],
                 [[0.2], [0.1], [0.22], [0.23], [0.25]]]
            ), dtype="float64")

        particles_visibility = constant(
            array(
                [[[0.96], [0.98], [0.97], [0.95], [1.0]],
                 [[0.955], [0.975], [0.95], [0.97], [0.98]],
                 [[0.965], [0.985], [0.99], [0.995], [1.0]]]
            ), dtype="float64")

        particles = concat([particles_phase, particles_visibility], 2)

        mean = self.pf.compute_mean(weights, particles)

        phase_mean = constant(
            array([[0.514], [0.6675], [0.225]]), dtype="float64",
        )

        visibility_mean = constant(
            array([[0.9735], [0.9515], [0.99475]]), dtype="float64",
        )

        target_mean = concat([phase_mean, visibility_mean], 1)

        self.assertEqual(
            mean.shape, TensorShape([3, 2]),
            msg="ParticleFilter: compute_mean gives wrong mean shape"
        )

        self.assertAllClose(
            mean,
            target_mean,
            atol=1e-10,
            rtol=0.0,
            msg="ParticleFilter: compute_mean gives wrong numerical result",
        )

    def test_compute_covariance(self):
        """Test of pf.compute covariance
        """

        weights = constant(
            array(
                [[0.0, 0.5, 0.5, 0.0, 0.0],
                 [0.0, 0.0, 0.5, 0.0, 0.5],
                 [0.5, 0.0, 0.0, 0.5, 0.0]]
            ), dtype="float64")

        particles_phase = constant(
            array(
                [[[0.10], [0.11], [0.12], [0.13], [0.14]],
                 [[0.15], [0.16], [0.17], [0.18], [0.19]],
                 [[0.20], [0.21], [0.22], [0.23], [0.24]]]
            ), dtype="float64")

        particles_visibility = ones((3, 5, 1), dtype="float64")

        particles = concat(
            [particles_phase, particles_visibility], 2,
        )

        target_covariance = constant(
            array(
                [[[2.5e-5, 0.0], [0.0, 0.0]],
                 [[1e-4, 0.0], [0.0, 0.0]],
                 [[2.25e-4, 0.0], [0.0, 0.0]]],
            ), dtype="float64"
        )

        covariance = self.pf.compute_covariance(weights, particles)

        self.assertEqual(
            covariance.shape, TensorShape([3, 2, 2]),
            msg="ParticleFilter: compute_variance returns the wrong shape."
        )

        self.assertAllClose(
            covariance,
            target_covariance,
            atol=1e-10,
            rtol=0.0,
            msg="ParticleFilter: compute_covariance gives wrong numerical result",
        )

    def test_compute_max(self):
        """Test of the compute_max_weights
        function of the particle filter.
        """

        weights = constant(
            array(
                [[0.1, 0.2, 0.2, 0.2, 0.3],
                 [0.1, 0.1, 0.2, 0.3, 0.3],
                 [0.9, 0.05, 0.05, 0.0, 0.0]]
            ), dtype="float64")

        particles_phase = constant(
            array(
                [[[0.10], [0.11], [0.12], [0.13], [0.14]],
                 [[0.15], [0.16], [0.17], [0.18], [0.19]],
                 [[0.20], [0.21], [0.22], [0.23], [0.24]]]
            ), dtype="float64")

        particles_visibility = ones((3, 5, 1), dtype="float64")

        particles = concat(
            [particles_phase, particles_visibility], 2,
        )

        target_max = constant(
            array([[0.14, 1.0], [0.18, 1.0], [0.20, 1.0]]),
            dtype="float64",
        )

        max_particles = self.pf.compute_max_weights(weights, particles)

        self.assertEqual(
            max_particles.shape, TensorShape([3, 2]),
            msg="ParticleFilter: compute_max returns the wrong shape."
        )

        self.assertAllClose(
            max_particles,
            target_max,
            atol=1e-10,
            rtol=0.0,
            msg="ParticleFilter: compute_max gives wrong numerical result",
        )

    def test_invalid_weights(self):

        weights = constant(
            array(
                [[0.0, 0.0, 0.0, 0.0, 0.0],
                 [0.1, 0.1, 0.2, 0.3, 0.3],
                 [0.9, 0.05, 0.05, 0.0, 0.0]]
            ), dtype="float64")

        inv_weights = self.pf._invalid_weights(weights)

        inv_weights_expected = constant(
            [[True], [False], [False]], dtype="bool",
        )

        self.assertAllEqual(
            inv_weights, inv_weights_expected,
            msg="ParticleFilter _invalid_weights returns the wrong values."
        )


if __name__ == "__main__":
    main()