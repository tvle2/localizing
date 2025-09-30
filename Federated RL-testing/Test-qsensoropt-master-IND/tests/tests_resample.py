#!/usr/bin/env python3
from tensorflow.test import TestCase, main
from tensorflow.random import set_seed, Generator
from tensorflow import constant, zeros, ones, concat, \
    expand_dims, TensorShape
from tensorflow.math import is_nan, reduce_any
from numpy import array
from numpy.random import seed
from math import pi

from qsensoropt import ParticleFilter, \
    Parameter

from interferometer import Interferometer

set_seed(1)
seed(1)


class ResamplingTest(TestCase):
    """Test of the resampling.
    """
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
            resample_fraction=0.7,
            alpha=1.0,
            beta=0.9,
            gamma=0.8,
        )

    def test_resample_soft(self):
        """Test of pf.resample_soft
        """

        particles_phase = constant(
            array(
                [[[0.10], [0.11], [0.12], [0.13], [0.14]],
                 [[0.15], [0.16], [0.17], [0.18], [0.19]],
                 [[0.20], [0.21], [0.22], [0.23], [0.24]]]
            ), dtype="float64")

        particles_visibility = ones((3, 5, 1), dtype="float64")

        particles = concat([particles_phase, particles_visibility], 2)

        weights = constant(
            array(
                [[0.0, 0.0, 1.0, 0.0, 0.0],
                 [1.0, 0.0, 0.0, 0.0, 0.0],
                 [0.0, 0.0, 0.0, 0.0, 1.0]]
            ), dtype="float64")

        target_particles_phase = constant(
            array(
                [[[0.12], [0.12], [0.12], [0.12]],
                 [[0.15], [0.15], [0.15], [0.15]],
                 [[0.24], [0.24], [0.24], [0.24]]]
            ), dtype="float64")

        particles_visibility = ones((3, 4, 1), dtype="float64")

        target_particles = concat(
            [target_particles_phase, particles_visibility], 2,
        )

        rangen = Generator.from_seed(0xdeadbeef)

        weights, particles = self.pf._resample_soft(
            weights, particles, rangen, 
        )

        self.assertEqual(
            reduce_any(is_nan(particles)), constant(False),
            msg="ParticleFilter: resample_soft produce nan particles",
        )

        self.assertEqual(
            reduce_any(is_nan(weights)), constant(False),
            msg="ParticleFilter: resample_soft produce nan weights",
        )

        self.assertEqual(
            particles.shape, TensorShape([3, 4, 2]),
            msg="ParticleFilter: resample_soft gives wrong particle shape",
        )

        self.assertEqual(
            weights.shape, TensorShape([3, 4]),
            msg="ParticleFilter: resample_soft gives wrong weights shape",
        )

        self.assertAllClose(
            particles,
            target_particles,
            atol=1e-10,
            rtol=0.0,
            msg="ParticleFilter: resample_soft is sampling the wrong particles",
        )

        target_weights = 0.2*ones((3, 4), dtype="float64")

        self.assertAllClose(
            weights,
            target_weights,
            atol=1e-10,
            rtol=0.0,
            msg="ParticleFilter: resample_soft gives the wrong weights",
        )

        particles_phase = constant(
            array(
                [[[0.10], [0.11], [0.12], [0.13], [0.14]],
                 [[0.15], [0.16], [0.17], [0.18], [0.19]],
                 [[0.20], [0.21], [0.22], [0.23], [0.24]]]
            ), dtype="float64")

        particles_visibility = ones((3, 5, 1), dtype="float64")

        particles = concat([particles_phase, particles_visibility], 2)

        weights = constant(
            array(
                [[0.0, 0.0, 1.0, 0.0, 0.0],
                 [1.0, 0.0, 0.0, 0.0, 0.0],
                 [0.0, 0.0, 0.0, 0.0, 1.0]]
            ), dtype="float64")

        self.pf.alpha = 0

        weights, _ = self.pf._resample_soft(
            weights, particles, rangen,
        )

        target_weights = constant(
            0.8*array(
                [[0.25, 0.25, 0.25, 0.25],
                 [0.25, 0.25, 0.25, 0.25],
                 [0.25, 0.25, 0.25, 0.25]]
            ), dtype="float64")

        self.assertAllClose(
            weights,
            target_weights,
            atol=1e-10,
            rtol=0.0,
            msg="ParticleFilter resample soft: hard resampling is not called when needed",
        )
        
    def test_partial_resample_soft(self):
        """Test of pf.resample_soft with partial
        batchsize.
        """

        particles_phase = constant(
            array(
                [[[0.10], [0.11], [0.12], [0.13], [0.14]],
                 [[0.15], [0.16], [0.17], [0.18], [0.19]]]
            ), dtype="float64")

        particles_visibility = ones((2, 5, 1), dtype="float64")

        particles = concat([particles_phase, particles_visibility], 2)

        weights = constant(
            array(
                [[0.0, 0.0, 1.0, 0.0, 0.0],
                 [1.0, 0.0, 0.0, 0.0, 0.0]]
            ), dtype="float64")

        target_particles_phase = constant(
            array(
                [[[0.12], [0.12], [0.12], [0.12]],
                 [[0.15], [0.15], [0.15], [0.15]]]
            ), dtype="float64")

        particles_visibility = ones((2, 4, 1), dtype="float64")

        target_particles = concat(
            [target_particles_phase, particles_visibility], 2,
        )

        rangen = Generator.from_seed(0xdeadbeef)

        weights, particles = self.pf._resample_soft(
            weights, particles, rangen, batchsize=2,
        )

        self.assertEqual(
            reduce_any(is_nan(particles)), constant(False),
            msg="ParticleFilter: resample_soft produce nan particles",
        )

        self.assertEqual(
            reduce_any(is_nan(weights)), constant(False),
            msg="ParticleFilter: resample_soft produce nan weights",
        )

        self.assertEqual(
            particles.shape, TensorShape([2, 4, 2]),
            msg="ParticleFilter: resample_soft gives wrong particle shape",
        )

        self.assertEqual(
            weights.shape, TensorShape([2, 4]),
            msg="ParticleFilter: resample_soft gives wrong weights shape",
        )

        self.assertAllClose(
            particles,
            target_particles,
            atol=1e-10,
            rtol=0.0,
            msg="ParticleFilter: resample_soft is sampling the wrong particles",
        )

        target_weights = 0.2*ones((2, 4), dtype="float64")

        self.assertAllClose(
            weights,
            target_weights,
            atol=1e-10,
            rtol=0.0,
            msg="ParticleFilter: resample_soft gives the wrong weights",
        )

        particles_phase = constant(
            array(
                [[[0.10], [0.11], [0.12], [0.13], [0.14]],
                 [[0.15], [0.16], [0.17], [0.18], [0.19]]]
            ), dtype="float64")

        particles_visibility = ones((2, 5, 1), dtype="float64")

        particles = concat([particles_phase, particles_visibility], 2)

        weights = constant(
            array(
                [[0.0, 0.0, 1.0, 0.0, 0.0],
                 [1.0, 0.0, 0.0, 0.0, 0.0]]
            ), dtype="float64")

        self.pf.alpha = 0

        weights, _ = self.pf._resample_soft(
            weights, particles, rangen, batchsize=2,
        )

        target_weights = constant(
            0.8*array(
                [[0.25, 0.25, 0.25, 0.25],
                 [0.25, 0.25, 0.25, 0.25]]
            ), dtype="float64")

        self.assertAllClose(
            weights,
            target_weights,
            atol=1e-10,
            rtol=0.0,
            msg="ParticleFilter resample soft: hard resampling is not called when needed",
        )

    def test_resample_hard(self):
        """Test of pf.resample_hard
        """
        particles_phase = constant(
            array(
                [[[0.10], [0.11], [0.12], [0.13], [0.14]],
                 [[0.15], [0.16], [0.17], [0.18], [0.19]],
                 [[0.20], [0.21], [0.22], [0.23], [0.24]]]
            ), dtype="float64")

        particles_visibility = ones((3, 5, 1), dtype="float64")

        particles = concat([particles_phase, particles_visibility], 2)

        weights = constant(
            array(
                [[0.0, 0.0, 1.0, 0.0, 0.0],
                 [1.0, 0.0, 0.0, 0.0, 0.0],
                 [0.0, 0.0, 0.0, 0.0, 1.0]]
            ), dtype="float64")

        target_particles_phase = constant(
            array(
                [[[0.12], [0.12], [0.12], [0.12]],
                 [[0.15], [0.15], [0.15], [0.15]],
                 [[0.24], [0.24], [0.24], [0.24]]]
            ), dtype="float64")

        particles_visibility = ones((3, 4, 1), dtype="float64")

        target_particles = concat(
            [target_particles_phase, particles_visibility], 2,
        )

        rangen = Generator.from_seed(0xdeadbeef)

        weights, selected_weights, particles = self.pf._resample_hard(
            weights, particles, rangen,
        )

        self.assertEqual(
            reduce_any(is_nan(particles)), constant(False),
            msg="ParticleFilter: resample_soft produce nan particles",
        )

        self.assertEqual(
            reduce_any(is_nan(weights)), constant(False),
            msg="ParticleFilter: resample_soft produce nan weights",
        )

        self.assertEqual(
            particles.shape, TensorShape([3, 4, 2]),
            msg="ParticleFilter: resample_soft gives wrong particle shape",
        )

        self.assertEqual(
            weights.shape, TensorShape([3, 4]),
            msg="ParticleFilter: resample_soft gives wrong weights shape",
        )

        self.assertAllClose(
            particles,
            target_particles,
            atol=1e-10,
            rtol=0.0,
            msg="ParticleFilter: resample_soft is sampling the wrong particles",
        )

        target_weights = 0.2*ones((3, 4), dtype="float64")

        target_selected_weights = ones((3, 4), dtype="float64")

        self.assertAllClose(
            weights,
            target_weights,
            atol=1e-10,
            rtol=0.0,
            msg="ParticleFilter: resample_soft gives the wrong weights",
        )

        self.assertAllClose(
            selected_weights,
            target_selected_weights,
            atol=1e-10,
            rtol=0.0,
            msg="ParticleFilter: resample_soft gives the wrong weights",
        )
        
    def test_partial_resample_hard(self):
        """Test of pf.resample_hard with partial
        batchsize.
        """
        particles_phase = constant(
            array(
                [[[0.10], [0.11], [0.12], [0.13], [0.14]],
                 [[0.15], [0.16], [0.17], [0.18], [0.19]]]
            ), dtype="float64")

        particles_visibility = ones((2, 5, 1), dtype="float64")

        particles = concat([particles_phase, particles_visibility], 2)

        weights = constant(
            array(
                [[0.0, 0.0, 1.0, 0.0, 0.0],
                 [1.0, 0.0, 0.0, 0.0, 0.0]]
            ), dtype="float64")

        target_particles_phase = constant(
            array(
                [[[0.12], [0.12], [0.12], [0.12]],
                 [[0.15], [0.15], [0.15], [0.15]]]
            ), dtype="float64")

        particles_visibility = ones((2, 4, 1), dtype="float64")

        target_particles = concat(
            [target_particles_phase, particles_visibility], 2,
        )

        rangen = Generator.from_seed(0xdeadbeef)

        weights, selected_weights, particles = self.pf._resample_hard(
            weights, particles, rangen, batchsize=2,
        )

        self.assertEqual(
            reduce_any(is_nan(particles)), constant(False),
            msg="ParticleFilter: resample_soft produce nan particles",
        )

        self.assertEqual(
            reduce_any(is_nan(weights)), constant(False),
            msg="ParticleFilter: resample_soft produce nan weights",
        )

        self.assertEqual(
            particles.shape, TensorShape([2, 4, 2]),
            msg="ParticleFilter: resample_soft gives wrong particle shape",
        )

        self.assertEqual(
            weights.shape, TensorShape([2, 4]),
            msg="ParticleFilter: resample_soft gives wrong weights shape",
        )

        self.assertAllClose(
            particles,
            target_particles,
            atol=1e-10,
            rtol=0.0,
            msg="ParticleFilter: resample_soft is sampling the wrong particles",
        )

        target_weights = 0.2*ones((2, 4), dtype="float64")

        target_selected_weights = ones((2, 4), dtype="float64")

        self.assertAllClose(
            weights,
            target_weights,
            atol=1e-10,
            rtol=0.0,
            msg="ParticleFilter: resample_soft gives the wrong weights",
        )

        self.assertAllClose(
            selected_weights,
            target_selected_weights,
            atol=1e-10,
            rtol=0.0,
            msg="ParticleFilter: resample_soft gives the wrong weights",
        )

    def test_gaussian(self):
        """Deterministic test of resample_gaussian
        """

        mean = constant(
            array(
                [[0.4, 0.99], [0.8, 0.98], [0.5, 0.97]]
            ), dtype="float64"
        )

        dev = zeros((3, 1, 1), dtype="float64")

        rangen = Generator.from_seed(0xdeadbeef)
        weights, particles = self.pf._resample_gaussian(
            mean, dev, rangen
        )

        self.assertEqual(
            particles.shape, TensorShape([3, 1, 2]),
            msg="ParticleFilter: resample_gaussian gives wrong particle shape"
        )

        self.assertEqual(
            weights.shape, TensorShape([3, 1]),
            msg="ParticleFilter: resample_gaussian gives wrong weights shape"
        )

        self.assertEqual(
            reduce_any(is_nan(particles)), constant(False),
            msg="ParticleFilter: resample_gaussian produce nan particles"
        )

        self.assertEqual(
            reduce_any(is_nan(weights)), constant(False),
            msg="ParticleFilter: resample_gaussian produce nan weights"
        )

        self.assertAllClose(
            particles,
            expand_dims(mean, axis=1),
            atol=1e-10,
            rtol=0.0,
            msg="Particle filter: resample_gaussian gives wrong particles"
        )

        self.assertAllClose(
            weights,
            0.2*ones((3, 1), dtype="float64"),
            atol=1e-10,
            rtol=0.0,
            msg="Particle filter: resample_gaussian gives wrong weights."
        )
        
    def test_partial_gaussian(self):
        """Deterministic test of resample_gaussian with
        partial batchsize.
        """

        mean = constant(
            array(
                [[0.4, 0.99], [0.8, 0.98]]
            ), dtype="float64"
        )

        dev = zeros((2, 1, 1), dtype="float64")

        rangen = Generator.from_seed(0xdeadbeef)
        weights, particles = self.pf._resample_gaussian(
            mean, dev, rangen, batchsize=2,
        )

        self.assertEqual(
            particles.shape, TensorShape([2, 1, 2]),
            msg="ParticleFilter: resample_gaussian gives wrong particle shape"
        )

        self.assertEqual(
            weights.shape, TensorShape([2, 1]),
            msg="ParticleFilter: resample_gaussian gives wrong weights shape"
        )

        self.assertEqual(
            reduce_any(is_nan(particles)), constant(False),
            msg="ParticleFilter: resample_gaussian produce nan particles"
        )

        self.assertEqual(
            reduce_any(is_nan(weights)), constant(False),
            msg="ParticleFilter: resample_gaussian produce nan weights"
        )

        self.assertAllClose(
            particles,
            expand_dims(mean, axis=1),
            atol=1e-10,
            rtol=0.0,
            msg="Particle filter: resample_gaussian gives wrong particles"
        )

        self.assertAllClose(
            weights,
            0.2*ones((2, 1), dtype="float64"),
            atol=1e-10,
            rtol=0.0,
            msg="Particle filter: resample_gaussian gives wrong weights."
        )

    def test_resample(self):
        """Test pf.resample.
        """

        particles_phase = constant(
            array(
                [[[0.10], [0.11], [0.12], [0.13], [0.14]],
                 [[0.15], [0.16], [0.17], [0.18], [0.19]],
                 [[0.20], [0.21], [0.22], [0.23], [0.24]]]
            ), dtype="float64")

        particles_visibility = 0.97*ones((3, 5, 1), dtype="float64")

        particles = concat([particles_phase, particles_visibility], 2)

        weights = constant(
            array(
                [[0.0, 0.0, 0.9, 0.0, 0.1],
                 [0.5, 0.1, 0.1, 0.0, 0.3],
                 [0.0, 0.0, 0.0, 0.0, 1.0]]
            ), dtype="float64")

        rangen = Generator.from_seed(0xdeadbeef)
        weights, particles = self.pf.resample(
            weights, particles, rangen,
        )

        self.assertEqual(
            reduce_any(is_nan(particles)), constant(False),
            msg="ParticleFilter: resample produce nan particles"
        )

        self.assertEqual(
            reduce_any(is_nan(particles)), constant(False),
            msg="ParticleFilter: resample produce nan particles"
        )

        self.assertEqual(
            particles.shape, TensorShape([3, 5, 2]),
            msg="ParticleFilter: resample gives wrong particle shape"
        )

        self.assertEqual(
            weights.shape, TensorShape([3, 5]),
            msg="ParticleFilter: resample gives wrong weights shape"

        )
        
    def test_partial_resample(self):
        """Test pf.resample with partial batchsize.
        """

        particles_phase = constant(
            array(
                [[[0.10], [0.11], [0.12], [0.13], [0.14]],
                 [[0.15], [0.16], [0.17], [0.18], [0.19]]]
            ), dtype="float64")

        particles_visibility = 0.97*ones((2, 5, 1), dtype="float64")

        particles = concat([particles_phase, particles_visibility], 2)

        weights = constant(
            array(
                [[0.0, 0.0, 0.9, 0.0, 0.1],
                 [0.5, 0.1, 0.1, 0.0, 0.3]]
            ), dtype="float64")

        rangen = Generator.from_seed(0xdeadbeef)
        weights, particles = self.pf.resample(
            weights, particles, rangen, batchsize=2,
        )

        self.assertEqual(
            reduce_any(is_nan(particles)), constant(False),
            msg="ParticleFilter: resample produce nan particles"
        )

        self.assertEqual(
            reduce_any(is_nan(particles)), constant(False),
            msg="ParticleFilter: resample produce nan particles"
        )

        self.assertEqual(
            particles.shape, TensorShape([2, 5, 2]),
            msg="ParticleFilter: resample gives wrong particle shape"
        )

        self.assertEqual(
            weights.shape, TensorShape([2, 5]),
            msg="ParticleFilter: resample gives wrong weights shape"

        )


if __name__ == "__main__":
    main()
