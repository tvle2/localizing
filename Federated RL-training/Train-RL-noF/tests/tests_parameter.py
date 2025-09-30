#!/usr/bin/env python3
from tensorflow.test import TestCase, main
from tensorflow import constant, ones, \
    reshape, unique_with_counts
from numpy import array
from math import pi

from qsensoropt import Parameter, \
    trim_single_param


class ParameterTest(TestCase):
    """Tests for the Parameter class.
    """

    def setUp(self):
        
        self.continuous_parameter = Parameter(
            bounds=(2.0, 3.0),
            name="Continuous1",
        )

        # This setting happen in the init of the PF,
        # the class Parameter is not projected to be
        # used outside the definition of a PF. 
        self.continuous_parameter.bs = 3
        self.continuous_parameter.prec = "float64"

        self.discrete_parameter_no_remainder = Parameter(
            values={1.0, 2.0, pi, pi**2, -1.0},
            randomize=False,
            name="Discrete1",
        )

        # This setting happen in the init of the PF,
        # the class Parameter is not projected to be
        # used outside the definition of a PF.
        self.discrete_parameter_no_remainder.bs = 3
        self.discrete_parameter_no_remainder.prec = "float64"

        self.discrete_parameter_remainder = Parameter(
            values={1.0, 2.0, pi, pi**2},
            randomize=False,
            name="Discrete2",
        )

        # This setting happen in the init of the PF,
        # the class Parameter is not projected to be
        # used outside the definition of a PF. 
        self.discrete_parameter_remainder.bs = 3
        self.discrete_parameter_remainder.prec = "float64"

    def test_reset_continuous(self):
        """Test of reset with continuous parameter.
        """
        # Prior of the continuous parameter
        seed = constant([0, 0], dtype="int32")
        prior = self.continuous_parameter.reset(seed, 5)

        expected_prior = constant(
            array(
                [[[2.6881243 ], [2.40050761], [2.10870968], [2.46550037], [2.79416594]],
                 [[2.08868071], [2.97677751], [2.86195413], [2.40910759], [2.74078764]],
                 [[2.31167404], [2.30469754], [2.76275653], [2.26128897], [2.37686001]]]
            ), dtype="float64")
        
        self.assertAllClose(
            prior,
            expected_prior,
            atol=1e-7,
            rtol=0.0,
            msg="Parameter: initialization of the continuous parameter prior wrong."
        )

    def test_reset_discrete(self):
        """Test reset for the discrete parameter.
        """
        # Prior of the continuous parameter
        seed = constant([0, 0], dtype="int32")
        prior = self.discrete_parameter_no_remainder.reset(seed, 5)

        expected_prior = constant(
            array(
                [[[-1.0], [1.0], [2.0], [pi], [pi**2]],
                 [[-1.0], [1.0], [2.0], [pi], [pi**2]],
                 [[-1.0], [1.0], [2.0], [pi], [pi**2]]]
            ), dtype="float64")
        
        self.assertAllClose(
            prior,
            expected_prior,
            atol=1e-10,
            rtol=0.0,
            msg="Parameter: initialization of the discrete parameter prior wrong."
        )

        prior = self.discrete_parameter_remainder.reset(seed, 5)

        expected_partial_prior = constant(
            array(
                [[[1.0], [2.0], [pi], [pi**2]],
                 [[1.0], [2.0], [pi], [pi**2]],
                 [[1.0], [2.0], [pi], [pi**2]]]
            ), dtype="float64")
        
        self.assertAllClose(
            prior[:, 0:4, :],
            expected_partial_prior,
            atol=1e-10,
            rtol=0.0,
            msg="Parameter: initialization of the discrete parameter prior wrong."
        )

    def test_reset_frequency(self):
        """Test of the frequency of the particles produced
        by the reset function.
        """
        seed = constant([0, 0], dtype="int32")
        prior = self.discrete_parameter_no_remainder.\
            _reset_uniform_discrete_deterministic(seed, 10000)

        flat_prior = reshape(prior, (30000, ))
        _, _, count = unique_with_counts(flat_prior)

        frequency = count/30000

        expected_frequency = constant(
            array(
                [0.20, 0.20, 0.20, 0.20, 0.20]
            )
        )

        self.assertAllClose(
            frequency,
            expected_frequency,
            atol=1e-10,
            rtol=0.0,
            msg="Parameter: initialization of the discrete parameter prior wrong."
        )

        seed = constant([0, 0], dtype="int32")
        prior = self.discrete_parameter_no_remainder.\
            _reset_uniform_discrete_random(seed, 1000000)

        flat_prior = reshape(prior, (3000000, ))
        _, _, count = unique_with_counts(flat_prior)

        frequency = count/3000000

        expected_frequency = constant(
            array(
                [0.20, 0.20, 0.20, 0.20, 0.20]
            )
        )

        self.assertAllClose(
            frequency,
            expected_frequency,
            atol=1e-3,
            rtol=0.0,
            msg="Parameter: initialization of the discrete parameter prior wrong."
        )

    def test_trim_single_param(self):

        param = self.discrete_parameter_no_remainder

        single_param = constant(
            array(
                [[1.0+0.1, pi+0.1, 2.0-0.1, pi**2-0.3, -1.0+pi/20],
                 [1.0+0.9, 2.0-0.8, pi/2, pi**2-0.1, -1.0+0.7],
                 [2.+0.1, 1.0-0.4, pi, pi**2, -1.0]]
        ), dtype="float64")
        
        new_single_param = trim_single_param(
            single_param, param,
            ones((3, 5, ), dtype="float64"),
            )
        
        expected_new_param = constant(
            array(
                [[[1.0], [pi], [2.0], [pi**2], [-1.0]],
                 [[2.0], [1.0], [2.0], [pi**2], [-1.0]],
                 [[2.0], [1.0], [pi], [pi**2], [-1.0]]]
        ), dtype="float64")

        self.assertAllClose(
            expected_new_param,
            new_single_param,
            atol=1e-3,
            rtol=0.0,
            msg="Parameter: trim_single_param doesn't work \
                for discrete parameters"
        )


if __name__ == "__main__":
    main()
