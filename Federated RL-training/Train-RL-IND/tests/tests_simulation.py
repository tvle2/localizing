#!/usr/bin/env python3
from tensorflow.test import TestCase, main
from tensorflow.random import set_seed, Generator
from tensorflow import constant, ones, concat, \
    zeros, reduce_sum, TensorShape
from numpy import array
from numpy.random import seed
from math import pi, sqrt, log, cos

from qsensoropt import ParticleFilter, \
    Parameter, SimulationParameters
from qsensoropt.utils import standard_model

from interferometer import Interferometer, \
    StatelessMetrologyModified

set_seed(1)
seed(1)

class SimulationTest(TestCase):

    def setUp(self):

        self.interferometer=Interferometer(
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

    def test_compute_scalar_loss(self):
        """Test of the compute_loss function
        """

        weights = constant(
            array([[0.1, 0.1, 0.8, 0.0, 0.0],
                   [0.0, 1.0, 0.0, 0.0, 0.0],
                   [0.0, 0.0, 0.5, 0.0, 0.5]]
                  ), dtype="float64",
        )

        particles_phase = constant(
            array(
                [[[pi], [pi/4], [pi/3], [pi/5], [pi/8]],
                 [[pi/5], [pi/6], [pi], [pi/2], [pi/3]],
                 [[pi/7], [pi/2], [pi/3], [pi/4], [pi/8]]]
            ), dtype="float64")

        particles_visibility = ones((3, 5, 1), dtype="float64")

        particles = concat(
            [particles_phase, particles_visibility], 2,
            )

        true_values = constant(
            array([[[pi/3, 1.0]],
                   [[pi/7, 1.0]],
                   [[pi/5, 1.0]]]
                  ), dtype="float64")

        state_filter = self.pf.phys_model.wrapper_initialize_state(
            particles, self.pf.np,
            )

        true_state = self.interferometer.wrapper_initialize_state(
            true_values, 1,
            )

        sum_log_prob = constant(
            array([[-0.1],
                   [-0.2],
                   [-0.3]]
                  ), dtype="float64")

        # The last estimation has been completed
        estimation_not_finished = constant(array(
            [[True], [True], [False]]), dtype="bool",
        )

        meas_step = ones((3, 1), dtype="float64")
        used_resources = ones((3, 1), dtype="float64")

        loss_diff, loss = self.sim_nn._compute_scalar_loss(
            weights, particles, true_state, state_filter, sum_log_prob,
            true_values, used_resources, meas_step,
            estimation_not_finished,
        )

        loss_expected = (pi**2)*((7/120)**2+1/(42**2)+(7/240)**2)/3
        loss_diff_expected = (pi**2)*((9/10)*(7/120)**2 +
                                      4/(5*(42**2))+7**3/(10*240**2))/3

        self.assertAllClose(
            loss_expected,
            loss,
            atol=1e-10,
            rtol=0.0,
            msg="Simulation _compute_scalar_loss: wrong loss",
        )

        self.assertAllClose(
            loss_diff_expected,
            loss_diff,
            atol=1e-10,
            rtol=0.0,
            msg="Simulation _compute_scalar_loss: wrong loss_diff",
        )

        self.sim_nn.simpars = SimulationParameters(
            sim_name="interferometer",
            num_steps=10,
            max_resources=10,
            cumulative_loss=True,
            loss_logl_outcomes=False,
        )

        loss_diff, loss = self.sim_nn._compute_scalar_loss(
            weights, particles, true_state, state_filter, sum_log_prob,
            true_values, used_resources, meas_step,
            estimation_not_finished,
        )

        loss_expected = (pi**2)*((7/120)**2+1/(42**2)+(7/240)**2)/3
        loss_diff_expected = loss_expected

        self.assertAllClose(
            loss_expected,
            loss,
            atol=1e-10,
            rtol=0.0,
            msg="Simulation _compute_scalar_loss: wrong loss without log_likelihood",
        )

        self.assertAllClose(
            loss_diff_expected,
            loss_diff,
            atol=1e-10,
            rtol=0.0,
            msg="Simulation _compute_scalar_loss: wrong loss_diff without log_likelihood",
        )

        self.sim_nn.simpars = SimulationParameters(
            sim_name="interferometer",
            num_steps=10,
            max_resources=10,
            cumulative_loss=True,
            log_loss=True,
        )

        loss_diff, loss = self.sim_nn._compute_scalar_loss(
            weights, particles, true_state, state_filter, sum_log_prob,
            true_values, used_resources, meas_step,
            estimation_not_finished,
        )

        loss_expected = log((pi**2)*((7/120)**2+1/(42**2)+(7/240)**2)/3)
        loss_diff_expected = ((pi**2)*((9/10)*(7/120)**2+4/(5*(42**2))+7**3/(10*240**2))) /\
            ((pi**2)*((7/120)**2+1/(42**2)+(7/240)**2))

        self.assertAllClose(
            loss_expected,
            loss,
            atol=1e-10,
            rtol=0.0,
            msg="Simulation _compute_scalar_loss: wrong loss with log_loss",
        )

        self.assertAllClose(
            loss_diff_expected,
            loss_diff,
            atol=1e-10,
            rtol=0.0,
            msg="Simulation _compute_scalar_loss: wrong loss_diff with log_loss",
        )

    def test_compute_finished(self):
        """Test of the _compute_finished function.
        """

        estimation_not_finished = constant(array(
            [[False], [False], [True]]), dtype="bool",
        )

        meas_step = constant(array(
            [[10], [9], [8]]), dtype="int32",
        )

        finished, num_finished = self.sim_nn._compute_finished(
            estimation_not_finished, meas_step,
        )

        finished_expected = constant(array(
            [[True], [True], [False]]), dtype="bool",
        )

        num_finished_expected = constant(2, dtype="int32")

        self.assertAllEqual(
            finished, finished_expected,
            msg="Simulation _compute_finished: wrong finished tensor"
        )

        self.assertAllEqual(
            num_finished, num_finished_expected,
            msg="Simulation _compute_finished: wrong finished tensor"
        )

    def test_loop_cond(self):
        """Test of the loop condition in the simulation class.
        """

        self.sim_nn.simpars = SimulationParameters(
            sim_name="interferometer",
            num_steps=20,
            max_resources=10,
            # The simulation steps if 2/3 of the simulation are complete,
            # but continues if only 1/3 has finished.
            resources_fraction=0.5,
            cumulative_loss=True
        )

        # The last estimation has been completed
        estimation_not_finished = constant(array(
            [[True], [True], [False]]), dtype="bool",
        )

        cont_true = self.sim_nn._loop_cond(
            self.sim_nn.simpars, estimation_not_finished)

        self.assertEqual(
            cont_true, constant(True),
            msg="Simulation _loop_cond: the loop stopping condition is wrong"
        )

        # The last estimation has been completed
        estimation_not_finished = constant(array(
            [[True], [False], [False]]), dtype="bool",
        )

        cont_false = self.sim_nn._loop_cond(
            self.sim_nn.simpars, estimation_not_finished)

        self.assertEqual(
            cont_false, constant(False),
            msg="Simulation _loop_cond: the loop stopping condition is wrong"
        )

        self.sim_nn.simpars = SimulationParameters(
            sim_name="interferometer",
            num_steps=10,
            max_resources=10,
            cumulative_loss=True,
        )

        cont_true = self.sim_nn._loop_cond(
            self.sim_nn.simpars, estimation_not_finished)

        self.assertEqual(
            cont_true, constant(True),
            msg="Simulation _loop_cond: the loop stopping condition is wrong"
        )

    def test_loop_body(self):
        """Test of the loop body.
        """

        num_steps = 20

        self.sim_nn.simpars = SimulationParameters(
            sim_name="interferometer",
            num_steps=num_steps,
            max_resources=10,
            resources_fraction=0.5,
            cumulative_loss=True,
        )

        # self.pf.rf = 1.5

        self.sim_nn.control_strategy = \
            lambda x: zeros((self.sim_nn.bs, 1), dtype="float64")

        # The last estimation has been completed
        continue_flag = constant(array(
            [[True], [True], [True]]), dtype="bool",
        )
        rangen = Generator.from_seed(0xdeadbeef)

        meas_step = constant(array(
            [[7], [9], [10]]), dtype="int32",
        )

        used_resources = constant(array(
            [[7.0], [9.0], [10.0]]), dtype="float64",
        )

        # Manual test of the application of the Bayes rule.
        weights, _ = self.pf.reset(rangen)

        particles_phase = constant(
            array(
                [[[pi/6], [pi/6], [pi/3], [pi/3], [pi/3]],
                 [[pi/6], [pi/6], [pi/3], [pi/3], [pi/3]],
                 [[pi/2], [pi/2], [pi/3], [pi/3], [pi/3]]]
            ), dtype="float64")

        particles_visibility = ones((3, 5, 1), dtype="float64")

        particles = concat([particles_phase, particles_visibility], 2)

        particles = self.pf._trim_particles(particles)

        state_filter = self.pf.phys_model.wrapper_initialize_state(
            particles, self.pf.np,
            )
    
        true_values = constant(
            array([[[pi/3, 1.0]],
                   [[pi/7, 1.0]],
                   [[pi/5, 1.0]]]
                  ), dtype="float64")
        
        true_state = self.interferometer.wrapper_initialize_state(
            true_values, 1,
            )
        
        outcomes = zeros((3,  1), dtype="float64")

        index = constant(0, dtype="int32")

        sum_log_prob = constant(
            array([[-0.1],
                   [-0.2],
                   [-0.3]]
                  ), dtype="float64")

        particles_expected = particles  # No resampling

        loss = zeros((), dtype="float64")
        loss_diff = zeros((), dtype="float64")

        history_input = zeros((num_steps, 3, 10), dtype="float64")
        history_control = zeros((num_steps, 3, 1), dtype="float64")
        history_resources = zeros((num_steps, 3, 1), dtype="float64")
        history_precision = zeros((num_steps, 3, 1), dtype="float64")

        hist_outcomes_res = zeros((num_steps, 3,  1), dtype="float64")
        hist_control_res = zeros((num_steps, 3, 1), dtype="float64")
        hist_enf_res = zeros((num_steps, 3, 1), dtype="int32")
        hist_step_res = zeros((num_steps, 3, 1), dtype="int32")

        # Second execution of the body of the loop
        continue_flag, index, meas_step, used_resources, outcomes, \
            sum_log_prob, true_values, weights, particles, true_state, state_filter, \
                loss_diff, loss, history_input, history_control, history_resources, \
                    history_precision, hist_outcomes_res, hist_control_res, hist_enf_res, \
                        hist_step_res = \
            self.sim_nn._loop_body(
                self.sim_nn.simpars, False, rangen,
                continue_flag, index, meas_step, used_resources, outcomes, sum_log_prob,
                true_values, weights, particles, true_state, state_filter, loss_diff, loss,
                history_input, history_control, history_resources, history_precision,
                hist_outcomes_res, hist_control_res, hist_enf_res, hist_step_res,
            )
        
        # There is no resampling triggered
        estimation_not_finished_expected = constant(array(
            [[True], [True], [False]]), dtype="bool",
        )

        meas_step_expected = constant(array(
            [[8], [10], [10]]), dtype="int32",
        )

        used_resources_expected = constant(array(
            [[8.0], [10.0], [10.0]]), dtype="float64",
        )

        sum_log_prob_expected = constant(
            array([[-0.1+log(0.5*(1+cos(pi/3)))],
                   [-0.2+log(0.5*(1+cos(pi/7)))],
                   [-0.3]]
                  ), dtype="float64")

        target_weights_unnormalized = constant(
            array(
                [[1/2+sqrt(3)/4, 1/2+sqrt(3)/4, 3/4, 3/4, 3/4],
                 [1/2+sqrt(3)/4, 1/2+sqrt(3)/4, 3/4, 3/4, 3/4],
                 [1, 1, 1, 1, 1]]  # No Bayesian update for the last estimation
            ), dtype="float64")

        normalizations = constant(
            array(
                [[(13+2*sqrt(3))/4],
                 [(13+2*sqrt(3))/4],
                 [5]]
            ), dtype="float64")

        weights_expected = target_weights_unnormalized/normalizations

        self.assertAllEqual(
            estimation_not_finished_expected, continue_flag,
            msg="Simulation _loop_body: estimation_not_finished wrongly updated"
        )

        self.assertAllEqual(
            meas_step_expected, meas_step,
            msg="Simulation _loop_body: meas_step wrongly updated"
        )

        self.assertAllClose(
            used_resources_expected,
            used_resources,
            atol=1e-10,
            rtol=0.0,
            msg="Simulation _loop_body: used_resources wrongly updated",
        )

        self.assertAllClose(
            sum_log_prob,
            sum_log_prob_expected,
            atol=1e-10,
            rtol=0.0,
            msg="Simulation _loop_body: wrong update sum_log_prob",
        )

        self.assertAllClose(
            weights,
            weights_expected,
            atol=1e-10,
            rtol=0.0,
            msg="Simulation _loop_body: wrong update weights",
        )

        self.assertAllClose(
            particles,
            particles_expected,
            atol=1e-10,
            rtol=0.0,
            msg="Simulation _loop_body: wrong update particle",
        )

        loss_expected = ((pi**2)/9*((2+sqrt(3))/(13+2*sqrt(3))) **
                         2+((pi/21)**2)*((38+sqrt(3))/(13+2*sqrt(3)))**2+pi**2/25)/3
        loss_diff_expected = (pi**2/9*((2+sqrt(3))/(13+2*sqrt(3)))**2*(1-0.1+log(0.5*(1+cos(pi/3)))) +
                              ((pi/21)**2)*((38+sqrt(3))/(13+2*sqrt(3)))**2*(1-0.2+log(0.5*(1+cos(pi/7))))+pi**2/25*0.7)/3

        self.assertAllClose(
            loss,
            loss_expected,
            atol=1e-10,
            rtol=0.0,
            msg="Simulation _loop_body: wrong update loss",
        )

        self.assertAllClose(
            loss_diff,
            loss_diff_expected,
            atol=1e-10,
            rtol=0.0,
            msg="Simulation _loop_body: wrong update loss_diff",
        )

        # Second execution of the body of the loop
        continue_flag, index, meas_step, used_resources, outcomes, \
            sum_log_prob, true_values, weights, particles, true_state, state_filter, \
                loss_diff, loss, history_input, history_control, history_resources, \
                    history_precision, hist_outcomes_res, hist_control_res, hist_enf_res, \
                        hist_step_res = \
            self.sim_nn._loop_body(
                self.sim_nn.simpars, False, rangen,
                continue_flag, index, meas_step, used_resources, outcomes, sum_log_prob,
                true_values, weights, particles, true_state, state_filter, loss_diff, loss,
                history_input, history_control, history_resources, history_precision,
                hist_outcomes_res, hist_control_res, hist_enf_res, hist_step_res,
            )

        # There is no resampling triggered

        continue_flag_expected = constant(array(
            [[True], [False], [False]]), dtype="bool",
        )

        meas_step_expected = constant(array(
            [[9], [10], [10]]), dtype="int32",
        )

        used_resources_expected = constant(array(
            [[9.0], [10.0], [10.0]]), dtype="float64",
        )

        sum_log_prob_expected = constant(
            array([[-0.1+2*log(0.5*(1+cos(pi/3)))],
                   [-0.2+log(0.5*(1+cos(pi/7)))],
                   [-0.3]]
                  ), dtype="float64")

        target_weights_unnormalized = constant(
            array(
                [[(1/2+sqrt(3)/4)**2, (1/2+sqrt(3)/4)**2, (3/4)**2, (3/4)**2, (3/4)**2],
                 [1/2+sqrt(3)/4, 1/2+sqrt(3)/4, 3/4, 3/4, 3/4],
                 [1, 1, 1, 1, 1]],
            ), dtype="float64")

        normalizations = constant(
            array(
                [[(41+8*sqrt(3))/16],
                 [(13+2*sqrt(3))/4],
                 [5]]
            ), dtype="float64")

        weights_expected = target_weights_unnormalized/normalizations

        self.assertAllEqual(
            continue_flag_expected, continue_flag,
            msg="Simulation _loop_body: estimation_not_finished wrongly updated"
        )

        self.assertAllEqual(
            meas_step_expected, meas_step,
            msg="Simulation _loop_body: meas_step wrongly updated"
        )

        self.assertAllClose(
            used_resources_expected,
            used_resources,
            atol=1e-10,
            rtol=0.0,
            msg="Simulation _loop_body: used_resources wrongly updated",
        )

        self.assertAllClose(
            sum_log_prob,
            sum_log_prob_expected,
            atol=1e-10,
            rtol=0.0,
            msg="Simulation _loop_body: wrong update sum_log_prob",
        )

        self.assertAllClose(
            weights,
            weights_expected,
            atol=1e-10,
            rtol=0.0,
            msg="Simulation _loop_body: wrong update weights",
        )

        self.assertAllClose(
            particles,
            particles_expected,
            atol=1e-10,
            rtol=0.0,
            msg="Simulation _loop_body: wrong update particle",
        )

        loss_expected += ((pi**2)/9*((7+4*sqrt(3))/(41+8*sqrt(3))) **
                          2+((pi/21)**2)*((38+sqrt(3))/(13+2*sqrt(3)))**2+pi**2/25)/3
        loss_diff_expected += ((pi**2)/9*((7+4*sqrt(3))/(41+8*sqrt(3)))**2*(1-0.1+2*log(0.5*(1+cos(pi/3)))) +
                               ((pi/21)**2)*((38+sqrt(3))/(13+2*sqrt(3)))**2*(1-0.2+log(0.5*(1+cos(pi/7))))+pi**2/25*0.7)/3

        self.assertAllClose(
            loss,
            loss_expected,
            atol=1e-10,
            rtol=0.0,
            msg="Simulation _loop_body: wrong update loss",
        )

        self.assertAllClose(
            loss_diff,
            loss_diff_expected,
            atol=1e-10,
            rtol=0.0,
            msg="Simulation _loop_body: wrong update loss_diff",
        )

        self.assertEqual(
            reduce_sum(meas_step),
            constant(29, dtype="int32"),
            msg="Simulation: wrong total_step_number",
        )

    def test_execute(self):
        """Test of the execute function
        """

        rangen = Generator.from_seed(0xdeadbeef)

        loss_diff, loss = self.sim_nn.execute(rangen)

        self.assertEqual(
            loss.shape, TensorShape([]),
            msg="Simulation execute: returned loss is not scalar",
        )

        self.assertEqual(
            loss_diff.shape, TensorShape([]),
            msg="Simulation execute: returned loss_diff is not scalar",
        )


if __name__ == "__main__":
    main()
