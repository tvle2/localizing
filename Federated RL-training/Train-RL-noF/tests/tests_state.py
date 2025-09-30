#!/usr/bin/env python3
from tensorflow.test import TestCase, main
from tensorflow.random import set_seed, Generator
from tensorflow import constant, ones, concat, \
    zeros, cond
from numpy import array
from numpy.random import seed
from math import pi

from qsensoropt import ParticleFilter, \
    Parameter, SimulationParameters, StatelessMetrology

from qsensoropt.utils import standard_model

from stateful_interferometer import ObservedDamageInterferometer

set_seed(1)
seed(1)

class TestStates(TestCase):

    def setUp(self):

        self.prec = "float32"

        self.network = standard_model(
            input_size=10, controls_size=1,
            neurons_per_layer=4,
            prec=self.prec,
            )
        self.network.compile()

        self.interferometer = ObservedDamageInterferometer(
            batchsize=3,
            params=[
                Parameter(bounds=(0.0, 2*pi),
                          name="phase"),
                Parameter(bounds=(0.95, 1.00),
                          name="visibility"),
            ],
            prec=self.prec,
        )

        self.pf = ParticleFilter(
            num_particles=5,
            phys_model=self.interferometer,
            resample_fraction=0.98, 
            prec=self.prec,
        )

        self.simpars = SimulationParameters(
            sim_name="interferometer_stateful",
            num_steps=12,
            max_resources=12,
            cumulative_loss=True,
            prec=self.prec
        )

        self.sim_nn = StatelessMetrology(
            particle_filter=self.pf,
            simpars=self.simpars,
            phys_model=self.interferometer,
            control_strategy=self.network,
        )

    def test_state_update(self):

        num_steps = 12

        # The last estimation has been completed
        continue_flag = constant(array(
            [[True], [True], [True]]), dtype="bool",
        )

        rangen = Generator.from_seed(0xdeadbeef)

        meas_step = constant(array(
            [[12], [7], [7]]), dtype="int32",
        )

        used_resources = constant(array(
            [[12.0], [7.0], [7.0]]), dtype=self.prec,
        )

        # Manual test of the application of the Bayes rule.
        weights, _ = self.pf.reset(rangen)

        particles_phase = constant(
            array(
                [[[pi/12], [pi/11], [pi/10], [pi/9], [pi/8]],
                 [[pi/13], [pi/12], [pi/11], [pi/10], [pi/9]],
                 [[pi/14], [pi/13], [pi/12], [pi/11], [pi/10]]]
            ), dtype=self.prec)

        particles_visibility = ones((3, 5, 1), dtype=self.prec)

        particles = concat([particles_phase, particles_visibility], 2)

        particles = self.pf._trim_particles(particles)

        true_values = constant(
            array([[[pi/3, 1.0]],
                   [[pi/7, 1.0]],
                   [[pi/5, 1.0]]]
                  ), dtype=self.prec)

        state_filter = self.pf.phys_model.wrapper_initialize_state(
            particles, self.pf.np,
        )

        true_state = self.interferometer.wrapper_initialize_state(
            true_values, 1,
            )

        index = constant(10, dtype="int32")

        sum_log_prob = constant(
            array([[-0.5],
                   [-0.5],
                   [-0.5]]
                  ), dtype=self.prec)
        
        outcomes = zeros((3,  1), dtype="float64")

        loss = zeros((), dtype=self.prec)

        loss_diff = zeros((), dtype=self.prec)

        history_input = zeros((num_steps, 3, 10), dtype=self.prec)
        history_control = zeros((num_steps, 3, 1), dtype=self.prec)
        history_resources = zeros((num_steps, 3, 1), dtype=self.prec)
        history_precision = zeros((num_steps, 3, 1), dtype=self.prec)

        hist_outcomes_res = zeros((num_steps, 3, 2), dtype=self.prec)
        hist_control_res = zeros((num_steps, 3, 1), dtype=self.prec)
        hist_enf_res = zeros((num_steps, 3, 1), dtype="int32")
        hist_step_res = zeros((num_steps, 3, 1), dtype="int32")

        self.sim_nn.recompute_state = False

        past_true_state = true_state
        past_state_filter = state_filter

        # First execution of the body of the loop
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

        self.assertAllClose(
            state_filter[0, :, :],
            past_state_filter[0, :, :],
            atol=1e-10,
            rtol=0.0,
            msg="Simulation: state_filter is not updated correctly",
        )

        self.assertAllClose(
            true_state[0, :, :],
            past_true_state[0, :, :],
            atol=1e-10,
            rtol=0.0,
            msg="Simulation: true_state is not updated correctly",
        )

        self.assertNotAllClose(
            true_state[1:3, :, :],
            past_true_state[1:3, :, :],
            atol=1e-10,
            rtol=0.0,
            msg="Simulation: true_state is not updated correctly",
        )

        self.assertNotAllClose(
            state_filter[1:3, :, :],
            past_state_filter[1:3, :, :],
            atol=1e-10,
            rtol=0.0,
            msg="Simulation: state_filter is not updated correctly",
        )

    def test_resample_state(self):

        num_steps = 10

        # self.sim_nn.control_strategy = \
        #     lambda x: zeros((self.sim_nn.bs, 1), dtype=self.prec)

        # The last estimation has been completed
        continue_flag = constant(array(
            [[True], [True], [True]]), dtype="bool",
        )

        rangen = Generator.from_seed(0xdeadbeef)

        meas_step = constant(array(
            [[0], [0], [0]]), dtype="int32",
        )

        used_resources = constant(array(
            [[0.0], [0.0], [0.0]]), dtype=self.prec,
        )

        # Manual test of the application of the Bayes rule.
        weights, _ = self.pf.reset(rangen)

        particles_phase = constant(
            array(
                [[[pi/12], [pi/11], [pi/10], [pi/9], [pi/8]],
                 [[pi/13], [pi/12], [pi/11], [pi/10], [pi/9]],
                 [[pi/14], [pi/13], [pi/12], [pi/11], [pi/10]]]
            ), dtype=self.prec)

        particles_visibility = ones((3, 5, 1), dtype=self.prec)

        particles = concat(
            [particles_phase, particles_visibility], 2,
        )

        particles = self.pf._trim_particles(particles)

        state_filter = self.pf.phys_model.wrapper_initialize_state(
            particles, self.pf.np,
        )

        true_values = constant(
            array([[[pi/3, 1.0]],
                   [[pi/7, 1.0]],
                   [[pi/5, 1.0]]]
                  ), dtype=self.prec)

        true_state = self.interferometer.wrapper_initialize_state(
            true_values, 1,
            )
        
        outcomes = zeros((3,  1), dtype="float64")

        index = constant(0, dtype="int32")

        sum_log_prob = constant(
            array([[0.0],
                   [0.0],
                   [0.0]]
                  ), dtype=self.prec)

        loss = zeros((), dtype=self.prec)

        loss_diff = zeros((), dtype=self.prec)

        history_input = zeros((num_steps, 3, 10), dtype=self.prec)
        history_control = zeros((num_steps, 3, 1), dtype=self.prec)
        history_resources = zeros((num_steps, 3, 1), dtype=self.prec)
        history_precision = zeros((num_steps, 3, 1), dtype=self.prec)

        hist_outcomes_res = zeros((num_steps, 3, 2), dtype=self.prec)
        hist_control_res = zeros((num_steps, 3, 1), dtype=self.prec)
        hist_enf_res = zeros((num_steps, 3, 1), dtype="int32")
        hist_step_res = zeros((num_steps, 3, 1), dtype="int32")

        self.sim_nn.recompute_state = False

        for _ in range(5):

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

        expected_state_filter = state_filter

        state_filter = cond(constant(True),
                            lambda: self.pf.recompute_state(
            index, particles, hist_control_res,
            hist_outcomes_res, hist_enf_res, hist_step_res,
            num_steps,
        ),
            lambda: zeros((3, 5, 1), dtype=self.prec),
            name="state_resampling",
        )

        self.assertAllClose(
            state_filter,
            expected_state_filter,
            atol=1e-10,
            rtol=0.0,
            msg="PhysModel: state_resampling doesn't work as intended",
        )

        self.assertAllClose(
            index,
            constant(5, dtype="int32"),
            atol=1e-10,
            rtol=0.0,
            msg="Simulation: index is not correctly updated",
        )

if __name__ == "__main__":
    main()
