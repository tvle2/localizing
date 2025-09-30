#!/usr/bin/env python3
from typing import Literal, Callable, List, Optional, \
    Tuple

from tensorflow import cast, stop_gradient, ones, \
    gather, concat, reshape, norm, expand_dims, \
        broadcast_to, constant, \
            Variable, Tensor
from tensorflow.math import exp, log, cos, abs, minimum, \
    reciprocal_no_nan, round
from tensorflow.linalg import trace, tensor_diag, matmul
from tensorflow.random import stateless_uniform, uniform, \
    Generator

from itertools import product
from argparse import ArgumentParser
from os.path import join
from math import pi

from src.qsensoropt import StatelessPhysicalModel, \
    StatelessMetrology, InverseSqrtDecay, ParticleFilter, \
        Parameter, Control, SimulationParameters
from src.qsensoropt.utils import train, \
    performance_evaluation, get_seed, store_input_control, \
        standard_model, denormalize

class NVCenter(StatelessPhysicalModel):
    r"""Model for the negatively charged NV center in
    diamond used for various quantum metrological
    task. A single measurement on the NV center
    consists in multiple Ramsey sequencies of the same
    controllable duration applied to the NV center,
    followed by photon counting of the photoluminescent
    photon and a majority voting to decide the
    binary outcome. The NV center is reinitialized
    after each photon counting.
    During the free evolution in the Ramsey sequence
    the NV center precesses freely in the external
    magnetic field, thereby encoding its value
    in its state. The two possible controls we
    have on the system are the duration of the
    free evolution :math:`\tau` and the phase
    :math:`\phi` applied before
    the photon counting.
    The resource of the estimation task
    can be either the total number of measurements or
    the total evolution time.

    **Achtung!** The :py:meth:`~.StatelessPhysicalModel.model`
    method must still be implemented in this class. It
    should describe the probability of getting
    :math:`+1` in the measurement after the majority voting
    from the collected photons.
    """
    def __init__(
        self, batchsize: int, params: List[Parameter],
        prec: Literal["float64", "float32"] = "float64",
        res: Literal["meas", "time"] = "time",
        control_phase: bool = False,
        ):
        r"""Constructor
        of the :py:obj:`~.NVCenter` class.

        Parameters
        ----------
        batchsize: int
            Batchsize of the simulation, i.e. number of
            estimations executed simultaneously.
        params: List[:py:obj:`~.Parameter`]
            List of unknown parameters to estimate in
            the NV center experiment, with their
            respective bounds. 
        prec : {"float64", "float32"}
            Precision of the floating point operations in the 
            simulation.
        res: {"meas", "time"}
            Resource type for the present metrological task, 
            can be either the total evolution time, i.e. `time`,
            or the total number of measurements on
            the NV center, i.e. `meas`.
        control_phase: bool = False
            If this flag is `True`, beside the free evolution time,
            also the phase applied
            before the photon counting is controlled by
            the agent.
        """
        self.control_phase = control_phase

        if not res in ("meas", "time"):
            raise ValueError("The allowed values of \
                             res are time and res.")
        
        if self.control_phase:
            # Controls the time and the phase.
            controls=[
                Control(name="EvolutionTime"),
                Control(name="Phase"),
            ]
        else:
            # The only control is the free precession
            # time of the NV center
            # in the magnetic field, between the two pi/2-pulses.
            controls=[
                Control(name="EvolutionTime"),
            ]
        
        super().__init__(
            batchsize, controls, params,
            prec=prec,
        )

        self.res = res

    def perform_measurement(
        self, controls: Tensor, parameters: Tensor,
        meas_step: Tensor,
        rangen: Generator,
        ):
        r"""Measurement on the NV center.
        
        The NV center is measured after having evolved freely in the
        magnetic field for a time specified by the parameter
        `control`. The possible outcomes
        are :math:`+1` and :math:`-1`,
        selected stochastically according to the probabilities
        :math:`p(\pm 1| \vec{\theta}, \tau)`, where :math:`\tau` is the
        evolution time (the control) and :math:`\vec{\theta}`
        the parameters to estimate. This method
        returns the observed outcomes
        and their log-likelihood.

        **Achtung!** The :py:meth:`model` method must return
        the probability :math:`p(\pm 1| \vec{\theta}, \tau)`
        """
        list_plus = ones((self.bs, 1, 1), dtype=self.prec)
        prob_plus = self.model(
            list_plus, controls, parameters, meas_step,
            )
        # Extraction of the actual outcomes
        seed = get_seed(rangen)
        outcomes = 2*cast((stateless_uniform((self.bs, 1),
                        seed, dtype=self.prec) <
                        stop_gradient(abs(prob_plus))),
                        dtype="int8") - 1
        outcomes = cast(
            expand_dims(outcomes, axis=1),
            dtype=self.prec, name="outcomes",
            )
        prob_outcomes = self.model(
            outcomes, controls, parameters, meas_step,
            )
        log_prob = cast(
            log(prob_outcomes),
            dtype=self.prec, name="log_prob",
            )
        return outcomes, log_prob

    def count_resources(
        self, resources: Tensor, outcomes: Tensor,
        controls: Tensor, true_values: Tensor,
        meas_step: Tensor,
        ):
        """The resources can be either the
        total number of measurements, or the total
        evolution time, according to the
        attribute `res` of the :py:obj:`NVCenter`
        class.
        """
        if self.res == "time":
            return resources+abs(controls[:, 0:1])+240#5#25.0
        return resources+240#5#25.0#1.0


class NVCenterDCMagnetometry(NVCenter):
    r"""Model describing the estimation of a
    static magnetic field with an NV center
    used as magnetometer.
    The spin-spin relaxation time :math:`T_2^{-1}` can
    be either known, or be a parameter to
    estimate. The estimation will be
    formulated in terms of the precession
    frequency :math:`\omega` of the NV center, which
    is proportional to the magnetic
    field :math:`B`.

    This physical model and the application of 
    Reinforcement Learning to the estimation
    of a static magnetic fields have been also studied in the
    seminal work of Fiderer, Schuff and Braun [1]_.

    .. [1] Lukas J. Fiderer, Jonas Schuff, and Daniel Braun
        PRX Quantum 2, 020303 (2021).
    """
    def __init__(
        self, batchsize: int, params: List[Parameter],
        prec: Literal["float64", "float32"] = "float64",
        res: Literal["meas", "time"] = "meas",
        invT2: Optional[float] = None,
        ):
        r"""Constructor
        of the :py:obj:`~.NVCenterDCMagnetometry` class.

        Parameters
        ----------
        batchsize: int
            Batchsize of the simulation, i.e. number of estimations
            executed simultaneously.
        params: List[:py:obj:`~.Parameter`]
            List of unknown parameters to estimate in
            the NV center experiment, with their
            respective bounds.
        prec : {"float64", "float32"}
            Precision of the floating point operations in the 
            simulation.
        res: {"meas", "time"}
            Resource type for the present metrological task, 
            can be either the total evolution time, i.e. `time`,
            or the total number of measurements on
            the NV center, i.e. `meas`.
        invT2: float, optional
            If this parameter is specified only the precession
            frequency :math:`\omega` is considered as an unknown
            parameter, while the inverse of the
            transverse relaxation time :math:`T_2^{-1}` is fixed
            to the value `invT2`. In this case the list `params`
            must contain a single parameter, i.e. `omega`.
            If no `invT2` is specified,
            it is assumed that it is an unknown parameter that
            will be estimated along the frequency in the Bayesian
            procedure. In this case `params` must contain two
            objects, the second of them should be the inverse of the
            transversal relaxation time.
        """
        super().__init__(
            batchsize, params, prec=prec, res=res,
        )

        self.invT2 = invT2

    def model(
        self, outcomes: Tensor, controls: Tensor,
        parameters: Tensor, meas_step: Tensor,
        num_systems: int = 1,
        ) -> Tensor:
        r"""Model for the outcome of a measurement
        on a NV center subject to free precession in a static
        magnetic field.  The probability of getting the
        outcome :math:`+1` is

        .. math::

            p(+1|\omega, T_2, \tau) := e^{-\frac{\tau}{T_2}}
            \cos^2 \left( \frac{\omega}{2} \tau \right) +
            \frac{1-e^{-\frac{\tau}{T_2}}}{2} \; .

        The evolution time :math:`\tau` is controlled
        by the trainable agent, and :math:`\omega`
        is the unknown precession frequency, which is
        proportional to the magnetic field.
        The parameter :math:`T_2`
        may or may not be an unknown in the estimation,
        according to the value of the attribute `invT2`
        of the :py:obj:`~.NVCenterDCMagnetometry` class.
        """
        evolution_time = controls[:, :, 0]
        omega = parameters[:, :, 0]
        if self.invT2 is not None:
            invT2 = self.invT2
        else:
            invT2 = parameters[:, :, 1]
        exp_decay = exp(-evolution_time*invT2)
        ramsey_out = outcomes[:, :, 0]
        noise_less = (1.0-ramsey_out*cos(omega*evolution_time))/2.0
        return exp_decay*noise_less + (1.0-exp_decay)/2.0
    
class Magnetometry(StatelessMetrology):
    r"""Simulates the estimation of a magnetic field
    with a mean square error loss. This class is suitable
    for a neural network agent, for a static strategy,
    and for other simple controls known in
    the literature, like the :math:`\sigma^{-1}` strategy
    and the particle guess heuristic (PGH).
    It works both for static and for oscillating
    magnetic fields.
    """
    def __init__(
            self, particle_filter: ParticleFilter,
            phys_model: NVCenter,
            control_strategy: Callable,
            simpars: SimulationParameters,
            cov_weight_matrix=None,
            eta_flag: bool = False,
            extraction_flag: bool = False,
            cov_flag: bool = False,
    ):
        r"""Constructor of the 
        :py:obj:`Magnetometry` class.

        Parameters
        ----------
        particle_filter: :py:obj:`~.ParticleFilter`
            Particle filter responsible for the update
            of the Bayesian posterior on the parameters
            and on the state of the probe. It
            contains the methods for applying the Bayes
            rule and computing Bayesian estimators
            from the posterior.
        phys_model: :py:obj:`~.NVCenter`
            Abstract description of the parameters
            encoding and of the measurement on the
            NV center.
        control_strategy: Callable
            Callable object (normally a
            function or a lambda function) that
            computes the values of the controls
            for the next measurement from
            the `Tensor` `input_strategy`.
            This class expects a callable
            with the following
            header

            ``controls = 
            control_strategy(input_strategy)``

            It is typically a wrapper for the
            neural network or a vector of
            static controls.
        simpars: :py:obj:`~.SimulationParameters`
            Contains the flags and parameters
            that regulate the stopping
            condition of the measurement loop
            and modify the loss function used in the
            training.
        cov_weight_matrix: List, optional
            Weight matrix that determines the relative
            contribution to the total error of the
            parameters in `phys_model.params`.
            It is list of `float` representing
            a positive semidefinite matrix.
            If this parameter is not passed then the
            default weight matrix is the identity, i.e.
            :math:`G=\text{Id}`.
        eta_flag: bool = False
            This flag
            controls the addition of a normalization factor
            to the MSE loss.

            If `eta_flag` is True, the MSE loss is divided by the
            normalization factor
            
            .. math::
                \eta = \min \left( \sum_{i=1}^d G_{ii}
                \frac{(b_i-a_i)}{12}, \frac{1}{T} \right) \; ,

            where :math:`(a_i, b_i)` are the bounds
            of the `i`-th parameter in `phys_model.params`
            and :math:`G_{ii}` are the diagonal entries of
            `cov_weight_matrix`. :math:`T` is the total
            elapsed evolution time, which can be different
            for each estimation in the batch.

            **Achtung!** This flag should be used only
            if the resource is the total estimation time.
        extraction_flag: bool = False
            If `extraction_flag=True` a couple of
            particles are sampled from the posterior
            and added to the `input_strategy` `Tensor`.
            This is useful to simulate the PGH control for
            the evolution time, according to which the `k-th`
            control should be

            .. math::
                \tau_k = \frac{1}{||\vec{x}_1 - \vec{x}_2||
                + \epsilon} \; ,
                :label: PGH_tau

            with :math:`\epsilon \ll 1`,
            where :math:`\vec{x}_1` and :math:`\vec{x}_2` are
            respectively the first and the second particle
            extracted from the ensemble and :math:`||\cdot||`
            is the cartesian norm.
        cov_flag: bool = False
            If `cov_flag=True` a flattened version of the
            covariance matrix of the particle filter
            ensemble is added to the `input_strategy` `Tensor`.
            This is useful to simulate the :math:`\sigma^{-1}`
            control strategy and its variant that accounts
            for a finite
            transversal relaxation time. They prescribe
            respectively for the `k-th` control

            .. math::
                \tau_k = \frac{1}{\left[ \text{tr} 
                (\Sigma) \right]^{\frac{1}{2}} } \; ,
                :label: cov_1
            
            and

            .. math::
                \tau_k = \frac{1}{\left[ \text{tr} 
                (\Sigma) \right]^{\frac{1}{2}} +
                \widehat{T_2^{-1}}} \; ,
                :label: cov_2
            
            being :math:`\Sigma` the covariance matrix
            of the posterior computed with the
            :py:meth:`~.ParticleFilter.compute_covariance`
            method.
        """
        super().__init__(
            particle_filter, phys_model, control_strategy,
            simpars, cov_weight_matrix,
            )

        d = self.pf.d
        self.eta_flag = eta_flag
        self.extraction_flag = extraction_flag
        self.cov_flag = cov_flag

        if self.extraction_flag:
            self.input_size += 2*d
            self.input_name +=  [f"Ext1_{par.name}" \
                                 for par in self.pf.phys_model.params] + \
                [f"Ext2_{par.name}" for par in self.pf.phys_model.params]
        if self.cov_flag:
            self.input_size += d**2
            self.input_name += [f"Cov_{par1.name}_{par2.name}" \
                                for par1, par2 \
                    in product(self.pf.phys_model.params,
                               self.pf.phys_model.params)]

    def generate_input(
        self, weights: Tensor,
        particles: Tensor,
        meas_step: Tensor,
        used_resources: Tensor,
        rangen: Generator,
        ):
        """Generates the `input_strategy` `Tensor`
        of the :py:meth:`~.StatelessMetrology.generate_input`
        method.  The returned `Tensor` is influenced
        by the parameters `extract_flag` and
        `cov_flag` of the constructor.
        """
        d = self.pf.d
        input_tensor = super().generate_input(
            weights, particles, meas_step, used_resources,
            rangen,
            )

        if self.extraction_flag:
            two_particles = self.pf.extract_particles(
                weights, particles, 2, rangen,
                )
            input_tensor = concat(
                [input_tensor, reshape(two_particles,
                                       (self.bs, 2*d))], 1,
                )
        if self.cov_flag:
            cov = self.pf.compute_covariance(weights, particles)
            input_tensor = concat(
                [input_tensor, reshape(cov, (self.bs, d**2))], 1,
            )
        return input_tensor
        
    def loss_function(
            self, weights: Tensor, particles: Tensor,
            true_values: Tensor, used_resources: Tensor,
            meas_step: Tensor,
    ):
        """Mean square error on the parameters,
        as computed in :py:meth:`~.StatelessMetrology.loss_function`.
        The behavior of this
        method is influence by the flag `eta_flag` passed to the
        constructor of the class.
        """
        pars = self.simpars
        loss_values = super().loss_function(
                weights, particles, true_values,
                used_resources, meas_step,
            )
        d = len(self.pf.phys_model.params)
        if self.eta_flag:
            init_var = []
            for param in self.pf.phys_model.params:
                bounds = param.bounds
                init_var.append(1/12*(bounds[1]-bounds[0])**2)
            initial_loss = broadcast_to(reshape(
                tensor_diag(constant(init_var, dtype=pars.prec)),
                (1, d, d),
                ), (self.bs, d, d),
            )
            initial_loss_scalar = expand_dims(
                trace(matmul(
                    self.cov_weight_matrix_tensor, initial_loss),
                    ), axis=1,
                    )
            eta = minimum(
                reciprocal_no_nan(used_resources),
                initial_loss_scalar,
            )
            loss_values = loss_values/eta
        return loss_values
