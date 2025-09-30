.. Quantum Sensor Optimization documentation master file, created by
   sphinx-quickstart on Tue Oct 25 18:47:13 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to Quantum Sensor Optimization's documentation!
=======================================================

``qsensoropt`` is a framework that has been created
to automate a broad class of optimizations that can be found in the
tasks of quantum parameter estimation, quantum metrology and
quantum hypothesis testing. It works both for Bayesian estimation
as well as for point estimation. ``qsensoropt`` is based
on model-aware Reinforcement Learning with policy gradient.


Getting Started
---------------

.. toctree::
   :maxdepth: 1
   :glob:

   install

Overview
--------

``qsensoropt`` is a framework written in Python
and based on Tensorflow. Its typical use case is to train
a neural network to optimally control a quantum sensor.
The framework is based on the interaction of the three classes
that are
:py:obj:`~.PhysicalModel`,
:py:obj:`~.Simulation`,
and
:py:obj:`~.ParticleFilter`.

.. image:: _static/pipeline.png
   :width: 800
   :alt: pipeline


The user is required to create a new class with the description
of the quantum probe that inherits either from
:py:obj:`~.StatefulPhysicalModel`
or
:py:obj:`~.StatelessPhysicalModel`
according to whether the probe is stateless (meaning that its state
is reinitialized after each encoding and measurement) or stateful
(the measurements affect each other through the quantum backreaction
on the probe). In this class the user must define the methods
:py:meth:`~.PhysicalModel.count_resources`,
:py:meth:`~.StatefulPhysicalModel.perform_measurement`,
:py:meth:`~.StatefulPhysicalModel.model`,
and
:py:meth:`~.StatefulPhysicalModel.initialize_state`
(this last one is only needed if the model is stateful).

If the system model is not representable analytically
in a compact form, it it possible to use a neural network
that has been calibrated to reproduce the statistics of
the physical system, thereby implementing both
applications of machine learning to quantum metrology
that have been described in the
literature [0]_.

Having specified the physics of the probe
the user is now asked to define a class
specifying how the particle filter should
interact with the neural network and what is
the error in the metrological task. This is
done by deriving either the class
:py:obj:`~.StatefulSimulation`
or
:py:obj:`~.StatelessSimulation`
and by implementing the two methods
:py:meth:`~.StatefulSimulation.generate_input`
and
:py:meth:`~.StatefulSimulation.loss_function`.

If the users tasks
is a typical quantum metrological problem, where the loss
is the Mean Square Error, and where we want the neural network
to produce the optimal control based on the first and
second moments of the particle filter,
then it is possible to use directly the classes
:py:obj:`~.StatelessMetrology`
or
:py:obj:`~.StatefulMetrology`
for the simulation, without the need of defining a
new class.

At this point we are ready to instantiate the
classes created, passing to the constructors the
parameters to tune the estimation.

At this point having specified the details of the
physical model and of the simulation, the user is ready
to instantiate the two define class and the
:py:obj:`~.ParticleFilter`. Calling then the function
:py:func:`~.utils.train` on the simulation will
train the network in controlling the sensor, while
the function :py:func:`~.utils.performance_evaluation`
evaluates its performances


.. [0] Advanced Photonics, Vol. 5, Issue 2, 020501 (March 2023).


API documentation
-----------------
All of the APIs are documented here

.. toctree::
   :maxdepth: 1
   :glob:

   qsensoropt

Examples
--------
Several examples can be found in this repository

.. toctree::
   :maxdepth: 1
   :glob:

   examples

Acknowledgement
==================
We gratefully acknowledge computational
resources of the Center for High Performance
Computing (CHPC) at SNS.

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
