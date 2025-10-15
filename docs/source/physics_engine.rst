Physics Engine Documentation
============================

The physics engine module provides the core computational framework for simulating analog Hawking radiation in laser-plasma systems.

Overview
--------

The physics engine is organized into several submodules that handle different aspects of the simulation:

* ``plasma_models`` - Fundamental plasma physics models and interactions
* ``optimization`` - Optimization algorithms for parameter search
* ``horizon`` - Horizon detection and surface gravity calculation
* ``multi_beam_superposition`` - Multi-beam field superposition calculations
* ``simulation`` - Main simulation runner and orchestration

Plasma Models
-------------

The ``plasma_models`` submodule contains the fundamental physics implementations:

plasma_physics
~~~~~~~~~~~~~~

.. automodule:: analog_hawking.physics_engine.plasma_models.plasma_physics
   :members:
   :undoc-members:
   :show-inheritance:

laser_plasma_interaction
~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: analog_hawking.physics_engine.plasma_models.laser_plasma_interaction
   :members:
   :undoc-members:
   :show-inheritance:

quantum_field_theory
~~~~~~~~~~~~~~~~~~~~

.. automodule:: analog_hawking.physics_engine.plasma_models.quantum_field_theory
   :members:
   :undoc-members:
   :show-inheritance:

Horizon Detection
-----------------

The ``horizon`` module provides robust algorithms for detecting analog event horizons in plasma flows.

horizon
~~~~~~~

.. automodule:: analog_hawking.physics_engine.horizon
   :members:
   :undoc-members:
   :show-inheritance:

Multi-Beam Superposition
------------------------

The ``multi_beam_superposition`` module calculates time-averaged intensity gradients from multiple coherent laser beams.

multi_beam_superposition
~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: analog_hawking.physics_engine.multi_beam_superposition
   :members:
   :undoc-members:
   :show-inheritance:

Optimization Framework
----------------------

The ``optimization`` submodule provides algorithms for optimizing experimental parameters.

merit_function
~~~~~~~~~~~~~~

.. automodule:: analog_hawking.physics_engine.optimization.merit_function
   :members:
   :undoc-members:
   :show-inheritance:

probabilistic_horizon
~~~~~~~~~~~~~~~~~~~~~

.. automodule:: analog_hawking.physics_engine.optimization.probabilistic_horizon
   :members:
   :undoc-members:
   :show-inheritance:

snr_model
~~~~~~~~~

.. automodule:: analog_hawking.physics_engine.optimization.snr_model
   :members:
   :undoc-members:
   :show-inheritance:

graybody_1d
~~~~~~~~~~~

.. automodule:: analog_hawking.physics_engine.optimization.graybody_1d
   :members:
   :undoc-members:
   :show-inheritance: