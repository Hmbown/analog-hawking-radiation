Overview
========

The Analog Hawking Radiation Simulation Framework is a comprehensive computational tool designed to simulate and analyze analog Hawking radiation in laser-plasma systems. This framework addresses the fundamental challenges in detecting quantum effects in laboratory analogs of black holes.

Computational Framework Architecture
------------------------------------

The framework follows a modular architecture that separates different physical domains while maintaining tight integration between components:

.. code-block:: text

   ┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
   │   Plasma Models │    │ Horizon Detection │    │  Quantum Field  │
   │                 │───▶│                   │───▶│     Theory      │
   │ - Fluid Backend │    │ - κ calculation   │    │ - Hawking Spec  │
   │ - WarpX Backend │    │ - Uncertainty est │    │ - Graybody corr │
   └─────────────────┘    └──────────────────┘    └─────────────────┘
            │                       │                       │
            ▼                       ▼                       ▼
   ┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
   │ Multi-Beam      │    │ Detection        │    │ Optimization    │
   │ Superposition   │    │ Modeling         │    │ Framework       │
   │                 │    │                  │    │                 │
   │ - Power consv   │    │ - Radio SNR      │    │ - Bayesian opt  │
   │ - Envelope scale│    │ - Integration t  │    │ - Merit func    │
   └─────────────────┘    └──────────────────┘    └─────────────────┘

Core Methodological Components
------------------------------

Horizon Detection System
~~~~~~~~~~~~~~~~~~~~~~~~

The horizon detection system provides robust algorithms for identifying analog event horizons in plasma flows:

* **Robust Root Finding**: Bracketing and bisection methods for solving :math:`|v(x)| = c_s(x)` condition
* **Surface Gravity Calculation**: :math:`\kappa = \frac{1}{2}\left|\frac{d}{dx}(|v| - c_s)\right|` with multi-stencil uncertainty quantification
* **Adaptive Smoothing**: κ-plateau diagnostics for optimal scale selection

Multi-Beam Field Superposition
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The multi-beam superposition module enables realistic modeling of complex laser configurations:

* **Power-Conserving Design**: Total peak power normalization across beam configurations
* **Envelope-Scale Coarse-Graining**: Realistic gradient modeling at skin-depth scales
* **Geometric Configurations**: Support for rings, crossings, standing waves with variable parameters

Detection Feasibility Modeling
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The detection modeling component provides practical assessments of experimental feasibility:

* **Quantum Field Theory Integration**: Direct Hawking spectrum calculation from first principles
* **Radiometer Equation**: :math:`\text{SNR} = \frac{T_{\text{sig}}}{T_{\text{sys}}}\sqrt{B \cdot t}` for detection time estimation
* **Graybody Transmission**: WKB-based transmission probability calculations

Bayesian Optimization Framework
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The optimization framework enables efficient parameter space exploration:

* **Unified Merit Function**: :math:`\text{Merit} = P_{\text{horizon}} \times E[\text{SNR}(T_H(\kappa))]`
* **Parameter Space Exploration**: Systematic exploration of plasma density, laser intensity, and temperature
* **Uncertainty Propagation**: Probabilistic horizon modeling with parameter uncertainties

Scientific Validation
---------------------

The framework has been extensively validated against known analytical solutions and demonstrates proper numerical convergence:

* **Unit/Formula Checks**: Plasma frequency, relativistic parameter :math:`a_0`, and Hawking temperature from :math:`\kappa` match analytical expressions
* **Convergence Testing**: Spatial and temporal convergence verified through grid refinement studies
* **Physical Consistency**: Parameter ranges and physical plausibility checks ensure realistic results

Reproducibility
---------------

The framework emphasizes reproducibility through:

* **Version Control**: All code and documentation maintained in a git repository
* **Dependency Management**: Explicit specification of required Python packages
* **Validation Protocols**: Automated testing suite to verify correct implementation
* **Documentation**: Comprehensive documentation of all modules and functions