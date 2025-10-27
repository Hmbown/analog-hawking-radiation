Limitations
===========

Computational Approximations
----------------------------

The v0.3 release still relies on surrogate fluid models rather than full particle-in-cell (PIC) simulations. The recent GPU campaign reinforces the following caveats:

* **Surrogate scope:** The gradient catastrophe sweep mapped where the fluid surrogate remains internally consistent (~50% of sampled configurations). Scaling exponents (kappa proportional to a0^-0.2 ne^-0.05) are therefore model-dependent and should not be interpreted as universal until PIC comparisons are completed.
* **Coarse-graining:** Envelope and skin-depth assumptions provide qualitative trends but may misstate coupling between optical and plasma scales in steep-gradient regimes.
* **Surface gravity estimates:** The mapping from intensity gradients to kappa uses simplified ponderomotive scaling. Numerical uncertainties are reported, yet physical uncertainty must still be propagated from upstream parameter distributions.

Physical Model Limitations
--------------------------

* **Sound-speed structure:** Many studies still assume uniform c_s. Temperature-dependent c_s(x) profiles can shift horizon locations and should be included when experimental data are available.
* **Magnetisation:** Fast magnetosonic corrections are implemented in a simplified form; validation against full magnetised PIC outputs is outstanding.
* **Higher dimensions and dissipation:** Graybody calculations are one-dimensional and ignore dissipative channels. Multi-dimensional scattering effects remain outside the current scope.

Validation and Data Gaps
------------------------

* **WarpX integration:** The code paths exist but the latest GPU campaign used surrogate data. Real openPMD outputs are required to benchmark the surrogate trends.
* **Analytical baselines:** Closed-form comparisons (e.g., square-well graybody, linear gradients) need to be added as regression tests to anchor the pipeline.
* **Experimental accessibility:** Present-day laser intensities (<=1e23 W/m^2) correspond to kappa <=1e8 Hz in the surrogate models; higher values remain forward-looking targets.

Future Work
-----------

1. Acquire WarpX datasets and perform side-by-side validation against the fluid surrogate results.
2. Propagate experimental uncertainties through the horizon and detection pipelines (Monte Carlo or adjoint techniques).
3. Extend graybody and fluctuation models to higher dimensions and dissipative channels.
4. Translate the accessibility analysis into concrete experimental proposals and diagnostic requirements.

The objective is to treat the GPU-era results as a map of the surrogate landscape while transparently highlighting the steps needed to connect that map to laboratory reality.
