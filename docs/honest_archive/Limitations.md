Limitations
===========

> **Status ⚠️** Mixed validated + experimental notes – review before citing
> **Navigate:** [Docs Index](index.md) · [Gradient Analysis](GradientCatastropheAnalysis.md)

Computational Approximations
----------------------------

The current implementation relies on several approximations that should be considered when interpreting results:

* **Fluid/Superposition Surrogates**: The framework uses surrogate models rather than full particle-in-cell (PIC) or fluid physics validation. While these surrogates capture key physics trends, they may not represent all aspects of real plasma behavior.

* **Coarse-Graining Scale**: The envelope/skin-depth scale assumption for coarse-graining may not accurately represent the true coupling between optical and plasma scales in all experimental configurations.

* **Surface Gravity Surrogate**: The mapping from intensity gradients to surface gravity (κ) uses simplified ponderomotive scaling with a characteristic response time (τ). These results provide trend-level insights rather than absolute quantitative predictions.

Physical Model Limitations
--------------------------

Several physical effects are simplified or not fully captured in the current model:

* **Sound Speed Profiles**: While the model supports position-dependent sound speed profiles, in many cases a uniform c_s is assumed. Realistic c_s(x) profiles, which can be induced by laser heating, can significantly shift horizon positions.

* **Magnetized Plasma Effects**: The current implementation primarily focuses on unmagnetized plasmas. Fast magnetosonic speed approximations require further validation for magnetized systems.

* **Nonlinear Plasma Effects**: The model may underestimate complex nonlinear interaction dynamics that could be important in high-intensity laser-plasma experiments.

* **Acoustic WKB Assumptions**: The acoustic-WKB graybody option constructs a tortoise coordinate via dx*/dx = 1/|c − |v|| and uses an effective, κ-scaled potential shape derived from the near-horizon gap. This captures the correct qualitative behavior (low-ω suppression, high-ω transparency) but remains a 1D, near-horizon approximation and does not include full multi-dimensional scattering or dissipation.

* **Exact Acoustic κ Evaluation**: The κ = |∂x(c_s² − v²)|/(2 c_H) form evaluates c_H via interpolation at the root; numerical grid effects can introduce small biases. Multi-stencil estimates are used to report an uncertainty band, but these are numerical rather than physical uncertainties.

* **Intensity Cap Convention**: Threshold sweeps enforce $I < 1\times10^{24}\,\text{W/m}^2$ as a theoretical 1D breakdown bound to keep comparisons consistent. This exceeds present ELI-class facilities (~$10^{23}\,\text{W/m}^2$) and should be treated as a modeling convenience rather than an experimental capability claim.

* **Experimental Enhanced Modules**: The `src/analog_hawking/physics_engine/enhanced_*` modules are experimental collaboration scaffolding. They include runtime warnings and docstring callouts until coefficients and scalings are benchmarked.

Experimental Validation Gap
---------------------------

The framework currently lacks full experimental validation:

* **WarpX Integration**: The WarpX backend implementation uses mock configurations that lack real reduced diagnostics from actual PIC simulations.

* **Fluctuation Seeding**: Full validation of fluctuation injection requires coupling with complete PIC simulations.

* **Magnetized Horizon Sweeps**: Experimental validation of magnetized horizon analysis depends on the availability of appropriate B-field diagnostics.

Future Work
-----------

Addressing these limitations represents important directions for future development:

1. **Full PIC/Fluid Validation**: Implement and validate against complete particle-in-cell and fluid simulations
2. **Advanced Coarse-Graining**: Develop more sophisticated coupling models between optical and plasma scales
3. **Experimental Implementation**: Apply the framework to design and analyze actual laser-plasma experiments
4. **Extended Physics Models**: Incorporate additional quantum field effects and nonlinear plasma phenomena

Despite these limitations, the framework provides a scientifically rigorous approach to analog Hawking radiation research that emphasizes conservative, well-validated claims over speculative results.

---

Back: [Gradient Analysis](GradientCatastropheAnalysis.md) · Next: [AnaBHEL Comparison »](AnaBHEL_Comparison.md)
