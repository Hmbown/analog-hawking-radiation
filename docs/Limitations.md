Limitations
===========

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