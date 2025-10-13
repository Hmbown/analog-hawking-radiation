Methods
-------

Horizon Finder
--------------
- f(x) = |v(x)| − c_s; roots via bracketing and bisection
- κ = 0.5 |d/dx(|v| − c_s)| at the root; uncertainty from multi-stencil (±1,2,3) central differences

Sound Speed Profile
--------------------
- The sound speed `c_s` can be initialized as a constant or as a 1D position-dependent profile `c_s(x)`.
- This allows for more realistic modeling where laser-plasma interactions create a non-uniform temperature profile, directly affecting the horizon condition `|v(x)| = c_s(x)`.

Multi-Beam Superposition
------------------------
- Coherent Gaussian beams; total peak power conserved (weights sum to 1)
- Time-averaged intensity on a 2D grid; coarse-grained using a Gaussian kernel (≈ envelope/skin-depth scale)
- Gradient enhancement = max |∇I| within a small radius vs. single-beam baseline (equal total power)
- Optional κ surrogate via U_p ∝ E²/ω², a ∝ −∇U_p, v ≈ a τ, κ ∝ 0.5 |∂|v|/∂x|
- Shapes: rings (N beams), small-angle crossings (θ), non-equal weights, lab-fixed elliptical waists, two-color beat (Δλ/λ)

Radio SNR Modeling
------------------
- Radiometer: SNR = (T_sig/T_sys) √(B t); t_5σ = (5 T_sys / (T_sig √B))²
- T_sig from QFT spectrum via band power P_sig and T_sig = P_sig/(k B)

Bayesian-Style Guidance (Surrogate)
-----------------------------------
- Envelope matching: enhancement peaks near Λ≈δ; κ ≈ κ0×enhancement
- Radiometer feasibility from T_sig and T_sys, B → t_5σ; score ∝ 1/t_5σ
