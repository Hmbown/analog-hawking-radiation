Final Answer: What This Work Shows (and Doesn’t)
================================================

Scope
-----
This repository does not prove the existence or detectability of analog Hawking radiation in a specific apparatus. Instead, it establishes a rigorous, conservative workflow that helps researchers prioritize where/when to look for signals and why horizon formation is the key bottleneck.

What We Built
-------------
- Robust horizon detection with uncertainty (|v| = c_s roots, κ ± err).
- Power‑conserving, envelope‑scale field superposition for multi‑beam configurations (no naïve multipliers).
- Radio‑band radiometer SNR modeling and sweeps driven directly from the QFT spectrum.

What It Doesn’t Prove
---------------------
- No claim of a formed analog horizon in the current parameter space.
- No claim that a specific geometry guarantees detection.
- No claim that the envelope‑scale gradient proxy equals true κ from full PIC/fluid simulations.

What It Does Demonstrate
------------------------
1) Formation, not detection, is the limiting physics.
   - The robust horizon finder (κ ± err) rarely returns horizons in current regimes; this shifts effort from “how to detect” to “how to form.”

2) Multi‑beam power splitting does not yield naïve N× enhancement at envelope scales.
   - Conserved‑power, coarse‑grained superposition yields ~1× (± modest) gradient enhancement depending on geometry and coarse‑graining scale. See figures/enhancement_bar.png and figures/phase_jitter_stability.png.

3) Where/when to look is quantifiable in the radio band.
   - QFT‑based radiometer sweeps (time‑to‑5σ vs T_sys and bandwidth B) give a sober feasibility map. See figures/radio_snr_from_qft.png and figures/radio_snr_sweep.png.

Practical Guidance: How This Helps Researchers
---------------------------------------------
- Horizon formation focus: Use the horizon finder on simulated or experimental velocity profiles to quantify κ and uncertainty. If |v| never meets c_s, adjust plasma/density/temperature/drive, not the detector.
- Envelope scale matters: Choose geometries and incidence angles that set envelope gradients (fringe or beat length) close to skin‑depth/envelope scales; optical‑fringe gradients don’t survive coarse‑graining.
- Radio detection planning: For plausible T_H in the mK regime, build SNR budgets using T_sys and B; the sweeps show where integration time is practical.
- Bayesian guidance (next): Combine horizon probability (from κ ± err), QFT power, and instrument priors into a posterior over parameters to identify the highest‑value regions to test.

Recommended Next Experiments
----------------------------
- Match envelope scales: Small‑angle crossings and two‑color beat‑waves with beat length ≈ δ (skin depth).
- Spatiotemporal focusing: Increase effective envelope gradients via short τ_response; evaluate κ surrogate sensitivity.
- Magnetized horizons: Replace c_s with fast magnetosonic speed and repeat horizon finder.
- Reduced PIC/fluid cross‑checks: Validate dv/dx near focus and refine surrogate κ mapping.

Reproducibility
---------------
- Make all figures: `make figures`
- Validate gating: `make validate`
- Enhancement summary: `make enhancements`

Figures
-------
- figures/enhancement_bar.png — Multi‑beam enhancement (conserved power, coarse‑grained)
- figures/phase_jitter_stability.png — Enhancement robustness under random phases
- figures/radio_snr_from_qft.png — Time‑to‑5σ heatmap from QFT spectrum
- figures/radio_snr_sweep.png — Synthetic sweep for rapid sanity checks

Bottom Line
-----------
This work doesn’t claim a detection pathway today; it provides principled tools that help decide where to invest effort tomorrow. By quantifying horizon formation, envelope‑scale gradient limits, and radio SNR feasibility from first principles, it reduces guesswork and concentrates experimental effort where it matters most.

