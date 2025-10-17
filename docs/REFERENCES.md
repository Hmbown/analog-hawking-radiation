# References

This repository draws on foundational literature in analog gravity, plasma physics, black hole radiation, and radiometry. Key references are listed for context and further reading.

- Hawking, S. W. (1974). Black hole explosions? Nature, 248, 30–31. doi:10.1038/248030a0
- Hawking, S. W. (1975). Particle creation by black holes. Communications in Mathematical Physics, 43, 199–220. doi:10.1007/BF02345020
- Unruh, W. G. (1981). Experimental black-hole evaporation? Physical Review Letters, 46(21), 1351–1353. doi:10.1103/PhysRevLett.46.1351
- Barceló, C., Liberati, S., & Visser, M. (2011). Analogue gravity. Living Reviews in Relativity, 14, 3. doi:10.12942/lrr-2011-3
- Weinfurtner, S., Tedford, E. W., Penrice, M. C. J., Unruh, W. G., & Lawrence, G. A. (2011). Measurement of stimulated Hawking emission in an analogue system. Physical Review Letters, 106(2), 021302. doi:10.1103/PhysRevLett.106.021302
- Steinhauer, J. (2016). Observation of quantum Hawking radiation and its entanglement in an analogue black hole. Nature Physics, 12, 959–965. doi:10.1038/nphys3863
- Drori, J., Rosenberg, Y., Bermudez, D., Silberberg, Y., & Leonhardt, U. (2019). Observation of stimulated Hawking radiation in an optical analogue. Physical Review Letters, 122(1), 010404. doi:10.1103/PhysRevLett.122.010404
- Page, D. N. (1976). Particle emission rates from a black hole: Massless particles from an uncharged, nonrotating hole. Physical Review D, 13(2), 198–206. doi:10.1103/PhysRevD.13.198
- Esarey, E., Schroeder, C. B., & Leemans, W. P. (2009). Physics of laser-driven plasma-based accelerators. Reviews of Modern Physics, 81(3), 1229–1285. doi:10.1103/RevModPhys.81.1229
- Chen, F. F. (2016). Introduction to Plasma Physics and Controlled Fusion (3rd ed.). Springer. doi:10.1007/978-3-319-22309-4
- Wilson, T. L., Rohlfs, K., & Hüttemeister, S. (2013). Tools of Radio Astronomy (6th ed.). Springer. doi:10.1007/978-3-642-39950-3

## Project-specific

- Analog Black Hole Evaporation via Lasers (AnaBHEL). See project overview and white paper(s). For a quick conceptual summary and local notes, refer to `AnaBHEL_Analog_Black_Hole_Evaporation_via_Lasers_E.md` in the repository root.

Notes:
- The graybody fallback used in this codebase follows Page-type low-frequency suppression heuristics and is implemented as `\mathcal{T}(\omega) = \omega^2/(\omega^2 + \kappa^2)` in `quantum_field_theory.py` when a profile-derived transmission is not supplied.
- Where possible, profile-derived WKB graybody factors computed from `(x, v(x), c_s(x))` near the horizon are preferred for quantitative predictions.
