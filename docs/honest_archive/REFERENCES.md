# References

> **Status ✅** Curated reference list for validated and experimental modules
> **Navigate:** [Docs Index](index.md) · [AnaBHEL Comparison](AnaBHEL_Comparison.md)

This repository draws on foundational literature in analog gravity, plasma physics, black hole radiation, and radiometry. Key references are listed for context and further reading.

## Foundational Analog Gravity Theory

- Hawking, S. W. (1974). Black hole explosions? Nature, 248, 30–31. doi:10.1038/248030a0
- Hawking, S. W. (1975). Particle creation by black holes. Communications in Mathematical Physics, 43, 199–220. doi:10.1007/BF02345020
- **Unruh, W. G. (1981). Experimental black-hole evaporation? Physical Review Letters, 46(21), 1351–1353. doi:10.1103/PhysRevLett.46.1351** - *Original proposal for analog Hawking radiation in flowing fluids*
- Barceló, C., Liberati, S., & Visser, M. (2011). Analogue gravity. Living Reviews in Relativity, 14, 3. doi:10.12942/lrr-2011-3

## Experimental Analog Gravity (Fluid Systems)

- **Steinhauer, J. (2016). Observation of quantum Hawking radiation and its entanglement in an analogue black hole. Nature Physics, 12, 959–965. doi:10.1038/nphys3863** - *First strong experimental evidence for analog Hawking radiation*
- Weinfurtner, S., Tedford, E. W., Penrice, M. C. J., Unruh, W. G., & Lawrence, G. A. (2011). Measurement of stimulated Hawking emission in an analogue system. Physical Review Letters, 106(2), 021302. doi:10.1103/PhysRevLett.106.021302
- Drori, J., Rosenberg, Y., Bermudez, D., Silberberg, Y., & Leonhardt, U. (2019). Observation of stimulated Hawking radiation in an optical analogue. Physical Review Letters, 122(1), 010404. doi:10.1103/PhysRevLett.122.010404

## AnaBHEL Framework & Plasma Mirror Theory

- **Chen, P., Mourou, G., Besancon, M., Fukuda, Y., Glicenstein, J.-F., et al. (2022). AnaBHEL (Analog Black Hole Evaporation via Lasers) Experiment: Concept, Design, and Status. Photonics, 9(12), 1003. doi:10.3390/photonics9121003** - *Primary AnaBHEL collaboration paper*
- **Chen, P., & Mourou, G. (2017). Accelerating plasma mirrors to investigate the black hole information loss paradox. Physical Review Letters, 118(4), 045001. doi:10.1103/PhysRevLett.118.045001** - *Foundational plasma mirror acceleration theory*
- **Chen, P., Besancon, M., Fukuda, Y., Glicenstein, J.-F., Horan, D., et al. (2021). Proposal for a laboratory test of the black hole information loss paradox. Classical and Quantum Gravity, 38(19), 195025. doi:10.1088/1361-6382/ac1bf9**

## Optical & Laser-Based Analog Systems

- **Faccio, D., & Wright, E. M. (2013). Nonlinear dynamics and quantum phenomena in optical fibers. In P. Chamorro-Posada & F. Fraile-Peláez (Eds.), Nonlinear Optical Phenomena in Fibers (pp. 39-62). Academic Press.** - *Bridge between fluid and laser-based analog systems*
- Belgiorno, F., Cacciatori, S. L., Clerici, M., Gorini, V., Ortenzi, G., Rizzi, L., ... & Faccio, D. (2010). Hawking radiation from ultrashort laser pulse filaments. Physical Review Letters, 105(20), 203901. doi:10.1103/PhysRevLett.105.203901
- Faccio, D., Cacciatori, S., Gorini, V., Sala, V. G., Averchi, A., Lotti, A., ... & Rubino, E. (2012). Analogue gravity phenomenology: Analogue spacetimes and horizons, from theory to experiment. Lecture Notes in Physics, 870. Springer.

## High-Intensity Laser Physics

- **Mourou, G., Tajima, T., & Bulanov, S. V. (2006). Optics in the relativistic regime. Reviews of Modern Physics, 78(2), 309–371. doi:10.1103/RevModPhys.78.309** - *Nobel laureate work on ultra-intense lasers enabling AnaBHEL*
- Esarey, E., Schroeder, C. B., & Leemans, W. P. (2009). Physics of laser-driven plasma-based accelerators. Reviews of Modern Physics, 81(3), 1229–1285. doi:10.1103/RevModPhys.81.1229

## Plasma Physics & Radiometry

- Chen, F. F. (2016). Introduction to Plasma Physics and Controlled Fusion (3rd ed.). Springer. doi:10.1007/978-3-319-22309-4
- Page, D. N. (1976). Particle emission rates from a black hole: Massless particles from an uncharged, nonrotating hole. Physical Review D, 13(2), 198–206. doi:10.1103/PhysRevD.13.198
- Wilson, T. L., Rohlfs, K., & Hüttemeister, S. (2013). Tools of Radio Astronomy (6th ed.). Springer. doi:10.1007/978-3-642-39950-3

---

Back: [AnaBHEL Comparison](AnaBHEL_Comparison.md) · Top: [Docs Index](index.md)

## Key Research Institutions

- **Leung Center for Cosmology and Particle Astrophysics (LeCosPA), National Taiwan University** - *Primary AnaBHEL theory development*
- **IZEST (International Zetta-Exawatt Science and Technology), École Polytechnique** - *Ultra-high intensity laser technology*
- **Kansai Institute for Photon Science (QST), Japan** - *Advanced laser-plasma interaction research*
- **Xtreme Light Group, University of Glasgow** - *Analog gravity with intense laser pulses*

## Implementation Notes

- The graybody fallback used in this codebase follows Page-type low-frequency suppression heuristics and is implemented as `\mathcal{T}(\omega) = \omega^2/(\omega^2 + \kappa^2)` in `quantum_field_theory.py` when a profile-derived transmission is not supplied.
- Where possible, profile-derived WKB graybody factors computed from `(x, v(x), c_s(x))` near the horizon are preferred for quantitative predictions.
- **The speculative hybrid coupling explored in this framework extends beyond established AnaBHEL theory and represents a computational thought experiment.**
