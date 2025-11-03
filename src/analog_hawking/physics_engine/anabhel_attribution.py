"""
Status: ⚠️ Experimental attribution scaffolding.

AnaBHEL Framework Attribution and Theoretical Foundation

This module provides the theoretical foundation and proper attribution for the
AnaBHEL (Analog Black Hole Evaporation via Lasers) concepts used in this framework.

Key Contributors and Institutions:
==================================

Primary AnaBHEL Collaboration:
- Prof. Pisin Chen (National Taiwan University, LeCosPA)
- Prof. Gerard Mourou (École Polytechnique, IZEST) - Nobel Laureate
- AnaBHEL Collaboration (2022) - "AnaBHEL Experiment: Concept, Design, and Status"

Foundational Theory:
- Chen & Mourou (2017) - "Accelerating plasma mirrors to investigate the
  black hole information loss paradox" - Physical Review Letters 118, 045001

Related Analog Gravity Experts:
- Prof. William Unruh (University of British Columbia) - Original analog gravity theory
- Prof. Jeff Steinhauer (Technion) - First experimental analog Hawking radiation
- Prof. Daniele Faccio (University of Glasgow) - Laser-based analog systems

IMPORTANT DISCLAIMER:
====================
This framework explores a SPECULATIVE extension of AnaBHEL concepts by coupling
plasma mirrors with fluid sonic horizons. This hybrid approach goes beyond
established AnaBHEL theory and represents a computational thought experiment
rather than established physics.

The original AnaBHEL work focuses on pure plasma mirror systems, not hybrid
fluid-mirror coupling as explored in this speculative framework.
"""

# AnaBHEL model parameters from Chen et al. (2022)
ANABHEL_MIRROR_MAPPING = {
    "model": "anabhel",
    "kappa_formula": "2π * η_a / D",  # From Chen & Mourou (2017)
    "temperature_relation": "k_B * T_H = (ħ/D) * η_a",
    "reference": "Chen, P., Mourou, G., et al. (2022). Photonics 9(12), 1003",
}

# Acknowledgment text for use in outputs
ATTRIBUTION_TEXT = """
This framework builds upon the AnaBHEL (Analog Black Hole Evaporation via Lasers)
concepts developed by P. Chen, G. Mourou, and collaborators. The speculative hybrid
coupling explored here extends beyond established AnaBHEL theory.

Key References:
- Chen & Mourou (2017). Phys. Rev. Lett. 118, 045001
- Chen et al. (2022). Photonics 9(12), 1003
"""
