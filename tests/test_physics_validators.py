import numpy as np
from scipy.constants import c

from analog_hawking.physics_engine.enhanced_ionization_physics import (
    ATOMIC_DATA,
    IonizationDynamics,
)
from analog_hawking.physics_engine.physics_validation_framework import (
    PhysicalConstraintsValidator,
)


def test_causality_validator_allows_phase_velocity_rounding():
    validator = PhysicalConstraintsValidator()
    group_velocity = 0.95 * c
    phase_velocity = c * (1.0 + 2.5e-8)  # Within relaxed tolerance

    result = validator.test_causality(group_velocity=group_velocity, phase_velocity=phase_velocity)

    assert result.passed, f"Causality check unexpectedly failed: {result.description}"
    assert result.value < 5e-8


def test_adk_strong_field_log_rate_monotonic():
    ionization = IonizationDynamics(ATOMIC_DATA["Al"])
    fields = np.logspace(11, 13, 5)
    log_rates = np.array([ionization.adk_model.log_adk_rate(E, 0) for E in fields])

    assert np.all(np.isfinite(log_rates)), "Log ADK rates should be finite in strong field regime"
    diffs = np.diff(log_rates)
    assert np.all(diffs > 0), f"Log-rate differences not positive: {diffs}"
