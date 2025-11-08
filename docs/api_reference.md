# API Reference

## Core Modules

### `analog_hawking.facilities.eli_capabilities`

**Status**: Implemented (NOT experimentally validated)

Facility parameter validation for ELI laser systems.

```python
from analog_hawking.facilities.eli_capabilities import (
    ELICapabilities, ELIFacility, validate_intensity_range
)

# Check if parameters are achievable
eli = ELICapabilities()
compatible = eli.get_compatible_systems(
    intensity_W_cm2=1e19,  # W/cm²
    wavelength_nm=800,     # nm
    pulse_duration_fs=100  # fs
)

# Returns list of compatible laser systems (or empty if none)
```

**Limitations**: Based on publicly available ELI specifications. Does not account for experimental constraints like target positioning, vacuum quality, or diagnostic availability.

---

### `analog_hawking.physics_engine.horizon`

**Status**: Implemented (NOT validated against PIC simulations)

Sonic horizon detection in velocity fields.

```python
from analog_hawking.physics_engine.horizon import find_horizon_surface

# Find where v(x) = c_s(x)
horizon = find_horizon_surface(
    x_grid, velocity_field, sound_speed_field
)

# Returns surface positions and normals
```

**Limitations**: Fluid approximation only. Assumes smooth velocity fields. Does not capture kinetic effects or turbulent fluctuations.

**Validation needed**: Compare with PIC simulations of known trapping conditions.

---

### `analog_hawking.detection.graybody_nd`

**Status**: Implemented (theoretical approximation only)

Graybody factor calculation for Hawking-like emission.

```python
from analog_hawking.detection.graybody_nd import (
    aggregate_patchwise_graybody, GraybodySpectrumND
)

# Calculate spectrum from horizon patches
result = aggregate_patchwise_graybody(
    grids=[x, y], 
    v_field=velocity_field,
    c_s=sound_speed_field,
    kappa_eff=surface_gravity,
    max_patches=16
)

# Returns aggregated spectrum (order-of-magnitude estimate)
```

**Limitations**: Uses dimensionless approximation Ω ↦ r²/(1+r²). Does not solve full wave equation. Precision: ±50% at best.

**Validation needed**: Compare with acoustic wave propagation in varying flow fields.

---

### `analog_hawking.facilities.eli_physics_validator`

**Status**: Implemented (NOT peer-reviewed)

Physics validation for ELI-scale experiments.

```python
from analog_hawking.facilities.eli_physics_validator import (
    ELIPhysicsValidator
)

validator = ELIPhysicsValidator()
results = validator.validate_comprehensive_configuration(
    intensity_W_m2=1e22,
    wavelength_nm=800,
    pulse_duration_fs=150,
    plasma_density_m3=1e25,
    gradient_scale_m=1e-6,
    flow_velocity_ms=2e6,
    facility=ELIFacility.ELI_BEAMLINES
)
```

**Limitations**: Thresholds based on theoretical estimates. No experimental calibration. Uncertainty propagation not validated.

**Validation needed**: Experimental measurement campaigns at ELI facilities.

---

## Usage Examples

See `examples/` directory for working demonstrations:

- `basic_horizon_detection.py`: Simple 1D horizon finding
- `eli_parameter_validation.py`: Check experimental feasibility
- `graybody_calculation_demo.py`: Hawking spectrum estimation

**Note**: All examples produce order-of-magnitude estimates only. Do not use for experimental planning without expert review.