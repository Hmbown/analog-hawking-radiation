"""Demonstrate SimulationRunner with FluidBackend and report metrics."""

from __future__ import annotations

import json

from physics_engine.plasma_models.fluid_backend import FluidBackend
from physics_engine.simulation import SimulationRunner


def main() -> None:
    config = {
        "plasma_density": 1e18,
        "laser_wavelength": 800e-9,
        "laser_intensity": 1e17,
        "grid": None,
    }
    backend = FluidBackend()
    backend.configure(config)
    runner = SimulationRunner(backend)
    outputs = runner.run_step(0.0)

    summary = {
        "omega_pe": float(backend._model.omega_pe),  # type: ignore[attr-defined]
        "a0": float(backend._model.a0),  # type: ignore[attr-defined]
        "density_mean": float(outputs.state.density.mean()),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
