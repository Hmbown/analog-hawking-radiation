"""Quantum fluctuation injection utilities for PIC backends.

Backwards-compatibility notes:
- Prior test code instantiated `FluctuationConfig` directly with legacy fields
  (seed, target_temperature, mode_cutoff, amplitude_scale, cadence, band_min,
  band_max, background_psd). The v0.3 refactor switched to a YAML-driven
  schema. To preserve compatibility, this module now accepts either the legacy
  constructor or the new schema and maps fields accordingly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Tuple, Union

import numpy as np
import yaml


@dataclass
class FluctuationConfig:
    # New schema (YAML-driven)
    seed_type: str = "mt19937"
    band_limit: Tuple[float, float] = (0.0, float("inf"))  # Hz
    amplitude: float = 1.0
    correlation_time: float = 1e-9  # s
    mean: float = 0.0
    target_field: str = "sound_speed"  # 'sound_speed', 'velocity', etc.
    cadence: int = 1
    seed: int = 0

    # Legacy fields (ignored in new flow; accepted for compatibility)
    target_temperature: float = 0.0
    mode_cutoff: float = float("inf")
    amplitude_scale: float = 1.0
    band_min: float = 0.0
    band_max: float = float("inf")
    background_psd: float = 0.0

    @staticmethod
    def from_legacy(**kwargs: Any) -> "FluctuationConfig":
        """Create config from legacy constructor fields used by tests.

        Expected legacy keys: seed, target_temperature, mode_cutoff,
        amplitude_scale, cadence, band_min, band_max, background_psd.
        """
        seed = int(kwargs.get("seed", 0))
        amp = float(kwargs.get("amplitude_scale", 1.0))
        fmin = float(kwargs.get("band_min", 0.0))
        fmax = float(kwargs.get("band_max", kwargs.get("mode_cutoff", float("inf"))))
        cadence = int(kwargs.get("cadence", 1))
        mean = 0.0
        target_field = "sound_speed"
        return FluctuationConfig(
            seed_type="mt19937",
            band_limit=(fmin, fmax),
            amplitude=amp,
            correlation_time=1e-9,
            mean=mean,
            target_field=target_field,
            cadence=cadence,
            seed=seed,
            # Preserve legacy attributes for introspection if needed
            target_temperature=float(kwargs.get("target_temperature", 0.0)),
            mode_cutoff=float(kwargs.get("mode_cutoff", fmax)),
            amplitude_scale=amp,
            band_min=fmin,
            band_max=fmax,
            background_psd=float(kwargs.get("background_psd", 0.0)),
        )

    @staticmethod
    def from_yaml_dict(data: Mapping[str, Any]) -> "FluctuationConfig":
        """Create config from YAML dict using new schema keys, tolerating legacy keys."""
        # Map legacy keys if present
        if any(k in data for k in ("amplitude_scale", "mode_cutoff", "band_min", "band_max")):
            return FluctuationConfig.from_legacy(**data)  # type: ignore[arg-type]
        # New-schema path
        band = data.get("band_limit")
        if band is None:
            # Tolerate separate min/max
            fmin = float(data.get("band_min", 0.0))
            fmax = float(data.get("band_max", float("inf")))
            band = (fmin, fmax)
        return FluctuationConfig(
            seed_type=str(data.get("seed_type", "mt19937")),
            band_limit=tuple(band),  # type: ignore[arg-type]
            amplitude=float(data.get("amplitude", 1.0)),
            correlation_time=float(data.get("correlation_time", 1e-9)),
            mean=float(data.get("mean", 0.0)),
            target_field=str(data.get("target_field", "sound_speed")),
            cadence=int(data.get("cadence", 1)),
            seed=int(data.get("seed", 0)),
        )


class QuantumFluctuationInjector:
    """Generates and injects broadband quantum-like fluctuations using Ornstein-Uhlenbeck process."""

    def __init__(self, config: Union[str, FluctuationConfig, Mapping[str, Any]]) -> None:
        """Initialize the injector with either a YAML path, a config object, or a dict.

        This preserves compatibility with previous tests that directly constructed
        `FluctuationConfig(...)` using legacy fields.
        """
        if isinstance(config, str):
            with open(config, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
            self._config = FluctuationConfig.from_yaml_dict(data)
        elif isinstance(config, FluctuationConfig):
            self._config = config
        else:
            # Mapping/dict provided
            self._config = FluctuationConfig.from_yaml_dict(config)
        self._rng = np.random.default_rng(self._config.seed)
        self._step_counter = 0
        self._backend = None
        self._current_noise = None  # For OU persistence
        self._dt = 1e-12  # Assume default dt; can be set in backend
        self._theta = 1.0 / self._config.correlation_time
        self._sigma = np.sqrt(
            2 * self._theta * (self._config.amplitude**2)
        )  # For variance = amplitude^2

    def attach_to_backend(self, backend) -> None:
        self._backend = backend
        backend.attach_fluctuation_injector(self)
        # Initialize noise array size from grid
        if hasattr(backend, "_grid") and backend._grid is not None:
            size = len(backend._grid)
            self._current_noise = np.zeros(size)

    def inject(self) -> None:
        if self._backend is None:
            return
        self._step_counter += 1
        if self._step_counter % self._config.cadence != 0:
            return

        # Get grid size
        if self._backend._grid is None:
            return
        size = len(self._backend._grid)

        # Generate Ornstein-Uhlenbeck noise
        dW = self._rng.normal(0, np.sqrt(self._dt), size)
        self._current_noise = (
            self._current_noise - self._theta * self._current_noise * self._dt + self._sigma * dW
        )

        # Band-limit: simple low-pass filter approximation for [f_min, f_max]
        f_min, f_max = self._config.band_limit
        if f_min > 0 or f_max < np.inf:
            # FFT filter (simple box filter in freq domain)
            fft_noise = np.fft.rfft(self._current_noise)
            freqs = np.fft.rfftfreq(size, self._dt)
            mask = (freqs >= f_min) & (freqs <= f_max)
            fft_noise[~mask] = 0.0
            noise_filtered = np.fft.irfft(fft_noise, n=size)
            noise = noise_filtered
        else:
            noise = self._current_noise

        # Add to target field in raw_observables (post-extraction addition)
        target = self._config.target_field
        if target in self._backend._raw_observables:
            self._backend._raw_observables[target] += self._config.amplitude * noise
        else:
            # If not present, add it
            self._backend._raw_observables[target] = (
                self._config.amplitude * noise
                + self._backend._call_moment_getter("electrons", target)
                if target in ["sound_speed", "velocity"]
                else self._config.amplitude * noise
            )

    def sample_fourier_modes(self, k_values: Iterable[float]) -> np.ndarray:
        k_array = np.asarray(list(k_values))
        amplitude = self._config.amplitude
        phases = self._rng.uniform(0, 2 * np.pi, size=k_array.shape)
        return amplitude * np.exp(1j * phases)
