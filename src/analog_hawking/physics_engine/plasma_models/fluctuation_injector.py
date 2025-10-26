"""Quantum fluctuation injection utilities for PIC backends."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np
import yaml


@dataclass
class FluctuationConfig:
    seed_type: str
    band_limit: Tuple[float, float]  # Hz
    amplitude: float
    correlation_time: float  # s
    mean: float
    target_field: str  # 'sound_speed', 'velocity', etc.
    cadence: int
    seed: int


class QuantumFluctuationInjector:
    """Generates and injects broadband quantum-like fluctuations using Ornstein-Uhlenbeck process."""

    def __init__(self, config_path: str) -> None:
        with open(config_path, 'r') as f:
            self._config = FluctuationConfig(**yaml.safe_load(f))
        self._rng = np.random.default_rng(self._config.seed)
        self._step_counter = 0
        self._backend = None
        self._current_noise = None  # For OU persistence
        self._dt = 1e-12  # Assume default dt; can be set in backend
        self._theta = 1.0 / self._config.correlation_time
        self._sigma = np.sqrt(2 * self._theta * (self._config.amplitude ** 2))  # For variance = amplitude^2

    def attach_to_backend(self, backend) -> None:
        self._backend = backend
        backend.attach_fluctuation_injector(self)
        # Initialize noise array size from grid
        if hasattr(backend, '_grid') and backend._grid is not None:
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
            self._current_noise - self._theta * self._current_noise * self._dt
            + self._sigma * dW
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
            self._backend._raw_observables[target] = self._config.amplitude * noise + self._backend._call_moment_getter("electrons", target) if target in ["sound_speed", "velocity"] else self._config.amplitude * noise

    def sample_fourier_modes(self, k_values: Iterable[float]) -> np.ndarray:
        k_array = np.asarray(list(k_values))
        amplitude = self._config.amplitude
        phases = self._rng.uniform(0, 2 * np.pi, size=k_array.shape)
        return amplitude * np.exp(1j * phases)


