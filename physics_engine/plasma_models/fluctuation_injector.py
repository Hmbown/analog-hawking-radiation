"""Quantum fluctuation injection utilities for PIC backends."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np


@dataclass
class FluctuationConfig:
    seed: int
    target_temperature: float
    mode_cutoff: float
    amplitude_scale: float = 1.0


class QuantumFluctuationInjector:
    """Generates and injects broadband quantum-like fluctuations."""

    def __init__(self, config: FluctuationConfig) -> None:
        self._config = config
        self._rng = np.random.default_rng(config.seed)

    def attach_to_backend(self, backend) -> None:
        backend.attach_fluctuation_injector(self)

    def inject(self) -> None:
        # Placeholder: interact with backend to perturb fields/particles
        return None

    def sample_fourier_modes(self, k_values: Iterable[float]) -> np.ndarray:
        k_array = np.asarray(list(k_values))
        amplitude = np.sqrt(self._config.target_temperature * self._config.amplitude_scale)
        phases = self._rng.uniform(0, 2 * np.pi, size=k_array.shape)
        return amplitude * np.exp(1j * phases)

