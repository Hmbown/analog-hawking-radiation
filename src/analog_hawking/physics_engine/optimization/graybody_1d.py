"""1D graybody transmission estimator for analog horizons."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Tuple

import numpy as np
from scipy.integrate import quad


@dataclass
class GraybodyResult:
    frequencies: np.ndarray
    transmission: np.ndarray
    uncertainties: np.ndarray


def _effective_potential(x: np.ndarray, v: np.ndarray, c_s: np.ndarray) -> np.ndarray:
    return (np.abs(v) - c_s) ** 2


def _wkb_transmission(omega: float, potential: Callable[[float], float], x_bounds: Tuple[float, float]) -> float:
    def integrand(x: float) -> float:
        V = potential(x)
        if omega <= V:
            return np.sqrt(V - omega)
        return 0.0

    integral, _ = quad(integrand, x_bounds[0], x_bounds[1], limit=200)
    if not np.isfinite(integral):
        return 0.0
    return float(np.exp(-2.0 * integral))


def compute_graybody(
    x: np.ndarray,
    velocity: np.ndarray,
    sound_speed: np.ndarray,
    frequencies: Iterable[float],
    perturbation: float = 0.05,
) -> GraybodyResult:
    frequencies = np.asarray(list(frequencies), dtype=float)
    potential = _effective_potential(x, velocity, sound_speed)
    potential_interp = lambda xi: float(np.interp(xi, x, potential))
    omega_values = 2.0 * np.pi * frequencies

    transmission = np.array([
        _wkb_transmission(omega, potential_interp, (x[0], x[-1])) for omega in omega_values
    ])

    upper = _effective_potential(x, velocity * (1.0 + perturbation), sound_speed * (1.0 - perturbation))
    lower = _effective_potential(x, velocity * (1.0 - perturbation), sound_speed * (1.0 + perturbation))
    upper_interp = lambda xi: float(np.interp(xi, x, upper))
    lower_interp = lambda xi: float(np.interp(xi, x, lower))

    transmission_upper = np.array([
        _wkb_transmission(omega, upper_interp, (x[0], x[-1])) for omega in omega_values
    ])
    transmission_lower = np.array([
        _wkb_transmission(omega, lower_interp, (x[0], x[-1])) for omega in omega_values
    ])

    uncertainties = 0.5 * np.abs(transmission_upper - transmission_lower)

    return GraybodyResult(
        frequencies=frequencies,
        transmission=np.clip(transmission, 0.0, 1.0),
        uncertainties=uncertainties,
    )


