"""1D graybody transmission estimator for analog horizons."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Tuple, Optional

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
        V = float(potential(x))
        if V > omega:
            dv = V - omega
            return float(np.sqrt(dv if dv > 0.0 else 0.0))
        return 0.0

    integral, _ = quad(integrand, x_bounds[0], x_bounds[1], limit=200, epsabs=1e-8, epsrel=1e-6)
    if not np.isfinite(integral):
        return 0.0
    return float(np.exp(-2.0 * integral))


def compute_graybody(
    x: np.ndarray,
    velocity: np.ndarray,
    sound_speed: np.ndarray,
    frequencies: Iterable[float],
    perturbation: float = 0.05,
    *,
    method: str = "dimensionless",
    kappa: Optional[float] = None,
    alpha: float = 1.0,
) -> GraybodyResult:
    """Compute graybody transmission.

    Default method (dimensionless) uses a conservative, Page-like low-frequency
    suppression shape T(ω) = (ω/ω_c)^2 / (1 + (ω/ω_c)^2) with ω_c = α·κ. When κ is
    not supplied, an estimate is derived from the local horizon gradients.

    The previous WKB routine (``method='wkb'``) is retained as experimental and
    compares a velocity-based surrogate potential against ω; this has unit-mismatch
    caveats and should be used with care.
    """
    frequencies = np.asarray(list(frequencies), dtype=float)
    omega_values = 2.0 * np.pi * frequencies

    method_l = method.lower()
    if method_l not in {"wkb", "acoustic_wkb"}:
        # Dimensionless fallback requiring κ
        if kappa is None or kappa <= 0.0:
            # Estimate κ from local profiles (acoustic definition)
            try:
                from ..horizon import find_horizons_with_uncertainty
                hr = find_horizons_with_uncertainty(x, velocity, sound_speed)
                kappa_est = float(np.max(hr.kappa)) if hr.kappa.size else 0.0
            except Exception:
                kappa_est = 0.0
            kappa_eff = kappa_est if kappa_est > 0.0 else 1.0
        else:
            kappa_eff = float(kappa)

        omega_c = max(alpha * kappa_eff, 1e-30)
        r = omega_values / omega_c
        transmission = (r**2) / (1.0 + r**2)
        # Simple uncertainty heuristic: larger near the turnover
        uncertainties = 0.1 * transmission * (1.0 - transmission)
        return GraybodyResult(
            frequencies=frequencies,
            transmission=np.clip(transmission, 0.0, 1.0),
            uncertainties=uncertainties,
        )

    if method_l == "wkb":
        # Experimental WKB path (legacy)
        potential = _effective_potential(x, velocity, sound_speed)
        potential_interp = lambda xi: float(np.interp(xi, x, potential))

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

    # Physically consistent acoustic WKB using tortoise coordinate
    x = np.asarray(x, dtype=float)
    v = np.asarray(velocity, dtype=float)
    cs = np.asarray(sound_speed, dtype=float)
    # Identify near-horizon index as minimal |c - |v||
    gap = np.abs(np.abs(v) - cs)
    idx = int(np.argmin(gap)) if gap.size else 0
    # Estimate kappa if not provided
    if kappa is None or kappa <= 0.0:
        try:
            from ..horizon import find_horizons_with_uncertainty
            hr = find_horizons_with_uncertainty(x, v, cs, kappa_method="acoustic_exact")
            kappa_eff = float(np.max(hr.kappa)) if hr.kappa.size else 1.0
        except Exception:
            kappa_eff = 1.0
    else:
        kappa_eff = float(kappa)

    # Construct tortoise coordinate x*: dx* = dx / |c - |v||
    denom = np.clip(gap, 1e-16, None)
    dx = np.gradient(x)
    dx_star = dx / denom
    x_star = np.cumsum(np.insert(dx_star[1:], 0, 0.0))  # anchored at x[0]

    # Build a dimensionless shape S(x*) that vanishes at the horizon and tends to 1 away
    s_raw = gap**2
    s_max = float(np.max(s_raw)) if np.any(np.isfinite(s_raw)) else 1.0
    if s_max <= 0:
        s_max = 1.0
    S = np.clip(s_raw / s_max, 0.0, 1.0)

    # Effective potential in ω^2 units: V = (α κ)^2 S(x*)
    k_scale = max(alpha * kappa_eff, 1e-30)
    V_vals = (k_scale ** 2) * S
    # Use compact barrier region where V is significant to avoid far-field artifacts
    barrier_mask = V_vals > 0.5 * (k_scale ** 2)
    xs_use = x_star[barrier_mask] if np.any(barrier_mask) else x_star
    V_arr = V_vals[barrier_mask] if np.any(barrier_mask) else V_vals
    # WKB transmission in tortoise coordinate using discrete trapezoidal integral
    def trans_from_threshold(thr: float) -> float:
        y = np.sqrt(np.clip(V_arr - thr, 0.0, None))
        integral = float(np.trapz(y, xs_use)) if xs_use.size > 1 else 0.0
        if not np.isfinite(integral):
            return 0.0
        return float(np.exp(-2.0 * integral))
    transmission = np.array([trans_from_threshold(omega**2) for omega in omega_values])

    # Uncertainty via small perturbations in v and c
    v_up = v * (1.0 + perturbation)
    cs_dn = cs * (1.0 - perturbation)
    gap_up = np.abs(np.abs(v_up) - cs_dn)
    denom_up = np.clip(gap_up, 1e-16, None)
    x_star_up = np.cumsum(np.insert((np.gradient(x) / denom_up)[1:], 0, 0.0))
    S_up = np.clip(gap_up**2 / max(float(np.max(gap_up**2)), 1e-30), 0.0, 1.0)
    V_up = (k_scale ** 2) * S_up
    mask_up = V_up > 0.5 * (k_scale ** 2)
    xs_up_use = x_star_up[mask_up] if np.any(mask_up) else x_star_up
    V_up_use = V_up[mask_up] if np.any(mask_up) else V_up
    # Upper/lower using discrete integral as well

    v_dn = v * (1.0 - perturbation)
    cs_up = cs * (1.0 + perturbation)
    gap_dn = np.abs(np.abs(v_dn) - cs_up)
    denom_dn = np.clip(gap_dn, 1e-16, None)
    x_star_dn = np.cumsum(np.insert((np.gradient(x) / denom_dn)[1:], 0, 0.0))
    S_dn = np.clip(gap_dn**2 / max(float(np.max(gap_dn**2)), 1e-30), 0.0, 1.0)
    V_dn = (k_scale ** 2) * S_dn
    mask_dn = V_dn > 0.5 * (k_scale ** 2)
    xs_dn_use = x_star_dn[mask_dn] if np.any(mask_dn) else x_star_dn
    V_dn_use = V_dn[mask_dn] if np.any(mask_dn) else V_dn
    def trans_up(thr: float) -> float:
        yu = np.sqrt(np.clip(V_up_use - thr, 0.0, None))
        integ = float(np.trapz(yu, xs_up_use)) if xs_up_use.size > 1 else 0.0
        return float(np.exp(-2.0 * integ)) if np.isfinite(integ) else 0.0
    def trans_dn(thr: float) -> float:
        yd = np.sqrt(np.clip(V_dn_use - thr, 0.0, None))
        integ = float(np.trapz(yd, xs_dn_use)) if xs_dn_use.size > 1 else 0.0
        return float(np.exp(-2.0 * integ)) if np.isfinite(integ) else 0.0
    T_up = np.array([trans_up(omega**2) for omega in omega_values])
    T_dn = np.array([trans_dn(omega**2) for omega in omega_values])
    uncertainties = 0.5 * np.abs(T_up - T_dn)

    return GraybodyResult(
        frequencies=frequencies,
        transmission=np.clip(transmission, 0.0, 1.0),
        uncertainties=uncertainties,
    )
