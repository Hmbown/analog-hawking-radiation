"""1D graybody transmission estimator for analog horizons."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Optional, Tuple

import numpy as np
from scipy.integrate import quad

from ...utils.array_module import (
    as_scalar,
    to_numpy,
    xp,
    xp_abs,
    xp_clip,
    xp_gradient,
    xp_trapezoid,
)


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

        def potential_interp(xi: float) -> float:
            return float(np.interp(xi, x, potential))

        transmission = np.array([
            _wkb_transmission(omega, potential_interp, (x[0], x[-1])) for omega in omega_values
        ])

        upper = _effective_potential(
            x, velocity * (1.0 + perturbation), sound_speed * (1.0 - perturbation)
        )
        lower = _effective_potential(
            x, velocity * (1.0 - perturbation), sound_speed * (1.0 + perturbation)
        )

        def upper_interp(xi: float) -> float:
            return float(np.interp(xi, x, upper))

        def lower_interp(xi: float) -> float:
            return float(np.interp(xi, x, lower))

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

    # Physically consistent acoustic WKB using tortoise coordinate with GPU support.
    dtype = getattr(xp, "float64", float)
    x_arr = xp.asarray(x, dtype=dtype)
    v_arr = xp.asarray(velocity, dtype=dtype)
    cs_arr = xp.asarray(sound_speed, dtype=dtype)
    x_np = to_numpy(x_arr)
    v_np = to_numpy(v_arr)
    cs_np = to_numpy(cs_arr)

    # Identify near-horizon index as minimal |c - |v||
    gap = xp_abs(xp_abs(v_arr) - cs_arr)

    # Estimate kappa if not provided
    if kappa is None or kappa <= 0.0:
        try:
            from ..horizon import find_horizons_with_uncertainty
            hr = find_horizons_with_uncertainty(x_np, v_np, cs_np, kappa_method="acoustic_exact")
            kappa_eff = float(np.max(hr.kappa)) if hr.kappa.size else 1.0
        except Exception:
            kappa_eff = 1.0
    else:
        kappa_eff = float(kappa)

    def _tortoise(gap_vals: np.ndarray) -> any:
        denom = xp_clip(gap_vals, 1e-16, None)
        dx = xp_gradient(x_arr)
        dx_arr = xp.asarray(dx)
        dx_star = dx_arr / denom
        if dx_star.size > 1:
            tail = dx_star[1:]
        else:
            tail = xp.zeros(0, dtype=dtype)
        padded = xp.concatenate((xp.zeros(1, dtype=dtype), tail))
        return xp.cumsum(padded)

    def _normalized_gap_sq(gap_vals: np.ndarray) -> any:
        s_raw = gap_vals ** 2
        s_max = float(np.max(to_numpy(s_raw))) if s_raw.size else 1.0
        if not np.isfinite(s_max) or s_max <= 0.0:
            s_max = 1.0
        return xp_clip(s_raw / s_max, 0.0, 1.0)

    # Construct tortoise coordinate x*: dx* = dx / |c - |v||
    x_star = _tortoise(gap)

    # Build a dimensionless shape S(x*) that vanishes at the horizon and tends to 1 away
    S = _normalized_gap_sq(gap)

    # Effective potential in ω^2 units: V = (α κ)^2 S(x*)
    k_scale = max(alpha * kappa_eff, 1e-30)
    V_vals = (k_scale ** 2) * S
    # Use compact barrier region where V is significant to avoid far-field artifacts
    barrier_np = to_numpy(V_vals > 0.5 * (k_scale ** 2))
    if barrier_np.any():
        xs_use = x_star[barrier_np]
        V_use = V_vals[barrier_np]
    else:
        xs_use = x_star
        V_use = V_vals

    # WKB transmission in tortoise coordinate using discrete trapezoidal integral
    def trans_from_threshold(thr: float) -> float:
        y = xp.sqrt(xp_clip(V_use - thr, 0.0, None))
        integral = as_scalar(xp_trapezoid(y, xs_use)) if xs_use.size > 1 else 0.0
        if not np.isfinite(integral):
            return 0.0
        return float(np.exp(-2.0 * integral))
    transmission = np.array([trans_from_threshold(omega**2) for omega in omega_values])

    # Uncertainty via small perturbations in v and c
    v_up = v_arr * (1.0 + perturbation)
    cs_dn = cs_arr * (1.0 - perturbation)
    gap_up = xp_abs(xp_abs(v_up) - cs_dn)
    x_star_up = _tortoise(gap_up)
    S_up = _normalized_gap_sq(gap_up)
    V_up = (k_scale ** 2) * S_up
    mask_up_np = to_numpy(V_up > 0.5 * (k_scale ** 2))
    xs_up_use = x_star_up[mask_up_np] if mask_up_np.any() else x_star_up
    V_up_use = V_up[mask_up_np] if mask_up_np.any() else V_up

    v_dn = v_arr * (1.0 - perturbation)
    cs_up = cs_arr * (1.0 + perturbation)
    gap_dn = xp_abs(xp_abs(v_dn) - cs_up)
    x_star_dn = _tortoise(gap_dn)
    S_dn = _normalized_gap_sq(gap_dn)
    V_dn = (k_scale ** 2) * S_dn
    mask_dn_np = to_numpy(V_dn > 0.5 * (k_scale ** 2))
    xs_dn_use = x_star_dn[mask_dn_np] if mask_dn_np.any() else x_star_dn
    V_dn_use = V_dn[mask_dn_np] if mask_dn_np.any() else V_dn

    def trans_up(thr: float) -> float:
        yu = xp.sqrt(xp_clip(V_up_use - thr, 0.0, None))
        integ = as_scalar(xp_trapezoid(yu, xs_up_use)) if xs_up_use.size > 1 else 0.0
        return float(np.exp(-2.0 * integ)) if np.isfinite(integ) else 0.0
    def trans_dn(thr: float) -> float:
        yd = xp.sqrt(xp_clip(V_dn_use - thr, 0.0, None))
        integ = as_scalar(xp_trapezoid(yd, xs_dn_use)) if xs_dn_use.size > 1 else 0.0
        return float(np.exp(-2.0 * integ)) if np.isfinite(integ) else 0.0
    T_up = np.array([trans_up(omega**2) for omega in omega_values])
    T_dn = np.array([trans_dn(omega**2) for omega in omega_values])
    uncertainties = 0.5 * np.abs(T_up - T_dn)

    return GraybodyResult(
        frequencies=frequencies,
        transmission=np.clip(transmission, 0.0, 1.0),
        uncertainties=uncertainties,
    )
