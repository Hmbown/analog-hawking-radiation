from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from analog_hawking.physics_engine.horizon import (
    HorizonResult,
    find_horizons_with_uncertainty,
)

from .plasma_mirror import MirrorDynamics


@dataclass
class HybridHorizonParams:
    coupling_strength: float = 0.3
    coupling_length: float = 5e-6
    alignment_power: float = 1.0
    use_localization: bool = True


@dataclass
class HybridHorizonResult:
    fluid: HorizonResult
    kappa_mirror: float
    hybrid_kappa: np.ndarray
    coupling_weight: np.ndarray
    alignment: np.ndarray


def _nearest_index(arr: np.ndarray, value: float) -> int:
    return int(np.clip(np.searchsorted(arr, value), 0, len(arr) - 1))


def find_hybrid_horizons(
    x: np.ndarray,
    v_fluid: np.ndarray,
    c_s_profile: np.ndarray,
    mirror: MirrorDynamics,
    params: HybridHorizonParams = HybridHorizonParams(),
) -> HybridHorizonResult:
    """
    Fuse fluid horizon detection with plasma-mirror dynamics by locally enhancing the
    surface gravity near horizons where mirror acceleration aligns and is proximal.

    - Compute fluid horizons via find_horizons_with_uncertainty().
    - For each fluid horizon at position x_h, find the closest mirror position xM(t*).
    - Weight w = coupling_strength * exp(-|x_h - xM(t*)|/L) if use_localization else coupling_strength.
    - If alignment between d/dx(|v|-c_s) and a_M(t*) is negative, set weight=0 (conservative).
    - κ_eff = κ_fluid + w * κ_mirror.
    """
    x = np.asarray(x, dtype=float)
    v = np.asarray(v_fluid, dtype=float)
    cs = np.asarray(c_s_profile, dtype=float)
    assert x.ndim == v.ndim == cs.ndim == 1 and x.size == v.size == cs.size

    fluid = find_horizons_with_uncertainty(x, v, cs)

    if fluid.positions.size == 0:
        return HybridHorizonResult(
            fluid=fluid,
            kappa_mirror=float(mirror.kappa_mirror),
            hybrid_kappa=np.array([]),
            coupling_weight=np.array([]),
            alignment=np.array([]),
        )

    # Precompute f(x) slope for alignment sign
    f = np.abs(v) - cs
    df_dx = np.gradient(f, x)

    hybrid_k = np.zeros_like(fluid.kappa)
    weights = np.zeros_like(fluid.kappa)
    aligns = np.zeros_like(fluid.kappa)

    for i, x_h in enumerate(fluid.positions):
        idx = _nearest_index(x, x_h)
        slope = float(df_dx[idx])
        # Closest mirror point to this horizon
        j = int(np.argmin(np.abs(mirror.xM - x_h))) if mirror.xM.size else 0
        a_local = float(mirror.aM[j]) if mirror.aM.size else 0.0

        # Alignment sign (+1 aligned, -1 anti-aligned, 0 neutral)
        align_sign = 0.0
        if slope != 0.0 and a_local != 0.0:
            align_sign = float(np.sign(slope) * np.sign(a_local))
        aligns[i] = align_sign

        # Proximity weight
        w = float(params.coupling_strength)
        if params.use_localization and mirror.xM.size:
            L = max(params.coupling_length, 1e-12)
            d = abs(float(mirror.xM[j] - x_h))
            w *= float(np.exp(-d / L))

        if align_sign <= 0:
            w = 0.0  # conservative: only enhance when aligned
        elif params.alignment_power != 1.0 and w > 0.0:
            w = float(w**params.alignment_power)
        weights[i] = w

        # Hybrid kappa (ensure non-negative)
        k_eff = float(fluid.kappa[i]) + w * float(mirror.kappa_mirror)
        hybrid_k[i] = max(k_eff, 0.0)

    return HybridHorizonResult(
        fluid=fluid,
        kappa_mirror=float(mirror.kappa_mirror),
        hybrid_kappa=hybrid_k,
        coupling_weight=weights,
        alignment=aligns,
    )
