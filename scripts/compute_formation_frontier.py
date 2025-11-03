#!/usr/bin/env python3
from __future__ import annotations

"""
Compute a horizon-formation frontier: for each (n_e, T) find the minimum
laser intensity that yields a sonic horizon and record the corresponding κ.

Outputs
- results/formation_frontier.json
- figures/formation_frontier.png

Notes
- Uses coarse grids and bounded bisection for speed; tweak ranges at call time.
- Units follow the rest of the codebase (densities consistent with FluidBackend usage).
"""

import argparse
import json
import os
from dataclasses import asdict, dataclass
from typing import Optional, Tuple

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from analog_hawking.physics_engine.horizon import find_horizons_with_uncertainty
from analog_hawking.physics_engine.plasma_models.fluid_backend import FluidBackend


@dataclass
class FrontierConfig:
    densities: np.ndarray  # e.g., np.geomspace(1e17, 1e19, 15)
    temperatures: np.ndarray  # e.g., np.geomspace(1e4, 1e6, 15)
    intensity_min: float = 1e15
    intensity_max: float = 1e19
    intensity_tol: float = 5e-2  # relative tolerance for bisection
    max_bisection_iter: int = 10
    wavelength: float = 800e-9
    magnetic_field: Optional[float] = 0.01
    use_fast_magnetosonic: bool = True
    scale_with_intensity: bool = True
    grid_min: float = 0.0
    grid_max: float = 50e-6
    grid_points: int = 512


def _has_horizon(backend: FluidBackend) -> Tuple[bool, float]:
    state = backend.step(0.0)
    hz = find_horizons_with_uncertainty(state.grid, state.velocity, state.sound_speed)
    if hz.positions.size:
        return True, float(hz.kappa[0])
    return False, 0.0


def _bisection_intensity(
    cfg: FrontierConfig, density: float, temp: float
) -> Tuple[Optional[float], float]:
    lo = cfg.intensity_min
    hi = cfg.intensity_max
    kappa_at_hi = 0.0

    # Ensure hi is feasible at least once
    backend = FluidBackend()
    grid = np.linspace(cfg.grid_min, cfg.grid_max, cfg.grid_points)
    backend.configure(
        {
            "plasma_density": float(density),
            "laser_wavelength": cfg.wavelength,
            "laser_intensity": float(hi),
            "grid": grid,
            "temperature_settings": {"constant": float(temp)},
            "use_fast_magnetosonic": bool(cfg.use_fast_magnetosonic),
            "scale_with_intensity": bool(cfg.scale_with_intensity),
            "magnetic_field": cfg.magnetic_field,
        }
    )
    ok, kappa_at_hi = _has_horizon(backend)
    if not ok:
        return None, 0.0

    # Expand/shrink to ensure lo is infeasible
    backend.configure(
        {
            "plasma_density": float(density),
            "laser_wavelength": cfg.wavelength,
            "laser_intensity": float(lo),
            "grid": grid,
            "temperature_settings": {"constant": float(temp)},
            "use_fast_magnetosonic": bool(cfg.use_fast_magnetosonic),
            "scale_with_intensity": bool(cfg.scale_with_intensity),
            "magnetic_field": cfg.magnetic_field,
        }
    )
    ok_lo, _ = _has_horizon(backend)
    if ok_lo:
        # Lower until it breaks
        for _ in range(3):
            hi = lo
            lo = lo * 0.3
            backend.configure(
                {
                    "plasma_density": float(density),
                    "laser_wavelength": cfg.wavelength,
                    "laser_intensity": float(lo),
                    "grid": grid,
                    "temperature_settings": {"constant": float(temp)},
                    "use_fast_magnetosonic": bool(cfg.use_fast_magnetosonic),
                    "scale_with_intensity": bool(cfg.scale_with_intensity),
                    "magnetic_field": cfg.magnetic_field,
                }
            )
            ok_lo, _ = _has_horizon(backend)
            if not ok_lo:
                break
        if ok_lo:
            # Never broke at lowered bounds; accept lowest as frontier
            return float(lo), kappa_at_hi

    # Bisection search
    left = lo
    right = hi
    best_kappa = kappa_at_hi
    for _ in range(cfg.max_bisection_iter):
        mid = np.sqrt(left * right)  # geometric mean for decades
        backend.configure(
            {
                "plasma_density": float(density),
                "laser_wavelength": cfg.wavelength,
                "laser_intensity": float(mid),
                "grid": grid,
                "temperature_settings": {"constant": float(temp)},
                "use_fast_magnetosonic": bool(cfg.use_fast_magnetosonic),
                "scale_with_intensity": bool(cfg.scale_with_intensity),
                "magnetic_field": cfg.magnetic_field,
            }
        )
        ok_mid, k_mid = _has_horizon(backend)
        if ok_mid:
            best_kappa = k_mid
            right = mid
        else:
            left = mid
        if right / left - 1.0 < cfg.intensity_tol:
            break
    return float(right), float(best_kappa)


def compute_frontier(cfg: FrontierConfig):
    ne = cfg.densities
    TT = cfg.temperatures
    Imin = np.full((ne.size, TT.size), np.nan, dtype=float)
    Kmin = np.full((ne.size, TT.size), np.nan, dtype=float)

    for i, n_e in enumerate(ne):
        for j, T in enumerate(TT):
            I_star, kappa_star = _bisection_intensity(cfg, float(n_e), float(T))
            if I_star is not None:
                Imin[i, j] = I_star
                Kmin[i, j] = kappa_star

    os.makedirs("results", exist_ok=True)
    cfg_dict = asdict(cfg)
    # Ensure arrays inside config are JSON-serializable
    if isinstance(cfg_dict.get("densities"), np.ndarray):
        cfg_dict["densities"] = ne.tolist()
    if isinstance(cfg_dict.get("temperatures"), np.ndarray):
        cfg_dict["temperatures"] = TT.tolist()

    out = {
        "densities": ne.tolist(),
        "temperatures": TT.tolist(),
        "I_min": Imin.tolist(),
        "kappa_at_I_min": Kmin.tolist(),
        "config": cfg_dict,
    }
    with open("results/formation_frontier.json", "w") as f:
        json.dump(out, f, indent=2)

    # Figure
    os.makedirs("figures", exist_ok=True)
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    # Prepare masked arrays to handle regions with no horizon (NaNs)
    Imin_log = np.log10(Imin)
    Kmin_log = np.log10(np.clip(Kmin, 1e-30, None))
    Imin_m = np.ma.array(Imin_log, mask=~np.isfinite(Imin_log))
    Kmin_m = np.ma.array(Kmin_log, mask=~np.isfinite(Kmin_log))
    cmap0 = plt.get_cmap("magma").copy()
    cmap1 = plt.get_cmap("viridis").copy()
    cmap0.set_bad(color="#f0f0f0")
    cmap1.set_bad(color="#f0f0f0")

    im0 = ax[0].imshow(
        Imin_m,
        origin="lower",
        aspect="auto",
        extent=[np.log10(TT[0]), np.log10(TT[-1]), np.log10(ne[0]), np.log10(ne[-1])],
        cmap=cmap0,
    )
    ax[0].set_title("log10 I_min [W/m²]")
    ax[0].set_xlabel("log10 T [K]")
    ax[0].set_ylabel("log10 n_e [m⁻³]")
    fig.colorbar(im0, ax=ax[0])

    im1 = ax[1].imshow(
        Kmin_m,
        origin="lower",
        aspect="auto",
        extent=[np.log10(TT[0]), np.log10(TT[-1]), np.log10(ne[0]), np.log10(ne[-1])],
        cmap=cmap1,
    )
    ax[1].set_title("log10 κ at frontier [s⁻¹]")
    ax[1].set_xlabel("log10 T [K]")
    ax[1].set_ylabel("log10 n_e [m⁻³]")
    fig.colorbar(im1, ax=ax[1])
    plt.tight_layout()
    plt.savefig("figures/formation_frontier.png", dpi=200)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dens-min", type=float, default=1e17)
    p.add_argument("--dens-max", type=float, default=1e19)
    p.add_argument("--dens-N", type=int, default=10)
    p.add_argument("--temp-min", type=float, default=1e4)
    p.add_argument("--temp-max", type=float, default=1e6)
    p.add_argument("--temp-N", type=int, default=10)

    p.add_argument("--I-min", type=float, default=1e15)
    p.add_argument("--I-max", type=float, default=1e19)
    p.add_argument("--I-tol", type=float, default=5e-2)
    p.add_argument("--max-iters", type=int, default=10)
    p.add_argument("--wavelength", type=float, default=800e-9)
    p.add_argument("--magnetic-field", type=float, default=0.01)
    p.add_argument("--no-magnetic-field", action="store_true")
    p.add_argument("--use-fast-ms", action="store_true")
    p.add_argument("--no-use-fast-ms", action="store_true")
    p.add_argument("--scale-with-intensity", action="store_true", default=True)
    p.add_argument("--no-scale-with-intensity", action="store_true")
    p.add_argument("--grid-min", type=float, default=0.0)
    p.add_argument("--grid-max", type=float, default=50e-6)
    p.add_argument("--grid-points", type=int, default=512)
    args = p.parse_args()

    ne = np.geomspace(args.dens_min, args.dens_max, args.dens_N)
    TT = np.geomspace(args.temp_min, args.temp_max, args.temp_N)
    mf = None if args.no_magnetic_field else args.magnetic_field
    use_ms = args.use_fast_ms and not args.no_use_fast_ms
    scale_int = not args.no_scale_with_intensity

    cfg = FrontierConfig(
        densities=ne,
        temperatures=TT,
        intensity_min=args.I_min,
        intensity_max=args.I_max,
        intensity_tol=args.I_tol,
        max_bisection_iter=args.max_iters,
        wavelength=args.wavelength,
        magnetic_field=mf,
        use_fast_magnetosonic=use_ms,
        scale_with_intensity=scale_int,
        grid_min=args.grid_min,
        grid_max=args.grid_max,
        grid_points=args.grid_points,
    )
    compute_frontier(cfg)
    print("Saved results/formation_frontier.json and figures/formation_frontier.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
