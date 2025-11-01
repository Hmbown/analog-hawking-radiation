from __future__ import annotations

"""Patch-wise graybody aggregation for nD horizon surfaces.

This utility samples 1D profiles along a chosen axis (or later, local normals)
at selected horizon patches and aggregates the resulting spectra. The goal is to
provide a pragmatic, reproducible nD graybody estimate without solving a full
multidimensional scattering problem.
"""

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np


@dataclass
class AggregatedSpectrum:
    success: bool
    frequencies: np.ndarray | None = None
    power_spectrum: np.ndarray | None = None
    power_std: np.ndarray | None = None
    peak_frequency: float | None = None
    n_patches: int = 0


def aggregate_patchwise_graybody(
    grids: Sequence[np.ndarray],
    v_field: np.ndarray,  # (..., D)
    c_s: np.ndarray,      # (...,)
    kappa_eff: float,
    *,
    graybody_method: str = "dimensionless",
    alpha_gray: float = 1.0,
    scan_axis: int = 0,
    patch_indices: np.ndarray | None = None,
    max_patches: int = 64,
    sample_mode: str = "scan_axis",  # or "normal"
) -> AggregatedSpectrum:
    """Aggregate patch-wise spectra along the scan axis.

    Args:
        grids: coordinate arrays [x0, x1, (x2)]
        v_field: vector velocity field with components last
        c_s: scalar sound speed field
        kappa_eff: effective κ to use for spectral calculations
        graybody_method: one of {dimensionless, wkb, acoustic_wkb}
        alpha_gray: graybody scaling parameter
        scan_axis: axis to sample along (0..D-1)
        patch_indices: optional array of indices into the horizon points list
        max_patches: cap on the number of patches to sample

    Returns:
        AggregatedSpectrum with mean power spectrum and standard deviation.
    """
    if kappa_eff <= 0.0:
        return AggregatedSpectrum(success=False)

    dims = len(grids)
    if v_field.shape[-1] != dims:
        return AggregatedSpectrum(success=False)

    # Lazy import to avoid circular dependencies in tests
    from analog_hawking.physics_engine.horizon_nd import find_horizon_surface_nd
    try:
        from scripts.hawking_detection_experiment import calculate_hawking_spectrum  # type: ignore
    except Exception:
        from hawking_detection_experiment import calculate_hawking_spectrum  # type: ignore

    surf = find_horizon_surface_nd(grids, v_field, c_s, scan_axis=scan_axis)
    if surf.positions.shape[0] == 0:
        return AggregatedSpectrum(success=False)

    n_patches = min(int(max_patches), surf.positions.shape[0])
    if patch_indices is None:
        patch_indices = np.linspace(0, surf.positions.shape[0] - 1, num=n_patches, dtype=int)
    else:
        patch_indices = np.asarray(patch_indices, dtype=int)[:n_patches]

    # Build spectra per patch by extracting 1D lines along scan_axis
    specs: List[Dict[str, np.ndarray]] = []
    x_axis = grids[scan_axis]
    for k in patch_indices:
        pos = surf.positions[k]
        if sample_mode.lower() != "normal":
            # Fix non-scan axes to nearest index to pos; slice along scan axis
            slicer = [slice(None)] * dims
            for ax in range(dims):
                if ax == scan_axis:
                    continue
                idx = int(np.clip(np.searchsorted(grids[ax], pos[ax]), 1, len(grids[ax]) - 2))
                slicer[ax] = idx
            sl = tuple(slicer)
            v_line_vec = v_field[sl]  # shape (n_line, D)
            if v_line_vec.ndim != 2:
                v_line_vec = np.reshape(v_line_vec, (-1, dims))
            v_line = np.sqrt(np.sum(v_line_vec ** 2, axis=-1))
            cs_line = c_s[sl]
            if cs_line.ndim != 1:
                cs_line = np.reshape(cs_line, (-1,))
            profile = {"x": x_axis, "v": v_line, "c_s": cs_line}
        else:
            # Sample along local normal using linear interpolation
            try:
                from scipy.ndimage import map_coordinates  # type: ignore
            except Exception:
                # Fallback to scan_axis sampling if SciPy unavailable
                slicer = [slice(None)] * dims
                for ax in range(dims):
                    if ax == scan_axis:
                        continue
                    idx = int(np.clip(np.searchsorted(grids[ax], pos[ax]), 1, len(grids[ax]) - 2))
                    slicer[ax] = idx
                sl = tuple(slicer)
                v_line_vec = v_field[sl]
                if v_line_vec.ndim != 2:
                    v_line_vec = np.reshape(v_line_vec, (-1, dims))
                v_line = np.sqrt(np.sum(v_line_vec ** 2, axis=-1))
                cs_line = c_s[sl]
                if cs_line.ndim != 1:
                    cs_line = np.reshape(cs_line, (-1,))
                profile = {"x": x_axis, "v": v_line, "c_s": cs_line}
            else:
                n = surf.normals[k]
                # Build param s around the point within a fraction of domain
                extents = [g[-1] - g[0] for g in grids]
                s_span = 0.125 * float(min(extents))
                n_pts = max(64, len(grids[scan_axis]))
                s_line = np.linspace(-s_span, s_span, n_pts)
                # Convert world coords -> index coords for map_coordinates
                idx_coords = []
                for ax in range(dims):
                    dx = float(np.mean(np.diff(grids[ax]))) if len(grids[ax]) > 1 else 1.0
                    origin = float(grids[ax][0])
                    idx_coords.append((pos[ax] + s_line * n[ax] - origin) / dx)
                coords = np.vstack(idx_coords)
                # Interpolate each velocity component and c_s
                v_comps = []
                for c in range(dims):
                    field = v_field[..., c]
                    v_comps.append(map_coordinates(field, coords, order=1, mode="nearest"))
                v_line = np.sqrt(np.sum(np.vstack(v_comps) ** 2, axis=0))
                cs_line = map_coordinates(c_s, coords, order=1, mode="nearest")
                # Use s_line as the local coordinate
                profile = {"x": s_line, "v": v_line, "c_s": cs_line}

        sp = calculate_hawking_spectrum(
            kappa_eff,
            graybody_profile=profile,
            graybody_method=str(graybody_method),
            alpha_gray=float(alpha_gray),
            emitting_area_m2=1e-6,
            solid_angle_sr=5e-2,
            coupling_efficiency=0.1,
        )
        if sp.get("success"):
            specs.append(sp)

    if not specs:
        return AggregatedSpectrum(success=False)

    # Align and average
    f0 = np.asarray(specs[0]["frequencies"])  # type: ignore[index]
    P_mat = []
    for sp in specs:
        f = np.asarray(sp["frequencies"])  # type: ignore[index]
        P = np.asarray(sp["power_spectrum"])  # type: ignore[index]
        if f.shape != f0.shape or not np.allclose(f, f0):
            P = np.interp(f0, f, P)
        P_mat.append(P)
    P_stack = np.vstack(P_mat)
    P_mean = np.mean(P_stack, axis=0)
    P_std = np.std(P_stack, axis=0)
    peak_f = float(f0[int(np.argmax(P_mean))])

    return AggregatedSpectrum(
        success=True,
        frequencies=f0,
        power_spectrum=P_mean,
        power_std=P_std,
        peak_frequency=peak_f,
        n_patches=len(specs),
    )
