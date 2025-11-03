"""nD horizon finding and surface gravity for analog Hawking flows.

Provides utilities to detect zero level-sets of f(x)=|v|-c_s in 2D/3D and
estimate κ on the horizon using the acoustic-exact form generalized to nD:

  κ(x_H) = |n · ∇(c_s^2 − |v|^2)| / (2 c_H)

where n is the unit normal to the surface f=0, and c_H is the sound speed
evaluated on the surface.

This implementation is designed to be dependency-light: it avoids external
marching cubes libraries by detecting sign changes along a chosen axis and
linearly interpolating root positions. It is intended for toy-model grids and
unit tests rather than production-quality meshing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import numpy as np


@dataclass
class HorizonSurfaceND:
    positions: np.ndarray        # (N, D) horizon points
    kappa: np.ndarray            # (N,) κ at points
    normals: np.ndarray          # (N, D) unit normals at points
    c_h: np.ndarray              # (N,) sound speed at horizon


def _spacing_from_grid(grid: np.ndarray) -> float:
    if grid.size <= 1:
        return 1.0
    return float(np.mean(np.diff(grid)))


def _compute_gradients(scalar: np.ndarray, grids: Sequence[np.ndarray]) -> List[np.ndarray]:
    """Compute ∇scalar using central differences with physical spacing."""
    spacings = tuple(_spacing_from_grid(g) for g in grids)
    grads = np.gradient(scalar, *spacings, edge_order=2)
    if not isinstance(grads, list):
        grads = list(grads)
    return grads  # [∂/∂x0, ∂/∂x1, ...]


def _unit(v: np.ndarray, eps: float = 1e-30) -> np.ndarray:
    nrm = np.linalg.norm(v, axis=-1, keepdims=True)
    nrm = np.maximum(nrm, eps)
    return v / nrm


def find_horizon_surface_nd(
    grids: Sequence[np.ndarray],
    v_field: np.ndarray,
    c_s: np.ndarray,
    scan_axis: int = 0,
) -> HorizonSurfaceND:
    """Detect horizon surface points and estimate κ in 2D/3D.

    Args:
        grids: list of coordinate arrays [x0, x1, (x2)] each 1D.
        v_field: array with shape (*grid_shape, D) where D=len(grids), components last.
        c_s: scalar sound speed array with shape (*grid_shape)
        scan_axis: axis along which to search for sign changes (default 0)

    Returns:
        HorizonSurfaceND with positions (N,D), kappa (N,), normals (N,D), c_h (N,)

    Notes:
        - Assumes at most one crossing per line along scan_axis; adequate for tanh-like profiles.
        - Uses nearest-grid gradients for κ and n estimation.
    """
    dims = len(grids)
    assert v_field.shape[-1] == dims
    grid_shape = tuple(len(g) for g in grids)
    assert v_field.shape[:-1] == grid_shape and c_s.shape == grid_shape

    # Magnitude fields
    vmag = np.sqrt(np.sum(v_field**2, axis=-1))
    f = vmag - c_s

    # Gradients for normal and κ evaluation
    grad_f = _compute_gradients(f, grids)
    cs2_minus_v2 = c_s**2 - vmag**2
    grad_cs2_mv2 = _compute_gradients(cs2_minus_v2, grids)

    # Iterate over all index lines orthogonal to scan_axis
    other_axes = [ax for ax in range(dims) if ax != scan_axis]

    positions: List[np.ndarray] = []
    normals: List[np.ndarray] = []
    kappas: List[float] = []
    ch_list: List[float] = []

    # Build index iterators for other axes
    other_ranges = [range(grid_shape[ax]) for ax in other_axes]
    for idx_other in np.ndindex(*(len(r) for r in other_ranges)):
        # Construct full slice along scan_axis for this line
        slicer = [slice(None)] * dims
        for k, ax in enumerate(other_axes):
            slicer[ax] = idx_other[k]
        f_line = f[tuple(slicer)]  # shape along scan_axis
        if f_line.ndim != 1:
            f_line = np.asarray(f_line).reshape(-1)
        # Find sign changes
        sign = np.sign(f_line)
        # Skip lines with no sign change
        prod = sign[:-1] * sign[1:]
        crossings = np.where(prod < 0)[0]
        if crossings.size == 0:
            # Handle rare exact zero match: pick those indices
            zeros = np.where(f_line == 0)[0]
            if zeros.size == 0:
                continue
            i0 = int(zeros[0])
            t = 0.5
        else:
            i0 = int(crossings[0])
            # Linear interpolation on f to find root fraction
            f0 = float(f_line[i0])
            f1 = float(f_line[i0 + 1])
            denom = f0 - f1
            t = float(f0 / denom) if denom != 0 else 0.5
            t = np.clip(t, 0.0, 1.0)

        # Position vector
        pos = []
        for ax in range(dims):
            g = grids[ax]
            if ax == scan_axis:
                x0 = float(g[i0])
                x1 = float(g[i0 + 1]) if i0 + 1 < len(g) else float(g[i0])
                pos.append((1.0 - t) * x0 + t * x1)
            else:
                pos.append(float(grids[ax][idx_other[other_axes.index(ax)]]))
        pos_arr = np.array(pos, dtype=float)

        # Normal from grad f at nearest grid index
        # Build nearest index for evaluating gradients
        nearest_index = [0] * dims
        for ax in range(dims):
            if ax == scan_axis:
                nearest_index[ax] = i0
            else:
                nearest_index[ax] = idx_other[other_axes.index(ax)]
        nearest_index_t = tuple(nearest_index)
        gradf_vec = np.array([garr[nearest_index_t] for garr in grad_f], dtype=float)
        n = _unit(gradf_vec.reshape(1, -1)).reshape(-1)

        # κ via n · ∇(c_s^2 − v^2) / (2 c_H)
        grad_cs2mv2_vec = np.array([garr[nearest_index_t] for garr in grad_cs2_mv2], dtype=float)
        # Interpolate c_H along the line between i0 and i0+1
        cs_line = c_s[tuple(slicer)]
        if cs_line.ndim != 1:
            cs_line = np.asarray(cs_line).reshape(-1)
        cs0 = float(cs_line[i0])
        cs1 = float(cs_line[i0 + 1]) if i0 + 1 < cs_line.size else cs0
        cH = max((1.0 - t) * cs0 + t * cs1, 1e-30)

        kappa = abs(float(np.dot(n, grad_cs2mv2_vec))) / (2.0 * cH)

        positions.append(pos_arr)
        normals.append(n)
        kappas.append(kappa)
        ch_list.append(cH)

    if not positions:
        return HorizonSurfaceND(
            positions=np.zeros((0, dims), dtype=float),
            kappa=np.zeros((0,), dtype=float),
            normals=np.zeros((0, dims), dtype=float),
            c_h=np.zeros((0,), dtype=float),
        )

    return HorizonSurfaceND(
        positions=np.vstack(positions),
        kappa=np.asarray(kappas, dtype=float),
        normals=np.vstack(normals),
        c_h=np.asarray(ch_list, dtype=float),
    )

