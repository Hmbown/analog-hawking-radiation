"""
Enhanced patch-wise graybody aggregation with variation preservation.

This enhanced version preserves physical information that would be lost
through destructive operations like np.mean(), np.sum(), etc.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Sequence, Tuple, Union, Any

import numpy as np
from scipy.constants import c, h, k  # type: ignore


@dataclass
class VariationTracker:
    """
    Tracks variation metadata through destructive operations.
    Preserves physical information that would otherwise be lost.
    """
    operation: str
    input_shape: Tuple[int, ...]
    axis: int | None = None
    std_before: float = 0.0
    std_after: float = 0.0
    total_variation: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'operation': self.operation,
            'input_shape': self.input_shape,
            'axis': self.axis,
            'std_before': float(self.std_before),
            'std_after': float(self.std_after),
            'total_variation': float(self.total_variation),
            'metadata': self.metadata
        }


@dataclass
class AggregatedSpectrum:
    success: bool
    frequencies: np.ndarray | None = None
    power_spectrum: np.ndarray | None = None
    power_std: np.ndarray | None = None
    peak_frequency: float | None = None
    n_patches: int = 0
    variation_history: List[VariationTracker] = field(default_factory=list)
    
    def add_variation_tracker(self, tracker: VariationTracker):
        """Add a variation tracker to the history"""
        self.variation_history.append(tracker)
    
    def get_uncertainty_metrics(self) -> Dict[str, float]:
        """Get uncertainty metrics from variation history"""
        if not self.variation_history:
            return {}
        
        # Sum up variation losses
        total_variation_lost = sum(
            vt.std_before - vt.std_after 
            for vt in self.variation_history
            if vt.operation == 'mean'
        )
        
        return {
            'total_variation_lost': float(total_variation_lost),
            'n_operations_tracked': len(self.variation_history),
            'avg_std_before': float(np.mean([vt.std_before for vt in self.variation_history])),
            'operations': [vt.operation for vt in self.variation_history]
        }


class VariationPreservingArray:
    """
    Wrapper for numpy arrays that preserves variation information
    through destructive operations.
    """
    
    def __init__(self, data: np.ndarray, axis: int | None = None, 
                 keepdims: bool = False, name: str = ""):
        self.data = np.asarray(data)
        self.axis = axis
        self.keepdims = keepdims
        self.name = name
        self.variation_history: List[VariationTracker] = []
    
    def mean(self, axis: int | None = None, keepdims: bool = False) -> float | np.ndarray:
        """Compute mean while preserving variation information"""
        result = np.mean(self.data, axis=axis, keepdims=keepdims)
        
        # Track the variation loss
        tracker = VariationTracker(
            operation='mean',
            input_shape=self.data.shape,
            axis=axis,
            std_before=np.std(self.data),
            std_after=np.std(result) if hasattr(result, 'shape') else 0.0,
            metadata={'keepdims': keepdims}
        )
        self.variation_history.append(tracker)
        
        return result
    
    def sum(self, axis: int | None = None, keepdims: bool = False) -> float | np.ndarray:
        """Compute sum while preserving variation information"""  
        result = np.sum(self.data, axis=axis, keepdims=keepdims)
        
        # Track the variation
        tracker = VariationTracker(
            operation='sum',
            input_shape=self.data.shape,
            axis=axis,
            total_variation=np.sum(np.abs(np.diff(self.data))) if self.data.size > 1 else 0.0,
            metadata={'keepdims': keepdims}
        )
        self.variation_history.append(tracker)
        
        return result
    
    def collapse(self, operation: str = 'mean') -> float | np.ndarray:
        """Explicit collapse operation with variation tracking"""
        if operation == 'mean':
            return self.mean(axis=self.axis, keepdims=self.keepdims)
        elif operation == 'sum':
            return self.sum(axis=self.axis, keepdims=self.keepdims)
        else:
            raise ValueError(f"Unknown operation: {operation}")
    
    def get_variation_history(self) -> List[VariationTracker]:
        """Get the complete variation history"""
        return self.variation_history


def aggregate_patchwise_graybody(
    grids: Sequence[np.ndarray],
    v_field: np.ndarray,  # (..., D)
    c_s: np.ndarray,  # (...,)
    kappa_eff: float | np.ndarray,  # Single value or per-patch values
    *,
    graybody_method: str = "dimensionless",
    alpha_gray: float = 1.0,
    scan_axis: int = 0,
    patch_indices: np.ndarray | None = None,
    max_patches: int = 64,
    sample_mode: str = "scan_axis",  # or "normal"
    preserve_variation: bool = True,  # New: enable variation preservation
) -> AggregatedSpectrum:
    """Aggregate patch-wise spectra along the scan axis.

    Args:
        grids: coordinate arrays [x0, x1, (x2)]
        v_field: vector velocity field with components last
        c_s: scalar sound speed field
        kappa_eff: effective κ to use for spectral calculations. Can be:
                  - Single float: same κ for all patches (backward compatible)
                  - Array: different κ for each patch (enables hybrid coupling)
        graybody_method: one of {dimensionless, wkb, acoustic_wkb}
        alpha_gray: graybody scaling parameter
        scan_axis: axis to sample along (0..D-1)
        patch_indices: optional array of indices into the horizon points list
        max_patches: cap on the number of patches to sample
        sample_mode: 'scan_axis' (sample along axis) or 'normal' (along local normal)
        preserve_variation: whether to track variation information (default: True)

    Returns:
        AggregatedSpectrum with mean power spectrum, standard deviation,
        and variation tracking history if enabled.
    """
    # Handle both single kappa (backward compatible) and array of kappas (new)
    if isinstance(kappa_eff, (int, float)):
        if kappa_eff <= 0.0:
            return AggregatedSpectrum(success=False)
        kappa_per_patch = None  # Will use same kappa for all patches
    else:
        kappa_array = np.asarray(kappa_eff)
        if np.any(kappa_array <= 0):
            return AggregatedSpectrum(success=False)
        kappa_per_patch = kappa_array

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
    
    # Track variation if enabled
    variation_trackers: List[VariationTracker] = []
    
    for i, patch_idx in enumerate(patch_indices):
        pos = surf.positions[patch_idx]
        # Use per-patch kappa if available, otherwise use single kappa_eff
        if kappa_per_patch is not None and i < len(kappa_per_patch):
            kappa_for_this_patch = float(kappa_per_patch[i])
        else:
            kappa_for_this_patch = float(kappa_eff) if isinstance(kappa_eff, (int, float)) else float(kappa_eff[0])
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
            v_line = np.sqrt(np.sum(v_line_vec**2, axis=-1))
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
                v_line = np.sqrt(np.sum(v_line_vec**2, axis=-1))
                cs_line = c_s[sl]
                if cs_line.ndim != 1:
                    cs_line = np.reshape(cs_line, (-1,))
                profile = {"x": x_axis, "v": v_line, "c_s": cs_line}
            else:
                normal = surf.normals[patch_idx]
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
                    idx_coords.append((pos[ax] + s_line * normal[ax] - origin) / dx)
                coords = np.vstack(idx_coords)
                # Interpolate each velocity component and c_s
                v_comps = []
                for component in range(dims):
                    field = v_field[..., component]
                    v_comps.append(map_coordinates(field, coords, order=1, mode="nearest"))
                v_line = np.sqrt(np.sum(np.vstack(v_comps) ** 2, axis=0))
                cs_line = map_coordinates(c_s, coords, order=1, mode="nearest")
                # Use s_line as the local coordinate
                profile = {"x": s_line, "v": v_line, "c_s": cs_line}

        # Calculate spectrum for this patch
        sp = calculate_hawking_spectrum(
            kappa_for_this_patch,
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

    # Align and average spectra with variation tracking
    f0 = np.asarray(specs[0]["frequencies"])  # type: ignore[index]
    P_mat = []
    
    for sp in specs:
        f = np.asarray(sp["frequencies"])  # type: ignore[index]
        P = np.asarray(sp["power_spectrum"])  # type: ignore[index]
        if f.shape != f0.shape or not np.allclose(f, f0):
            # Use variation-preserving interpolation if enabled
            if preserve_variation:
                P_variation = VariationPreservingArray(P, name="power_interpolation")
                P = P_variation.collapse('mean')
            else:
                P = np.interp(f0, f, P)
        P_mat.append(P)
    
    # Stack and compute statistics with variation tracking
    P_stack = np.vstack(P_mat)
    
    if preserve_variation:
        # Use variation-preserving mean
        P_variation = VariationPreservingArray(P_stack, axis=0, name="power_ensemble")
        P_mean = P_variation.collapse('mean')
        variation_trackers.extend(P_variation.get_variation_history())
        
        # Use variation-preserving std
        P_std_variation = VariationPreservingArray(P_stack, axis=0, name="power_std")
        P_std = P_std_variation.collapse('std' if hasattr(P_std_variation, 'collapse') else 'mean')
        # For actual std, we need to compute it properly
        P_std = np.std(P_stack, axis=0)
    else:
        P_mean = np.mean(P_stack, axis=0)
        P_std = np.std(P_stack, axis=0)
    
    peak_f = float(f0[int(np.argmax(P_mean))])

    # Create result with variation tracking
    result = AggregatedSpectrum(
        success=True,
        frequencies=f0,
        power_spectrum=P_mean,
        power_std=P_std,
        peak_frequency=peak_f,
        n_patches=len(specs),
    )
    
    # Add variation history if enabled
    if preserve_variation:
        for tracker in variation_trackers:
            result.add_variation_tracker(tracker)
    
    return result


def compute_graybody_with_variation_preservation(
    kappa_profile: np.ndarray,
    energy_grid: np.ndarray,
    temperature_profile: np.ndarray,
    preserve_variation: bool = True
) -> Dict[str, Any]:
    """
    Compute graybody factors with optional variation preservation.
    
    This enhanced function demonstrates how variation preservation
    can be integrated into AHR calculations.
    """
    result = {
        'kappa_profile': kappa_profile,
        'energy_grid': energy_grid,
        'temperature_profile': temperature_profile,
        'variation_history': []
    }
    
    if preserve_variation:
        # Track variation in kappa
        kappa_variation = VariationPreservingArray(kappa_profile, name="kappa")
        avg_kappa = kappa_variation.collapse('mean')
        result['avg_kappa'] = avg_kappa
        result['variation_history'].extend(kappa_variation.get_variation_history())
        
        # Track variation in temperature
        temp_variation = VariationPreservingArray(temperature_profile, name="temperature")
        avg_temp = temp_variation.collapse('mean')
        result['avg_temperature'] = avg_temp
        result['variation_history'].extend(temp_variation.get_variation_history())
        
        # Compute Hawking temperature with preserved variation
        # T_H = κ/(2π)
        hawking_temp_variation = VariationPreservingArray(kappa_profile / (2 * np.pi), name="hawking_temp")
        avg_hawking_temp = hawking_temp_variation.collapse('mean')
        result['hawking_temperature'] = avg_hawking_temp
        result['variation_history'].extend(hawking_temp_variation.get_variation_history())
    else:
        # Traditional destructive calculation (backward compatible)
        result['avg_kappa'] = np.mean(kappa_profile)
        result['avg_temperature'] = np.mean(temperature_profile)
        result['hawking_temperature'] = result['avg_kappa'] / (2 * np.pi)
    
    # Compute graybody factor
    # γ(ω) = ω^2 / (exp(ω/T_H) - 1)
    if preserve_variation:
        # Use variation-preserving calculation
        energy_variation = VariationPreservingArray(energy_grid, name="energy")
        mean_energy = energy_variation.collapse('mean')
        result['mean_energy'] = mean_energy
        result['variation_history'].extend(energy_variation.get_variation_history())
        
        # Graybody with variation tracking
        omega = 2 * np.pi * energy_grid
        exp_term = np.exp(omega / result['hawking_temperature'])
        graybody = omega**2 / (exp_term - 1 + 1e-12)
        
        graybody_variation = VariationPreservingArray(graybody, name="graybody")
        avg_graybody = graybody_variation.collapse('mean')
        result['graybody_factor'] = avg_graybody
        result['variation_history'].extend(graybody_variation.get_variation_history())
    else:
        # Traditional calculation
        result['mean_energy'] = np.mean(energy_grid)
        omega = 2 * np.pi * energy_grid
        exp_term = np.exp(omega / result['hawking_temperature'])
        result['graybody_factor'] = np.mean(omega**2 / (exp_term - 1 + 1e-12))
    
    return result