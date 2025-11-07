"""
Enhanced Coupling Module for Hybrid Plasma-Mirror Models

This module addresses the spatial coupling variation issue in hybrid models where
the plasma mirror coupling weight varies across the horizon surface, but the graybody
calculation uses a single effective kappa value.

The key insight: coupling_weight from horizon_hybrid.py is spatially varying, but
graybody_nd.py's aggregate_patchwise_graybody() uses a single kappa_eff parameter.
This creates a computational mirage - we compute spatially varying coupling weights
but then ignore them in the spectral calculation.

Author: bern2025-k2 (Patent Clerk Edition)
Date: 1905-11-06 (in spirit)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np

from analog_hawking.physics_engine.horizon_hybrid import HybridHorizonResult


@dataclass
class SpatialCouplingProfile:
    """Container for spatially resolved coupling information."""
    
    positions: np.ndarray  # Horizon positions [m]
    fluid_kappa: np.ndarray  # Fluid surface gravity at each position [Hz]
    mirror_kappa: float  # Mirror surface gravity [Hz] 
    coupling_weights: np.ndarray  # Spatially varying weights [dimensionless]
    alignment: np.ndarray  # Alignment factors [-1, 0, 1]
    
    @property
    def effective_kappa(self) -> np.ndarray:
        """Compute effective kappa at each position including coupling."""
        return self.fluid_kappa + self.coupling_weights * self.mirror_kappa
    
    @property 
    def mean_coupling_weight(self) -> float:
        """Mean coupling weight across horizon surface."""
        if len(self.coupling_weights) == 0:
            return 0.0
        return float(np.mean(self.coupling_weights))
    
    @property
    def max_coupling_weight(self) -> float:
        """Maximum coupling weight (for validation bounds)."""
        if len(self.coupling_weights) == 0:
            return 0.0
        return float(np.max(self.coupling_weights))


def create_spatial_coupling_profile(
    hybrid_result: HybridHorizonResult,
) -> SpatialCouplingProfile:
    """
    Convert HybridHorizonResult to spatial coupling profile.
    
    This bridges the gap between the hybrid horizon detection (which computes
    spatially varying coupling weights) and the graybody calculation (which
    needs to apply different effective kappa values at different patches).
    """
    return SpatialCouplingProfile(
        positions=hybrid_result.fluid.positions,
        fluid_kappa=hybrid_result.fluid.kappa,
        mirror_kappa=hybrid_result.kappa_mirror,
        coupling_weights=hybrid_result.coupling_weight,
        alignment=hybrid_result.alignment,
    )


def compute_patchwise_effective_kappa(
    profile: SpatialCouplingProfile,
    patch_indices: np.ndarray | None = None,
) -> np.ndarray:
    """
    Compute effective kappa values for specific patches.
    
    Args:
        profile: Spatial coupling profile from hybrid model
        patch_indices: Indices of patches to compute kappa for. If None, uses all.
        
    Returns:
        Array of effective kappa values for each patch [Hz]
    """
    if patch_indices is None:
        patch_indices = np.arange(len(profile.positions))
    
    effective_kappas = np.zeros(len(patch_indices))
    
    for i, idx in enumerate(patch_indices):
        if idx < len(profile.fluid_kappa):
            kappa_fluid = profile.fluid_kappa[idx]
            weight = profile.coupling_weights[idx] if idx < len(profile.coupling_weights) else 0.0
            effective_kappas[i] = kappa_fluid + weight * profile.mirror_kappa
        else:
            effective_kappas[i] = 0.0
            
    return effective_kappas


def validate_coupling_profile(profile: SpatialCouplingProfile) -> Dict[str, float]:
    """
    Validate that coupling profile makes physical sense.
    
    Returns validation metrics and flags potential issues.
    """
    validation_results = {}
    
    # Check for negative kappa values (unphysical)
    effective_kappas = profile.effective_kappa
    negative_kappas = np.sum(effective_kappas < 0)
    validation_results["negative_kappa_count"] = int(negative_kappas)
    
    # Check coupling weight distribution
    if len(profile.coupling_weights) > 0:
        validation_results["mean_coupling_weight"] = float(np.mean(profile.coupling_weights))
        validation_results["std_coupling_weight"] = float(np.std(profile.coupling_weights))
        validation_results["max_coupling_weight"] = float(np.max(profile.coupling_weights))
        validation_results["min_coupling_weight"] = float(np.min(profile.coupling_weights))
        
        # Flag if all weights are identical (suspicious - might be computational artifact)
        weight_std = np.std(profile.coupling_weights)
        if weight_std < 1e-12:
            validation_results["uniform_weight_flag"] = 1.0
            validation_results["uniform_weight_warning"] = "All coupling weights are identical - potential computational artifact"
        else:
            validation_results["uniform_weight_flag"] = 0.0
    else:
        validation_results["mean_coupling_weight"] = 0.0
        validation_results["warning"] = "No coupling weights available"
        
    # Check alignment distribution
    if len(profile.alignment) > 0:
        aligned_count = np.sum(profile.alignment > 0)
        anti_aligned_count = np.sum(profile.alignment < 0)
        neutral_count = np.sum(profile.alignment == 0)
        
        validation_results["aligned_fraction"] = float(aligned_count / len(profile.alignment))
        validation_results["anti_aligned_fraction"] = float(anti_aligned_count / len(profile.alignment))
        validation_results["neutral_fraction"] = float(neutral_count / len(profile.alignment))
    
    return validation_results


def diagnose_coupling_artifact(profile: SpatialCouplingProfile) -> Dict[str, any]:
    """
    Diagnose whether the hybrid signal enhancement is physical or computational artifact.
    
    The frontier problem asks: "Why does hybrid model predict 4× higher signal temperature,
    yet validation flags perfect correlations as 'by construction'?"
    
    This function investigates whether the enhancement comes from:
    1. Genuine physical coupling (spatially varying weights)
    2. Computational artifact (uniform weights, deterministic relationships)
    """
    diagnosis = {
        "is_artifact": False,
        "artifact_type": None,
        "confidence": 0.0,
        "recommendations": []
    }
    
    validation = validate_coupling_profile(profile)
    
    # Red flag: All coupling weights are identical
    if validation.get("uniform_weight_flag", 0) > 0.5:
        diagnosis["is_artifact"] = True
        diagnosis["artifact_type"] = "uniform_weights"
        diagnosis["confidence"] = 0.9
        diagnosis["explanation"] = "All coupling weights are identical - this creates deterministic relationship between fluid and hybrid kappa"
        diagnosis["recommendations"].append("Implement spatially varying coupling based on physical parameters")
        diagnosis["recommendations"].append("Check horizon_hybrid.py localization logic")
        
    # Red flag: Perfect correlation between fluid and hybrid kappa
    if len(profile.fluid_kappa) > 1 and len(profile.coupling_weights) > 1:
        # If weights are constant, hybrid_kappa = fluid_kappa + constant*mirror_kappa
        # This creates perfect linear correlation
        weight_std = validation.get("std_coupling_weight", 0)
        if weight_std < 1e-6:  # Nearly uniform weights
            correlation = np.corrcoef(profile.fluid_kappa, profile.effective_kappa)[0, 1]
            if correlation > 0.999:
                diagnosis["is_artifact"] = True
                diagnosis["artifact_type"] = "perfect_correlation"
                diagnosis["confidence"] = 0.95
                diagnosis["explanation"] = f"Perfect correlation (r={correlation:.3f}) between fluid and hybrid kappa - by construction artifact"
                diagnosis["recommendations"].append("Validation framework correctly flagged this as 'by construction'")
                diagnosis["recommendations"].append("Need spatially varying weights to break deterministic relationship")
    
    # Red flag: No spatial variation in coupling (only if not already flagged as uniform)
    if diagnosis["artifact_type"] is None and len(np.unique(profile.coupling_weights)) <= 2:
        diagnosis["is_artifact"] = True
        diagnosis["artifact_type"] = "insufficient_spatial_variation"
        diagnosis["confidence"] = 0.8
        diagnosis["explanation"] = f"Only {len(np.unique(profile.coupling_weights))} unique coupling weight values across {len(profile.coupling_weights)} patches"
        diagnosis["recommendations"].append("Increase spatial resolution or improve coupling model")
    
    # Green flags: Physical coupling
    if validation.get("std_coupling_weight", 0) > 0.01:  # Significant variation
        if validation.get("aligned_fraction", 0) > 0.3:  # Some alignment
            diagnosis["physical_evidence"] = "Significant weight variation and alignment detected"
            diagnosis["confidence"] = min(diagnosis["confidence"], 0.3)  # Lower artifact confidence
    
    # Set explanation if not already set
    if "explanation" not in diagnosis:
        if diagnosis["is_artifact"]:
            diagnosis["explanation"] = "No specific explanation generated"
        else:
            diagnosis["explanation"] = "No artifact detected - coupling appears physical"
    
    return diagnosis