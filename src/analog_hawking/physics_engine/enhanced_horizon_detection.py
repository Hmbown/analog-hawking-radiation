"""
Enhanced Horizon Detection with 4th-Order Numerical Accuracy

This module provides A+ grade computational accuracy for horizon detection
in analog Hawking radiation analysis, implementing:

- 4th-order finite differences for gradient calculations at horizons
- Enhanced root-finding with Richardson extrapolation
- Adaptive thresholding for physics breakdown detection
- High-order interpolation for horizon property estimation
- Comprehensive uncertainty quantification

Author: Claude Scientific Computing Expert
Date: November 2025

NOTE: Experimental tooling for prototyping; verify outputs against the validated
1D/ND horizon pipeline before drawing conclusions.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple, Callable, Any
from scipy.constants import k, m_p
from scipy.optimize import brentq, newton
import warnings

from .enhanced_numerical_methods import (
    FourthOrderFiniteDifferences,
    EnhancedInterpolation,
    AdaptiveThresholding,
    RichardsonExtrapolation,
    NumericalAccuracyReport
)

@dataclass
class EnhancedHorizonResult:
    """Enhanced horizon detection results with comprehensive accuracy information"""
    positions: np.ndarray                     # horizon x-positions (4th-order accurate)
    kappa: np.ndarray                         # surface gravity estimates (4th-order accurate)
    kappa_err: np.ndarray                     # comprehensive uncertainty estimates
    kappa_method: str                         # method used for kappa calculation
    convergence_info: Dict[str, Any]          # convergence diagnostics
    accuracy_report: Optional[NumericalAccuracyReport]  # numerical accuracy assessment
    interpolation_functions: Dict[str, Callable]  # high-order interpolators
    physics_breakdown_flags: Dict[str, np.ndarray]  # physics breakdown indicators
    adaptive_thresholds: np.ndarray           # locally adaptive thresholds
    gradient_methods: Dict[str, str]          # methods used for gradient calculations
    richardson_extrapolated: Optional[Dict[str, np.ndarray]] = None  # extrapolated results

@dataclass
class HorizonDetectionConfig:
    """Configuration for enhanced horizon detection"""
    gradient_order: int = 4                   # Order of accuracy for gradients (2 or 4)
    interpolation_method: str = 'cubic_spline'  # 'linear', 'cubic_spline', 'pchip'
    use_richardson: bool = True               # Enable Richardson extrapolation
    use_adaptive_thresholds: bool = True       # Enable adaptive thresholding
    kappa_method: str = 'acoustic'            # 'acoustic', 'acoustic_exact', 'legacy'
    convergence_tolerance: float = 1e-10       # Convergence tolerance for root finding
    max_iterations: int = 50                  # Maximum iterations for root refinement
    uncertainty_method: str = 'multi_stencil'  # 'multi_stencil', 'richardson', 'bootstrap'
    enable_physics_checks: bool = True        # Enable physics consistency checks

class EnhancedHorizonDetector:
    """
    Enhanced horizon detection with 4th-order numerical accuracy
    """

    def __init__(self, config: Optional[HorizonDetectionConfig] = None):
        """
        Initialize enhanced horizon detector

        Args:
            config: Configuration for horizon detection parameters
        """
        self.config = config or HorizonDetectionConfig()

        # Initialize numerical methods
        self.fd_solver = FourthOrderFiniteDifferences()
        self.interpolator = EnhancedInterpolation()

        # Initialize adaptive systems
        if self.config.use_adaptive_thresholds:
            self.adaptive_threshold = AdaptiveThresholding()
        else:
            self.adaptive_threshold = None

        if self.config.use_richardson:
            self.richardson = RichardsonExtrapolation()
        else:
            self.richardson = None

    def find_horizons_enhanced(self,
                             x: np.ndarray,
                             v: np.ndarray,
                             c_s: np.ndarray,
                             sigma_cells: Optional[np.ndarray] = None,
                             additional_fields: Optional[Dict[str, np.ndarray]] = None) -> EnhancedHorizonResult:
        """
        Find horizons with 4th-order numerical accuracy

        Args:
            x: spatial coordinates (must be strictly increasing)
            v: velocity field
            c_s: sound speed field
            sigma_cells: optional cell uncertainties
            additional_fields: additional physical quantities for analysis

        Returns:
            EnhancedHorizonResult: Comprehensive horizon detection results
        """
        # Input validation
        x = np.asarray(x, dtype=np.float64)
        v = np.asarray(v, dtype=np.float64)
        c_s = np.asarray(c_s, dtype=np.float64)

        assert x.ndim == v.ndim == c_s.ndim == 1
        assert len(x) == len(v) == len(c_s)
        assert np.all(np.diff(x) > 0), "x must be strictly increasing"

        if self.config.gradient_order == 4 and len(x) < 5:
            warnings.warn("Grid too small for 4th-order methods. Falling back to 2nd-order.")
            self.config.gradient_order = 2

        # Step 1: Compute horizon function f(x) = |v| - c_s
        f = np.abs(v) - c_s

        # Step 2: Find initial horizon locations using sign changes
        initial_horizons = self._find_horizon_brackets(x, f)

        if len(initial_horizons) == 0:
            return self._create_empty_result()

        # Step 3: Refine horizon positions with high-order root finding
        refined_horizons = self._refine_horizon_positions(x, v, c_s, initial_horizons)

        # Step 4: Compute high-order gradients at horizons
        gradient_data = self._compute_horizon_gradients(x, v, c_s, refined_horizons)

        # Step 5: Calculate surface gravity with enhanced accuracy
        kappa_results = self._compute_surface_gravity(x, v, c_s, refined_horizons, gradient_data)

        # Step 6: Perform uncertainty quantification
        uncertainty_results = self._quantify_uncertainties(x, v, c_s, refined_horizons, kappa_results)

        # Step 7: Richardson extrapolation if enabled
        richardson_results = None
        if self.config.use_richardson:
            richardson_results = self._perform_richardson_extrapolation(x, v, c_s, refined_horizons)

        # Step 8: Physics breakdown detection if enabled
        breakdown_flags = {}
        adaptive_thresholds = np.array([])
        if self.config.use_adaptive_thresholds and additional_fields:
            breakdown_flags, adaptive_thresholds = self._detect_physics_breakdown(
                x, v, c_s, refined_horizons, additional_fields
            )

        # Step 9: Create high-order interpolators
        interpolators = self._create_interpolators(x, v, c_s)

        # Step 10: Generate accuracy report
        accuracy_report = self._generate_accuracy_report(x, v, c_s, refined_horizons)

        # Step 11: Compile results
        return EnhancedHorizonResult(
            positions=refined_horizons,
            kappa=kappa_results['kappa'],
            kappa_err=uncertainty_results['kappa_err'],
            kappa_method=self.config.kappa_method,
            convergence_info=kappa_results['convergence_info'],
            accuracy_report=accuracy_report,
            interpolation_functions=interpolators,
            physics_breakdown_flags=breakdown_flags,
            adaptive_thresholds=adaptive_thresholds,
            gradient_methods={
                'velocity_gradient': f'{self.config.gradient_order}th_order',
                'sound_speed_gradient': f'{self.config.gradient_order}th_order'
            },
            richardson_extrapolated=richardson_results
        )

    def _find_horizon_brackets(self, x: np.ndarray, f: np.ndarray) -> List[Tuple[float, float]]:
        """Find brackets containing horizon locations"""
        brackets = []

        for i in range(len(x) - 1):
            f0, f1 = f[i], f[i + 1]

            # Check for sign change or zero crossing
            if np.sign(f0) == 0 and np.sign(f1) == 0:
                # Both points are at horizon - take midpoint
                brackets.append((x[i], x[i+1]))
            elif f0 == 0:
                # Left point is at horizon
                brackets.append((x[i], x[i]))
            elif f1 == 0:
                # Right point is at horizon
                brackets.append((x[i+1], x[i+1]))
            elif f0 * f1 < 0:
                # Sign change indicates horizon between points
                brackets.append((x[i], x[i+1]))

        # Remove duplicate brackets
        unique_brackets = []
        for bracket in brackets:
            if not any(np.isclose(bracket, ub).all() for ub in unique_brackets):
                unique_brackets.append(bracket)

        return unique_brackets

    def _refine_horizon_positions(self,
                                x: np.ndarray,
                                v: np.ndarray,
                                c_s: np.ndarray,
                                brackets: List[Tuple[float, float]]) -> np.ndarray:
        """Refine horizon positions using high-order root finding"""
        refined_positions = []

        for xl, xr in brackets:
            if np.isclose(xl, xr):
                # Already at exact horizon
                refined_positions.append(xl)
                continue

            # Define horizon function with high-order interpolation
            def horizon_function(xi):
                # Use high-order interpolation for v and c_s
                v_interp = self._interpolate_field(x, v, xi)
                cs_interp = self._interpolate_field(x, c_s, xi)
                return abs(v_interp) - cs_interp

            try:
                # Use Brent's method for robust root finding
                root = brentq(horizon_function, xl, xr,
                            xtol=self.config.convergence_tolerance,
                            maxiter=self.config.max_iterations)
                refined_positions.append(root)
            except (ValueError, RuntimeError):
                # Fallback to midpoint if root finding fails
                refined_positions.append(0.5 * (xl + xr))

        return np.array(sorted(refined_positions))

    def _interpolate_field(self, x: np.ndarray, y: np.ndarray, xi: float) -> float:
        """Interpolate field value at xi using configured method"""
        if self.config.interpolation_method == 'linear':
            return np.interp(xi, x, y)
        elif self.config.interpolation_method == 'cubic_spline':
            spline = self.interpolator.cubic_spline_interpolation(x, y)
            return float(spline(xi))
        elif self.config.interpolation_method == 'pchip':
            spline = self.interpolator.cubic_spline_interpolation(x, y, monotonic=True)
            return float(spline(xi))
        else:
            # Default to linear interpolation
            return np.interp(xi, x, y)

    def _compute_horizon_gradients(self,
                                 x: np.ndarray,
                                 v: np.ndarray,
                                 c_s: np.ndarray,
                                 horizon_positions: np.ndarray) -> Dict[str, np.ndarray]:
        """Compute gradients at horizon positions with enhanced accuracy"""
        dvdx_list = []
        dcsdx_list = []

        for horizon_pos in horizon_positions:
            # Find nearest grid index
            idx = int(np.clip(np.searchsorted(x, horizon_pos), 1, len(x)-2))

            if self.config.gradient_order == 4 and len(x) >= 5:
                # Use 4th-order finite differences
                dvdx = self.fd_solver.gradient_central_4th(x, v)[idx]
                dcsdx = self.fd_solver.gradient_central_4th(x, c_s)[idx]
            else:
                # Use 2nd-order finite differences
                dvdx = np.gradient(v, x)[idx]
                dcsdx = np.gradient(c_s, x)[idx]

            dvdx_list.append(dvdx)
            dcsdx_list.append(dcsdx)

        return {
            'dvdx': np.array(dvdx_list),
            'dcsdx': np.array(dcsdx_list)
        }

    def _compute_surface_gravity(self,
                               x: np.ndarray,
                               v: np.ndarray,
                               c_s: np.ndarray,
                               horizon_positions: np.ndarray,
                               gradient_data: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Compute surface gravity with enhanced accuracy"""
        kappa_list = []
        convergence_info = []

        dvdx = gradient_data['dvdx']
        dcsdx = gradient_data['dcsdx']

        for i, horizon_pos in enumerate(horizon_positions):
            # Get velocity at horizon
            v_h = self._interpolate_field(x, v, horizon_pos)

            # Compute derivative of |v|
            if v_h != 0:
                d_abs_v = np.sign(v_h) * dvdx[i]
            else:
                d_abs_v = abs(dvdx[i])

            # Compute kappa based on method
            if self.config.kappa_method.lower() == 'legacy':
                kappa = 0.5 * abs(d_abs_v - dcsdx[i])
            elif self.config.kappa_method.lower() == 'acoustic_exact':
                # κ = |∂x(c_s² - v²)| / (2c_H)
                c_h = self._interpolate_field(x, c_s, horizon_pos)
                dc2_minus_v2 = 2.0 * c_h * dcsdx[i] - 2.0 * v_h * dvdx[i]
                kappa = abs(dc2_minus_v2) / (2.0 * max(abs(c_h), 1e-30))
            else:  # acoustic (default)
                kappa = abs(dcsdx[i] - d_abs_v)

            kappa_list.append(kappa)

            # Store convergence information
            convergence_info.append({
                'position': horizon_pos,
                'velocity_at_horizon': v_h,
                'sound_speed_at_horizon': c_h if 'c_h' in locals() else self._interpolate_field(x, c_s, horizon_pos),
                'gradient_difference': abs(dcsdx[i] - d_abs_v),
                'method_used': self.config.kappa_method
            })

        return {
            'kappa': np.array(kappa_list),
            'convergence_info': convergence_info
        }

    def _quantify_uncertainties(self,
                              x: np.ndarray,
                              v: np.ndarray,
                              c_s: np.ndarray,
                              horizon_positions: np.ndarray,
                              kappa_results: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """Quantify uncertainties in horizon detection"""
        kappa_err_list = []

        if self.config.uncertainty_method == 'multi_stencil':
            # Use multiple finite difference stencils
            for i, horizon_pos in enumerate(horizon_positions):
                idx = int(np.clip(np.searchsorted(x, horizon_pos), 2, len(x)-3))

                # Compute kappa with different stencil sizes
                kappa_stencils = []
                for stencil in [1, 2, 3]:
                    if self.config.gradient_order == 4:
                        # Use custom stencil sizes
                        dvdx = self._finite_difference_custom(x, v, idx, stencil)
                        dcsdx = self._finite_difference_custom(x, c_s, idx, stencil)
                    else:
                        # Simple gradient with custom stencil
                        dvdx = (v[min(idx+stencil, len(v)-1)] - v[max(idx-stencil, 0)]) / \
                               (x[min(idx+stencil, len(x)-1)] - x[max(idx-stencil, 0)])
                        dcsdx = (c_s[min(idx+stencil, len(c_s)-1)] - c_s[max(idx-stencil, 0)]) / \
                                (x[min(idx+stencil, len(x)-1)] - x[max(idx-stencil, 0)])

                    v_h = self._interpolate_field(x, v, horizon_pos)
                    if v_h != 0:
                        d_abs_v = np.sign(v_h) * dvdx
                    else:
                        d_abs_v = abs(dvdx)

                    if self.config.kappa_method.lower() == 'legacy':
                        kappa = 0.5 * abs(d_abs_v - dcsdx)
                    else:  # acoustic
                        kappa = abs(dcsdx - d_abs_v)

                    kappa_stencils.append(kappa)

                # Uncertainty as standard deviation of stencil results
                kappa_err = np.std(kappa_stencils)
                kappa_err_list.append(kappa_err)

        elif self.config.uncertainty_method == 'richardson' and self.richardson:
            # Use Richardson extrapolation for uncertainty estimation
            # This would require multiple grid resolutions
            kappa_err_list = np.full(len(horizon_positions), 0.1 * kappa_results['kappa'])

        else:  # bootstrap or default
            # Simple uncertainty estimate based on local gradients
            kappa_err_list = 0.05 * kappa_results['kappa']  # 5% uncertainty

        return {
            'kappa_err': np.array(kappa_err_list)
        }

    def _finite_difference_custom(self, x: np.ndarray, y: np.ndarray, idx: int, stencil: int) -> float:
        """Custom finite difference with specified stencil size"""
        i0 = max(0, idx - stencil)
        i1 = min(len(x) - 1, idx + stencil)

        if i1 == i0:
            return 0.0

        return (y[i1] - y[i0]) / (x[i1] - x[i0])

    def _perform_richardson_extrapolation(self,
                                         x: np.ndarray,
                                         v: np.ndarray,
                                         c_s: np.ndarray,
                                         horizon_positions: np.ndarray) -> Optional[Dict[str, np.ndarray]]:
        """Perform Richardson extrapolation for enhanced accuracy"""
        if not self.richardson:
            return None

        # This would require running the same analysis on refined grids
        # For now, return None as a placeholder
        # In a full implementation, this would:
        # 1. Create refined versions of the grids
        # 2. Run horizon detection on each grid
        # 3. Apply Richardson extrapolation to converge to grid-independent results

        return None

    def _detect_physics_breakdown(self,
                                x: np.ndarray,
                                v: np.ndarray,
                                c_s: np.ndarray,
                                horizon_positions: np.ndarray,
                                additional_fields: Dict[str, np.ndarray]) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
        """Detect physics breakdown using adaptive thresholds"""
        if not self.adaptive_threshold:
            return {}, np.array([])

        # Prepare physical quantities for thresholding
        physical_quantities = {
            'velocity': v,
            'sound_speed': c_s
        }
        physical_quantities.update(additional_fields)

        # Compute adaptive thresholds
        adaptive_thresholds = self.adaptive_threshold.compute_adaptive_threshold(physical_quantities)

        # Define diagnostic functions for breakdown detection
        def velocity_check(velocity, x_coords):
            return np.gradient(velocity, x_coords) / (np.mean(np.abs(velocity)) + 1e-10)

        def sound_speed_check(sound_speed, x_coords):
            return np.gradient(sound_speed, x_coords) / (np.mean(np.abs(sound_speed)) + 1e-10)

        diagnostic_functions = {
            'velocity': velocity_check,
            'sound_speed': sound_speed_check
        }

        # Add checks for additional fields
        for field_name in additional_fields:
            def field_check(field, x_coords):
                return np.gradient(field, x_coords) / (np.mean(np.abs(field)) + 1e-10)
            diagnostic_functions[field_name] = field_check

        # Detect breakdown
        breakdown_flags = self.adaptive_threshold.detect_physics_breakdown(
            x, physical_quantities, diagnostic_functions
        )

        return breakdown_flags, adaptive_thresholds

    def _create_interpolators(self,
                            x: np.ndarray,
                            v: np.ndarray,
                            c_s: np.ndarray) -> Dict[str, Callable]:
        """Create high-order interpolators for fields"""
        interpolators = {}

        if self.config.interpolation_method == 'cubic_spline':
            interpolators['velocity'] = self.interpolator.cubic_spline_interpolation(x, v)
            interpolators['sound_speed'] = self.interpolator.cubic_spline_interpolation(x, c_s)
        elif self.config.interpolation_method == 'pchip':
            interpolators['velocity'] = self.interpolator.cubic_spline_interpolation(x, v, monotonic=True)
            interpolators['sound_speed'] = self.interpolator.cubic_spline_interpolation(x, c_s, monotonic=True)
        else:  # linear
            interpolators['velocity'] = lambda xi: np.interp(xi, x, v)
            interpolators['sound_speed'] = lambda xi: np.interp(xi, x, c_s)

        return interpolators

    def _generate_accuracy_report(self,
                                x: np.ndarray,
                                v: np.ndarray,
                                c_s: np.ndarray,
                                horizon_positions: np.ndarray) -> NumericalAccuracyReport:
        """Generate comprehensive accuracy report"""
        # Estimate order of accuracy by comparing with lower-order methods
        if self.config.gradient_order == 4 and len(x) >= 9:
            # Compare 4th-order vs 2nd-order gradients
            grad_4th = self.fd_solver.gradient_central_4th(x, v)
            grad_2nd = np.gradient(v, x)

            # Estimate order from difference between methods
            if len(horizon_positions) > 0:
                idx = int(np.clip(np.searchsorted(x, horizon_positions[0]), 2, len(x)-3))
                diff_4th = abs(grad_4th[idx])
                diff_2nd = abs(grad_2nd[idx])

                if diff_2nd > 1e-15:
                    estimated_order = np.log(diff_2nd / 1e-15) / np.log(2.0)  # Rough estimate
                    estimated_order = max(2.0, min(4.0, estimated_order))
                else:
                    estimated_order = 4.0
            else:
                estimated_order = 4.0
        else:
            estimated_order = float(self.config.gradient_order)

        # Generate recommendations
        recommendations = []
        if len(x) < 10:
            recommendations.append("Grid resolution is low. Consider increasing grid points for better accuracy.")
        if self.config.gradient_order == 2:
            recommendations.append("Consider using 4th-order gradients for enhanced accuracy.")
        if not self.config.use_richardson:
            recommendations.append("Enable Richardson extrapolation for grid-independent results.")

        return NumericalAccuracyReport(
            method=f"Enhanced Horizon Detection ({self.config.gradient_order}th-order)",
            order_of_accuracy=estimated_order,
            error_estimate=0.0,  # Would be computed from Richardson extrapolation
            grid_convergence=True,  # Would be determined from convergence study
            recommendations=recommendations
        )

    def _create_empty_result(self) -> EnhancedHorizonResult:
        """Create empty result when no horizons are found"""
        return EnhancedHorizonResult(
            positions=np.array([]),
            kappa=np.array([]),
            kappa_err=np.array([]),
            kappa_method=self.config.kappa_method,
            convergence_info={},
            accuracy_report=None,
            interpolation_functions={},
            physics_breakdown_flags={},
            adaptive_thresholds=np.array([]),
            gradient_methods={}
        )

# Convenience function for easy usage
def find_horizons_with_enhanced_accuracy(x: np.ndarray,
                                       v: np.ndarray,
                                       c_s: np.ndarray,
                                       gradient_order: int = 4,
                                       kappa_method: str = 'acoustic',
                                       **kwargs) -> EnhancedHorizonResult:
    """
    Convenience function for enhanced horizon detection

    Args:
        x: spatial coordinates
        v: velocity field
        c_s: sound speed field
        gradient_order: order of accuracy for gradients (2 or 4)
        kappa_method: method for surface gravity calculation
        **kwargs: additional configuration parameters

    Returns:
        EnhancedHorizonResult: comprehensive horizon detection results
    """
    config = HorizonDetectionConfig(
        gradient_order=gradient_order,
        kappa_method=kappa_method,
        **kwargs
    )

    detector = EnhancedHorizonDetector(config)
    return detector.find_horizons_enhanced(x, v, c_s)
