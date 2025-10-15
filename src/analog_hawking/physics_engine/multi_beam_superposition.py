"""
Multi-beam field superposition (2D) for gradient enhancement estimation.

This module computes time-averaged intensity from multiple coherent Gaussian beams
focused at the origin and estimates the magnitude of the intensity gradient near the
focus as a proxy for velocity-gradient-driven surface gravity (κ) scaling.

Key choices for physical sanity:
- Conserve total peak power by splitting among N beams (I_total / N per beam).
- Coherent combination with optional phase alignment at the focus.
- Time-average over several optical cycles.
- Use max |∇I| within a small radius around focus as the metric (center gradient is
  zero by symmetry), then normalize to a 1-beam baseline under the same total power.

This does not replace PIC/fluid simulations, but provides a grounded alternative
to naive multiplicative “n×” heuristics.
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from scipy.constants import c, epsilon_0, pi
from scipy import ndimage as ndi


@dataclass
class BeamConfig:
    direction: np.ndarray  # unit vector (2D) pointing toward the focus
    phase: float           # initial phase offset
    weight: float          # fractional power weight for this beam (sum to 1)


def _unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def _r_perp_sq(x: np.ndarray, y: np.ndarray, n: np.ndarray) -> np.ndarray:
    """Return squared distance perpendicular to direction n for each (x,y)."""
    # r = (x,y); r·n = xn + yn; |r_perp|^2 = r^2 - (r·n)^2
    r2 = x**2 + y**2
    dot = x * n[0] + y * n[1]
    return r2 - dot**2


def _time_average_intensity(x: np.ndarray,
                            y: np.ndarray,
                            beams: List[BeamConfig],
                            wavelength: float,
                            w0: float,
                            I_total: float,
                            n_time: int = 16,
                            w0_lab: Optional[Tuple[float, float]] = None,
                            two_color: Optional[Dict[str, float]] = None) -> np.ndarray:
    """
    Time-average intensity from coherent sum of Gaussian beams.
    - Gaussian envelope transverse to each beam axis: exp(-r_perp^2/w0^2)
    - Split total peak power among beams by weight; E0 ~ sqrt(I).
    - Phase aligned at focus when phase=0 to explore best-case constructive combining.
    """
    omega = 2 * pi * c / wavelength
    k = 2 * pi / wavelength
    # Peak electric field for total intensity I_total: E0_tot = sqrt(2 I_total / (c eps0))
    E0_tot = np.sqrt(2 * I_total / (c * epsilon_0))

    # Build per-beam amplitude scalings
    E0_beams = [E0_tot * np.sqrt(max(beam.weight, 0.0)) for beam in beams]

    # Time sampling across one period
    T = 2 * pi / omega
    ts = np.linspace(0, T, n_time, endpoint=False)

    # Accumulate time-averaged E^2 (intensity ~ E^2 time-averaged for this purpose)
    E2_acc = np.zeros_like(x)
    for t in ts:
        Etot = np.zeros_like(x)
        for E0_i, beam in zip(E0_beams, beams):
            n = beam.direction
            # Envelope: either beam-axis transverse Gaussian or lab-fixed elliptical Gaussian
            if w0_lab is not None:
                w0x, w0y = w0_lab
                env = np.exp(-((x / max(w0x, 1e-20))**2 + (y / max(w0y, 1e-20))**2))
            else:
                rps = _r_perp_sq(x, y, n)
                env = np.exp(-rps / (w0**2))
            phase = omega * t - k * (x * n[0] + y * n[1]) + beam.phase
            # Two-color envelope modulation (beat) if provided
            if two_color is not None:
                dl = two_color.get('delta_lambda_fraction', 0.0)
                if dl != 0.0:
                    lam2 = wavelength * (1.0 + dl)
                    k2 = 2 * pi / lam2
                    # modulation factor ≈ cos(Δk * projection)
                    proj = x * n[0] + y * n[1]
                    mod = np.cos((k - k2) * proj)
                    env = env * (0.5 * (1.0 + mod))
            Etot += E0_i * env * np.cos(phase)
        E2_acc += Etot**2

    # Time-average of E^2; proportional to intensity (omitting constant factors)
    E2_avg = E2_acc / n_time
    return E2_avg


def _grad_mag(Z: np.ndarray, dx: float, dy: float) -> np.ndarray:
    gx, gy = np.gradient(Z, dx, dy)
    return np.sqrt(gx**2 + gy**2)


def _make_beams_from_config(config_name: str,
                            weights: Optional[List[float]] = None,
                            ring_N: Optional[int] = None,
                            angle_deg: Optional[float] = None,
                            phase_align: bool = True) -> List[BeamConfig]:
    """Build beam list for common geometries.
    Supported config_name:
      - 'single', 'two_opposed', 'triangular', 'square', 'pentagram', 'hexagon', 'standing_wave'
      - 'ring' with ring_N (N >= 3)
      - 'angled_crossing' with angle_deg (two beams crossing at ±angle/2)
    """
    def unit(v):
        v = np.asarray(v)
        n = np.linalg.norm(v)
        return v / n if n > 0 else v

    angs_deg: List[float]
    if config_name == 'ring':
        N = ring_N if (ring_N and ring_N >= 3) else 6
        angs_deg = list(np.linspace(0.0, 360.0, N, endpoint=False))
    elif config_name == 'angled_crossing':
        a = float(angle_deg) if angle_deg is not None else 20.0
        # split symmetrical around x-axis
        angs_deg = [a/2.0, 360.0 - a/2.0]
    else:
        base = {
            'single': [0.0],
            'two_opposed': [0.0, 180.0],
            'triangular': [0.0, 120.0, 240.0],
            'square': [0.0, 90.0, 180.0, 270.0],
            'pentagram': [0.0, 72.0, 144.0, 216.0, 288.0],
            'hexagon': [0.0, 60.0, 120.0, 180.0, 240.0, 300.0],
            'standing_wave': [0.0, 180.0],
        }
        if config_name not in base:
            raise ValueError(f"Unknown configuration: {config_name}")
        angs_deg = base[config_name]

    N = len(angs_deg)
    if weights is None:
        weights = [1.0 / N] * N
    if len(weights) != N:
        raise ValueError("weights length must match number of beams")

    beams: List[BeamConfig] = []
    for ang_deg, w in zip(angs_deg, weights):
        ang = np.deg2rad(ang_deg)
        n = unit(np.array([np.cos(ang), np.sin(ang)]))
        phase = 0.0 if phase_align else 2 * pi * np.random.rand()
        beams.append(BeamConfig(direction=n, phase=phase, weight=max(float(w), 0.0)))
    # normalize weights in case of rounding
    total_w = sum(b.weight for b in beams)
    if total_w > 0:
        for b in beams:
            b.weight /= total_w
    return beams


def simulate_gradient_enhancement(config_name: str,
                                  wavelength: float = 800e-9,
                                  w0: float = 5e-6,
                                  I_total: float = 1.0,  # arbitrary units (ratios only)
                                  grid_half_width: float = 12e-6,
                                  n_grid: int = 201,
                                  n_time: int = 16,
                                  radius_for_max: float = 2.5e-6,
                                  coarse_grain_length: float = None,
                                  phase_align: bool = True,
                                  # Optional κ surrogate mapping controls
                                  tau_response: float = None,   # s, e.g., 10e-15
                                  c_s_value: float = None,      # m/s; if provided, compute κ surrogate
                                  # Optional custom geometry controls
                                  weights: Optional[List[float]] = None,
                                  ring_N: Optional[int] = None,
                                  angle_deg: Optional[float] = None
                                  ) -> Dict[str, float]:
    """
    Compute gradient enhancement for a multi-beam configuration relative to
    a single-beam baseline, holding total peak power fixed.

    Returns a dict with:
    - enhancement: ratio of max |∇I| within radius around focus vs. single-beam
    - grad_max_multi, grad_max_single: absolute values (arbitrary units)
    - config_name
    - meta parameters for reproducibility
    """
    # Geometry handled by _make_beams_from_config()

    # Build grid
    xs = np.linspace(-grid_half_width, grid_half_width, n_grid)
    ys = np.linspace(-grid_half_width, grid_half_width, n_grid)
    dx = xs[1] - xs[0]
    dy = ys[1] - ys[0]
    X, Y = np.meshgrid(xs, ys, indexing='xy')

    # Coarse-graining kernel (to suppress optical-scale fringes):
    # Default to w0/2 if not provided
    if coarse_grain_length is None:
        coarse_grain_length = w0 / 2.0
    sigma_pix = max(coarse_grain_length / max(dx, dy), 1.0)

    # Baseline: single beam (same total power)
    beams_single = _make_beams_from_config('single', phase_align=phase_align)
    I_single = _time_average_intensity(X, Y, beams_single, wavelength, w0, I_total, n_time,
                                       w0_lab=None, two_color=None)
    I_single = ndi.gaussian_filter(I_single, sigma=sigma_pix)
    G_single = _grad_mag(I_single, dx, dy)

    # Optionally forward lab-fixed waists and two-color beat parameters via weights dict sentinel
    w0_lab = None
    two_color = None
    weights_list = weights
    if isinstance(weights, dict):
        w0_lab = weights.get('w0_lab')
        two_color = weights.get('two_color')
        weights_list = None

    # Multi-beam: requested config
    if config_name in ('ring', 'angled_crossing'):
        beams_multi = _make_beams_from_config(config_name, weights=weights_list, ring_N=ring_N,
                                              angle_deg=angle_deg, phase_align=phase_align)
    else:
        beams_multi = _make_beams_from_config(config_name, weights=weights_list, phase_align=phase_align)
    I_multi = _time_average_intensity(X, Y, beams_multi, wavelength, w0, I_total, n_time,
                                       w0_lab=w0_lab, two_color=two_color)
    I_multi = ndi.gaussian_filter(I_multi, sigma=sigma_pix)
    G_multi = _grad_mag(I_multi, dx, dy)

    # Measure max gradient magnitude within a small radius from focus
    R = np.sqrt(X**2 + Y**2)
    mask = R <= radius_for_max
    grad_max_single = float(np.max(G_single[mask])) if np.any(mask) else float(np.max(G_single))
    grad_max_multi = float(np.max(G_multi[mask])) if np.any(mask) else float(np.max(G_multi))

    enhancement = grad_max_multi / grad_max_single if grad_max_single > 0 else np.nan

    # Optional κ surrogate mapping via ponderomotive-like response
    # U_p ~ E^2/ω^2; a ~ -∇U_p/m_e; v ~ a * tau_response; κ ~ 0.5|∂(|v|-c_s)/∂x|
    kappa_surr_single = np.nan
    kappa_surr_multi = np.nan
    kappa_surr_enh = np.nan
    if (tau_response is not None) and (c_s_value is not None):
        omega = 2 * pi * c / wavelength
        U_single = I_single / (omega**2)  # proportional
        U_multi = I_multi / (omega**2)
        # acceleration fields (proportional), in x-direction derivative
        dUdx_single = np.gradient(U_single, dx, axis=1)
        dUdx_multi = np.gradient(U_multi, dx, axis=1)
        a_single = - dUdx_single
        a_multi = - dUdx_multi
        v_single = a_single * tau_response
        v_multi = a_multi * tau_response
        # compute κ surrogate field (center row) and take max within radius
        row = v_single.shape[0] // 2
        vs_row = v_single[row, :]
        vm_row = v_multi[row, :]
        # c_s constant across x for surrogate
        cs = c_s_value
        # κ_surr(x) = 0.5 |d/dx(|v|-c_s)| = 0.5 |d|v|/dx| (since c_s const)
        dvs_dx = np.gradient(np.abs(vs_row), dx)
        dvm_dx = np.gradient(np.abs(vm_row), dx)
        kappa_s_row = 0.5 * np.abs(dvs_dx)
        kappa_m_row = 0.5 * np.abs(dvm_dx)
        # max within radius
        col_center = v_single.shape[1] // 2
        # map radius in meters to columns
        cols_radius = int(max(radius_for_max / dx, 1))
        sl = slice(max(col_center - cols_radius, 0), min(col_center + cols_radius + 1, v_single.shape[1]))
        kappa_surr_single = float(np.max(kappa_s_row[sl]))
        kappa_surr_multi = float(np.max(kappa_m_row[sl]))
        if kappa_surr_single > 0:
            kappa_surr_enh = kappa_surr_multi / kappa_surr_single

    return {
        'config_name': config_name,
        'enhancement': enhancement,
        'grad_max_multi': grad_max_multi,
        'grad_max_single': grad_max_single,
        'kappa_surrogate_single': kappa_surr_single,
        'kappa_surrogate_multi': kappa_surr_multi,
        'kappa_surrogate_enhancement': kappa_surr_enh,
        'params': {
            'wavelength': wavelength,
            'w0': w0,
            'I_total': I_total,
            'grid_half_width': grid_half_width,
            'n_grid': n_grid,
            'n_time': n_time,
            'radius_for_max': radius_for_max,
            'coarse_grain_length': coarse_grain_length,
            'phase_align': phase_align,
            'tau_response': tau_response,
            'c_s_value': c_s_value,
        }
    }


def compare_configurations(configs: List[str] = None,
                           **kwargs) -> Dict[str, Dict[str, float]]:
    """Compute enhancement metrics for multiple configurations."""
    if configs is None:
        configs = ['single', 'two_opposed', 'triangular', 'square', 'pentagram', 'hexagon', 'standing_wave']
    results: Dict[str, Dict[str, float]] = {}
    for name in configs:
        results[name] = simulate_gradient_enhancement(name, **kwargs)
    return results
