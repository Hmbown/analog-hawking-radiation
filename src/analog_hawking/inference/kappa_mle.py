"""Maximum-likelihood estimator for κ using Bayesian optimisation (scikit-optimize)."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

try:
    from skopt import gp_minimize  # type: ignore
    from skopt.space import Real  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    gp_minimize = None
    Real = None

try:  # pragma: no cover - import convenience
    from hawking_detection_experiment import calculate_hawking_spectrum  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - defer error until use
    calculate_hawking_spectrum = None  # type: ignore


ModelFn = Callable[[float], np.ndarray]
PriorFn = Callable[[float], float]


@dataclass
class KappaInferenceResult:
    kappa_hat: float
    kappa_err: float
    credible_interval: Tuple[float, float]
    trace: List[Tuple[float, float]]
    posterior_grid: np.ndarray
    posterior_density: np.ndarray
    diagnostics: Dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, object]:
        return {
            "kappa_hat": float(self.kappa_hat),
            "kappa_err": float(self.kappa_err),
            "credible_interval": tuple(float(x) for x in self.credible_interval),
            "trace": [(float(x), float(y)) for x, y in self.trace],
            "posterior_grid": self.posterior_grid,
            "posterior_density": self.posterior_density,
            "diagnostics": dict(self.diagnostics),
        }


def make_graybody_model(
    frequencies: Sequence[float],
    *,
    graybody_profile: Optional[Mapping[str, np.ndarray]] = None,
    graybody_method: str = "dimensionless",
    alpha_gray: float = 1.0,
    emitting_area_m2: float = 1e-6,
    solid_angle_sr: float = 5e-2,
    coupling_efficiency: float = 0.1,
) -> ModelFn:
    """Create a PSD model callable that maps κ to a Hawking spectrum.

    Parameters mirror :func:`calculate_hawking_spectrum`. The returned callable
    interpolates the generated spectrum onto the supplied frequency grid so the
    optimiser can compare it directly against measured PSDs.
    """
    if calculate_hawking_spectrum is None:
        raise ImportError(
            "calculate_hawking_spectrum could not be imported. Ensure the project root "
            "is on PYTHONPATH (run scripts via `python -m scripts.<name>` or set PYTHONPATH)."
        )
    freqs = np.asarray(frequencies, dtype=float)

    def _model(kappa_val: float) -> np.ndarray:
        spec = calculate_hawking_spectrum(
            float(kappa_val),
            graybody_profile=graybody_profile,
            graybody_method=graybody_method,
            alpha_gray=alpha_gray,
            emitting_area_m2=emitting_area_m2,
            solid_angle_sr=solid_angle_sr,
            coupling_efficiency=coupling_efficiency,
        )
        if not spec.get("success"):
            raise RuntimeError(spec.get("error", "graybody model failure"))
        f_model = np.asarray(spec["frequencies"], dtype=float)
        psd_model = np.asarray(spec["power_spectrum"], dtype=float)
        if f_model.size == 0 or psd_model.size == 0:
            return np.zeros_like(freqs)
        return np.interp(
            freqs,
            f_model,
            psd_model,
            left=float(psd_model[0]),
            right=float(psd_model[-1]),
        )

    return _model


def infer_kappa(
    frequencies: Sequence[float],
    power_spectrum: Sequence[float],
    model: ModelFn,
    *,
    bounds: Tuple[float, float] = (1e4, 1e12),
    prior: Optional[PriorFn] = None,
    noise_floor: float = 1e-24,
    n_calls: int = 40,
    random_state: Optional[int] = None,
    posterior_points: int = 256,
) -> KappaInferenceResult:
    """Infer κ from a PSD by minimising a weighted residual using gp_minimize.

    Args:
        frequencies: Observed frequency grid (Hz).
        power_spectrum: Observed PSD (W/Hz).
        model: Callable returning model PSD for a given κ.
        bounds: Search bounds for κ (Hz).
        prior: Optional penalty function added to the objective (e.g., λ·(κ-κ₀)²).
        noise_floor: Added to the PSD when forming weights to avoid singularities.
        n_calls: Number of skopt evaluations (trade off runtime vs. fidelity).
        random_state: Deterministic seed for reproducibility.
        posterior_points: Number of κ samples used for posterior estimation.

    Returns:
        :class:`KappaInferenceResult` with κ̂, σ_κ, trace, and posterior samples.
    """
    if gp_minimize is None or Real is None:
        raise RuntimeError(
            "scikit-optimize is required for κ inference. Install via 'pip install scikit-optimize'."
        )

    freqs = np.asarray(frequencies, dtype=float)
    psd = np.asarray(power_spectrum, dtype=float)
    mask = np.isfinite(freqs) & np.isfinite(psd)
    if not np.any(mask):
        raise ValueError("power spectrum contains no finite samples")
    freqs = freqs[mask]
    psd = psd[mask]
    weights = 1.0 / np.maximum(psd + noise_floor, noise_floor)

    evaluations: List[Tuple[float, float]] = []

    def _objective(x: Sequence[float]) -> float:
        kappa_val = float(x[0])
        model_psd = np.asarray(model(kappa_val), dtype=float)
        if model_psd.shape != psd.shape:
            model_psd = np.interp(freqs, freqs, model_psd, left=model_psd[0], right=model_psd[-1])
        residual = (model_psd - psd) * weights
        mse = float(np.mean(residual**2))
        if prior is not None:
            mse += float(prior(kappa_val))
        evaluations.append((kappa_val, mse))
        return mse

    space = [Real(bounds[0], bounds[1], prior="log-uniform")]
    result = gp_minimize(
        _objective,
        space,
        n_calls=int(max(n_calls, 10)),
        random_state=random_state,
        n_initial_points=min(10, int(max(n_calls // 3, 5))),
    )

    kappa_hat = float(result.x[0])
    f_opt = float(result.fun)

    # Approximate curvature for σ estimate
    span = max(1e-6, 0.05 * kappa_hat)
    left = max(bounds[0], kappa_hat - span)
    right = min(bounds[1], kappa_hat + span)
    if right == left:
        right = min(bounds[1], left + span)
    f_left = _objective([left])
    f_right = _objective([right])
    denom = (right - left) ** 2 / 4.0
    curvature = (f_left - 2.0 * f_opt + f_right) / max(denom, 1e-24)
    if curvature > 0:
        sigma = math.sqrt(1.0 / curvature)
    else:
        sigma = float("inf")

    grid = np.geomspace(bounds[0], bounds[1], posterior_points)
    obj_vals = np.asarray([_objective([float(val)]) for val in grid], dtype=float)
    obj_vals -= float(np.min(obj_vals))
    eps = max(float(np.std(obj_vals)), 1e-24)
    log_like = -0.5 * obj_vals / eps
    density = np.exp(log_like - log_like.max())
    density /= np.trapezoid(density, grid)
    cdf = np.cumsum(density)
    cdf /= cdf[-1]
    # Report central 95% credible interval
    lower_idx = int(np.searchsorted(cdf, 0.025))
    upper_idx = int(np.searchsorted(cdf, 0.975))
    kappa_low = float(grid[max(0, lower_idx)])
    kappa_high = float(grid[min(len(grid) - 1, upper_idx)])
    credible = (kappa_low, kappa_high)
    sigma_sym = 0.5 * ((kappa_hat - kappa_low) + (kappa_high - kappa_hat))

    diagnostics = {
        "objective_opt": f_opt,
        "curvature": curvature,
        "kappa_left": left,
        "kappa_right": right,
        "objective_left": f_left,
        "objective_right": f_right,
    }

    return KappaInferenceResult(
        kappa_hat=kappa_hat,
        kappa_err=sigma_sym,
        credible_interval=credible,
        trace=evaluations,
        posterior_grid=grid,
        posterior_density=density,
        diagnostics=diagnostics,
    )


__all__ = ["KappaInferenceResult", "infer_kappa", "make_graybody_model"]
