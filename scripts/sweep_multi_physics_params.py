#!/usr/bin/env python3
"""Parameter sweep script for multi-physics simulations targeting Phase 3 success criteria."""

from __future__ import annotations

import itertools
import json

# Ensure package imports
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# compute_universality_r2 handled in nonlinear_plasma.py
from analog_hawking.physics_engine.horizon import find_horizons_with_uncertainty
from analog_hawking.physics_engine.plasma_models.nonlinear_plasma import NonlinearPlasmaSolver
from analog_hawking.physics_engine.plasma_models.warpx_backend import WarpXBackend


def run_sweep(param_ranges: dict[str, list[float]], n_steps: int = 10, mock: bool = True) -> dict:
    """Run parameter sweep over multi-physics configs."""
    results = []
    total_configs = np.prod([len(vals) for vals in param_ranges.values()])
    
    with tqdm(total=total_configs, desc="Sweeping params") as pbar:
        for config_params in itertools.product(*(param_ranges[k] for k in param_ranges)):
            params = dict(zip(param_ranges.keys(), config_params))
            # Configure backend
            run_config = {
                "mock": mock,
                "mhd_enabled": True,
                "qft_3d": True,
                "nonlinear_config": {
                    "nonlinear_strength": params["nonlinear_strength"],
                    "qft_modes": int(params["qft_modes"]),
                    "kappa_enhancement": params["kappa_enhancement"],
                },
                "default_density": params["plasma_density"],
                "default_temperature": 1e4,
                "species": [{"name": "electrons", "charge": -1.6e-19, "mass": 9.1e-31}],
                "grid": np.linspace(0, 1e-4, 100),
                "field_getters": {"electric_field": {"type": "mock_data", "data": np.zeros(100)}},
                "moment_getters": {"electrons": {"density": {"type": "mock_data", "data": np.full(100, params["plasma_density"])}}},
            }
            
            backend = WarpXBackend()
            backend.configure(run_config)
            
            # Run simulation steps
            state = backend.step(1e-15)  # dt ~1 fs
            for _ in range(n_steps - 1):
                state = backend.step(1e-15)
            
            # Compute metrics using nonlinear solver
            solver = NonlinearPlasmaSolver(run_config["nonlinear_config"])
            enhanced_obs = solver.solve(state.observables)
            
            # Universality R² (PSD collapse to blackbody)
            r2 = enhanced_obs.get("universality_r2", 0.0)
            
            # Kappa stability <3%
            horizons = find_horizons_with_uncertainty(
                state.grid, state.velocity, state.sound_speed
            )
            kappa_mean = np.mean(horizons.kappa) if horizons.kappa.size > 0 else 1.0
            kappa_std = np.std(horizons.kappa) if horizons.kappa.size > 1 else 0.0
            kappa_stability = (kappa_std / kappa_mean) if kappa_mean > 0 else 0.0
            
            # t_5σ <1s (simplified from SNR)
            t_5sigma = enhanced_obs.get("t_5sigma", 1.0)
            
            # Check criteria
            meets_criteria = (r2 > 0.98) and (kappa_stability < 0.03) and (t_5sigma < 1.0)
            
            result = {
                **params,
                "r2": r2,
                "kappa_stability": kappa_stability,
                "t_5sigma": t_5sigma,
                "meets_criteria": meets_criteria,
                "kappa_enhanced": enhanced_obs.get("enhanced_kappa", kappa_mean),
                "t_hawking": enhanced_obs.get("t_hawking", 1e-3),
            }
            results.append(result)
            pbar.update(1)
            
            backend.shutdown()
    
    # Summary
    success_rate = np.mean([r["meets_criteria"] for r in results])
    print(f"Success rate: {success_rate:.2%} ({success_rate * len(results):.0f}/{len(results)} configs meet criteria)")
    
    return {"results": results, "success_rate": float(success_rate), "param_ranges": param_ranges}


def main() -> int:
    """Main entry point for sweeps."""
    # CPU-optimized param ranges (small grids, mock mode)
    param_ranges = {
        "plasma_density": np.logspace(17, 19, 3) * 1e6,  # m^-3
        "nonlinear_strength": [0.05, 0.1, 0.2],
        "qft_modes": [5, 10, 15],
        "kappa_enhancement": [10.0, 50.0, 100.0],
    }
    
    output_dir = Path("results/phase3_sweeps")
    output_dir.mkdir(exist_ok=True)
    
    sweep_results = run_sweep(param_ranges, n_steps=5, mock=True)  # Short run for desktop
    
    # Save results
    with open(output_dir / "sweep_results.json", "w") as f:
        json.dump(sweep_results, f, indent=2, default=str)
    
    print(f"Results saved to {output_dir / 'sweep_results.json'}")
    print(f"Achieved R²>0.98 in {np.mean([r['r2'] > 0.98 for r in sweep_results['results']]):.2%}")
    print(f"κ stability <3% in {np.mean([r['kappa_stability'] < 0.03 for r in sweep_results['results']]):.2%}")
    print(f"t_{{5σ}} <1s in {np.mean([r['t_5sigma'] < 1.0 for r in sweep_results['results']]):.2%}")
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())