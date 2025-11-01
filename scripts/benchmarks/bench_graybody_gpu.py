#!/usr/bin/env python3
from __future__ import annotations

"""
Quick CPU vs GPU benchmark for acoustic-WKB graybody transmission.

Saves results to results/benchmarks/graybody_gpu_bench.json.
"""

import json
import os
import time
from pathlib import Path

import numpy as np


def _run_once(use_gpu: bool) -> dict:
    # Toggle backend via environment and reload modules
    if use_gpu:
        os.environ.pop("ANALOG_HAWKING_FORCE_CPU", None)
        os.environ["ANALOG_HAWKING_USE_CUPY"] = "1"
    else:
        os.environ["ANALOG_HAWKING_FORCE_CPU"] = "1"

    # Lazy import after env change
    from importlib import reload
    from analog_hawking.utils import array_module as am
    reload(am)
    from analog_hawking.physics_engine.optimization import graybody_1d
    reload(graybody_1d)

    # Profile: simple crossing with 4096 points
    N = 4096
    x = np.linspace(-1.0, 1.0, N)
    a = 1.0
    c0 = 0.2
    v = a * x
    c = np.full_like(x, c0)
    kappa = abs(a)
    freqs = np.logspace(-2, 2, 512)

    t0 = time.perf_counter()
    gb = graybody_1d.compute_graybody(x, v, c, freqs, method="acoustic_wkb", kappa=kappa, alpha=0.3)
    t1 = time.perf_counter()
    return {
        "backend": "GPU" if use_gpu else "CPU",
        "duration_s": t1 - t0,
        "n_points": N,
        "n_freqs": len(freqs),
        "transmission_checksum": float(np.sum(gb.transmission)),
    }


def main() -> int:
    results = []
    try:
        import cupy  # noqa: F401
        has_cupy = True
    except Exception:
        has_cupy = False

    # Always run CPU; run GPU if available
    results.append(_run_once(use_gpu=False))
    if has_cupy:
        results.append(_run_once(use_gpu=True))

    outdir = Path("results/benchmarks")
    outdir.mkdir(parents=True, exist_ok=True)
    with (outdir / "graybody_gpu_bench.json").open("w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)
    print("Saved results/benchmarks/graybody_gpu_bench.json")
    for r in results:
        print(f"{r['backend']}: {r['duration_s']*1e3:.2f} ms, checksum={r['transmission_checksum']:.3e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

