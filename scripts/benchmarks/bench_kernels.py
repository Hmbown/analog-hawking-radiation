#!/usr/bin/env python3
"""Benchmark harness for core kernels.

Measures horizon finder runtime vs grid size and prints JSON results.
"""

from __future__ import annotations

import json
import sys
import time

import numpy as np

if __name__ == "__main__" and "src" not in sys.path:
    # Allow running from repo root without install
    sys.path.insert(0, "src")

from analog_hawking.physics_engine.horizon import find_horizons_with_uncertainty, sound_speed


def run_suite(sizes=(500, 1000, 2000, 4000, 8000)):
    results = []
    for nx in sizes:
        x = np.linspace(0, 100e-6, nx)
        x0 = 50e-6
        L = 10e-6
        v0 = 0.1 * 3e8
        j = int(np.argmin(np.abs(x - x0)))
        x[j] = x0
        v = v0 * np.tanh((x - x0) / L)
        Te = np.full_like(x, 8e5)
        cs = sound_speed(Te)
        t0 = time.perf_counter()
        _ = find_horizons_with_uncertainty(x, v, cs)
        ms = (time.perf_counter() - t0) * 1e3
        results.append({"kernel": "horizon_finder", "nx": int(nx), "ms": ms})
    return {"results": results}


if __name__ == "__main__":
    data = run_suite()
    print(json.dumps(data, indent=2))

