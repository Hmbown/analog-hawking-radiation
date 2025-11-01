# Command Line Interface (CLI)

The `ahr` command provides quick access to common workflows.

- `ahr quickstart` — generate a synthetic example, find horizons, and write a results manifest.
- `ahr validate` — run the comprehensive physics validation suite.
- `ahr bench` — run a small benchmark of the horizon finder.

## Installation

After installing the package (editable or from source):

```bash
pip install -e .[dev]
```

## Quickstart

```bash
ahr quickstart --nx 1200 --x-min 0 --x-max 1e-4 --v0 3e7 --x0 5e-5 --L 1e-5 --Te 1e6 --out results/quickstart
```

Outputs:
- `results/quickstart/horizons.json`: horizon positions and κ estimates
- `results/quickstart/manifest.json`: provenance and configuration
- `results/quickstart/quickstart_profile.png`: plot (unless `ANALOG_HAWKING_NO_PLOTS=1`)

## Validation

```bash
ahr validate --report results/validation_summary.json
```

Runs the physics validation framework and exits non‑zero if critical tests fail.

## Benchmark

```bash
ahr bench
```

Times the horizon finder on a representative synthetic profile.

## Environment Flags

- `ANALOG_HAWKING_NO_PLOTS=1` — disable figure creation (CI‑friendly)
- `ANALOG_HAWKING_USE_CUPY=1` — enable CuPy if installed
- `ANALOG_HAWKING_FORCE_CPU=1` — force CPU even if GPU is available
