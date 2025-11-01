# Benchmarks

Use the built-in CLI or the benchmark harness to measure performance.

- Quick micro-benchmark:

```bash
ahr bench --json
```

- Full suite across grid sizes:

```bash
make bench-suite
```

This writes `results/bench_horizon.json` with timing per `nx`.

Notes
- CPU baseline is NumPy; optional GPU acceleration uses CuPy when available.
- For CUDA runs in Docker, see docs/Docker.md.

