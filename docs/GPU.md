# GPU Acceleration

This project supports optional GPU acceleration via CuPy. If CuPy is installed and usable, the
array backend switches automatically; otherwise it stays on NumPy.

- Enable (default): `ANALOG_HAWKING_USE_CUPY=1`
- Force CPU: `ANALOG_HAWKING_FORCE_CPU=1`

On macOS (Apple Silicon), CUDA/CuPy is generally unavailable. The CPU path remains the
reference implementation and is validated by CI.

On Linux with NVIDIA GPUs (CUDA 12+), install the GPU extra:

```bash
pip install -e .[gpu]   # installs cupy-cuda12x
```

Or install a specific CuPy build:

```bash
pip install cupy-cuda12x   # CUDA 12 wheels
# Alternatives: cupy-cuda11x, cupy-cuda12x==<pin>, etc.
```

Verify backend:

```bash
ahr gpu-info
```

Notes
- GPU kernels are used where implemented (e.g., graybody integration in 1D); other paths fall back to CPU.
- CI skips GPU-only tests if CuPy is not present.
- If CuPy is installed but a driver is unavailable, the code auto-falls back to NumPy.

See `src/analog_hawking/utils/array_module.py` for backend selection logic.
