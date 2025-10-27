# GPU Acceleration Guide (RTX 3080 Playbook)

This document explains how to push the repository's heaviest workloads on a single-GPU workstation (e.g., RTX 3080) using CuPy acceleration. The goal is to run large parameter sweeps, universality experiments, and kappa-inference benchmarks quickly while avoiding memory pitfalls.

## 1. Environment Preparation

1. **Install CuPy** with the wheel matching your CUDA toolkit (for CUDA 12.x on Windows/Linux):
   ```bash
   pip install cupy-cuda12x
   ```
   Verify installation:
   ```bash
   python -c "import cupy; cupy.cuda.runtime.deviceSynchronize(); print(cupy.cuda.runtime.getDeviceProperties(0)['name'])"
   ```

2. **Force the code to use CuPy** by default:
   ```powershell
   setx ANALOG_HAWKING_USE_CUPY 1
   ```
   (Use `export` on Unix shells. Omit if you only want to enable GPU selectively.)

3. **Optional:** Pin the repository inside the Docker image and install CuPy at build time:
   ```bash
   docker build -t analog-hawking:gpu --build-arg INSTALL_EXTRAS=true .
   docker run --gpus all -it analog-hawking:gpu bash
   pip install cupy-cuda12x
   ```

## 2. Quick Check

Run the CuPy smoke test bundled with the array helper:
```bash
python - <<'PY'
from analog_hawking.utils.array_module import using_cupy
print("Using CuPy:", using_cupy())
PY
```
Expect `Using CuPy: True` when the GPU is active.

## 3. Launch the GPU Campaign

Use the orchestration script to drive several heavy workloads. The defaults are tuned for an RTX 3080 (10 GB VRAM).

```bash
ANALOG_HAWKING_USE_CUPY=1 python scripts/run_gpu_campaign.py \
  --tasks gradient universality detection \
  --gradient-samples 1800 \
  --universality-families 64 \
  --detection-trials 128 \
  --results-dir results/gpu_rtx3080
```

### What Each Task Does

- `gradient`: Runs the gradient catastrophe sweep with 1,800 samples, generating plots and JSON summaries under `results/gpu_rtx3080/gradient_limits_gpu/`.
- `universality`: Recomputes kappa-normalized spectrum collapse across 64 flow families (useful to stress the graybody solver).
- `detection`: Benchmarks the kappa-inference pipeline with 128 noisy PSD realisations.

Progress is streamed to the console; if any task fails, the campaign stops so you can inspect logs immediately.

## 4. Memory Tuning Tips

- **Adjust sample counts** if you approach the RTX 3080 memory ceiling:
  - `--gradient-samples 1200` keeps usage modest (~6 GiB).
  - `--universality-families 48` consumes ~4 GiB.
  - `--detection-trials 96` stays within ~5 GiB.
- **Set precision** by overriding environment variables in scripts that support it (e.g., use float32 in custom kernels).
- **Monitor with `nvidia-smi`** to ensure utilisation hovers near 80-100% while kernels run.

## 5. Advanced Runs

### Split Sweeps into Chunks

For multi-hour explorations, run multiple campaigns with different seeds:
```bash
for SEED in 11 23 37 53; do
  ANALOG_HAWKING_USE_CUPY=1 python scripts/run_gpu_campaign.py \
    --tasks gradient \
    --gradient-samples 2000 \
    --seed $SEED \
    --results-dir results/gpu_rtx3080_seed$SEED
done
```
Aggregate results afterwards by merging the JSON summaries.

### Universality Stress Test

Increase the grid resolution inside `scripts/experiment_universality_collapse.py` by passing `--fft-points 4096` (available flag) to better resolve spectra:
```bash
ANALOG_HAWKING_USE_CUPY=1 python scripts/experiment_universality_collapse.py \
  --out results/universality_highres \
  --n 96 \
  --fft-points 4096 \
  --alpha 0.9 \
  --seed 21
```

### WarpX Integration (after datasets available)

Once real openPMD datasets are on disk, use the planned CLI (see `docs/upgrade_plan/external_actions.md`) to ingest slices and run cross-validation with the GPU-accelerated horizon finder.

## 6. Collecting Results

After each campaign, summaries are emitted in JSON:
- `gradient_catastrophe_sweep.json`
- `universality_summary.json`
- `kappa_inference_summary.json`

Use the provided helper to dump highlights quickly:
```bash
python scripts/run_gpu_campaign.py --tasks gradient --results-dir results/gpu_rtx3080 --force-cpu
```
(The `--force-cpu` flag skips recomputation, only printing summaries if they exist.)

## 7. Troubleshooting

- **CuPy import errors:** Ensure the wheel matches your CUDA version (`cupy-cuda11x`, `cupy-cuda12x`, etc.).
- **Out-of-memory:** Reduce sample counts or families, or run tasks sequentially rather than simultaneously.
- **Driver issues:** Update NVIDIA drivers and CUDA toolkit to versions supported by the CuPy wheel.
- **Fallback to CPU:** Set `ANALOG_HAWKING_FORCE_CPU=1` to confirm behaviour matches when GPU is disabled.

## 8. Next Steps

- Combine GPU runs with analytical benchmarks (see `docs/upgrade_plan/implementation_roadmap.md`).
- Capture performance metrics (wall time, memory usage) for publication artefacts.
- Share configuration scripts with collaborators running higher-end GPUs or clusters.

This guide should enable you to get the most out of an RTX 3080 while staying within reproducible, documented workflows.
