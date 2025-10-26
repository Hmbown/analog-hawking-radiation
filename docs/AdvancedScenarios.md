# Advanced Scenarios

This guide collects the extended command sequences for workflows that go beyond the basic demo run. Each section provides the shell commands you can paste into a terminal along with a short description of what each workflow accomplishes.

## Universality Experiments

Run the collapse test across analytic profiles, with optional inclusion of particle-in-cell (PIC) data.

```bash
# Analytic-only spectrum collapse
python scripts/experiment_universality_collapse.py \
  --out results/experiments/universality --n 32 --alpha 0.8 --seed 7 --include-controls

# Include PIC-derived profiles in the universality analysis
python scripts/experiment_universality_collapse.py \
  --out results/experiments/universality --n 16 --alpha 0.8 \
  --pic-profiles results/*.npz --include-controls
```

## PIC/OpenPMD Integration

Convert particle-in-cell output into the profile format used by the Hawking radiation pipeline and feed it through the full analysis.

```bash
# Convert an HDF5 slice to the profile format
python scripts/openpmd_slice_to_profile.py --in data/slice.h5 \
  --x-dataset /x --vel-dataset /vel --Te-dataset /Te \
  --out results/warpx_profile.npz

# Run the full pipeline on the converted PIC profile
python scripts/run_pic_pipeline.py --profile results/warpx_profile.npz \
  --kappa-method acoustic_exact --graybody acoustic_wkb

# Optionally reuse the PIC profiles when running universality collapse studies
python scripts/experiment_universality_collapse.py \
  --out results/experiments/universality --pic-profiles results/*.npz
```

### Direct ingestion via the Python API

You can now skip the intermediate NPZ step and load openPMD diagnostics directly from Python:

```python
from analog_hawking.pipelines import from_openpmd

profile = from_openpmd(
    "diags/openpmd/openpmd_%T.h5",
    t="latest",
    kappa_method="acoustic_exact",
)

grid = profile.grid            # positions in meters
velocity = profile.velocity    # bulk flow velocity v(x)
sound_speed = profile.sound_speed
horizons = profile.horizon     # HorizonResult with κ, κ_err, etc.
```

The adapter accepts optional ``velocity_source``/``sound_speed_source`` mappings if your openPMD
records use non-standard names. See the docstring of :func:`analog_hawking.pipelines.from_openpmd`
for the full list of tunable parameters.

## Exploratory Hybrid Coupling

Toggle the speculative hybrid branch that couples fluid horizons to accelerating plasma mirrors.

```bash
python scripts/run_full_pipeline.py --demo --hybrid \
  --hybrid-model anabhel --mirror-D 1e-5 --mirror-eta 1.0
```

For additional context and theoretical background on these workflows, review `docs/Experiments.md` and `docs/Methods.md`.
