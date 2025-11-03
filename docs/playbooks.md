# Collaboration Playbooks

Short recipes for the workflows collaborators run most often. Commands assume
you have activated the project virtual environment.

## Validated Workflows ✅

### Run the core test suite
```bash
pytest -q
```

### Regenerate analysis figures and tables
```bash
make comprehensive   # Produces figures used in docs
make results-pack    # Bundles artifacts for sharing
```

### Produce the gradient catastrophe sweep
```bash
python scripts/sweep_gradient_catastrophe.py \
  --config configs/gradient_sweep_v0_3.yaml \
  --out results/gradient_catastrophe
```

## Experimental Workflows ⚠️

### Plasma mirror demo (hybrid coupling)
```bash
python scripts/run_full_pipeline.py --demo --kappa-method acoustic_exact \
  --mirror-model anabhel
```

### PIC ingestion (mock configuration)
```bash
python scripts/run_pic_pipeline.py --nd-npz results/grid_nd_profile.npz \
  --nd-scan-axis 0 --graybody acoustic_wkb --alpha-gray 1.0
```

## Environment Management

### Install verified dependencies
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-verified.txt
```

### Optional extras for demos
```bash
pip install -r requirements-optional.txt
```

---

Back: [Docs Index](index.md) · Next: [Gradient Catastrophe Analysis »](GradientCatastropheAnalysis.md)
