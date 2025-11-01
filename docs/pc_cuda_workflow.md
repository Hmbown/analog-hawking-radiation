# PC CUDA Workflow for RTX 3080

This guide provides a comprehensive, pragmatic plan to run the analog Hawking radiation analysis pipeline on your PC with an RTX 3080, including CUDA acceleration, GPU testing, and scientific outreach strategy.

## TL;DR: Quick Start

**Goal**: Produce convincing, shareable artifacts (JSON summaries, figures, reports) to demonstrate the analog Hawking pipeline on commodity hardware, then reach out to world experts for collaboration.

**Time investment**: 2-4 hours to get everything running, 1 hour per outreach email.

---

## 1) Repository Overview

The repo is a practitioner-focused simulator that links:

- **Analytical fluid models** → horizon finding & κ (surface gravity) estimators
- **Graybody filtering** → transmission through the near‑horizon barrier
- **Detection modeling** → radio‑band SNR & 5σ timelines
- **PIC/OpenPMD post‑processing** → so you can feed in WarpX/other PIC outputs
- **Campaign/orchestration tools** → sweeps, monitoring, and reports

The public README shows end‑to‑end pipelines and orchestration, with JSON summaries of `kappa`, `T_H_K`, `t5sigma_s` and more, and documents the PIC/OpenPMD ingestion path.

> **Bottom line**: You can explore "are sonic horizons plausible in X profile?" and "would the Hawking‑like signal be remotely measurable under conservative assumptions?" on a single GPU box—no cluster required—then scale up with PIC data when you have it.

---

## 2) Make Your RTX 3080 Actually Work (Clean, Reproducible Setup)

### Recommended Configuration

**OS**: Linux (Ubuntu 22.04/24.04) or Windows via WSL2 Ubuntu
**CUDA/driver**: Install an NVIDIA driver that supports **CUDA 12.x** and use CuPy's CUDA‑12 wheels
**GPU**: RTX 3080 (compute capability 8.6 - Ampere architecture, fully supported)

### Installation Steps

#### 1. Clone & Base Install

```bash
git clone https://github.com/Hmbown/analog-hawking-radiation.git
cd analog-hawking-radiation
python -m venv .venv && source .venv/bin/activate     # or use uv/conda
pip install -e .
```

#### 2. GPU Extras

```bash
# CuPy prebuilt wheels for CUDA 12.x (works on Ampere like your 3080)
pip install cupy-cuda12x

# Optional: plotting / optimization extras used by some scripts
pip install scikit-optimize seaborn
```

CuPy's CUDA-12 wheels are the simplest path on consumer Ampere GPUs. For context, RTX 3080 is **compute capability 8.6 (Ampere)**, well within CuPy support.

#### 3. Sanity Check the GPU from Python

```bash
python - <<'PY'
import cupy as cp
x = cp.arange(10_000_000, dtype=cp.float32)
print("GPU OK:", float(x.sum()))
print(cp.show_config())
PY
```

Expected output should show your RTX 3080 and CUDA 12.x configuration.

#### 4. Run Tests

```bash
pytest -q
# CPU↔GPU parity test runs when CuPy is installed; skipped otherwise
pytest -q tests/test_gpu_parity_graybody.py
```

The README advertises a test matrix and integration tests; if your environment has CuPy, parity tests exercise accelerated codepaths.

---

## 3) What to Run TODAY on a Single RTX 3080

The goal is to produce convincing, shareable artifacts: JSON summaries, figures, and short reports you can drop into an email to a lab PI.

### A. Baseline Fluid → Spectrum → Detectability (Quick Win)

```bash
python scripts/run_full_pipeline.py \
  --demo --kappa-method acoustic_exact --graybody acoustic_wkb --save-figures
cat results/full_pipeline_summary.json
```

**What you get**: κ estimates, graybody transmission, and 5σ integration‑time forecasts
**Why it matters**: This is your baseline proof-of-concept run
**Time**: ~2-5 minutes

### B. Universality "Spectrum Collapse" Check (Physics Signal of Generality)

```bash
python scripts/sweep_multi_physics_params.py \
  --config configs/orchestration/pic_downramp.yml
```

**What you get**: κ‑normalized overlays that test whether disparate profiles align on a universal curve—one of the key arguments in the analogue‑gravity literature
**Why it matters**: Shows the physics is universal, not model-dependent
**Time**: ~10-15 minutes

### C. Gradient-Catastrophe / Breakdown Sweep (How Far Can κ Be Pushed?)

```bash
python scripts/sweep_gradient_catastrophe.py --n-samples 300 \
  --output results/gradient_limits_analysis --save-figures
```

**What you get**: Map pre‑relativistic breakdown envelopes and extract the *max* κ your model admits under constraints. The sweep uses the acoustic‑exact κ definition and enforces v, |dv/dx|, and intensity thresholds.
**Why it matters**: Theorists care about "how close to the limit are we?" when assessing plausibility
**Time**: ~15-30 minutes

### D. PIC ↔ OpenPMD Post-Processing (Without Running PIC Yourself)

If you have any PIC outputs in **openPMD** format, or want to test with sample data:

```bash
# Get example datasets (optional):
# git clone https://github.com/openPMD/openPMD-example-datasets
# tar -zxvf openPMD-example-datasets/example-2d.tar.gz

python scripts/run_pic_pipeline.py \
  --input-path ./openPMD-example-datasets/example-2d \
  --output-dir results/pic_run --plot
```

**What you get**: Convert/align/horizon‑find and compare spectra from real PIC data
**Why it matter**: Validates your pipeline against first-principles simulations
**Time**: ~5-10 minutes

### E. κ-Inference from PSDs (Bayesian "Invert the Experiment")

```bash
python scripts/infer_kappa_from_psd.py results/psd_*.npz \
  --graybody-method acoustic_wkb --calls 40
```

**What you get**: Posterior samples + corner plots to communicate uncertainty credibly
**Why it matters**: Senior researchers appreciate honest uncertainty quantification
**Time**: ~5-10 minutes

> **Pro tip for stability on a 3080**: Keep arrays in `float32` where available; CuPy will happily accelerate the numerics. If a run OOMs, downsize sweep counts (e.g., `--n-samples 150`) and disable optional figure generation.

---

## 4) What You Can (and Can't) Claim

### What You CAN Claim

- This toolkit explores **analog** Hawking scenarios (sonic horizons and their greybody‑filtered spectra), not astrophysical black holes
- If you generate convincing universality‑collapse plots plus pre‑breakdown κ‑maps and honest 5σ windows, you're contributing **useful comparative analysis** the community can react to
- The repo is designed for exactly this kind of pre‑experiment vetting

### What You CAN'T Claim

- Direct detection of astrophysical Hawking radiation (the signals are far too weak)
- Definitive proof without experimental validation
- Exclusivity (many groups work on this)

### Important Context

For background and context across platforms (BEC, fluids, optics), see the major reviews and exemplars: analogue gravity reviews; water‑tank and BEC evidence; optical analogues.

---

## 5) A Compact "Results Pack" to Email Experts

### Exportables to Include

1. **Core results**:
   - `results/full_pipeline_summary.json` + κ/graybody/detection figures
   - `results/universality/*` overlays from the collapse run
   - `results/gradient_limits_analysis/*` figures + findings markdown
   - (Optional) `results/pic_run/*` if you processed any openPMD set

2. **Summary document**: 1-page PDF with:
   - What you did
   - Key figures (2-3 plots)
   - JSON snippets showing quantitative results
   - What you think this means
   - What feedback you want

These are the right artifacts to solicit serious feedback.

---

## 6) Who to Reach Out To (and Why They'll Care)

### Laser/Plasma & AnaBHEL (Closest to Your Remit)

**Pisin Chen (NTU; AnaBHEL Lead)**
- *Why they'll care*: If anyone will engage with a plasma‑mirror / radiometer‑forecast workflow, it's this lab
- *Profile*: https://www.phys.ntu.edu.tw/enphysics/pisinchen.html
- *Key work*: AnaBHEL concept papers (open-access)

**Gérard Mourou (Nobel in CPA; Extreme-Light Programs)**
- *Why they'll care*: Not a day‑to‑day PI for your code, but his network (LOA, ELI ecosystem) is where "can the laser chain do this?" gets reality‑checked
- *Key work*: Nobel lecture offers authoritative context when framing laser feasibility
- *Reference*: NobelPrize.org lecture PDFs

### BEC Analogue Black Holes

**Jeff Steinhauer (Technion)**
- *Why they'll care*: Ran the most cited BEC analogue Hawking experiments; clear writing on spectral thermality/entanglement
- *Profile*: https://phsites.technion.ac.il/atomlab/
- *Key work*: Lab page summarizes status & interests

**Iacopo Carusotto (Trento/CNR‑INO)**
- *Why they'll care*: Theory/numerics for BEC analogues; recent work on spin‑sonic horizons shows ongoing interest in new signatures
- *Key work*: https://arxiv.org/abs/2408.17292 (spin-sonic horizons)

### Water-Waves / Fluids (Classical & Stimulated Signatures)

**Silke Weinfurtner (Nottingham Gravity Lab)**
- *Why they'll care*: Ran canonical water‑tank experiments and continues cross‑platform analogue research; excellent for universality arguments and diagnostics
- *Profile*: https://www.gravitylaboratory.com/people/weinfurtner-silke

### Optical Analogues

**Daniele Faccio (Glasgow, Extreme-Light Group)**
- *Why they'll care*: Pioneering work in optical analogue horizons and negative‑frequency radiation; bridge to ultrafast/filament optics
- *Profile*: https://www.physics.gla.ac.uk/xtremeLight/cv.html

### Foundations & Theory Across Platforms

**Ralf Schützhold (TU Dresden/Uni‑Duisburg‑Essen)**
- *Why they'll care*: Analogue Hawking theory across media (with Unruh), dispersion, and robustness
- *Profile*: https://www.uni-due.de/sfb1242/schuetzhold.php

**Stefano Liberati (SISSA/IFPU Director)**
- *Why they'll care*: Long‑standing analogue‑gravity theory; universality and phenomenology focus aligns with your collapse tests
- *Profile*: https://scholar.google.com/citations?hl=en&user=KB966c8AAAAJ

**Matt Visser (Victoria University of Wellington)**
- *Why they'll care*: Co‑author of the standard analogue‑gravity reviews; valuable on conceptual framing and what counts as robust evidence
- *Profile*: https://sms.wgtn.ac.nz/Groups/GravityGroup/ProfessorMattVisser

### Honorable Mentions (Historical Anchors)

**W. G. Unruh (Origin of the Acoustic Analogy)**
- *Why they matter*: Origin of the acoustic analogy; still active and thoughtful on feasibility across platforms
- *Note*: CC for context if appropriate
- *Profile*: https://scholar.google.com/citations?hl=en&user=udKlmAMAAAAJ

---

## 7) How to Reach Out (Citizen-Physicist / Vibe-Coder Edition)

### Email Template

**Subject**: *Open-source analog Hawking toolkit (fluid→graybody→radio) – early results & request for feedback*

**Body**:

```
Dear Prof. <Name>,

I'm an independent/citizen physicist building an open-source pipeline that
connects fluid profiles → horizon/κ diagnostics → graybody transmission →
radio-band detectability. It also ingests openPMD/PIC outputs (e.g., WarpX)
to test "universality" via κ-normalized spectra.

In a single-GPU setup (RTX 3080), I can reproduce baseline horizons, run
universality-collapse sweeps, and map pre-breakdown κ limits. I'd love
your candid feedback on (a) methodological gaps, (b) plausibility of the
assumptions, and (c) which diagnostics would be most convincing to your community.

Links and 1–2 representative figures are below; a 1‑page summary is attached.
If you see value, I'd be grateful for 15 minutes of guidance on what to test next.

Repo: https://github.com/Hmbown/analog-hawking-radiation
Selected outputs: <attach PNGs>  |  JSON summaries: <paste snips>

Thank you for considering,
<Your name>
```

### Tactics That Work Well

✓ **Lead with figures + JSON** (κ vs. position, graybody curve with band, 5σ window with error bars)
✓ **Be explicit about limitations** (1‑D graybody, numerical uncertainty definition, PIC read‑only post‑processing) so critics don't need to guess
✓ **Offer "apples‑to‑apples"**: If they share one profile (or a public dataset), you'll run your pipeline and return a comparable spectrum/detectability panel
✓ **Target the right venues**: APS‑DPP community (plasma), ELI/Extreme‑Light circles (lasers), analogue‑gravity workshops

### Who to Contact First (Priority Order)

1. **Silke Weinfurtner** - Water-tank experiments, great for feedback on universality
2. **Iacopo Carusotto** - Theory person who appreciates new analysis tools
3. **Jeff Steinhauer** - Most cited BEC work, influential
4. **Pisin Chen** - AnaBHEL relevance, direct experimental application
5. **Daniele Faccio** - Optical analogue expert
6. **Ralf Schützhold** - Theory foundations
7. **Stefano Liberati** - Phenomenology focus
8. **Matt Visser** - Conceptual framing expert

---

## 8) Stretch Goals as You Iterate

### Technical Improvements

- **PIC-backed universality**: Process small public openPMD sets first, then ask a lab contact for a single diagnostic slice from a real run
- **Detection realism**: Swap in system temperatures/bandwidths from specific radio front‑ends the labs actually use (makes your 5σ number much more meaningful)
- **Reproducibility polish**: Add a `make gpu-smoketest` that (a) prints device capability, (b) runs a 60‑second demo producing one figure and one JSON
- **Doc note**: Capture exactly what you did on the 3080 (driver, CUDA, `cupy-cuda12x`), so others can reproduce on commodity GPUs

### Outreach & Collaboration

- **Build a benchmark kit**: Publish a compact openPMD dataset + scripts that reproduce universality collapse and a horizon correlation band
- **Conference presentations**: APS-DPP, analogue-gravity workshops
- **Preprint**: Once you have 2-3 interesting results, consider writing up for arXiv

---

## 9) Handy References

### Technical

- **Repo landing page / capabilities & outputs**: https://github.com/Hmbown/analog-hawking-radiation
- **CuPy install (CUDA‑12 wheels)**: https://pypi.org/project/cupy-cuda12x/
- **Compute capability (RTX 3080 = 8.6)**: https://developer.nvidia.com/cuda-gpus
- **OpenPMD example datasets**: https://github.com/openPMD/openPMD-example-datasets
- **WarpX examples**: https://warpx.readthedocs.io/en/23.11/usage/examples.html

### Background Reading

- **Analogue‑gravity across platforms**: https://link.springer.com/content/pdf/10.12942/lrr-2011-3.pdf
- **AnaBHEL context**: https://arxiv.org/abs/2205.12195

---

## 10) Final Nudge

**Run three scripts on your 3080, save the figures/JSON, and send a tight results pack to 3–4 experts.**

That's the shortest path from "vibe coder" to a thoughtful reply from a world expert—with code and plots that invite collaboration.

### Your Action Items (Priority Order)

1. **Set up environment** (30 minutes): clone, venv, install CuPy
2. **Run baseline pipeline** (10 minutes): Scripts A + E above
3. **Run universality sweep** (15 minutes): Script B
4. **Package results** (15 minutes): collect figures + JSON, write 1-page summary
5. **Email 3 experts** (30 minutes): Use template above, prioritize Weinfurtner, Carusotto, Steinhauer
6. **Iterate based on feedback** (ongoing)

**You've got this!** 🖤⚡️
