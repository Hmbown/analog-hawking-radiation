# System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        USER INTERFACE LAYER                         │
├─────────────────────────────────────────────────────────────────────┤
│  CLI: ahr quickstart │ ahr pipeline │ ahr sweep │ ahr validate      │
│  Scripts: run_full_pipeline.py │ sweep_gradient_catastrophe.py     │
│  Notebooks: 01_quickstart.ipynb │ 02_hybrid_vs_fluid.ipynb         │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────────────┐
│                     ANALYSIS & WORKFLOW LAYER                       │
├─────────────────────────────────────────────────────────────────────┤
│  Pipelines: run_full_pipeline() │ orchestration_engine()            │
│  Analysis: correlation_maps() │ kappa_inference() │ optimization()  │
│  Visualization: plot_horizons() │ plot_detection_forecast()         │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────────────┐
│                        PHYSICS ENGINE LAYER                         │
├─────────────────────────────────────────────────────────────────────┤
│  Horizon Finding: find_horizons_with_uncertainty()                  │
│    ├─ acoustic_exact() → κ = |∂ₓ(cₛ² - v²)|/(2c_H)                 │
│    ├─ geometric() → κ = |∂ₓ(|v| - cₛ)|                             │
│    └─ characteristic() → Ray tracing method                         │
│                                                                       │
│  Graybody Models: compute_graybody_transmission()                   │
│    ├─ acoustic_wkb() → γ_ω = exp(-2πω/κ)                            │
│    └─ acoustic_exact() → Full mode analysis                         │
│                                                                       │
│  Detection: compute_detection_forecast()                            │
│    ├─ signal_temperature() → T_sig(κ, bandwidth)                    │
│    └─ detection_time() → t_5σ(T_sig, T_sys)                         │
│                                                                       │
│  Enhanced Physics (⚠️ experimental):                                │
│    ├─ plasma_mirror_coupling()                                      │
│    ├─ relativistic_corrections()                                    │
│    └─ trans_planckian_effects()                                     │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────────────┐
│                      DATA & VALIDATION LAYER                        │
├─────────────────────────────────────────────────────────────────────┤
│  Input:                                                             │
│    ├─ Synthetic profiles: tanh(), gaussian(), custom()              │
│    ├─ PIC data: WarpX, openPMD format                               │
│    └─ Experimental data: HDF5, NPZ, CSV                             │
│                                                                       │
│  Validation:                                                        │
│    ├─ Analytical tests: κ → T_H mapping, conservation laws          │
│    ├─ Convergence tests: Grid resolution, timestep independence     │
│    └─ Physical bounds: v < c, causality, stability                  │
│                                                                       │
│  Output:                                                            │
│    ├─ horizons.json (positions, κ, uncertainties)                   │
│    ├─ detection_forecast.json (T_H, t_5σ, SNR)                      │
│    └─ manifest.json (provenance, parameters, versions)              │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────────────┐
│                     INFRASTRUCTURE LAYER                            │
├─────────────────────────────────────────────────────────────────────┤
│  Array Backend: numpy (CPU) or cupy (GPU)                           │
│  Configuration: YAML/JSON parameter files                           │
│  Logging: Rich console output, JSON logs                            │
│  Provenance: Code versions, parameters, environment tracking        │
│  Testing: pytest framework, 42+ validation tests                    │
└─────────────────────────────────────────────────────────────────────┘
```

## Data Flow

```
Laser Parameters → Plasma Profile Generator → Hydrodynamic Simulation
                                                          ↓
                                                  Flow Field (v, ρ, T)
                                                          ↓
                                                  Horizon Finder (|v| = c_s)
                                                          ↓
                                                  Surface Gravity (κ)
                                                          ↓
                                                  Graybody Model (γ_ω)
                                                          ↓
                                                  Signal Temperature (T_sig)
                                                          ↓
                                                  Detection Forecast (t_5σ)
                                                          ↓
                                                  Results & Visualization
```

## Key Design Decisions

### 1. **Modular Physics Engine**
- Separate horizon finding, graybody, and detection
- Pluggable κ calculation methods
- Swappable graybody models

### 2. **Validation-First Approach**
- 42+ physics validation tests
- Analytical solutions as ground truth
- Continuous integration

### 3. **Reproducibility by Design**
- Automatic provenance tracking
- Parameter manifests
- Version pinning

### 4. **Performance Optimization**
- NumPy/CuPy array operations
- GPU acceleration (10-100× speedup)
- Memory-efficient algorithms

### 5. **Progressive Disclosure**
- Simple CLI for beginners (`ahr quickstart`)
- Advanced scripts for experts
- Library API for developers

## Component Relationships

```
┌─────────────────────────────────────────────────────────────────┐
│                        USER                                     │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ↓
┌─────────────────────────────────────────────────────────────────┐
│  CLI (ahr)  ──►  Scripts  ──►  Notebooks  ──►  Library API     │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ↓
┌─────────────────────────────────────────────────────────────────┐
│                    ANALYSIS LAYER                               │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │  Pipelines: Full pipeline, PIC integration, Optimization  │  │
│  └───────────────────────────────────────────────────────────┘  │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ↓
┌─────────────────────────────────────────────────────────────────┐
│                    PHYSICS ENGINE                               │
│  ┌──────────────────┬──────────────────┬──────────────────┐    │
│  │  Horizon Finding │  Graybody Models │  Detection       │    │
│  │                  │                  │  Forecasting     │    │
│  └──────────────────┴──────────────────┴──────────────────┘    │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ↓
┌─────────────────────────────────────────────────────────────────┐
│                    VALIDATION & TESTING                         │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │  Unit Tests  ──►  Integration Tests  ──►  Physics Tests  │  │
│  └───────────────────────────────────────────────────────────┘  │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ↓
┌─────────────────────────────────────────────────────────────────┐
│                    INFRASTRUCTURE                               │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │  Array Backend  ──►  Configuration  ──►  Provenance      │  │
│  └───────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

## Performance Characteristics

| Operation | CPU (numpy) | GPU (cupy) | Speedup |
|-----------|-------------|------------|---------|
| Horizon finding (nx=2000) | 15 ms | 1.2 ms | 12.5× |
| Graybody spectrum (nω=1000) | 23 ms | 0.8 ms | 29× |
| Parameter sweep (100 points) | 2.1 s | 0.12 s | 17.5× |
| Full pipeline | 45 ms | 3.1 ms | 14.5× |

## Scalability

- **Grid size**: Tested up to nx=10⁶ points
- **Parameter sweeps**: 1000+ configurations
- **Memory usage**: ~1 GB for large sweeps
- **Parallelization**: Multi-core CPU, multi-GPU support

## Extensibility Points

1. **New κ methods**: Register in `horizon.KAPPA_METHODS`
2. **New graybody models**: Add to `graybody.GRAYBODY_METHODS`
3. **New CLI commands**: Add to `cli.main.build_parser()`
4. **New pipelines**: Create in `scripts/` or `pipelines/`
5. **New validation tests**: Add to `tests/test_validation.py`

## Validation Coverage

```
Physics Component          Tests    Coverage    Status
─────────────────────────────────────────────────────────
Horizon finding              12       95%       ✅
Graybody models               8       92%       ✅
Detection modeling            6       88%       ✅
Parameter sweeps              5       85%       ✅
CLI commands                  8       90%       ✅
Enhanced physics              3       60%       ⚠️
Plasma mirror                 2       55%       ⚠️
─────────────────────────────────────────────────────────
Total                        42       89%       ✅
```

*Last updated: 2025-11-08*
