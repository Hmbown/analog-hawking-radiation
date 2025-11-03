# Workflow Diagram for Phased Upgrade Plan

The following Mermaid diagram visualizes the sequential workflow across the four phases, highlighting key inputs, processes, outputs, and validation gates. Each phase builds on the previous, ensuring incremental progress toward full-fidelity simulations and experimental readiness.

```mermaid
graph TD
    A[Current State: 1D Fluid Surrogates<br/>Mock WarpX<br/>No Fluctuations<br/>No 3D Effects] --> B[Phase 1: Full WarpX Integration<br/>- OpenPMD Ingestion<br/>- Real PIC Profiles<br/>- Horizon Validation]
    B -->|Validation: Tests & Benchmarks<br/><5% κ Error| C[Phase 2: 3D Fluctuation Enhancements<br/>- 3D Graybody Extension<br/>- Vacuum Noise Seeding<br/>- κ Optimization for T_H >1 mK<br/>- Plasma Mirror Boosts]
    C -->|Validation: Convergence & Statistics<br/>Gaussian Fluctuations| D[Phase 3: Experimental Design<br/>- Laser Plasma Specs<br/>- Diagnostics Protocol<br/>- Facility Integration ELI-NP NIF<br/>- SNR Forecasts]
    D -->|Validation: Feasibility Sims<br/>Risk Assessment| E[Phase 4: Validation Publication Prep<br/>- AnaBHEL Benchmarks<br/>- Full Noise Models<br/>- Paper Draft Structure<br/>- 5σ Detection Sims]
    E --> F[Final Outputs: Publication-Ready<br/>Validated Roadmap<br/>Executable Experiments<br/>Conservative Claims]

    style A fill:#f9f,stroke:#333
    style F fill:#bbf,stroke:#333
```

This flowchart illustrates the linear progression with validation checkpoints to maintain rigor. Arrows represent dependencies, and nodes detail core activities per phase.
