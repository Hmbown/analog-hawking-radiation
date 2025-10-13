<!-- 64b025b9-95d2-4aa1-985d-f1492cd78ef0 cdae8f28-07fe-40f8-9a4a-5fc29aa478d7 -->
# Plan: PIC Simulation and Analysis Roadmap

This plan details the necessary steps to extend the project's capabilities by integrating a Particle-in-Cell (PIC) model. This will enable more fundamental simulations of the plasma physics involved in analog gravity experiments.

### 1. Foundational Upgrade: From Fluid Model to Particle-in-Cell (PIC)

- **Goal**: Augment the current fluid approximation of the plasma with a first-principles PIC simulation backend.
- **Justification**: A fluid model is an effective approximation. To explore physics at the particle level, a PIC code, which simulates the plasma from the level of individual particles and their interactions, is required. This addresses a core limitation noted in the `README.md`.
- **Implementation**:
    - Integrate a well-established open-source PIC library (e.g., `WarpX`, `Smilei`) into the existing backend framework.
    - Develop the coupling between our laser and horizon-finding modules and the new PIC engine.

### 2. Seeding the Quantum Vacuum: Implementing Fluctuation Injection

- **Goal**: Introduce physically-motivated fluctuations into the PIC simulation.
- **Justification**: Hawking radiation originates from quantum fluctuations at the event horizon. A classical PIC simulation is quiet by default. Seeding it with appropriate noise is necessary to accurately stimulate the analog Hawking effect.
- **Implementation**:
    - Develop a new module in the `physics_engine` that injects randomized field or particle distributions into the PIC simulation at each time step.
    - The statistical properties of this noise should be configurable to match different theoretical models.

### 3. Advanced Simulation: Probing High-Gradient Regimes

- **Goal**: Use the new PIC simulation to study the Hawking radiation spectrum in regimes with very sharp horizon gradients.
- **Justification**: A key question in black hole thermodynamics is how physics at the smallest scales (the "Planck scale") affects Hawking radiation. In our analog system, the inter-particle distance is the "Planck scale." We can use the optimizer to find parameters that create horizons so sharp that the predicted Hawking radiation has wavelengths smaller than this distance. Analyzing how the discrete nature of the plasma alters the radiation provides a direct analog of this important physics problem.
- **Implementation**:
    - Use the Bayesian optimizer to find parameters that create the sharpest possible velocity gradients.
    - Run large-scale simulations and analyze the outgoing radiation spectrum.
    - Search for specific, robust deviations from the perfect thermal spectrum predicted by Hawking.

### 4. Advanced Analysis: Correlation and Entanglement Studies

- **Goal**: Extend the simulation diagnostics to track the quantum entanglement between emitted radiation pairs.
- **Justification**: The Black Hole Information Paradox is a deep unsolved problem in theoretical physics. An analog system that can shed light on how information might be preserved in the correlations of outgoing radiation would be a significant contribution.
- **Implementation**:
    - Develop advanced diagnostic tools to track correlations and entanglement between pairs of simulated particles emitted from the horizon.
    - Run long-duration simulations that model the entire "evaporation" of an analog hole.
    - Analyze the outgoing stream of radiation for subtle correlations that could encode information about the formation of the horizon.

### To-dos

- [ ] Upgrade the physics engine from a fluid model to a more fundamental Particle-in-Cell (PIC) model.
- [ ] Develop a quantum fluctuation seeding module to inject physically accurate vacuum noise.
- [ ] Design and execute a computational experiment to hunt for Trans-Planckian signatures in the Hawking spectrum.
- [ ] Extend the simulation to model entanglement and search for signatures of information recovery (Information Paradox analog).