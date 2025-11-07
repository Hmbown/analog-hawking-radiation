# ELI Experimental Planning Guide for Analog Hawking Radiation

**Version:** 1.0.0
**Date:** November 2025
**Target Audience:** Experimental physicists planning analog Hawking radiation experiments at ELI facilities

## Executive Summary

This guide provides comprehensive experimental planning recommendations for analog Hawking radiation experiments at the Extreme Light Infrastructure (ELI) facilities. Based on detailed facility compatibility analysis, we recommend a phased approach across multiple ELI facilities to optimize experimental success and scientific output.

### Key Findings

- **Primary Recommendation:** ELI-Beamlines for highest intensity capabilities
- **Best for Statistics:** ELI-ALPS for high-repetition-rate parameter optimization
- **Advanced Diagnostics:** ELI-NP for comprehensive characterization with dual-beam capability
- **Optimal Parameters:** Intensity 1e22-1e24 W/m², plasma density 1e25-1e26 m⁻³
- **Critical Factors:** Plasma mirror timing, density gradient optimization, diagnostic integration

## Facility Comparison and Selection

### ELI-Beamlines (Czech Republic) - High-Performance Option

**Advantages:**
- Highest available intensity (10 PW, 1.5 kJ)
- Proven plasma physics infrastructure
- Good focal spot quality (M² = 1.2)
- Established experimental protocols

**Limitations:**
- Low repetition rate (1 shot/minute)
- Limited shot statistics
- Long experimental campaigns required

**Best Use Cases:**
- Proof-of-concept demonstrations
- Maximum signal strength experiments
- Single-shot precision measurements

### ELI-NP (Romania) - Advanced Diagnostics Option

**Advantages:**
- Dual-beam capability for enhanced diagnostics
- Excellent temporal contrast (1e-11)
- Magnetic field expertise
- Comprehensive nuclear physics diagnostics

**Limitations:**
- Very low repetition rate (1 shot/5 minutes)
- Complex safety procedures
- Limited beam time availability

**Best Use Cases:**
- Advanced characterization studies
- Correlation experiments with nuclear products
- Magnetic field effect studies

### ELI-ALPS (Hungary) - High-Statistics Option

**Advantages:**
- High repetition rate (up to 10 Hz)
- Excellent for parameter optimization
- Fast diagnostic capabilities
- Automated experimental control

**Limitations:**
- Lower maximum intensity (2 PW)
- Challenging plasma mirror timing
- Limited by 17 fs pulse duration

**Best Use Cases:**
- Parameter space mapping
- Statistical validation
- Rapid prototyping

## Recommended Experimental Phases

### Phase 1: Proof of Concept (ELI-ALPS)
**Duration:** 2 weeks
**Goals:**
- Validate plasma mirror formation
- Demonstrate basic horizon detection
- Optimize timing and diagnostics

**Parameters:**
- Intensity: 5e21 W/m² (conservative)
- Pulse duration: 17 fs
- Repetition rate: 5 Hz
- Target: Translating solid target

**Success Criteria:**
- Plasma mirror reflectivity > 50%
- Horizon detection on > 30% of shots
- Diagnostic validation complete

### Phase 2: High-Performance Measurements (ELI-Beamlines)
**Duration:** 4 weeks
**Goals:**
- Achieve maximum κ values
- Detailed spectrum analysis
- Reproducibility verification

**Parameters:**
- Intensity: 8e23 W/m² (near maximum)
- Pulse duration: 150 fs
- Repetition rate: 0.017 Hz
- Target: High-quality optical surface

**Success Criteria:**
- κ measurement with < 20% uncertainty
- Hawking-like spectrum detection
- 5σ detection confidence

### Phase 3: Advanced Characterization (ELI-NP)
**Duration:** 3 weeks
**Goals:**
- Correlation studies with nuclear products
- Magnetic field effects
- Comprehensive validation

**Parameters:**
- Intensity: 7e23 W/m²
- Pulse duration: 150 fs
- Dual-beam configuration
- Magnetic field: 20 T

**Success Criteria:**
- Nuclear product correlation
- Magnetic field effect characterization
- Complete physics validation

## Parameter Optimization Strategy

### Intensity Optimization

**Optimal Range:** 1e22-1e24 W/m²

**Guidelines:**
- Start conservative (1e22 W/m²) and gradually increase
- Monitor for plasma mirror formation quality
- Watch for relativistic effects (a₀ > 1)
- Ensure gradient catastrophe thresholds not exceeded

**Risks:**
- Above 1e24 W/m²: Physics breakdown likely
- Below 1e22 W/m²: Weak Hawking signal
- Non-uniform intensity: Poor horizon formation

### Plasma Density Optimization

**Optimal Range:** 1e25-1e26 m⁻³ (near-critical for 800nm)

**Guidelines:**
- Target 0.1-10 × critical density
- Ensure good ionization for plasma mirror
- Optimize density gradient scale (0.3-1.0 μm)
- Consider target material and pre-pulse effects

**Target Types:**
- **Solid targets:** Best for plasma mirror formation
- **Gas jets:** Challenging but possible with high density
- **Cluster targets:** Good for high-density plasma

### Density Gradient Control

**Optimal Scale:** 0.3-1.0 μm

**Control Methods:**
- Pre-pulse timing and intensity
- Target surface preparation
- Plasma mirror timing optimization
- Diagnostic interferometry

**Validation:**
- Optical probing of density profile
- Shadowgraphy measurements
- Interferometric characterization

## Diagnostic Requirements

### Essential Diagnostics

1. **Plasma Characterization**
   - Interferometry (density profile)
   - Shadowgraphy (gradient structure)
   - Optical probing (temporal evolution)

2. **Hawking Signal Detection**
   - Radio detection system (30K system temperature)
   - High-bandwidth oscilloscopes (>10 GHz)
   - Low-noise RF amplifiers
   - Spectrum analyzers

3. **Laser Characterization**
   - Focal spot analysis
   - Temporal contrast measurement
   - Pulse duration measurement
   - Energy monitoring

### Advanced Diagnostics (ELI-NP specific)

1. **Nuclear Product Detection**
   - Gamma ray detectors
   - Nuclear activation analysis
   - Positron detection
   - High-energy particle spectrometry

2. **Correlation Measurements**
   - Particle coincidence detectors
   - Time-of-flight systems
   - Angular distribution measurements

## Plasma Mirror Implementation

### Formation Requirements

**Intensity Threshold:** >1e14 W/cm²
**Formation Time:** 10-50 fs (depends on intensity)
**Optimal Delay:** Formation time + 10% buffer

### Timing Control

**Precision Required:** <5 fs
**Control Methods:**
- Optical delay lines
- Plasma mirror timing diagnostics
- Real-time feedback systems

**Quality Metrics:**
- Reflectivity > 60%
- Stability < 10% shot-to-shot
- Surface smoothness preservation

### Target Systems

**ELI-ALPS (High Rep Rate):**
- Translating target system
- 10 mm/s translation speed
- 50×50×10 mm target size
- 1000 shots per location

**ELI-Beamlines/NP (Low Rep Rate):**
- Fixed target with precision positioning
- Motorized XYZ stages
- 1 μm positioning accuracy
- Single-shot target replacement

## Risk Management

### Technical Risks

1. **Plasma Mirror Formation Failure**
   - **Mitigation:** Conservative intensity ramp-up
   - **Backup:** Alternative target materials
   - **Monitoring:** Real-time reflectivity measurement

2. **Detection System Limitations**
   - **Mitigation:** Redundant diagnostic systems
   - **Backup:** Alternative detection frequencies
   - **Monitoring:** Real-time noise floor tracking

3. **Laser Stability Issues**
   - **Mitigation:** Pre-experiment laser characterization
   - **Backup:** Multiple experimental configurations
   - **Monitoring:** Shot-to-shot parameter logging

### Facility Risks

1. **Beam Time Limitations**
   - **Mitigation:** Efficient experimental planning
   - **Backup:** Alternative facility agreements
   - **Monitoring:** Real-time efficiency tracking

2. **Equipment Failures**
   - **Mitigation:** Redundant systems
   - **Backup:** Spare parts inventory
   - **Monitoring:** Preventive maintenance schedule

## Data Analysis Strategy

### Real-time Analysis

**Goals:**
- Validate experimental success
- Optimize parameters between shots
- Identify system issues early

**Methods:**
- Automated data processing pipelines
- Real-time κ calculation
- Signal-to-noise estimation
- Quality flagging system

### Post-Processing

**Analysis Steps:**
1. **Raw Data Validation**
   - Shot selection criteria
   - Quality flag assessment
   - Outlier identification

2. **Physical Parameter Extraction**
   - Surface gravity calculation
   - Temperature estimation
   - Spectrum characterization

3. **Statistical Analysis**
   - Uncertainty quantification
   - Reproducibility assessment
   - Correlation studies

4. **Theoretical Comparison**
   - Model comparison
   - Parameter fitting
   - Deviation analysis

## Success Metrics and Validation

### Minimum Success Criteria

1. **Plasma Mirror Formation**
   - Reflectivity > 50% for > 80% of shots
   - Formation time consistency < 20%
   - Surface quality preservation

2. **Horizon Detection**
   - Sonic horizon identification
   - κ measurement capability
   - Gradient structure validation

3. **Signal Detection**
   - Hawking-like signal identification
   - 3σ confidence minimum
   - Reproducibility across shots

### Target Success Criteria

1. **High-Precision Measurements**
   - κ uncertainty < 20%
   - Temperature detection with 5σ confidence
   - Spectrum shape validation

2. **Reproducibility**
   - Results consistent across > 70% of shots
   - Parameter dependencies verified
   - Theoretical agreement within 50%

3. **Publication Quality**
   - Complete uncertainty analysis
   - Theoretical model validation
   - Novel physics demonstration

## Budget and Timeline Planning

### Estimated Costs

**Phase 1 (ELI-ALPS):** €200,000
- Beam time: €100,000
- Target fabrication: €30,000
- Diagnostic equipment: €50,000
- Personnel: €20,000

**Phase 2 (ELI-Beamlines):** €350,000
- Beam time: €200,000
- Advanced targets: €50,000
- Enhanced diagnostics: €70,000
- Personnel: €30,000

**Phase 3 (ELI-NP):** €400,000
- Beam time: €250,000
- Dual-beam configuration: €50,000
- Nuclear diagnostics: €60,000
- Personnel: €40,000

**Total Estimated Budget:** €950,000

### Timeline

**Months 1-3:** Preparation and simulations
**Months 4-5:** Phase 1 execution at ELI-ALPS
**Months 6-9:** Phase 2 execution at ELI-Beamlines
**Months 10-12:** Phase 3 execution at ELI-NP
**Months 13-15:** Data analysis and publication

## Collaboration and Expertise Requirements

### Required Expertise

1. **Laser-Plasma Physics**
   - High-intensity laser-matter interaction
   - Plasma mirror physics
   - Diagnostic development

2. **Analog Gravity**
   - Hawking radiation theory
   - Sonic horizon physics
   - Quantum field theory in curved spacetime

3. **Experimental Design**
   - High-power laser systems
   - Ultra-fast diagnostics
   - Data analysis techniques

### Recommended Collaborations

1. **ELI Facility Staff**
   - Laser operations expertise
   - Facility-specific knowledge
   - Safety protocol development

2. **Theory Groups**
   - Analog gravity modeling
   - Numerical simulation support
   - Interpretation guidance

3. **Diagnostic Experts**
   - Radio detection systems
   - High-speed data acquisition
   - Signal processing

## Next Steps

1. **Immediate Actions (1-2 months)**
   - Contact ELI facilities for beam time inquiries
   - Form collaboration teams
   - Secure initial funding
   - Begin detailed simulations

2. **Short-term Preparation (2-6 months)**
   - Complete experimental proposal
   - Finalize diagnostic specifications
   - Develop data analysis pipeline
   - Prepare safety documentation

3. **Experimental Execution (6-15 months)**
   - Execute three-phase experimental campaign
   - Real-time optimization and analysis
   - Iterative parameter improvement
   - Comprehensive data collection

4. **Analysis and Publication (12-18 months)**
   - Complete data analysis
   - Prepare publications
   - Present at conferences
   - Plan follow-up experiments

## Conclusion

The analog Hawking radiation experiments at ELI facilities represent a cutting-edge opportunity to test fundamental physics in laboratory conditions. The recommended three-phase approach across multiple ELI facilities provides the best chance of experimental success while managing risks and optimizing resource utilization.

Key factors for success include:
- Careful parameter optimization within physical limits
- Robust plasma mirror implementation
- Comprehensive diagnostic integration
- Real-time data analysis and feedback
- Strong collaboration with facility experts

With proper planning and execution, these experiments have the potential to provide the first laboratory detection of analog Hawking radiation and open new frontiers in experimental quantum gravity research.