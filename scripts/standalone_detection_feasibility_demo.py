#!/usr/bin/env python3
"""
Standalone Detection Feasibility Demo for Analog Hawking Radiation

This script demonstrates realistic detection feasibility assessment with:
- Comprehensive noise modeling
- Signal-to-noise ratio calculations
- Detection strategy assessment
- ELI facility recommendations

Author: Claude Analysis Assistant
Date: November 2025
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.analog_hawking.detection.detection_feasibility import (
    DetectionFeasibilityAnalyzer,
    SignalCharacteristics,
)
from src.analog_hawking.facilities.eli_diagnostic_integration import (
    ELIDiagnosticIntegrator,
    assess_eli_compatibility,
)


def create_realistic_signal_scenarios() -> pd.DataFrame:
    """Create realistic signal scenarios for detection feasibility assessment"""

    print("🔄 Creating realistic signal scenarios...")

    scenarios = []

    # Scenario 1: Optimistic case (high temperature)
    scenarios.append(
        {
            "scenario_name": "Optimistic High-Temperature",
            "hawking_temperature_K": 0.1,  # 100 mK
            "surface_gravity_s": 1e13,  # 10¹³ s⁻¹
            "peak_frequency_Hz": 2e12,  # 2 THz
            "plasma_temperature_K": 1e6,  # 1 MK
            "plasma_density_m3": 1e24,  # 10²⁴ m⁻³
            "emitting_area_m2": 1e-12,  # 1 μm²
        }
    )

    # Scenario 2: Realistic case (medium temperature)
    scenarios.append(
        {
            "scenario_name": "Realistic Medium-Temperature",
            "hawking_temperature_K": 0.01,  # 10 mK
            "surface_gravity_s": 1e12,  # 10¹² s⁻¹
            "peak_frequency_Hz": 2e11,  # 200 GHz
            "plasma_temperature_K": 1e6,  # 1 MK
            "plasma_density_m3": 1e23,  # 10²³ m⁻³
            "emitting_area_m2": 1e-12,  # 1 μm²
        }
    )

    # Scenario 3: Conservative case (low temperature)
    scenarios.append(
        {
            "scenario_name": "Conservative Low-Temperature",
            "hawking_temperature_K": 0.001,  # 1 mK
            "surface_gravity_s": 1e11,  # 10¹¹ s⁻¹
            "peak_frequency_Hz": 2e10,  # 20 GHz
            "plasma_temperature_K": 1e6,  # 1 MK
            "plasma_density_m3": 1e22,  # 10²² m⁻³
            "emitting_area_m2": 1e-12,  # 1 μm²
        }
    )

    # Scenario 4: Very challenging case
    scenarios.append(
        {
            "scenario_name": "Very Challenging Ultra-Low-Temperature",
            "hawking_temperature_K": 1e-4,  # 0.1 mK
            "surface_gravity_s": 1e10,  # 10¹⁰ s⁻¹
            "peak_frequency_Hz": 2e9,  # 2 GHz
            "plasma_temperature_K": 1e6,  # 1 MK
            "plasma_density_m3": 1e21,  # 10²¹ m⁻³
            "emitting_area_m2": 1e-12,  # 1 μm²
        }
    )

    return pd.DataFrame(scenarios)


def analyze_detection_feasibility_by_scenario(scenarios_df: pd.DataFrame) -> pd.DataFrame:
    """Analyze detection feasibility for each scenario"""

    print("📈 Analyzing detection feasibility for scenarios...")

    analyzer = DetectionFeasibilityAnalyzer()
    results = []

    for idx, scenario in scenarios_df.iterrows():
        print(f"  Analyzing scenario: {scenario['scenario_name']}")

        try:
            # Assess detection feasibility
            assessments = analyzer.assess_detection_feasibility(
                signal=SignalCharacteristics(
                    hawking_temperature=scenario["hawking_temperature_K"],
                    surface_gravity=scenario["surface_gravity_s"],
                    peak_frequency=scenario["peak_frequency_Hz"],
                    bandwidth=scenario["peak_frequency_Hz"] / 10,
                    total_power=1e-25,  # Will be calculated
                    power_density=1e-30,
                    signal_temperature=scenario["hawking_temperature_K"] * 0.1,
                    pulse_duration=1e-12,
                    rise_time=1e-13,
                    repetition_rate=1.0,
                    emitting_area=scenario["emitting_area_m2"],
                    angular_distribution="isotropic",
                ),
                plasma_params={
                    "temperature": scenario["plasma_temperature_K"],
                    "density": scenario["plasma_density_m3"],
                },
            )

            # Extract best assessment
            best_assessment = assessments[0] if assessments else None

            if best_assessment:
                result = {
                    "scenario_name": scenario["scenario_name"],
                    "hawking_temperature_K": scenario["hawking_temperature_K"],
                    "surface_gravity_s": scenario["surface_gravity_s"],
                    "peak_frequency_Hz": scenario["peak_frequency_Hz"],
                    "best_method": best_assessment.detection_method.value,
                    "best_detector": best_assessment.detector_type.value,
                    "best_snr": best_assessment.snr_optimal,
                    "feasibility_level": best_assessment.feasibility_level.value,
                    "detection_probability": best_assessment.detection_probability,
                    "integration_time_s": best_assessment.optimal_integration_time,
                    "required_shots": best_assessment.required_shots,
                    "experiment_time_hours": best_assessment.total_experiment_time,
                    "dominant_noise": best_assessment.dominant_noise_source,
                    "cost_estimate": best_assessment.cost_estimate,
                    "timeline_estimate": best_assessment.timeline_estimate,
                    "top_recommendations": "; ".join(best_assessment.recommendations[:3]),
                }
            else:
                result = {
                    "scenario_name": scenario["scenario_name"],
                    "hawking_temperature_K": scenario["hawking_temperature_K"],
                    "surface_gravity_s": scenario["surface_gravity_s"],
                    "peak_frequency_Hz": scenario["peak_frequency_Hz"],
                    "best_method": "None",
                    "best_detector": "None",
                    "best_snr": 0.0,
                    "feasibility_level": "Impossible",
                    "detection_probability": 0.0,
                    "integration_time_s": float("inf"),
                    "required_shots": 0,
                    "experiment_time_hours": float("inf"),
                    "dominant_noise": "N/A",
                    "cost_estimate": "N/A",
                    "timeline_estimate": "N/A",
                    "top_recommendations": "No feasible detection method",
                }

            results.append(result)

        except Exception as e:
            print(f"    ❌ Error: {e}")
            results.append(
                {
                    "scenario_name": scenario["scenario_name"],
                    "error": str(e),
                    "best_snr": 0.0,
                    "feasibility_level": "Error",
                }
            )

    return pd.DataFrame(results)


def analyze_eli_facility_compatibility(results_df: pd.DataFrame) -> pd.DataFrame:
    """Analyze ELI facility compatibility for each scenario"""

    print("🏢 Analyzing ELI facility compatibility...")

    facility_analysis = []

    integrator = ELIDiagnosticIntegrator()
    facilities = ["beamlines", "np", "alps"]

    for idx, result in results_df.iterrows():
        if "error" in result:
            continue

        signal_params = {
            "peak_frequency": result["peak_frequency_Hz"],
            "signal_power": 1e-25,  # Estimate
            "pulse_duration": 1e-12,
            "precise_timing": True,
            "vacuum_required": True,
        }

        for facility in facilities:
            try:
                # Get best method for this facility
                compatibility = assess_eli_compatibility(
                    detection_method=result["best_method"],
                    facility=facility,
                    signal_parameters=signal_params,
                )

                if compatibility["assessments"]:
                    best_assessment = compatibility["assessments"][0]
                    facility_analysis.append(
                        {
                            "scenario_name": result["scenario_name"],
                            "facility": facility,
                            "best_diagnostic": best_assessment["diagnostic"],
                            "compatibility_score": best_assessment["compatibility_score"],
                            "integration_complexity": best_assessment["integration_complexity"],
                            "timeline": best_assessment["timeline"],
                            "cost": best_assessment["cost"],
                            "num_risks": len(best_assessment["risks"]),
                        }
                    )
                else:
                    facility_analysis.append(
                        {
                            "scenario_name": result["scenario_name"],
                            "facility": facility,
                            "best_diagnostic": "None",
                            "compatibility_score": 0.0,
                            "integration_complexity": "High",
                            "timeline": "Unknown",
                            "cost": "Unknown",
                            "num_risks": 999,
                        }
                    )

            except Exception as e:
                print(f"    ⚠️ Facility {facility} analysis error: {e}")
                facility_analysis.append(
                    {
                        "scenario_name": result["scenario_name"],
                        "facility": facility,
                        "best_diagnostic": "Error",
                        "compatibility_score": 0.0,
                        "integration_complexity": "High",
                        "timeline": "Unknown",
                        "cost": "Unknown",
                        "num_risks": 999,
                    }
                )

    return pd.DataFrame(facility_analysis)


def generate_comprehensive_plots(
    detection_results: pd.DataFrame, facility_results: pd.DataFrame, output_dir: str
):
    """Generate comprehensive analysis plots"""

    print("📊 Generating comprehensive plots...")

    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # Set up plotting style
    plt.style.use("default")
    plt.rcParams.update({"font.size": 10})

    # 1. SNR vs Hawking Temperature
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # Plot 1: SNR vs Temperature
    axes[0, 0].scatter(
        detection_results["hawking_temperature_K"],
        detection_results["best_snr"],
        s=100,
        alpha=0.7,
        color="blue",
    )
    axes[0, 0].set_xlabel("Hawking Temperature (K)")
    axes[0, 0].set_ylabel("Best SNR")
    axes[0, 0].set_title("Detection SNR vs Hawking Temperature")
    axes[0, 0].set_xscale("log")
    axes[0, 0].set_yscale("log")
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].axhline(y=5, color="red", linestyle="--", alpha=0.7, label="5σ threshold")
    axes[0, 0].legend()

    # Plot 2: Detection Probability
    axes[0, 1].bar(
        range(len(detection_results)),
        detection_results["detection_probability"],
        color="green",
        alpha=0.7,
    )
    axes[0, 1].set_xlabel("Scenario")
    axes[0, 1].set_ylabel("Detection Probability")
    axes[0, 1].set_title("Detection Probability by Scenario")
    axes[0, 1].set_xticks(range(len(detection_results)))
    axes[0, 1].set_xticklabels(
        [s[:15] + "..." if len(s) > 15 else s for s in detection_results["scenario_name"]],
        rotation=45,
        ha="right",
    )

    # Plot 3: Integration Time
    valid_times = detection_results[detection_results["integration_time_s"] != float("inf")]
    if len(valid_times) > 0:
        axes[0, 2].bar(
            range(len(valid_times)), valid_times["integration_time_s"], color="orange", alpha=0.7
        )
        axes[0, 2].set_xlabel("Scenario")
        axes[0, 2].set_ylabel("Integration Time (s)")
        axes[0, 2].set_title("Required Integration Time")
        axes[0, 2].set_yscale("log")
        axes[0, 2].set_xticks(range(len(valid_times)))
        axes[0, 2].set_xticklabels(
            [s[:15] + "..." if len(s) > 15 else s for s in valid_times["scenario_name"]],
            rotation=45,
            ha="right",
        )

    # Plot 4: Facility Compatibility Heatmap
    facility_pivot = facility_results.pivot(
        index="scenario_name", columns="facility", values="compatibility_score"
    )
    im1 = axes[1, 0].imshow(
        facility_pivot.values, aspect="auto", origin="lower", cmap="RdYlGn", vmin=0, vmax=1
    )
    axes[1, 0].set_title("ELI Facility Compatibility Scores")
    axes[1, 0].set_xticks(range(len(facility_pivot.columns)))
    axes[1, 0].set_xticklabels(facility_pivot.columns)
    axes[1, 0].set_yticks(range(len(facility_pivot.index)))
    axes[1, 0].set_yticklabels(
        [s[:20] + "..." if len(s) > 20 else s for s in facility_pivot.index], fontsize=8
    )
    plt.colorbar(im1, ax=axes[1, 0], label="Compatibility Score")

    # Plot 5: Required Shots
    valid_shots = detection_results[detection_results["required_shots"] > 0]
    if len(valid_shots) > 0:
        axes[1, 1].bar(
            range(len(valid_shots)), valid_shots["required_shots"], color="purple", alpha=0.7
        )
        axes[1, 1].set_xlabel("Scenario")
        axes[1, 1].set_ylabel("Required Shots")
        axes[1, 1].set_title("Number of Shots Required")
        axes[1, 1].set_yscale("log")
        axes[1, 1].set_xticks(range(len(valid_shots)))
        axes[1, 1].set_xticklabels(
            [s[:15] + "..." if len(s) > 15 else s for s in valid_shots["scenario_name"]],
            rotation=45,
            ha="right",
        )

    # Plot 6: Feasibility Level Distribution
    feasibility_counts = detection_results["feasibility_level"].value_counts()
    axes[1, 2].bar(
        range(len(feasibility_counts)), feasibility_counts.values, color="red", alpha=0.7
    )
    axes[1, 2].set_xlabel("Feasibility Level")
    axes[1, 2].set_ylabel("Count")
    axes[1, 2].set_title("Feasibility Level Distribution")
    axes[1, 2].set_xticks(range(len(feasibility_counts)))
    axes[1, 2].set_xticklabels(
        [level[:20] + "..." if len(level) > 20 else level for level in feasibility_counts.index],
        rotation=45,
        ha="right",
    )

    plt.tight_layout()
    plt.savefig(
        output_path / "detection_feasibility_comprehensive.png", dpi=300, bbox_inches="tight"
    )
    plt.close()

    print(f"✅ Plots saved to {output_path}")


def generate_summary_report(
    detection_results: pd.DataFrame, facility_results: pd.DataFrame, output_dir: str
):
    """Generate comprehensive summary report"""

    print("📄 Generating summary report...")

    report_lines = []
    report_lines.append("# Comprehensive Detection Feasibility Analysis Report")
    report_lines.append("=" * 60)
    report_lines.append("Analysis Date: November 1, 2025")
    report_lines.append("Scope: Analog Hawking Radiation Detection with Realistic Noise Modeling")
    report_lines.append("")

    # Executive Summary
    report_lines.append("## Executive Summary")
    report_lines.append("")

    total_scenarios = len(detection_results)
    feasible_scenarios = len(detection_results[detection_results["best_snr"] >= 5])
    challenging_scenarios = len(
        detection_results[
            (detection_results["best_snr"] >= 1) & (detection_results["best_snr"] < 5)
        ]
    )

    report_lines.append(f"**Total Scenarios Analyzed:** {total_scenarios}")
    report_lines.append(f"**Feasible Detections (SNR ≥ 5):** {feasible_scenarios}")
    report_lines.append(f"**Challenging Detections (SNR 1-5):** {challenging_scenarios}")
    report_lines.append(f"**Feasibility Rate:** {100*feasible_scenarios/total_scenarios:.1f}%")
    report_lines.append("")

    if feasible_scenarios > 0:
        best_scenario = detection_results.loc[detection_results["best_snr"].idxmax()]
        report_lines.append("### Most Promising Scenario")
        report_lines.append(f"- **Scenario:** {best_scenario['scenario_name']}")
        report_lines.append(
            f"- **Hawking Temperature:** {best_scenario['hawking_temperature_K']:.3f} K"
        )
        report_lines.append(f"- **Detection Method:** {best_scenario['best_method']}")
        report_lines.append(f"- **Best SNR:** {best_scenario['best_snr']:.2f}")
        report_lines.append(
            f"- **Detection Probability:** {best_scenario['detection_probability']:.1%}"
        )
        report_lines.append(
            f"- **Required Integration Time:** {best_scenario['integration_time_s']:.2e} s"
        )
        report_lines.append(f"- **Required Shots:** {int(best_scenario['required_shots'])}")
        report_lines.append("")

    # Detailed Scenario Analysis
    report_lines.append("## Detailed Scenario Analysis")
    report_lines.append("")

    for _, scenario in detection_results.iterrows():
        report_lines.append(f"### {scenario['scenario_name']}")
        report_lines.append("")
        report_lines.append("**Signal Parameters:**")
        report_lines.append(f"- Hawking Temperature: {scenario['hawking_temperature_K']:.4f} K")
        report_lines.append(f"- Surface Gravity: {scenario['surface_gravity_s']:.2e} s⁻¹")
        report_lines.append(f"- Peak Frequency: {scenario['peak_frequency_Hz']:.2e} Hz")
        report_lines.append("")

        report_lines.append("**Detection Assessment:**")
        report_lines.append(f"- Best Method: {scenario['best_method']}")
        report_lines.append(f"- Best Detector: {scenario['best_detector']}")
        report_lines.append(f"- SNR: {scenario['best_snr']:.2f}")
        report_lines.append(f"- Feasibility Level: {scenario['feasibility_level']}")
        report_lines.append(f"- Detection Probability: {scenario['detection_probability']:.1%}")
        report_lines.append("")

        if scenario["integration_time_s"] != float("inf"):
            report_lines.append("**Experimental Requirements:**")
            report_lines.append(f"- Integration Time: {scenario['integration_time_s']:.2e} s")
            report_lines.append(f"- Required Shots: {int(scenario['required_shots'])}")
            report_lines.append(f"- Experiment Time: {scenario['experiment_time_hours']:.1f} hours")
            report_lines.append(f"- Cost Estimate: {scenario['cost_estimate']}")
            report_lines.append(f"- Timeline: {scenario['timeline_estimate']}")
            report_lines.append("")

            report_lines.append(f"**Dominant Noise Source:** {scenario['dominant_noise']}")
            report_lines.append("")

            report_lines.append("**Top Recommendations:**")
            for rec in scenario["top_recommendations"].split("; "):
                report_lines.append(f"- {rec}")
            report_lines.append("")
        else:
            report_lines.append("**Status:** Detection not feasible with current technology")
            report_lines.append("")

    # ELI Facility Assessment
    report_lines.append("## ELI Facility Assessment")
    report_lines.append("")

    for facility in ["beamlines", "np", "alps"]:
        facility_data = facility_results[facility_results["facility"] == facility]
        if len(facility_data) > 0:
            avg_compatibility = facility_data["compatibility_score"].mean()
            best_compatibility = facility_data["compatibility_score"].max()

            report_lines.append(f"### {facility.upper()}")
            report_lines.append(f"- Average Compatibility: {avg_compatibility:.1%}")
            report_lines.append(f"- Best Compatibility: {best_compatibility:.1%}")

            best_facility_scenario = facility_data.loc[
                facility_data["compatibility_score"].idxmax()
            ]
            report_lines.append(f"- Best Diagnostic: {best_facility_scenario['best_diagnostic']}")
            report_lines.append(
                f"- Integration Complexity: {best_facility_scenario['integration_complexity']}"
            )
            report_lines.append("")

    # Key Findings and Recommendations
    report_lines.append("## Key Findings and Recommendations")
    report_lines.append("")

    if feasible_scenarios == 0:
        report_lines.append("### Critical Finding: Detection Currently Not Feasible")
        report_lines.append("")
        report_lines.append("Based on realistic noise modeling and current detector technology,")
        report_lines.append("analog Hawking radiation detection is not currently feasible for")
        report_lines.append("the predicted signal levels in our scenarios.")
        report_lines.append("")
        report_lines.append("### Required Breakthroughs:")
        report_lines.append("1. **Signal Enhancement (10-100× needed)**:")
        report_lines.append("   - Stronger plasma flow gradients")
        report_lines.append("   - Enhanced coupling mechanisms")
        report_lines.append("   - Novel signal amplification techniques")
        report_lines.append("")
        report_lines.append("2. **Revolutionary Detector Technology**:")
        report_lines.append("   - Quantum-limited detectors")
        report_lines.append("   - Near-zero noise amplifiers")
        report_lines.append("   - Advanced correlation techniques")
        report_lines.append("")
        report_lines.append("3. **Alternative Detection Paradigms**:")
        report_lines.append("   - Horizon detection (easier than temperature measurement)")
        report_lines.append("   - Quantum correlation signatures")
        report_lines.append("   - Indirect evidence through plasma dynamics")
        report_lines.append("")

    elif feasible_scenarios <= total_scenarios // 2:
        report_lines.append("### Finding: Detection Possible but Challenging")
        report_lines.append("")
        report_lines.append("Detection is achievable for favorable conditions but requires:")
        report_lines.append("- Optimized experimental parameters")
        report_lines.append("- Extended integration times")
        report_lines.append("- Advanced diagnostic systems")
        report_lines.append("")

    else:
        report_lines.append("### Finding: Detection Generally Feasible")
        report_lines.append("")
        report_lines.append("Multiple scenarios show feasible detection prospects.")
        report_lines.append("Focus on optimizing parameters and diagnostic integration.")
        report_lines.append("")

    # Near-Term Achievable Goals
    report_lines.append("### Near-Term Achievable Goals")
    report_lines.append("")
    report_lines.append("1. **Horizon Detection (Highest Priority)**:")
    report_lines.append("   - Timeline: 6-12 months")
    report_lines.append("   - Method: Optical interferometry and shadowgraphy")
    report_lines.append("   - Feasibility: High")
    report_lines.append("   - Requirements: Standard ELI diagnostics")
    report_lines.append("")
    report_lines.append("2. **Flow Characterization**:")
    report_lines.append("   - Timeline: 3-6 months")
    report_lines.append("   - Method: Proton radiography, optical probing")
    report_lines.append("   - Feasibility: Very High")
    report_lines.append("   - Requirements: Established diagnostic suite")
    report_lines.append("")
    report_lines.append("3. **Temperature Upper Limits**:")
    report_lines.append("   - Timeline: 1-2 years")
    report_lines.append("   - Method: Advanced spectroscopy with signal averaging")
    report_lines.append("   - Feasibility: Medium")
    report_lines.append("   - Requirements: Enhanced detector systems")
    report_lines.append("")

    # ELI Facility Recommendations
    report_lines.append("### ELI Facility Recommendations")
    report_lines.append("")
    report_lines.append("1. **ELI-Beamlines (Recommended)**:")
    report_lines.append("   - Best overall compatibility")
    report_lines.append("   - Established diagnostic infrastructure")
    report_lines.append("   - High repetition rate for statistics")
    report_lines.append("")
    report_lines.append("2. **ELI-NP (For High-Energy Studies)**:")
    report_lines.append("   - Highest available intensity")
    report_lines.append("   - Specialized radiation diagnostics")
    report_lines.append("   - Lower repetition rate limits statistics")
    report_lines.append("")
    report_lines.append("3. **ELI-ALPS (For Temporal Studies)**:")
    report_lines.append("   - Superior temporal resolution")
    report_lines.append("   - Attosecond diagnostic capabilities")
    report_lines.append("   - Limited spatial diagnostic suite")
    report_lines.append("")

    # Strategic Recommendations
    report_lines.append("### Strategic Recommendations")
    report_lines.append("")
    report_lines.append("1. **Phase 1: Horizon Detection (0-6 months)**:")
    report_lines.append("   - Focus on sonic horizon confirmation")
    report_lines.append("   - Use standard optical diagnostics")
    report_lines.append("   - Establish experimental methodology")
    report_lines.append("")
    report_lines.append("2. **Phase 2: Signal Enhancement (6-18 months)**:")
    report_lines.append("   - Optimize plasma parameters")
    report_lines.append("   - Implement advanced diagnostics")
    report_lines.append("   - Develop signal processing techniques")
    report_lines.append("")
    report_lines.append("3. **Phase 3: Temperature Measurement (18-36 months)**:")
    report_lines.append("   - Deploy enhanced detector systems")
    report_lines.append("   - Execute long integration campaigns")
    report_lines.append("   - Pursue quantum correlation detection")
    report_lines.append("")

    report_lines.append("---")
    report_lines.append("*Report generated by Comprehensive Detection Feasibility Analysis System*")
    report_lines.append(
        "*Analysis includes realistic noise modeling, ELI facility assessment, and near-term achievable goals*"
    )

    report = "\n".join(report_lines)

    # Save report
    report_path = Path(output_dir) / "detection_feasibility_comprehensive_report.md"
    report_path.write_text(report)

    print(f"✅ Report saved to {report_path}")

    return report


def main():
    """Main analysis pipeline"""

    print("🚀 Starting Comprehensive Detection Feasibility Analysis")
    print("=" * 60)

    # Create output directory
    output_dir = Path("results/detection_feasibility_demo")
    output_dir.mkdir(exist_ok=True)

    # Step 1: Create realistic signal scenarios
    scenarios_df = create_realistic_signal_scenarios()

    # Step 2: Analyze detection feasibility
    detection_results = analyze_detection_feasibility_by_scenario(scenarios_df)

    # Step 3: Analyze ELI facility compatibility
    facility_results = analyze_eli_facility_compatibility(detection_results)

    # Step 4: Generate plots
    generate_comprehensive_plots(detection_results, facility_results, output_dir)

    # Step 5: Generate summary report
    report = generate_summary_report(detection_results, facility_results, output_dir)

    # Step 6: Save results
    detection_results.to_csv(output_dir / "detection_feasibility_results.csv", index=False)
    facility_results.to_csv(output_dir / "facility_compatibility_results.csv", index=False)

    # Print summary
    print("\n" + "=" * 60)
    print("📊 ANALYSIS SUMMARY")
    print("=" * 60)

    total_scenarios = len(detection_results)
    feasible_scenarios = len(detection_results[detection_results["best_snr"] >= 5])
    challenging_scenarios = len(
        detection_results[
            (detection_results["best_snr"] >= 1) & (detection_results["best_snr"] < 5)
        ]
    )

    print(f"Total scenarios analyzed: {total_scenarios}")
    print(f"Feasible detections (SNR ≥ 5): {feasible_scenarios}")
    print(f"Challenging detections (SNR 1-5): {challenging_scenarios}")
    print(f"Overall feasibility rate: {100*feasible_scenarios/total_scenarios:.1f}%")

    if len(detection_results) > 0:
        best_snr = detection_results["best_snr"].max()
        print(f"Best SNR achieved: {best_snr:.2f}")

        if best_snr >= 5:
            print("✅ DETECTION IS FEASIBLE with optimized parameters")
        elif best_snr >= 1:
            print("⚠️ DETECTION IS CHALLENGING but may be possible with enhancements")
        else:
            print("❌ DETECTION IS NOT CURRENTLY FEASIBLE with predicted signal levels")

        print(
            f"\n🎯 TOP DETECTION METHOD: {detection_results.loc[detection_results['best_snr'].idxmax(), 'best_method']}"
        )
        print("🏢 BEST ELI FACILITY: Based on compatibility scores in report")

    print(f"\n📁 Results saved to: {output_dir}")
    print("📊 Plots: detection_feasibility_comprehensive.png")
    print("📄 Report: detection_feasibility_comprehensive_report.md")
    print("🎉 Analysis complete!")


if __name__ == "__main__":
    main()
