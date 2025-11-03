#!/usr/bin/env python3
"""
ELI-Compliant Analysis Pipeline for Analog Hawking Radiation

This script provides a comprehensive analysis pipeline that incorporates ELI facility
constraints, experimental feasibility assessment, and facility-specific optimization
for analog Hawking radiation experiments.

Author: Claude Analysis Assistant
Date: November 2025
"""

import argparse
import json
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from analog_hawking.analysis.gradient_sweep import run_single_configuration
from analog_hawking.facilities.eli_capabilities import (
    ELIFacility,
    get_eli_capabilities,
    validate_intensity_range,
)
from analog_hawking.facilities.experimental_feasibility import assess_experimental_feasibility


class ELICompliantAnalysisPipeline:
    """Comprehensive analysis pipeline with ELI facility constraints"""

    def __init__(self, facility: Optional[str] = None):
        self.eli = get_eli_capabilities()
        self.target_facility = None
        if facility:
            facility_map = {
                "beamlines": ELIFacility.ELI_BEAMLINES,
                "np": ELIFacility.ELI_NP,
                "alps": ELIFacility.ELI_ALPS
            }
            self.target_facility = facility_map.get(facility.lower())

        self.results = []
        self.analysis_log = []

    def run_parameter_sweep(self,
                          n_samples: int = 100,
                          intensity_range: Optional[Tuple[float, float]] = None,
                          density_range: Optional[Tuple[float, float]] = None,
                          **kwargs) -> pd.DataFrame:
        """
        Run ELI-compliant parameter sweep with facility constraints

        Args:
            n_samples: Number of parameter samples to evaluate
            intensity_range: Laser intensity range (W/m²)
            density_range: Plasma density range (m⁻³)
            **kwargs: Additional parameter constraints

        Returns:
            DataFrame with sweep results including ELI compatibility scores
        """

        print("🚀 STARTING ELI-COMPLIANT PARAMETER SWEEP")
        print(f"   Target facility: {self.target_facility.value if self.target_facility else 'All facilities'}")
        print(f"   Sample size: {n_samples}")

        # Define ELI-compliant parameter ranges
        if self.target_facility:
            facility_config = self.eli.facility_constraints[self.target_facility]
            if intensity_range is None:
                intensity_range = (
                    facility_config["max_intensity_W_cm2"] / 1e4 * 0.01,  # 1% of max
                    facility_config["max_intensity_W_cm2"] / 1e4 * 0.8     # 80% of max
                )
        else:
            if intensity_range is None:
                intensity_range = (1e19, 1e22)  # Safe range for all facilities

        if density_range is None:
            density_range = (1e23, 1e25)  # Optimal for horizon formation

        # Generate parameter samples
        np.random.seed(42)  # For reproducibility
        samples = self._generate_parameter_samples(
            n_samples, intensity_range, density_range, **kwargs
        )

        print(f"📊 Generated {len(samples)} parameter samples")

        # Run analysis with ELI compatibility checks
        results = []
        feasible_count = 0

        for i, sample in enumerate(samples):
            if i % 10 == 0:
                print(f"   Processing sample {i+1}/{len(samples)}...")

            # Validate ELI compatibility first
            eli_validation = validate_intensity_range(
                sample["laser_intensity"], self.target_facility
            )

            if not eli_validation["valid"]:
                continue  # Skip infeasible configurations

            feasible_count += 1

            # Run physics analysis
            try:
                physics_result = run_single_configuration(
                    laser_intensity=sample["laser_intensity"],
                    plasma_density=sample["plasma_density"],
                    **{k: v for k, v in sample.items()
                       if k not in ["laser_intensity", "plasma_density"]}
                )

                # Combine with ELI assessment
                sample_result = {
                    **sample,
                    **physics_result,
                    "eli_compatible": True,
                    "eli_feasibility_score": 1.0 if eli_validation["feasibility_level"].startswith("HIGH") else 0.7,
                    "compatible_facilities": eli_validation.get("compatible_facilities", [])
                }

                # Add experimental feasibility assessment
                feasibility = assess_experimental_feasibility(sample,
                    self.target_facility.value if self.target_facility else None)
                sample_result["experimental_feasibility"] = feasibility.overall_feasibility_score
                sample_result["detection_probability"] = feasibility.detection_probability
                sample_result["required_shots"] = feasibility.required_shots
                sample_result["experiment_time_hours"] = feasibility.estimated_experiment_time_hours

                results.append(sample_result)

            except Exception as e:
                self.analysis_log.append(
                    f"Physics analysis failed for sample {i}: {str(e)}"
                )
                continue

        # Create results DataFrame
        df = pd.DataFrame(results)

        print("✅ Analysis completed:")
        print(f"   Total samples: {len(samples)}")
        print(f"   ELI-compatible: {feasible_count}")
        print(f"   Successfully analyzed: {len(results)}")
        print(f"   Success rate: {100*len(results)/len(samples):.1f}%")

        return df

    def run_facility_comparison(self, parameters: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Compare the same parameters across different ELI facilities

        Args:
            parameters: List of parameter configurations to compare

        Returns:
            DataFrame with facility comparison results
        """

        print("🏢 RUNNING FACILITY COMPARISON ANALYSIS")
        print(f"   Configurations: {len(parameters)}")
        print(f"   Facilities: {len(self.eli.laser_systems)}")

        results = []

        for i, params in enumerate(parameters):
            print(f"   Analyzing configuration {i+1}/{len(parameters)}...")

            for facility in ELIFacility:
                # Get facility-specific feasibility
                intensity_W_cm2 = params["laser_intensity_W_m2"] / 1e4
                wavelength_nm = params.get("wavelength_nm", 800)
                pulse_duration_fs = params.get("pulse_duration_fs", 150)

                eli_feasibility = self.eli.calculate_feasibility_score(
                    intensity_W_cm2, wavelength_nm, pulse_duration_fs, facility
                )

                # Experimental feasibility assessment
                feasibility = assess_experimental_feasibility(params, facility.value)

                result = {
                    "configuration_id": f"config_{i+1}",
                    "facility": facility.value,
                    **params,
                    "eli_feasibility_score": eli_feasibility["score"],
                    "best_system": eli_feasibility.get("best_system", "None"),
                    "intensity_margin": eli_feasibility.get("intensity_margin", 1.0),
                    "experimental_feasibility": feasibility.overall_feasibility_score,
                    "detection_probability": feasibility.detection_probability,
                    "required_shots": feasibility.required_shots,
                    "experiment_time_hours": feasibility.estimated_experiment_time_hours,
                    "overall_score": eli_feasibility["score"] * 0.5 + feasibility.overall_feasibility_score * 0.5
                }

                results.append(result)

        df = pd.DataFrame(results)
        return df

    def generate_optimization_report(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate comprehensive optimization report with ELI constraints

        Args:
            df: Analysis results DataFrame

        Returns:
            Optimization report with recommendations
        """

        print("📈 GENERATING OPTIMIZATION REPORT")

        if df.empty:
            return {
                "status": "error",
                "message": "No feasible configurations found"
            }

        # Find optimal configurations
        if "overall_score" in df.columns:
            best_configs = df.nlargest(5, "overall_score")
        else:
            # Fallback to experimental feasibility
            best_configs = df.nlargest(5, "experimental_feasibility")

        # Parameter sensitivity analysis
        sensitivity = self._analyze_parameter_sensitivity(df)

        # Facility recommendations
        facility_recommendations = self._generate_facility_recommendations(df)

        # Risk assessment
        risk_assessment = self._assess_experimental_risks(df)

        # Resource requirements
        resource_requirements = self._estimate_resource_requirements(df)

        report = {
            "summary": {
                "total_configurations": len(df),
                "best_score": best_configs["overall_score"].iloc[0] if "overall_score" in best_configs.columns else 0.0,
                "feasible_configurations": len(df[df["experimental_feasibility"] > 0.6]),
                "average_detection_probability": df["detection_probability"].mean(),
                "average_experiment_time": df["experiment_time_hours"].mean()
            },
            "optimal_configurations": best_configs.to_dict("records"),
            "parameter_sensitivity": sensitivity,
            "facility_recommendations": facility_recommendations,
            "risk_assessment": risk_assessment,
            "resource_requirements": resource_requirements,
            "analysis_log": self.analysis_log
        }

        return report

    def _generate_parameter_samples(self,
                                 n_samples: int,
                                 intensity_range: Tuple[float, float],
                                 density_range: Tuple[float, float],
                                 **kwargs) -> List[Dict[str, Any]]:
        """Generate parameter samples within ELI constraints"""

        samples = []

        for i in range(n_samples):
            # Log-uniform sampling for intensity and density
            intensity = 10 ** np.random.uniform(
                np.log10(intensity_range[0]), np.log10(intensity_range[1])
            )
            density = 10 ** np.random.uniform(
                np.log10(density_range[0]), np.log10(density_range[1])
            )

            # Default parameter values
            sample = {
                "laser_intensity": intensity,
                "plasma_density": density,
                "temperature": 1e4,  # 10,000 K
                "magnetic_field": 0.0,  # No magnetic field baseline
                "wavelength": 800e-9,  # 800 nm
                "pulse_duration": 150e-15,  # 150 fs
                "grid_size": 50
            }

            # Add any additional parameters
            sample.update(kwargs)
            samples.append(sample)

        return samples

    def _analyze_parameter_sensitivity(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze parameter sensitivity for optimization"""

        if len(df) < 10:
            return {"error": "Insufficient data for sensitivity analysis"}

        sensitivity = {}

        # Intensity sensitivity
        if "laser_intensity" in df.columns and "experimental_feasibility" in df.columns:
            intensity_corr = np.corrcoef(np.log10(df["laser_intensity"]), df["experimental_feasibility"])[0, 1]
            sensitivity["intensity_sensitivity"] = {
                "correlation": intensity_corr,
                "optimal_range": [
                    df["laser_intensity"].quantile(0.25),
                    df["laser_intensity"].quantile(0.75)
                ]
            }

        # Density sensitivity
        if "plasma_density" in df.columns and "experimental_feasibility" in df.columns:
            density_corr = np.corrcoef(np.log10(df["plasma_density"]), df["experimental_feasibility"])[0, 1]
            sensitivity["density_sensitivity"] = {
                "correlation": density_corr,
                "optimal_range": [
                    df["plasma_density"].quantile(0.25),
                    df["plasma_density"].quantile(0.75)
                ]
            }

        return sensitivity

    def _generate_facility_recommendations(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Generate facility-specific recommendations"""

        recommendations = []

        if "facility" in df.columns:
            # Group by facility
            facility_stats = df.groupby("facility").agg({
                "experimental_feasibility": "mean",
                "detection_probability": "mean",
                "experiment_time_hours": "mean",
                "overall_score": "mean" if "overall_score" in df.columns else lambda x: 0
            }).sort_values("overall_score", ascending=False)

            for facility, stats in facility_stats.iterrows():
                recommendation = {
                    "facility": facility,
                    "average_feasibility": stats["experimental_feasibility"],
                    "average_detection_probability": stats["detection_probability"],
                    "average_experiment_time": stats["experiment_time_hours"],
                    "recommendation": self._get_facility_recommendation_text(facility, stats)
                }
                recommendations.append(recommendation)

        else:
            # General recommendations based on parameter ranges
            avg_intensity = df["laser_intensity"].mean()
            if avg_intensity > 1e22:
                recommendations.append({
                    "facility": "ELI-Beamlines",
                    "reason": "High intensity requirements (>10^22 W/m²)",
                    "confidence": 0.9
                })
            elif avg_intensity > 1e21:
                recommendations.append({
                    "facility": "ELI-NP",
                    "reason": "Moderate-high intensity with nuclear physics capability",
                    "confidence": 0.8
                })
            else:
                recommendations.append({
                    "facility": "ELI-ALPS",
                    "reason": "Moderate intensity suitable for high-repetition rate experiments",
                    "confidence": 0.7
                })

        return recommendations

    def _get_facility_recommendation_text(self, facility: str, stats: pd.Series) -> str:
        """Get recommendation text for facility"""

        feasibility = stats["experimental_feasibility"]
        detection_prob = stats["detection_probability"]
        exp_time = stats["experiment_time_hours"]

        if feasibility > 0.8 and detection_prob > 0.6:
            return f"EXCELLENT: High feasibility ({feasibility:.2f}) and detection probability ({detection_prob:.2f})"
        elif feasibility > 0.6:
            return f"GOOD: Moderate feasibility ({feasibility:.2f}) with reasonable experiment time ({exp_time:.1f} hours)"
        else:
            return f"CHALLENGING: Lower feasibility ({feasibility:.2f}) - consider optimization"

    def _assess_experimental_risks(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Assess experimental risks across configurations"""

        risks = {
            "technical_risks": [],
            "physics_risks": [],
            "operational_risks": []
        }

        # Check for high-intensity configurations
        if "laser_intensity" in df.columns:
            max_intensity = df["laser_intensity"].max()
            if max_intensity > 1e23:
                risks["technical_risks"].append(
                    "High intensity (>10^23 W/m²) may pose equipment damage risks"
                )

        # Check for low detection probability
        if "detection_probability" in df.columns:
            low_prob_count = len(df[df["detection_probability"] < 0.3])
            if low_prob_count > len(df) * 0.5:
                risks["physics_risks"].append(
                    "More than 50% of configurations have low detection probability (<30%)"
                )

        # Check for long experiment times
        if "experiment_time_hours" in df.columns:
            long_exp_count = len(df[df["experiment_time_hours"] > 100])
            if long_exp_count > 0:
                risks["operational_risks"].append(
                    f"{long_exp_count} configurations require >100 hours of beam time"
                )

        return risks

    def _estimate_resource_requirements(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Estimate experimental resource requirements"""

        if df.empty:
            return {"error": "No data available"}

        requirements = {}

        # Beam time requirements
        if "experiment_time_hours" in df.columns:
            requirements["beam_time_hours"] = {
                "minimum": df["experiment_time_hours"].min(),
                "average": df["experiment_time_hours"].mean(),
                "maximum": df["experiment_time_hours"].max(),
                "recommended_allocation": df["experiment_time_hours"].quantile(0.75)
            }

        # Shot requirements
        if "required_shots" in df.columns:
            requirements["required_shots"] = {
                "minimum": int(df["required_shots"].min()),
                "average": int(df["required_shots"].mean()),
                "maximum": int(df["required_shots"].max())
            }

        # Personnel requirements (estimated)
        avg_exp_time = df["experiment_time_hours"].mean()
        requirements["personnel"] = {
            "experimental_scientists": 2,
            "laser_operators": 1,
            "target_specialists": 1,
            "diagnostic_engineers": 1,
            "safety_officers": 1,
            "estimated_total_person_hours": avg_exp_time * 6  # 6 people total
        }

        return requirements


def main():
    """Main execution function"""

    parser = argparse.ArgumentParser(
        description="Run ELI-compliant analysis for analog Hawking radiation experiments"
    )
    parser.add_argument(
        "--mode",
        choices=["sweep", "comparison", "single"],
        default="sweep",
        help="Analysis mode"
    )
    parser.add_argument(
        "--facility",
        choices=["beamlines", "np", "alps"],
        help="Target ELI facility"
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=100,
        help="Number of parameter samples"
    )
    parser.add_argument(
        "--intensity-range",
        nargs=2,
        type=float,
        help="Intensity range (W/m²)"
    )
    parser.add_argument(
        "--density-range",
        nargs=2,
        type=float,
        help="Plasma density range (m⁻³)"
    )
    parser.add_argument(
        "--config-file",
        type=str,
        help="JSON file with parameter configurations"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="eli_analysis_results.json",
        help="Output file for results"
    )
    parser.add_argument(
        "--csv-output",
        type=str,
        help="CSV output for detailed results"
    )

    args = parser.parse_args()

    # Initialize pipeline
    pipeline = ELICompliantAnalysisPipeline(facility=args.facility)

    # Run analysis based on mode
    if args.mode == "sweep":
        print("🔄 RUNNING PARAMETER SWEEP MODE")
        df = pipeline.run_parameter_sweep(
            n_samples=args.n_samples,
            intensity_range=tuple(args.intensity_range) if args.intensity_range else None,
            density_range=tuple(args.density_range) if args.density_range else None
        )

    elif args.mode == "comparison":
        print("🏢 RUNNING FACILITY COMPARISON MODE")
        if not args.config_file:
            print("❌ ERROR: --config-file required for comparison mode")
            return 1

        with open(args.config_file, 'r') as f:
            configurations = json.load(f)

        df = pipeline.run_facility_comparison(configurations)

    else:  # single mode
        print("🎯 RUNNING SINGLE CONFIGURATION MODE")
        if not args.config_file:
            print("❌ ERROR: --config-file required for single mode")
            return 1

        with open(args.config_file, 'r') as f:
            config = json.load(f)

        # Convert to list for compatibility
        df = pipeline.run_facility_comparison([config])

    # Generate report
    report = pipeline.generate_optimization_report(df)

    # Save results
    results = {
        "analysis_mode": args.mode,
        "target_facility": args.facility,
        "parameters": {
            "n_samples": args.n_samples,
            "intensity_range": args.intensity_range,
            "density_range": args.density_range
        },
        "data": df.to_dict("records") if not df.empty else [],
        "report": report
    }

    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n💾 Results saved to: {args.output}")

    # Save CSV if requested
    if args.csv_output and not df.empty:
        df.to_csv(args.csv_output, index=False)
        print(f"📊 Detailed data saved to: {args.csv_output}")

    # Print summary
    print("\n📋 ANALYSIS SUMMARY:")
    print(f"   Mode: {args.mode}")
    print(f"   Configurations analyzed: {len(df)}")
    print(f"   Best score: {report['summary']['best_score']:.3f}")
    print(f"   Average detection probability: {report['summary']['average_detection_probability']:.3f}")
    print(f"   Average experiment time: {report['summary']['average_experiment_time']:.1f} hours")

    return 0


if __name__ == "__main__":
    exit(main())