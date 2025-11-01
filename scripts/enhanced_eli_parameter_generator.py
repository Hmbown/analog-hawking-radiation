#!/usr/bin/env python3
"""
ELI-Compliant Enhanced Parameter Generator for Analog Hawking Radiation Analysis

This script generates diverse, physically realistic parameter configurations
with strict ELI facility compatibility constraints and experimental feasibility scoring.

Author: Claude Analysis Assistant
Date: November 2025
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from scipy.stats import qmc, uniform, loguniform, norm
from scipy.constants import c, e, epsilon_0, k, m_e, m_p

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from analog_hawking.facilities.eli_capabilities import (
    ELICapabilities, ELIFacility, get_eli_capabilities
)
from analog_hawking.analysis.gradient_sweep import run_single_configuration
from analog_hawking.config.thresholds import Thresholds


class ELICompliantParameterGenerator:
    """Parameter generator with strict ELI facility constraints"""

    def __init__(self, facility: Optional[str] = None):
        self.eli = get_eli_capabilities()
        self.target_facility = None
        if facility:
            facility_map = {
                "beamlines": ELIFacility.ELI_BEAMLINES,
                "np": ELIFacility.ELI_NP,
                "alps": ELIFacility.ELI_ALPS,
                "eli-beamlines": ELIFacility.ELI_BEAMLINES,
                "eli-np": ELIFacility.ELI_NP,
                "eli-alps": ELIFacility.ELI_ALPS
            }

            if facility.lower() in facility_map:
                self.target_facility = facility_map[facility.lower()]
            else:
                print(f"Warning: Unknown facility '{facility}', using all facilities")
                self.target_facility = None

        # Define ELI-compliant parameter ranges
        self.parameter_ranges = self._initialize_eli_ranges()

    def _initialize_eli_ranges(self) -> Dict[str, Any]:
        """Initialize ELI-compliant parameter ranges"""

        # Base physics parameters (always included)
        base_ranges = {
            # Additional physics parameters
            "plasma_density": {
                "min": 1e23,   # Optimal for sonic horizon formation
                "max": 1e25,   # Solid density regime
                "log_scale": True
            },
            "magnetic_field": {
                "min": 0.0,    # No field baseline
                "max": 50.0,   # Realistic laboratory fields
                "log_scale": False
            },
            "temperature": {
                "min": 1e3,    # Cold plasma start
                "max": 1e5,    # Warm plasma
                "log_scale": True
            },
            "grid_size": {
                "min": 20,     # Minimum for physics accuracy
                "max": 100,    # Maximum for computational efficiency
                "log_scale": False
            }
        }

        if self.target_facility:
            # Facility-specific ranges + base physics parameters
            facility_config = self.eli.facility_constraints[self.target_facility]
            facility_ranges = {
                "laser_intensity": {
                    "min": facility_config["max_intensity_W_cm2"] / 1e4 * 0.01,  # 1% of max
                    "max": facility_config["max_intensity_W_cm2"] / 1e4 * 0.8,    # 80% of max
                    "log_scale": True
                },
                "wavelength": {
                    "min": facility_config["wavelength_range_nm"][0],
                    "max": facility_config["wavelength_range_nm"][1],
                    "log_scale": False
                },
                "pulse_duration": {
                    "min": facility_config["pulse_duration_range_fs"][0],
                    "max": facility_config["pulse_duration_range_fs"][1],
                    "log_scale": False
                },
                "repetition_rate": {
                    "min": facility_config["repetition_rate_limits_Hz"][0],
                    "max": facility_config["repetition_rate_limits_Hz"][1],
                    "log_scale": True
                }
            }
            return {**facility_ranges, **base_ranges}
        else:
            # Unified optimal ranges for analog Hawking experiments
            unified_ranges = {
                "laser_intensity": {
                    "min": 1e19,   # Conservative minimum for plasma mirror formation
                    "max": 1e22,   # Well within ELI capabilities
                    "log_scale": True
                },
                "wavelength": {
                    "min": 800,    # Ti:Sapphire standard
                    "max": 810,    # Small range for compatibility
                    "log_scale": False
                },
                "pulse_duration": {
                    "min": 100,    # Good balance for intensity and temporal resolution
                    "max": 200,    # Compatible with 10 PW systems
                    "log_scale": False
                },
                "repetition_rate": {
                    "min": 0.1,    # Balance data collection and intensity
                    "max": 10.0,   # High rep rate for good statistics
                    "log_scale": True
                }
            }
            return {**unified_ranges, **base_ranges}

    def generate_parameter_set(self, n_samples: int = 100,
                             design_type: str = "sobol") -> pd.DataFrame:
        """Generate ELI-compliant parameter sets"""

        print(f"🎯 Generating {n_samples} ELI-compliant parameter sets...")
        if self.target_facility:
            print(f"🏢 Target facility: {self.target_facility.value}")

        # Create quasi-random design
        if design_type.lower() == "sobol":
            sampler = qmc.Sobol(d=len(self.parameter_ranges), scramble=True)
        elif design_type.lower() == "halton":
            sampler = qmc.Halton(d=len(self.parameter_ranges), scramble=True)
        else:
            sampler = qmc.LatinHypercube(d=len(self.parameter_ranges))

        # Generate samples
        sample = sampler.random(n=n_samples)

        # Map samples to parameter ranges
        param_names = list(self.parameter_ranges.keys())
        df = pd.DataFrame()

        for i, param_name in enumerate(param_names):
            param_config = self.parameter_ranges[param_name]
            min_val = param_config["min"]
            max_val = param_config["max"]

            if param_config["log_scale"]:
                # Logarithmic sampling
                log_min = np.log10(min_val)
                log_max = np.log10(max_val)
                values = 10 ** (log_min + sample[:, i] * (log_max - log_min))
            else:
                # Linear sampling
                values = min_val + sample[:, i] * (max_val - min_val)

            df[param_name] = values

        # Add ELI facility compatibility information
        df["eli_feasibility_score"] = 0.0
        df["best_eli_facility"] = ""
        df["eli_compatible_systems"] = ""
        df["intensity_margin"] = 0.0

        for idx, row in df.iterrows():
            # Calculate ELI feasibility
            intensity_W_m2 = row["laser_intensity"]
            wavelength_nm = row["wavelength"]
            pulse_duration_fs = row["pulse_duration"]

            feasibility = self.eli.calculate_feasibility_score(
                intensity_W_m2 / 1e4,  # Convert to W/cm²
                wavelength_nm,
                pulse_duration_fs,
                self.target_facility
            )

            df.loc[idx, "eli_feasibility_score"] = feasibility["score"]
            df.loc[idx, "best_eli_facility"] = feasibility.get("facility", "Unknown")
            df.loc[idx, "eli_compatible_systems"] = ",".join(feasibility.get("all_compatible_systems", []))
            df.loc[idx, "intensity_margin"] = feasibility.get("intensity_margin", 1.0)

        # Filter out infeasible configurations
        feasible_df = df[df["eli_feasibility_score"] > 0.3].copy()

        print(f"✅ Generated {len(feasible_df)} feasible configurations out of {n_samples} samples")
        print(f"📊 Average feasibility score: {feasible_df['eli_feasibility_score'].mean():.3f}")

        # Generate physics metrics
        feasible_df = self._calculate_physics_metrics(feasible_df)

        return feasible_df

    def _calculate_physics_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate relevant physics metrics for parameter sets"""

        print("📐 Calculating physics metrics...")

        # Calculate normalized vector potential a0
        for idx, row in df.iterrows():
            intensity_W_m2 = row["laser_intensity"]
            wavelength_m = row["wavelength"] * 1e-9

            # a0 = e*E/(m_e*c*omega) = sqrt(2*I*ε₀)/(m_e*c*omega)
            omega = 2 * np.pi * c / wavelength_m
            E_field = np.sqrt(2 * intensity_W_m2 / (c * epsilon_0))
            a0 = e * E_field / (m_e * c * omega)

            df.loc[idx, "normalized_vector_potential"] = a0

            # Calculate plasma frequency
            plasma_density = row["plasma_density"]
            omega_p = np.sqrt(plasma_density * e**2 / (epsilon_0 * m_e))
            df.loc[idx, "plasma_frequency_Hz"] = omega_p

            # Calculate critical density
            critical_density = epsilon_0 * m_e * omega**2 / e**2
            df.loc[idx, "critical_density_m3"] = critical_density
            df.loc[idx, "density_overcritical_ratio"] = plasma_density / critical_density

            # Estimate sound speed (assuming ion temperature)
            T_e = row["temperature"]
            c_s = np.sqrt(k * T_e / m_p)  # Ion acoustic speed
            df.loc[idx, "sound_speed_m_s"] = c_s

            # Calculate characteristic velocity (from ponderomotive force)
            v_char = c * a0 / np.sqrt(1 + a0**2)
            df.loc[idx, "characteristic_velocity_m_s"] = v_char

            # Estimate horizon formation potential
            horizon_potential = min(1.0, v_char / c_s) if c_s > 0 else 0.0
            df.loc[idx, "horizon_formation_potential"] = horizon_potential

        return df

    def rank_by_experimental_feasibility(self, df: pd.DataFrame) -> pd.DataFrame:
        """Rank parameter sets by experimental feasibility"""

        print("🏆 Ranking configurations by experimental feasibility...")

        # Define scoring criteria
        def calculate_experimental_score(row):
            score = 0.0

            # ELI compatibility (40% weight)
            score += 0.4 * row["eli_feasibility_score"]

            # Physics quality (30% weight)
            a0 = row["normalized_vector_potential"]
            horizon_potential = row["horizon_formation_potential"]

            # Good a0 range (1-50 for relativistic but not extreme)
            if 1 <= a0 <= 50:
                a0_score = 1.0
            elif 0.1 <= a0 < 1 or 50 < a0 <= 100:
                a0_score = 0.7
            else:
                a0_score = 0.3

            score += 0.15 * a0_score
            score += 0.15 * horizon_potential

            # Practical considerations (20% weight)
            rep_rate = row["repetition_rate"]
            if rep_rate >= 1.0:
                rep_score = 1.0  # Good for statistics
            elif rep_rate >= 0.1:
                rep_score = 0.7  # Reasonable
            else:
                rep_score = 0.3  # Low statistics

            score += 0.2 * rep_score

            # Computational feasibility (10% weight)
            grid_size = row["grid_size"]
            if grid_size <= 50:
                comp_score = 1.0  # Fast computation
            elif grid_size <= 100:
                comp_score = 0.7  # Moderate computation
            else:
                comp_score = 0.3  # Heavy computation

            score += 0.1 * comp_score

            return score

        df["experimental_feasibility_score"] = df.apply(calculate_experimental_score, axis=1)

        # Sort by experimental feasibility
        ranked_df = df.sort_values("experimental_feasibility_score", ascending=False).copy()

        print(f"📊 Top 10 configurations by experimental feasibility:")
        for i, (idx, row) in enumerate(ranked_df.head(10).iterrows(), 1):
            print(f"   {i:2d}. Score: {row['experimental_feasibility_score']:.3f} | "
                  f"Facility: {row['best_eli_facility']} | "
                  f"I: {row['laser_intensity']:.1e} W/m² | "
                  f"a₀: {row['normalized_vector_potential']:.1f}")

        return ranked_df

    def generate_optimized_configurations(self, n_top: int = 20) -> Dict[str, Any]:
        """Generate optimized configurations for analog Hawking experiments"""

        print(f"🎯 Generating {n_top} optimized ELI-compliant configurations...")

        # Generate large parameter space
        df = self.generate_parameter_set(n_samples=500, design_type="sobol")

        # Rank by experimental feasibility
        ranked_df = self.rank_by_experimental_feasibility(df)

        # Select top configurations
        top_configs = ranked_df.head(n_top).copy()

        # Generate detailed recommendations for each configuration
        recommendations = []
        for idx, row in top_configs.iterrows():
            config = {
                "parameters": {
                    "laser_intensity_W_m2": float(row["laser_intensity"]),
                    "wavelength_nm": float(row["wavelength"]),
                    "pulse_duration_fs": float(row["pulse_duration"]),
                    "repetition_rate_Hz": float(row["repetition_rate"]),
                    "plasma_density_m3": float(row["plasma_density"]),
                    "magnetic_field_T": float(row["magnetic_field"]),
                    "temperature_K": float(row["temperature"]),
                    "grid_size": int(row["grid_size"])
                },
                "eli_assessment": {
                    "feasibility_score": float(row["eli_feasibility_score"]),
                    "best_facility": str(row["best_eli_facility"]),
                    "compatible_systems": str(row["eli_compatible_systems"]),
                    "intensity_margin": float(row["intensity_margin"])
                },
                "physics_metrics": {
                    "normalized_vector_potential": float(row["normalized_vector_potential"]),
                    "plasma_frequency_Hz": float(row["plasma_frequency_Hz"]),
                    "density_overcritical_ratio": float(row["density_overcritical_ratio"]),
                    "sound_speed_m_s": float(row["sound_speed_m_s"]),
                    "horizon_formation_potential": float(row["horizon_formation_potential"])
                },
                "experimental_score": float(row["experimental_feasibility_score"])
            }
            recommendations.append(config)

        return {
            "facility_target": self.target_facility.value if self.target_facility else "all_facilities",
            "total_configurations": len(recommendations),
            "parameter_ranges_used": self.parameter_ranges,
            "configurations": recommendations,
            "summary_statistics": {
                "avg_feasibility_score": float(top_configs["eli_feasibility_score"].mean()),
                "avg_experimental_score": float(top_configs["experimental_feasibility_score"].mean()),
                "intensity_range": [
                    float(top_configs["laser_intensity"].min()),
                    float(top_configs["laser_intensity"].max())
                ],
                "facility_distribution": top_configs["best_eli_facility"].value_counts().to_dict()
            }
        }


def main():
    """Main execution function"""

    parser = argparse.ArgumentParser(
        description="Generate ELI-compliant parameters for analog Hawking radiation experiments"
    )
    parser.add_argument(
        "--facility",
        choices=["beamlines", "np", "alps"],
        help="Target ELI facility (beamlines=ELI-Beamlines, np=ELI-NP, alps=ELI-ALPS)"
    )
    parser.add_argument(
        "--n-configs",
        type=int,
        default=20,
        help="Number of optimized configurations to generate"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="eli_compliant_configurations.json",
        help="Output file for configurations"
    )
    parser.add_argument(
        "--csv-output",
        type=str,
        help="Optional CSV output for detailed parameter sets"
    )

    args = parser.parse_args()

    # Initialize generator
    generator = ELICompliantParameterGenerator(facility=args.facility)

    # Generate optimized configurations
    print("🚀 STARTING ELI-COMPLIANT PARAMETER GENERATION")
    print("=" * 60)

    results = generator.generate_optimized_configurations(n_top=args.n_configs)

    # Save results
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n💾 Configurations saved to: {args.output}")

    # Save CSV if requested
    if args.csv_output:
        # Convert configurations to DataFrame
        config_data = []
        for config in results["configurations"]:
            row = {**config["parameters"], **config["eli_assessment"],
                   **config["physics_metrics"], "experimental_score": config["experimental_score"]}
            config_data.append(row)

        df = pd.DataFrame(config_data)
        df.to_csv(args.csv_output, index=False)
        print(f"📊 Detailed parameters saved to: {args.csv_output}")

    # Print summary
    print(f"\n📋 GENERATION SUMMARY:")
    print(f"   Target facility: {results['facility_target']}")
    print(f"   Configurations generated: {results['total_configurations']}")
    print(f"   Average ELI feasibility: {results['summary_statistics']['avg_feasibility_score']:.3f}")
    print(f"   Average experimental score: {results['summary_statistics']['avg_experimental_score']:.3f}")
    print(f"   Intensity range: {results['summary_statistics']['intensity_range'][0]:.1e} - "
          f"{results['summary_statistics']['intensity_range'][1]:.1e} W/m²")

    print(f"\n🏢 FACILITY DISTRIBUTION:")
    for facility, count in results['summary_statistics']['facility_distribution'].items():
        print(f"   {facility}: {count} configurations")

    return 0


if __name__ == "__main__":
    exit(main())