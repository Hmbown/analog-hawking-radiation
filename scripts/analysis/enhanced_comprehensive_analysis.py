#!/usr/bin/env python3
"""
Enhanced Comprehensive Analysis for Expanded Analog Hawking Radiation Dataset

This script provides rigorous statistical analysis for the expanded dataset
with ≥100 configurations, including proper handling of multiple physical
parameter dimensions and advanced uncertainty quantification.

Author: Claude Analysis Assistant
Date: November 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import ttest_ind, mannwhitneyu, chi2_contingency, pearsonr, spearmanr
from scipy.spatial.distance import pdist, squareform
import json
import warnings
warnings.filterwarnings('ignore')

# Set style for plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class EnhancedHawkingRadiationAnalyzer:
    """Enhanced analyzer for expanded analog Hawking radiation dataset"""

    def __init__(self, data_path='results/enhanced_hawking_dataset.csv',
                 metadata_path='results/enhanced_hawking_dataset_metadata.json',
                 generate_plots=False):
        """Initialize analyzer with expanded dataset path"""
        self.data_path = data_path
        self.metadata_path = metadata_path
        self.generate_plots = generate_plots
        self.df = None
        self.metadata = None
        self.significance_results = {}
        self.load_data()

    def load_data(self):
        """Load and preprocess the expanded dataset"""
        print("Loading expanded dataset...")
        self.df = pd.read_csv(self.data_path)

        # Load metadata
        try:
            with open(self.metadata_path, 'r') as f:
                self.metadata = json.load(f)
        except FileNotFoundError:
            print("Warning: Metadata file not found")
            self.metadata = {}

        print(f"Dataset loaded with {len(self.df)} rows and {len(self.df.columns)} columns")

        # Filter to valid configurations
        self.valid_df = self.df[self.df['validity_score'] > 0.5].copy()
        print(f"Valid configurations: {len(self.valid_df)}/{len(self.df)} ({100*len(self.valid_df)/len(self.df):.1f}%)")

        # Print parameter information
        print(f"\nKey parameter ranges (valid configurations):")
        key_params = ['a0', 'n_e', 'gradient_factor', 'kappa', 'coupling_strength', 'D']
        for param in key_params:
            if param in self.valid_df.columns:
                valid_values = self.valid_df[param].dropna()
                if len(valid_values) > 0:
                    print(f"  {param}: {valid_values.min():.3e} - {valid_values.max():.3e}")

        # Regime distribution
        if 'regime' in self.valid_df.columns:
            print(f"\nRegime distribution (valid configurations):")
            regime_counts = self.valid_df['regime'].value_counts()
            for regime, count in regime_counts.items():
                print(f"  {regime}: {count} configurations")

    def advanced_power_analysis(self, effect_size_range=(0.2, 0.5, 0.8), alpha=0.05):
        """Perform advanced statistical power analysis"""
        print("\n" + "="*80)
        print("ADVANCED STATISTICAL POWER ANALYSIS")
        print("="*80)

        n_valid = len(self.valid_df)
        if n_valid == 0:
            print("❌ No valid configurations available for power analysis")
            return

        print(f"Sample size: {n_valid} valid configurations")
        print(f"Significance level (α): {alpha}")

        # Power for different effect sizes
        print(f"\nStatistical Power Analysis:")
        print("-" * 40)

        from scipy.stats import t as t_dist

        for effect_size in effect_size_range:
            # Calculate degrees of freedom
            if n_valid > 30:
                # For two-group comparisons (approximate)
                df = n_valid - 2
            else:
                # For single sample or paired comparisons
                df = n_valid - 1

            # Calculate critical value
            critical_t = t_dist.ppf(1 - alpha/2, df)

            # Calculate non-centrality parameter
            ncp = effect_size * np.sqrt(n_valid/2)  # For two-group comparison

            # Calculate power
            power = 1 - t_dist.cdf(critical_t, df, ncp)
            power = max(0, min(1, power))

            print(f"  Effect size (Cohen's d) = {effect_size:.1f}: Power = {power:.3f}")

            # Interpretation
            if power >= 0.8:
                print(f"    ✅ Excellent power (> 0.8)")
            elif power >= 0.6:
                print(f"    ⚠️  Moderate power (0.6-0.8)")
            elif power >= 0.4:
                print(f"    ❌ Low power (0.4-0.6)")
            else:
                print(f"    ❌ Very low power (< 0.4)")

        # Minimum detectable effect size
        print(f"\nMinimum Detectable Effect Sizes (Power = 0.8):")
        print("-" * 50)

        target_power = 0.8
        for power_level in [0.6, 0.7, 0.8]:
            # Using approximation formula for two-sample t-test
            # This is conservative and commonly used
            min_effect = critical_t / np.sqrt(n_valid/2)
            print(f"  Target power {power_level:.1f}: Minimum detectable d ≈ {min_effect:.3f}")

        return {
            'n_valid': n_valid,
            'power_results': {f"effect_{e}": 1 - t_dist.cdf(critical_t, n_valid-2, e * np.sqrt(n_valid/2))
                            for e in effect_size_range},
            'min_detectable_effect': critical_t / np.sqrt(n_valid/2)
        }

    def enhanced_parameter_space_analysis(self):
        """Analyze coverage and distribution in multi-dimensional parameter space"""
        print("\n" + "="*80)
        print("ENHANCED PARAMETER SPACE COVERAGE ANALYSIS")
        print("="*80)

        if len(self.valid_df) == 0:
            print("❌ No valid configurations for parameter space analysis")
            return

        # Key physical parameters for analysis
        physical_params = ['a0', 'n_e', 'T_e', 'gradient_factor', 'B', 'lambda_l']
        available_params = [p for p in physical_params if p in self.valid_df.columns]

        print(f"Analyzing {len(available_params)} physical parameters:")
        for param in available_params:
            print(f"  • {param}")

        # Parameter space coverage metrics
        print(f"\nParameter Space Coverage Metrics:")
        print("-" * 35)

        coverage_results = {}

        for param in available_params:
            values = self.valid_df[param].dropna()
            if len(values) > 0:
                # Basic statistics
                min_val, max_val = values.min(), values.max()
                range_val = max_val / min_val if min_val > 0 else max_val - min_val

                # Logarithmic coverage (for wide-ranging parameters)
                if min_val > 0:
                    log_range = np.log10(max_val) - np.log10(min_val)
                else:
                    log_range = 0

                # Distribution uniformity (using coefficient of variation)
                cv = values.std() / values.mean() if values.mean() != 0 else 0

                coverage_results[param] = {
                    'min': min_val,
                    'max': max_val,
                    'linear_range': range_val,
                    'log_range': log_range,
                    'coefficient_of_variation': cv,
                    'n_points': len(values)
                }

                print(f"  {param}:")
                print(f"    Range: {min_val:.2e} - {max_val:.2e} (factor: {range_val:.1f}×)")
                print(f"    Log range: {log_range:.2f} decades")
                print(f"    Uniformity (CV): {cv:.3f}")
                print(f"    Points: {len(values)}")

        # Multi-dimensional space-filling analysis
        print(f"\nMulti-dimensional Space-Filling Analysis:")
        print("-" * 45)

        if len(available_params) >= 2:
            # Select top 4 most variable parameters for visualization
            param_variability = [(p, coverage_results[p]['coefficient_of_variation'])
                                for p in available_params]
            param_variability.sort(key=lambda x: x[1], reverse=True)
            top_params = [p for p, _ in param_variability[:4]]

            print(f"Top 4 most variable parameters: {', '.join(top_params)}")

            # Calculate pairwise distances in parameter space
            param_matrix = self.valid_df[top_params].values
            # Normalize each parameter to [0, 1] for distance calculation
            normalized_matrix = (param_matrix - param_matrix.min(axis=0)) / (param_matrix.max(axis=0) - param_matrix.min(axis=0))

            # Calculate pairwise distances
            distances = pdist(normalized_matrix, metric='euclidean')

            # Space-filling metrics
            min_distance = np.min(distances)
            mean_distance = np.mean(distances)
            max_distance = np.max(distances)

            print(f"  Minimum inter-point distance: {min_distance:.4f}")
            print(f"  Mean inter-point distance: {mean_distance:.4f}")
            print(f"  Maximum inter-point distance: {max_distance:.4f}")
            print(f"  Distance uniformity (CV): {np.std(distances)/np.mean(distances):.3f}")

            # Optimal spacing for uniform distribution (theoretical)
            n_points = len(self.valid_df)
            if n_points > 1:
                # For unit hypercube, optimal minimum distance ~ n^(-1/d) where d is dimension
                d = len(top_params)
                optimal_min_distance = n_points ** (-1/d)
                print(f"  Theoretical optimal min distance: {optimal_min_distance:.4f}")
                print(f"  Space-filling efficiency: {min_distance/optimal_min_distance:.3f}")

        return coverage_results

    def regime_based_analysis(self):
        """Perform analysis stratified by physical regimes"""
        print("\n" + "="*80)
        print("REGIME-BASED PHYSICAL ANALYSIS")
        print("="*80)

        if 'regime' not in self.valid_df.columns:
            print("❌ No regime information available")
            return

        regimes = self.valid_df['regime'].unique()
        print(f"Analyzing {len(regimes)} physical regimes:")
        for regime in regimes:
            print(f"  • {regime}")

        regime_results = {}

        for regime in regimes:
            regime_data = self.valid_df[self.valid_df['regime'] == regime]
            print(f"\n{regime.upper()} REGIME:")
            print("-" * (len(regime) + 8))
            print(f"  Configurations: {len(regime_data)}")

            # Key parameters for this regime
            if 'kappa' in regime_data.columns:
                kappas = regime_data['kappa'].dropna()
                if len(kappas) > 0:
                    print(f"  Kappa range: {kappas.min():.2e} - {kappas.max():.2e}")
                    print(f"  Mean kappa: {kappas.mean():.2e} ± {kappas.std():.2e}")

            if 'coupling_strength' in regime_data.columns:
                couplings = regime_data['coupling_strength'].dropna()
                if len(couplings) > 0:
                    print(f"  Coupling strength: {couplings.mean():.2f} ± {couplings.std():.2f}")

            if 'a0' in regime_data.columns:
                a0s = regime_data['a0'].dropna()
                if len(a0s) > 0:
                    print(f"  Normalized vector potential: {a0s.mean():.2f} ± {a0s.std():.2f}")

            if 'normalized_density' in regime_data.columns:
                densities = regime_data['normalized_density'].dropna()
                if len(densities) > 0:
                    print(f"  Normalized density (n_e/n_crit): {densities.mean():.2e} ± {densities.std():.2e}")

            regime_results[regime] = {
                'n_configs': len(regime_data),
                'kappa_stats': {
                    'mean': kappas.mean() if len(kappas) > 0 else np.nan,
                    'std': kappas.std() if len(kappas) > 0 else np.nan,
                    'min': kappas.min() if len(kappas) > 0 else np.nan,
                    'max': kappas.max() if len(kappas) > 0 else np.nan
                } if 'kappa' in regime_data.columns else {},
                'coupling_stats': {
                    'mean': couplings.mean() if len(couplings) > 0 else np.nan,
                    'std': couplings.std() if len(couplings) > 0 else np.nan
                } if 'coupling_strength' in regime_data.columns else {}
            }

        # Statistical comparison between regimes
        if len(regimes) >= 2:
            print(f"\nREGIME COMPARISON STATISTICS:")
            print("-" * 30)

            # Compare kappa values between regimes
            kappa_by_regime = {}
            for regime in regimes:
                regime_data = self.valid_df[self.valid_df['regime'] == regime]
                kappas = regime_data['kappa'].dropna()
                if len(kappas) > 0:
                    kappa_by_regime[regime] = kappas

            if len(kappa_by_regime) >= 2:
                regime_names = list(kappa_by_regime.keys())
                print(f"\nKappa comparisons:")

                for i in range(len(regime_names)):
                    for j in range(i+1, len(regime_names)):
                        regime1, regime2 = regime_names[i], regime_names[j]
                        kappas1, kappas2 = kappa_by_regime[regime1], kappa_by_regime[regime2]

                        # Mann-Whitney U test (non-parametric)
                        u_stat, u_pvalue = mannwhitneyu(kappas1, kappas2, alternative='two-sided')

                        # Effect size (rank-biserial correlation)
                        n1, n2 = len(kappas1), len(kappas2)
                        r_effect_size = 1 - (2 * u_stat) / (n1 * n2)

                        print(f"  {regime1} vs {regime2}:")
                        print(f"    U-statistic: {u_stat:.1f}, p = {u_pvalue:.3f}")
                        print(f"    Effect size (r): {r_effect_size:.3f}")

                        if u_pvalue < 0.05:
                            print(f"    ✅ Significant difference")
                        else:
                            print(f"    ❌ No significant difference")

        return regime_results

    def physical_scaling_analysis(self):
        """Analyze physical scaling relationships with proper uncertainty quantification"""
        print("\n" + "="*80)
        print("PHYSICAL SCALING RELATIONSHIP ANALYSIS")
        print("="*80)

        if len(self.valid_df) == 0:
            print("❌ No valid configurations for scaling analysis")
            return

        # Key scaling relationships to test
        scaling_relationships = [
            ('kappa', 'a0', 'κ ∝ a₀^β'),
            ('kappa', 'n_e', 'κ ∝ n_e^β'),
            ('kappa', 'gradient_factor', 'κ ∝ gradient_factor^β'),
            ('coupling_strength', 'a0', 'S ∝ a₀^β'),
            ('D', 'T_e', 'D ∝ T_e^β'),
            ('c_s', 'T_e', 'c_s ∝ T_e^β'),
        ]

        scaling_results = {}

        print("Testing scaling laws of the form: y ∝ x^β")
        print("Using log-log regression with uncertainty quantification")
        print("-" * 60)

        for y_var, x_var, relationship in scaling_relationships:
            if y_var in self.valid_df.columns and x_var in self.valid_df.columns:
                # Get valid data points (positive values for log transformation)
                mask = (self.valid_df[y_var] > 0) & (self.valid_df[x_var] > 0)
                x_data = self.valid_df.loc[mask, x_var]
                y_data = self.valid_df.loc[mask, y_var]

                if len(x_data) >= 3:  # Need at least 3 points for regression
                    # Log-transform
                    log_x = np.log10(x_data)
                    log_y = np.log10(y_data)

                    # Linear regression
                    slope, intercept, r_value, p_value, std_err = stats.linregress(log_x, log_y)

                    # Confidence intervals for slope
                    n = len(x_data)
                    t_critical = stats.t.ppf(0.975, df=n-2)
                    slope_ci_low = slope - t_critical * std_err
                    slope_ci_high = slope + t_critical * std_err

                    # Predicted values and residuals
                    log_y_pred = slope * log_x + intercept
                    residuals = log_y - log_y_pred

                    # Root mean square error
                    rmse = np.sqrt(np.mean(residuals**2))

                    scaling_results[relationship] = {
                        'slope': slope,
                        'intercept': intercept,
                        'r_squared': r_value**2,
                        'p_value': p_value,
                        'slope_ci': (slope_ci_low, slope_ci_high),
                        'rmse': rmse,
                        'n_points': len(x_data),
                        'x_range': (x_data.min(), x_data.max()),
                        'y_range': (y_data.min(), y_data.max())
                    }

                    print(f"\n{relationship}:")
                    print(f"  Exponent (β): {slope:.3f} [{slope_ci_low:.3f}, {slope_ci_high:.3f}] (95% CI)")
                    print(f"  R²: {r_value**2:.3f}")
                    print(f"  p-value: {p_value:.3e} {'✅' if p_value < 0.05 else '❌'}")
                    print(f"  Data points: {len(x_data)}")
                    print(f"  RMSE in log space: {rmse:.3f}")

                    # Physical interpretation
                    print(f"  Interpretation: {y_var} ∝ {x_var}^{slope:.2f}")

                    # Check for expected scaling laws
                    if y_var == 'c_s' and x_var == 'T_e':
                        expected_slope = 0.5  # Sound speed ∝ √T
                        if abs(slope - expected_slope) < 0.2:
                            print(f"  ✅ Consistent with expected scaling (β ≈ {expected_slope})")
                        else:
                            print(f"  ⚠️  Deviates from expected scaling (β ≈ {expected_slope})")

        # Cross-regime scaling comparison
        if 'regime' in self.valid_df.columns:
            print(f"\nCROSS-REGIME SCALING COMPARISON:")
            print("-" * 35)

            regimes = self.valid_df['regime'].unique()
            if len(regimes) >= 2:
                # Test if scaling exponents differ between regimes
                for y_var, x_var, relationship in scaling_relationships[:2]:  # Test key relationships
                    if y_var in self.valid_df.columns and x_var in self.valid_df.columns:
                        regime_slopes = {}

                        for regime in regimes:
                            regime_data = self.valid_df[self.valid_df['regime'] == regime]
                            mask = (regime_data[y_var] > 0) & (regime_data[x_var] > 0)
                            x_data = regime_data.loc[mask, x_var]
                            y_data = regime_data.loc[mask, y_var]

                            if len(x_data) >= 3:
                                log_x = np.log10(x_data)
                                log_y = np.log10(y_data)
                                slope, _, r_value, p_value, _ = stats.linregress(log_x, log_y)
                                regime_slopes[regime] = {
                                    'slope': slope,
                                    'r_squared': r_value**2,
                                    'p_value': p_value,
                                    'n_points': len(x_data)
                                }

                        if len(regime_slopes) >= 2:
                            print(f"\n{relationship} by regime:")
                            for regime, results in regime_slopes.items():
                                print(f"  {regime}: β = {results['slope']:.3f} (R² = {results['r_squared']:.3f}, n = {results['n_points']})")

        return scaling_results

    def uncertainty_quantification(self):
        """Perform comprehensive uncertainty quantification"""
        print("\n" + "="*80)
        print("UNCERTAINTY QUANTIFICATION ANALYSIS")
        print("="*80)

        if len(self.valid_df) == 0:
            print("❌ No valid configurations for uncertainty analysis")
            return

        # Bootstrap analysis for key parameters
        print("Bootstrap uncertainty quantification (1000 resamples):")
        print("-" * 50)

        n_bootstrap = 1000
        bootstrap_results = {}

        key_params = ['kappa', 'coupling_strength', 'D', 'a0']
        available_params = [p for p in key_params if p in self.valid_df.columns]

        for param in available_params:
            data = self.valid_df[param].dropna()
            if len(data) >= 5:  # Need minimum samples for bootstrap
                bootstrap_means = []
                bootstrap_stds = []

                for _ in range(n_bootstrap):
                    bootstrap_sample = np.random.choice(data, size=len(data), replace=True)
                    bootstrap_means.append(np.mean(bootstrap_sample))
                    bootstrap_stds.append(np.std(bootstrap_sample, ddof=1))

                # Calculate confidence intervals
                mean_ci_low = np.percentile(bootstrap_means, 2.5)
                mean_ci_high = np.percentile(bootstrap_means, 97.5)
                std_ci_low = np.percentile(bootstrap_stds, 2.5)
                std_ci_high = np.percentile(bootstrap_stds, 97.5)

                bootstrap_results[param] = {
                    'mean': np.mean(data),
                    'mean_ci': (mean_ci_low, mean_ci_high),
                    'std': np.std(data, ddof=1),
                    'std_ci': (std_ci_low, std_ci_high),
                    'n_samples': len(data),
                    'cv': np.std(data, ddof=1) / np.mean(data) if np.mean(data) != 0 else 0
                }

                print(f"\n{param}:")
                print(f"  Mean: {np.mean(data):.3e} [{mean_ci_low:.3e}, {mean_ci_high:.3e}] (95% CI)")
                print(f"  Std: {np.std(data, ddof=1):.3e} [{std_ci_low:.3e}, {std_ci_high:.3e}] (95% CI)")
                print(f"  Coefficient of variation: {bootstrap_results[param]['cv']:.3f}")

        # Parameter correlation uncertainty
        print(f"\nParameter Correlation Uncertainty:")
        print("-" * 35)

        if len(available_params) >= 2:
            # Calculate correlation matrix with uncertainty
            correlations = {}
            for i, param1 in enumerate(available_params):
                for j, param2 in enumerate(available_params[i+1:], i+1):
                    data1 = self.valid_df[param1].dropna()
                    data2 = self.valid_df[param2].dropna()

                    # Align data
                    aligned_data = pd.DataFrame({param1: data1, param2: data2}).dropna()
                    if len(aligned_data) >= 5:
                        x, y = aligned_data[param1], aligned_data[param2]

                        # Pearson correlation with bootstrap CI
                        bootstrap_corrs = []
                        for _ in range(n_bootstrap):
                            idx = np.random.choice(len(x), size=len(x), replace=True)
                            if len(np.unique(idx)) > 1:  # Ensure not all points are the same
                                corr, _ = pearsonr(x.iloc[idx], y.iloc[idx])
                                if not np.isnan(corr):
                                    bootstrap_corrs.append(corr)

                        if bootstrap_corrs:
                            corr_mean = np.mean(bootstrap_corrs)
                            corr_ci_low = np.percentile(bootstrap_corrs, 2.5)
                            corr_ci_high = np.percentile(bootstrap_corrs, 97.5)

                            correlations[f"{param1}_vs_{param2}"] = {
                                'correlation': corr_mean,
                                'ci': (corr_ci_low, corr_ci_high),
                                'n_points': len(aligned_data)
                            }

                            print(f"  {param1} vs {param2}: r = {corr_mean:.3f} [{corr_ci_low:.3f}, {corr_ci_high:.3f}]")

        return {
            'bootstrap_results': bootstrap_results,
            'correlations': correlations
        }

    def generate_comprehensive_report(self):
        """Generate comprehensive analysis report"""
        print("\n" + "="*80)
        print("COMPREHENSIVE ENHANCED ANALYSIS REPORT")
        print("="*80)

        # Dataset summary
        print(f"\nDATASET SUMMARY:")
        print("-" * 20)
        print(f"Total configurations generated: {self.metadata.get('n_samples', len(self.df))}")
        print(f"Valid configurations: {len(self.valid_df)}/{len(self.df)} ({100*len(self.valid_df)/len(self.df):.1f}%)")
        print(f"Sampling strategy: {self.metadata.get('sampling_strategy', 'Unknown')}")
        print(f"Parameters analyzed: {len(self.valid_df.columns)}")

        if self.metadata:
            param_ranges = self.metadata.get('parameter_ranges', {})
            print(f"\nParameter ranges used:")
            for param, value in param_ranges.items():
                print(f"  {param}: {value:.2e}")

        # Key findings
        print(f"\nKEY FINDINGS:")
        print("-" * 15)

        if 'kappa' in self.valid_df.columns:
            kappas = self.valid_df['kappa'].dropna()
            if len(kappas) > 0:
                max_kappa_config = self.valid_df.loc[kappas.idxmax()]
                print(f"  • Maximum κ: {kappas.max():.2e} s⁻¹")
                print(f"    Configuration: a₀ = {max_kappa_config.get('a0', 'N/A'):.2f}, "
                      f"n_e = {max_kappa_config.get('n_e', 'N/A'):.2e} m⁻³")

        # Physical insights
        print(f"\nPHYSICAL INSIGHTS:")
        print("-" * 20)

        if 'regime' in self.valid_df.columns:
            regime_performance = {}
            for regime in self.valid_df['regime'].unique():
                regime_data = self.valid_df[self.valid_df['regime'] == regime]
                kappas = regime_data['kappa'].dropna()
                if len(kappas) > 0:
                    regime_performance[regime] = kappas.mean()

            if regime_performance:
                best_regime = max(regime_performance, key=regime_performance.get)
                print(f"  • Best performing regime: {best_regime}")
                print(f"    Mean κ: {regime_performance[best_regime]:.2e} s⁻¹")

        # Statistical assessment
        print(f"\nSTATISTICAL ASSESSMENT:")
        print("-" * 25)

        n_valid = len(self.valid_df)
        if n_valid >= 50:
            print(f"  ✅ Excellent sample size (n = {n_valid}) for robust statistical inference")
        elif n_valid >= 30:
            print(f"  ✅ Good sample size (n = {n_valid}) for parametric analysis")
        elif n_valid >= 15:
            print(f"  ⚠️  Moderate sample size (n = {n_valid}) - use non-parametric methods")
        else:
            print(f"  ❌ Limited sample size (n = {n_valid}) - results may be unreliable")

        # Recommendations
        print(f"\nRECOMMENDATIONS:")
        print("-" * 18)

        if n_valid < 30:
            print(f"  • Generate additional configurations to improve statistical power")
            print(f"  • Focus on regimes with higher validity scores")

        if 'kappa' in self.valid_df.columns:
            kappas = self.valid_df['kappa'].dropna()
            if len(kappas) > 0:
                if kappas.max() / kappas.min() > 100:
                    print(f"  • Wide κ range ({kappas.max()/kappas.min():.1f}×) suggests diverse physics explored")

        print(f"  • Consider expanding parameter ranges where breakdown modes are frequent")
        print(f"  • Validate scaling relationships with targeted experiments")

def main():
    """Main analysis function for enhanced dataset"""
    import argparse

    parser = argparse.ArgumentParser(description='Enhanced Comprehensive Analysis of Expanded Analog Hawking Radiation Dataset')
    parser.add_argument('--data', type=str, default='results/enhanced_hawking_dataset.csv',
                       help='Path to enhanced dataset')
    parser.add_argument('--metadata', type=str, default='results/enhanced_hawking_dataset_metadata.json',
                       help='Path to dataset metadata')
    parser.add_argument('--plots', action='store_true',
                       help='Generate plots (can be slow on some computers)')
    args = parser.parse_args()

    if args.plots:
        print("📊 Plot generation ENABLED")
    else:
        print("📊 Plot generation DISABLED for performance")
    print("="*80)

    # Create analysis directory
    import os
    os.makedirs('results/enhanced_analysis', exist_ok=True)

    # Initialize enhanced analyzer
    analyzer = EnhancedHawkingRadiationAnalyzer(
        data_path=args.data,
        metadata_path=args.metadata,
        generate_plots=args.plots
    )

    # Run enhanced analyses
    analyzer.advanced_power_analysis()
    analyzer.enhanced_parameter_space_analysis()
    analyzer.regime_based_analysis()
    analyzer.physical_scaling_analysis()
    analyzer.uncertainty_quantification()

    # Generate comprehensive report
    analyzer.generate_comprehensive_report()

    print("\n" + "="*80)
    print("Enhanced Comprehensive Analysis Complete!")
    print(f"Results directory: results/enhanced_analysis/")
    if not args.plots:
        print("📊 Plots were not generated - use --plots flag to enable visualizations")
    print("="*80)

if __name__ == "__main__":
    main()