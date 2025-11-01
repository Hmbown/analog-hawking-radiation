#!/usr/bin/env python3
"""
Comprehensive Analysis of Analog Hawking Radiation Dataset
Analysis of hybrid_sweep.csv containing fluid vs hybrid model comparisons

Author: Claude Analysis Assistant
Date: November 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set style for plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class HawkingRadiationAnalyzer:
    """Comprehensive analyzer for analog Hawking radiation simulation data"""

    def __init__(self, data_path='results/hybrid_sweep.csv'):
        """Initialize analyzer with dataset path"""
        self.data_path = data_path
        self.df = None
        self.load_data()

    def load_data(self):
        """Load and preprocess the dataset"""
        print("Loading dataset...")
        self.df = pd.read_csv(self.data_path)
        print(f"Dataset loaded with {len(self.df)} rows and {len(self.df.columns)} columns")
        print("\nColumn names:")
        for i, col in enumerate(self.df.columns):
            print(f"  {i+1}. {col}")

        # Check for missing values
        print(f"\nMissing values: {self.df.isnull().sum().sum()}")

        # Basic statistics
        print(f"\nBasic dataset statistics:")
        print(f"  - Range of coupling_strength: {self.df['coupling_strength'].min():.3f} - {self.df['coupling_strength'].max():.3f}")
        print(f"  - Range of D: {self.df['D'].min():.2e} - {self.df['D'].max():.2e}")
        print(f"  - Range of ratio_fluid_over_hybrid: {self.df['ratio_fluid_over_hybrid'].min():.3f} - {self.df['ratio_fluid_over_hybrid'].max():.3f}")

    def statistical_characterization(self):
        """Perform comprehensive statistical analysis"""
        print("\n" + "="*60)
        print("STATISTICAL CHARACTERIZATION")
        print("="*60)

        # Descriptive statistics
        print("\nDescriptive Statistics:")
        stats_desc = self.df.describe()
        print(stats_desc.round(6))

        # Parameter distributions
        print("\nParameter Distributions:")
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Parameter Distributions in Analog Hawking Radiation Dataset', fontsize=16)

        params = ['coupling_strength', 'D', 'w_effective', 'kappa_mirror', 'ratio_fluid_over_hybrid']

        for i, param in enumerate(params):
            row, col = i // 3, i % 3
            ax = axes[row, col]
            ax.hist(self.df[param], bins=10, alpha=0.7, edgecolor='black')
            ax.set_xlabel(param)
            ax.set_ylabel('Frequency')
            ax.set_title(f'Distribution of {param}')
            ax.grid(True, alpha=0.3)

        # Temperature comparison
        ax = axes[1, 2]
        ax.scatter(self.df['T_sig_fluid'], self.df['T_sig_hybrid'], alpha=0.7, s=60)
        ax.plot([self.df['T_sig_fluid'].min(), self.df['T_sig_fluid'].max()],
                [self.df['T_sig_fluid'].min(), self.df['T_sig_fluid'].max()],
                'r--', label='y=x (perfect agreement)')
        ax.set_xlabel('T_sig_fluid (K)')
        ax.set_ylabel('T_sig_hybrid (K)')
        ax.set_title('Fluid vs Hybrid Temperature Comparison')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('results/analysis/parameter_distributions.png', dpi=300, bbox_inches='tight')
        plt.show()

        # Parameter space coverage
        print("\nParameter Space Coverage Analysis:")
        print(f"  - Unique coupling_strength values: {len(self.df['coupling_strength'].unique())}")
        print(f"  - Unique D values: {len(self.df['D'].unique())}")
        print(f"  - Parameter grid structure: {len(self.df['coupling_strength'].unique())} x {len(self.df['D'].unique())} = {len(self.df['coupling_strength'].unique()) * len(self.df['D'].unique())} total combinations")
        print(f"  - Actual data points: {len(self.df)}")

        return stats_desc

    def correlation_analysis(self):
        """Analyze correlations between all parameters"""
        print("\n" + "="*60)
        print("CORRELATION ANALYSIS")
        print("="*60)

        # Compute correlation matrix (exclude constant/zero-variance columns)
        numeric_df = self.df.select_dtypes(include=[np.number]).copy()
        zero_var_cols = [c for c in numeric_df.columns if numeric_df[c].nunique() <= 1 or np.isclose(numeric_df[c].std(ddof=0), 0.0)]
        if zero_var_cols:
            print("\nNote: Excluding constant columns from correlation (zero variance):")
            for c in zero_var_cols:
                print(f"  - {c}")
        used_df = numeric_df.drop(columns=zero_var_cols, errors='ignore')
        corr_matrix = used_df.corr()

        print("\nCorrelation Matrix:")
        print(corr_matrix.round(3))

        # Find strongest correlations
        print("\nStrongest Correlations (|r| > 0.5):")
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) > 0.5:
                    print(f"  {corr_matrix.columns[i]} ↔ {corr_matrix.columns[j]}: r = {corr_val:.3f}")

        # Create correlation heatmap
        plt.figure(figsize=(12, 10))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='RdBu_r', center=0,
                    square=True, fmt='.3f', cbar_kws={"shrink": .8})
        plt.title('Correlation Matrix: Analog Hawking Radiation Parameters', fontsize=16)
        plt.tight_layout()
        plt.savefig('results/analysis/correlation_heatmap.png', dpi=300, bbox_inches='tight')
        plt.show()

        # Scatter plots for key relationships
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Key Parameter Relationships', fontsize=16)

        # coupling_strength vs ratio
        ax = axes[0, 0]
        scatter = ax.scatter(self.df['coupling_strength'], self.df['ratio_fluid_over_hybrid'],
                           c=self.df['D'], cmap='viridis', s=60, alpha=0.7)
        ax.set_xlabel('coupling_strength')
        ax.set_ylabel('ratio_fluid_over_hybrid')
        ax.set_title('Coupling Strength vs Fluid/Hybrid Ratio')
        plt.colorbar(scatter, ax=ax, label='D')

        # D vs w_effective
        ax = axes[0, 1]
        ax.scatter(self.df['D'], self.df['w_effective'], c=self.df['coupling_strength'],
                  cmap='plasma', s=60, alpha=0.7)
        ax.set_xlabel('D')
        ax.set_ylabel('w_effective')
        ax.set_title('Diffusion vs Effective Frequency')
        ax.set_xscale('log')
        ax.set_yscale('log')

        # T_sig_fluid vs T_sig_hybrid
        ax = axes[1, 0]
        ax.scatter(self.df['T_sig_fluid'], self.df['T_sig_hybrid'],
                  c=self.df['coupling_strength'], cmap='coolwarm', s=60, alpha=0.7)
        ax.set_xlabel('T_sig_fluid (K)')
        ax.set_ylabel('T_sig_hybrid (K)')
        ax.set_title('Fluid vs Hybrid Temperature')
        ax.set_xscale('log')
        ax.set_yscale('log')

        # Detection time comparison
        ax = axes[1, 1]
        ax.scatter(self.df['t5_fluid'], self.df['t5_hybrid'],
                  c=self.df['ratio_fluid_over_hybrid'], cmap='RdYlBu', s=60, alpha=0.7)
        ax.set_xlabel('t5_fluid (s)')
        ax.set_ylabel('t5_hybrid (s)')
        ax.set_title('Fluid vs Hybrid Detection Time')
        ax.set_xscale('log')
        ax.set_yscale('log')

        plt.tight_layout()
        plt.savefig('results/analysis/key_relationships.png', dpi=300, bbox_inches='tight')
        plt.show()

        return corr_matrix

    def model_comparison_analysis(self):
        """Compare fluid vs hybrid model performance"""
        print("\n" + "="*60)
        print("FLUID VS HYBRID MODEL COMPARISON")
        print("="*60)

        # Temperature comparison
        temp_diff = self.df['T_sig_hybrid'] - self.df['T_sig_fluid']
        temp_ratio = self.df['T_sig_hybrid'] / self.df['T_sig_fluid']

        print("\nTemperature Signature Analysis:")
        print(f"  - Mean difference (Hybrid - Fluid): {temp_diff.mean():.6e} K")
        print(f"  - Std difference: {temp_diff.std():.6e} K")
        print(f"  - Mean ratio (Hybrid/Fluid): {temp_ratio.mean():.6f}")
        print(f"  - Std ratio: {temp_ratio.std():.6f}")

        # Detection time comparison
        time_diff = self.df['t5_hybrid'] - self.df['t5_fluid']
        time_ratio = self.df['t5_hybrid'] / self.df['t5_fluid']

        print("\nDetection Time Analysis:")
        print(f"  - Mean difference (Hybrid - Fluid): {time_diff.mean():.2e} s")
        print(f"  - Std difference: {time_diff.std():.2e} s")
        print(f"  - Mean ratio (Hybrid/Fluid): {time_ratio.mean():.6f}")
        print(f"  - Std ratio: {time_ratio.std():.6f}")

        # Performance metrics
        print("\nModel Performance Metrics:")
        hybrid_better_temp = (self.df['T_sig_hybrid'] > self.df['T_sig_fluid']).sum()
        hybrid_better_time = (self.df['t5_hybrid'] < self.df['t5_fluid']).sum()

        print(f"  - Hybrid yields higher T_sig: {hybrid_better_temp}/{len(self.df)} ({100*hybrid_better_temp/len(self.df):.1f}%)")
        print(f"  - Hybrid yields faster detection: {hybrid_better_time}/{len(self.df)} ({100*hybrid_better_time/len(self.df):.1f}%)")

        # Create comparison plots
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        fig.suptitle('Fluid vs Hybrid Model Performance Comparison', fontsize=16)

        # Temperature ratio vs parameters
        ax = axes[0, 0]
        scatter = ax.scatter(self.df['coupling_strength'], temp_ratio,
                           c=self.df['D'], cmap='viridis', s=80, alpha=0.7)
        ax.axhline(y=1, color='r', linestyle='--', label='Equal performance')
        ax.set_xlabel('coupling_strength')
        ax.set_ylabel('T_sig_hybrid / T_sig_fluid')
        ax.set_title('Temperature Ratio vs Coupling Strength')
        ax.legend()
        plt.colorbar(scatter, ax=ax, label='D')

        # Detection time ratio vs parameters
        ax = axes[0, 1]
        scatter = ax.scatter(self.df['coupling_strength'], time_ratio,
                           c=self.df['D'], cmap='plasma', s=80, alpha=0.7)
        ax.axhline(y=1, color='r', linestyle='--', label='Equal performance')
        ax.set_xlabel('coupling_strength')
        ax.set_ylabel('t5_hybrid / t5_fluid')
        ax.set_title('Detection Time Ratio vs Coupling Strength')
        ax.legend()
        plt.colorbar(scatter, ax=ax, label='D')
        ax.set_yscale('log')

        # Performance summary
        ax = axes[1, 0]
        performance_data = [
            hybrid_better_temp, len(self.df) - hybrid_better_temp,
            hybrid_better_time, len(self.df) - hybrid_better_time
        ]
        labels = ['Higher T_sig\n(Hybrid)', 'Higher T_sig\n(Fluid)',
                 'Faster Detection\n(Hybrid)', 'Faster Detection\n(Fluid)']
        colors = ['lightcoral', 'lightblue', 'lightgreen', 'lightyellow']

        ax.pie(performance_data, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        ax.set_title('Model Performance Summary')

        # Detailed comparison scatter
        ax = axes[1, 1]
        scatter = ax.scatter(self.df['ratio_fluid_over_hybrid'], temp_ratio,
                           c=self.df['coupling_strength'], cmap='coolwarm', s=80, alpha=0.7)
        ax.axhline(y=1, color='r', linestyle='--', alpha=0.5)
        ax.axvline(x=1, color='r', linestyle='--', alpha=0.5)
        ax.set_xlabel('ratio_fluid_over_hybrid')
        ax.set_ylabel('T_sig_hybrid / T_sig_fluid')
        ax.set_title('Cross-Model Performance Comparison')
        plt.colorbar(scatter, ax=ax, label='coupling_strength')
        ax.set_xscale('log')
        ax.set_yscale('log')

        plt.tight_layout()
        plt.savefig('results/analysis/model_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()

        return {
            'temp_ratio_mean': temp_ratio.mean(),
            'temp_ratio_std': temp_ratio.std(),
            'time_ratio_mean': time_ratio.mean(),
            'time_ratio_std': time_ratio.std(),
            'hybrid_better_temp_pct': 100*hybrid_better_temp/len(self.df),
            'hybrid_better_time_pct': 100*hybrid_better_time/len(self.df)
        }

    def scaling_analysis(self):
        """Analyze physical scaling relationships"""
        print("\n" + "="*60)
        print("PHYSICAL SCALING RELATIONSHIPS")
        print("="*60)

        # Temperature scaling with coupling strength
        print("\nTemperature Scaling Analysis:")

        # Power law fits for T_sig vs coupling_strength
        temp_scalings = {}
        for model, temp_col in [('Fluid', 'T_sig_fluid'), ('Hybrid', 'T_sig_hybrid')]:
            # Fit power law: T = a * coupling_strength^b
            log_coupling = np.log10(self.df['coupling_strength'])
            log_temp = np.log10(self.df[temp_col])

            slope, intercept, r_value, p_value, std_err = stats.linregress(log_coupling, log_temp)
            temp_scalings[model] = {
                'slope': slope,
                'intercept': intercept,
                'r2': r_value**2,
                'p_value': p_value,
            }

            print(f"\n{model} Model Temperature Scaling:")
            print(f"  T_sig ∝ coupling_strength^{slope:.3f}")
            print(f"  R² = {r_value**2:.4f}")
            print(f"  p-value = {p_value:.2e}")
            print(f"  T_sig = {10**intercept:.2e} * coupling_strength^{slope:.3f}")

        # Detection time scaling
        print("\nDetection Time Scaling Analysis:")

        time_scalings = {}
        for model, time_col in [('Fluid', 't5_fluid'), ('Hybrid', 't5_hybrid')]:
            log_coupling = np.log10(self.df['coupling_strength'])
            log_time = np.log10(self.df[time_col])

            slope, intercept, r_value, p_value, std_err = stats.linregress(log_coupling, log_time)
            time_scalings[model] = {
                'slope': slope,
                'intercept': intercept,
                'r2': r_value**2,
                'p_value': p_value,
            }

            print(f"\n{model} Model Detection Time Scaling:")
            print(f"  t5 ∝ coupling_strength^{slope:.3f}")
            print(f"  R² = {r_value**2:.4f}")
            print(f"  p-value = {p_value:.2e}")
            print(f"  t5 = {10**intercept:.2e} * coupling_strength^{slope:.3f}")

        # Frequency scaling with diffusion
        print("\nFrequency Scaling with Diffusion:")
        log_D = np.log10(self.df['D'])
        log_w = np.log10(self.df['w_effective'])

        slope_wD, intercept_wD, r_value_wD, p_value_wD, std_err_wD = stats.linregress(log_D, log_w)
        print(f"  w_effective ∝ D^{slope_wD:.3f}")
        print(f"  R² = {r_value_wD**2:.4f}")
        print(f"  w_effective = {10**intercept_wD:.2e} * D^{slope_wD:.3f}")

        # Create scaling plots
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        fig.suptitle('Physical Scaling Relationships', fontsize=16)

        # Temperature scaling
        ax = axes[0, 0]
        ax.loglog(self.df['coupling_strength'], self.df['T_sig_fluid'], 'o-',
                 label='Fluid Model', markersize=8, alpha=0.7)
        ax.loglog(self.df['coupling_strength'], self.df['T_sig_hybrid'], 's-',
                 label='Hybrid Model', markersize=8, alpha=0.7)
        ax.set_xlabel('coupling_strength')
        ax.set_ylabel('T_sig (K)')
        ax.set_title('Temperature Scaling with Coupling Strength')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Detection time scaling
        ax = axes[0, 1]
        ax.loglog(self.df['coupling_strength'], self.df['t5_fluid'], 'o-',
                 label='Fluid Model', markersize=8, alpha=0.7)
        ax.loglog(self.df['coupling_strength'], self.df['t5_hybrid'], 's-',
                 label='Hybrid Model', markersize=8, alpha=0.7)
        ax.set_xlabel('coupling_strength')
        ax.set_ylabel('t5 (s)')
        ax.set_title('Detection Time Scaling with Coupling Strength')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Frequency vs diffusion
        ax = axes[1, 0]
        scatter = ax.scatter(self.df['D'], self.df['w_effective'],
                           c=self.df['coupling_strength'], cmap='viridis', s=80, alpha=0.7)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('D')
        ax.set_ylabel('w_effective')
        ax.set_title('Effective Frequency vs Diffusion Coefficient')
        plt.colorbar(scatter, ax=ax, label='coupling_strength')

        # Kappa mirror relationships
        ax = axes[1, 1]
        scatter = ax.scatter(self.df['kappa_mirror'], self.df['ratio_fluid_over_hybrid'],
                           c=self.df['w_effective'], cmap='plasma', s=80, alpha=0.7)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('kappa_mirror')
        ax.set_ylabel('ratio_fluid_over_hybrid')
        ax.set_title('Model Performance vs Surface Gravity')
        plt.colorbar(scatter, ax=ax, label='w_effective')

        plt.tight_layout()
        plt.savefig('results/analysis/scaling_relationships.png', dpi=300, bbox_inches='tight')
        plt.show()

        return {
            'temp_scaling_fluid': float(temp_scalings['Fluid']['slope']),
            'temp_scaling_hybrid': float(temp_scalings['Hybrid']['slope']),
            'time_scaling_fluid': float(time_scalings['Fluid']['slope']),
            'time_scaling_hybrid': float(time_scalings['Hybrid']['slope']),
            'freq_scaling_diffusion': float(slope_wD),
        }

def main():
    """Main analysis function"""
    print("Starting Comprehensive Analysis of Analog Hawking Radiation Dataset")
    print("="*80)

    # Create analysis directory
    import os
    os.makedirs('results/analysis', exist_ok=True)

    # Initialize analyzer
    analyzer = HawkingRadiationAnalyzer()

    # Run all analyses
    analyzer.statistical_characterization()
    analyzer.correlation_analysis()
    analyzer.model_comparison_analysis()
    analyzer.scaling_analysis()

    print("\n" + "="*80)
    print("Analysis Complete! Results saved to results/analysis/")
    print("="*80)

if __name__ == "__main__":
    main()
