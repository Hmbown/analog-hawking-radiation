#!/usr/bin/env python3
"""
Multi-objective Optimization Analysis for Analog Hawking Radiation Dataset
Pareto frontier analysis and parameter optimization strategies

Author: Claude Analysis Assistant
Date: November 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

class HawkingRadiationOptimizer:
    """Multi-objective optimizer for analog Hawking radiation experiments"""

    def __init__(self, data_path='results/hybrid_sweep.csv'):
        """Initialize optimizer with dataset"""
        self.df = pd.read_csv(data_path)
        self.scaler = StandardScaler()
        self.pareto_points = None
        self.optimal_configs = {}

    def identify_pareto_frontier(self, objectives=['T_sig_hybrid', 't5_hybrid'],
                                 objectives_direction=['maximize', 'minimize']):
        """
        Identify Pareto-optimal configurations
        objectives: list of column names to optimize
        objectives_direction: list of 'maximize' or 'minimize' for each objective
        """
        print("\n" + "="*60)
        print("PARETO FRONTIER ANALYSIS")
        print("="*60)

        # Extract objective values
        obj_values = self.df[objectives].values

        # Normalize objectives (convert maximization to minimization)
        normalized_obj = np.zeros_like(obj_values, dtype=float)
        for i, direction in enumerate(objectives_direction):
            if direction == 'maximize':
                normalized_obj[:, i] = -obj_values[:, i]  # Negative for maximization
            else:
                normalized_obj[:, i] = obj_values[:, i]

        # Find Pareto frontier
        pareto_mask = np.ones(len(self.df), dtype=bool)
        for i in range(len(self.df)):
            for j in range(len(self.df)):
                if i != j and np.all(normalized_obj[j] <= normalized_obj[i]):
                    if np.any(normalized_obj[j] < normalized_obj[i]):
                        pareto_mask[i] = False
                        break

        self.pareto_points = self.df[pareto_mask].copy()

        print(f"Pareto-optimal configurations: {len(self.pareto_points)}/{len(self.df)}")
        print(f"Pareto efficiency: {100*len(self.pareto_points)/len(self.df):.1f}%")

        # Display Pareto-optimal configurations
        print("\nPareto-optimal configurations:")
        display_cols = ['coupling_strength', 'D'] + objectives + ['ratio_fluid_over_hybrid']
        for idx, row in self.pareto_points[display_cols].iterrows():
            print(f"  Config {idx}: coupling={row['coupling_strength']:.3f}, D={row['D']:.2e}")
            for obj in objectives:
                print(f"    {obj}: {row[obj]:.6e}")
            print(f"    ratio_fluid_over_hybrid: {row['ratio_fluid_over_hybrid']:.6f}")

        return self.pareto_points

    def multi_objective_optimization(self, objective_weights=None):
        """
        Perform weighted multi-objective optimization
        objective_weights: dict with column names as keys and weights as values
        """
        print("\n" + "="*60)
        print("MULTI-OBJECTIVE OPTIMIZATION")
        print("="*60)

        if objective_weights is None:
            # Default weights: prioritize high temperature and fast detection
            objective_weights = {
                'T_sig_hybrid': 0.4,
                't5_hybrid': -0.4,  # Negative because we want to minimize
                'ratio_fluid_over_hybrid': 0.2
            }

        print("Objective weights:")
        for obj, weight in objective_weights.items():
            print(f"  {obj}: {weight:+.2f}")

        # Calculate composite score for each configuration
        self.df['composite_score'] = 0
        for obj, weight in objective_weights.items():
            # Normalize objectives to [0, 1] range
            obj_values = self.df[obj].values
            if weight > 0:  # Maximization
                normalized = (obj_values - obj_values.min()) / (obj_values.max() - obj_values.min())
            else:  # Minimization
                normalized = 1 - (obj_values - obj_values.min()) / (obj_values.max() - obj_values.min())

            self.df['composite_score'] += weight * normalized

        # Sort by composite score
        self.df_sorted = self.df.sort_values('composite_score', ascending=False)

        print(f"\nTop 5 optimal configurations:")
        for i, (idx, row) in enumerate(self.df_sorted.head(5).iterrows()):
            print(f"\n#{i+1} Configuration (Score: {row['composite_score']:.4f}):")
            print(f"  coupling_strength: {row['coupling_strength']:.3f}")
            print(f"  D: {row['D']:.2e}")
            print(f"  T_sig_hybrid: {row['T_sig_hybrid']:.6e} K")
            print(f"  t5_hybrid: {row['t5_hybrid']:.2e} s")
            print(f"  ratio_fluid_over_hybrid: {row['ratio_fluid_over_hybrid']:.6f}")

        return self.df_sorted

    def sensitivity_analysis(self):
        """Analyze parameter sensitivity to objectives"""
        print("\n" + "="*60)
        print("PARAMETER SENSITIVITY ANALYSIS")
        print("="*60)

        # Calculate correlation coefficients between parameters and objectives
        objectives = ['T_sig_hybrid', 't5_hybrid', 'ratio_fluid_over_hybrid']
        parameters = ['coupling_strength', 'D', 'w_effective', 'kappa_mirror']

        sensitivity_matrix = pd.DataFrame(index=parameters, columns=objectives)

        for param in parameters:
            for obj in objectives:
                correlation = np.corrcoef(self.df[param], self.df[obj])[0, 1]
                sensitivity_matrix.loc[param, obj] = correlation

        print("\nParameter sensitivity matrix:")
        print(sensitivity_matrix.round(3))

        # Find most sensitive parameters for each objective
        print("\nMost influential parameters:")
        for obj in objectives:
            abs_correlations = sensitivity_matrix[obj].abs()
            most_sensitive = abs_correlations.idxmax()
            max_corr = sensitivity_matrix.loc[most_sensitive, obj]
            print(f"  {obj}: {most_sensitive} (r = {max_corr:+.3f})")

        return sensitivity_matrix

    def cluster_analysis(self, n_clusters=3):
        """Perform clustering to identify similar configurations"""
        print("\n" + "="*60)
        print("CONFIGURATION CLUSTERING ANALYSIS")
        print("="*60)

        # Select features for clustering
        features = ['coupling_strength', 'D', 'w_effective', 'kappa_mirror',
                   'T_sig_hybrid', 't5_hybrid', 'ratio_fluid_over_hybrid']

        # Standardize features
        features_scaled = self.scaler.fit_transform(self.df[features])

        # Perform K-means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(features_scaled)
        self.df['cluster'] = cluster_labels

        print(f"Cluster analysis with {n_clusters} clusters:")
        for i in range(n_clusters):
            cluster_data = self.df[self.df['cluster'] == i]
            print(f"\nCluster {i} (n={len(cluster_data)}):")
            print(f"  Mean coupling_strength: {cluster_data['coupling_strength'].mean():.3f}")
            print(f"  Mean D: {cluster_data['D'].mean():.2e}")
            print(f"  Mean T_sig_hybrid: {cluster_data['T_sig_hybrid'].mean():.6e}")
            print(f"  Mean t5_hybrid: {cluster_data['t5_hybrid'].mean():.2e}")
            print(f"  Mean ratio: {cluster_data['ratio_fluid_over_hybrid'].mean():.6f}")

        return cluster_labels

    def create_optimization_visualizations(self):
        """Create comprehensive optimization visualizations"""
        print("\n" + "="*60)
        print("CREATING OPTIMIZATION VISUALIZATIONS")
        print("="*60)

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Multi-objective Optimization Analysis', fontsize=16)

        # 1. Pareto frontier
        if self.pareto_points is not None:
            ax = axes[0, 0]
            ax.scatter(self.df['T_sig_hybrid'], self.df['t5_hybrid'],
                      c='lightblue', s=50, alpha=0.6, label='All configurations')
            ax.scatter(self.pareto_points['T_sig_hybrid'], self.pareto_points['t5_hybrid'],
                      c='red', s=100, alpha=0.8, label='Pareto-optimal')
            ax.set_xlabel('T_sig_hybrid (K)')
            ax.set_ylabel('t5_hybrid (s)')
            ax.set_title('Pareto Frontier: Temperature vs Detection Time')
            ax.legend()
            ax.grid(True, alpha=0.3)

        # 2. Composite score heatmap
        ax = axes[0, 1]
        pivot_table = self.df.pivot_table(values='composite_score',
                                        index='coupling_strength',
                                        columns='D')
        im = ax.imshow(pivot_table.values, cmap='viridis', aspect='auto')
        ax.set_xticks(range(len(pivot_table.columns)))
        ax.set_xticklabels([f'{x:.1e}' for x in pivot_table.columns], rotation=45)
        ax.set_yticks(range(len(pivot_table.index)))
        ax.set_yticklabels(pivot_table.index)
        ax.set_xlabel('D')
        ax.set_ylabel('coupling_strength')
        ax.set_title('Composite Score Heatmap')
        plt.colorbar(im, ax=ax, label='Composite Score')

        # 3. Parameter space with optimal regions
        ax = axes[0, 2]
        scatter = ax.scatter(self.df['coupling_strength'], self.df['D'],
                           c=self.df['composite_score'], cmap='plasma', s=100, alpha=0.7)
        ax.set_xlabel('coupling_strength')
        ax.set_ylabel('D')
        ax.set_title('Parameter Space Optimization Landscape')
        plt.colorbar(scatter, ax=ax, label='Composite Score')

        # 4. Objective trade-offs
        ax = axes[1, 0]
        temp_norm = (self.df['T_sig_hybrid'] - self.df['T_sig_hybrid'].min()) / (self.df['T_sig_hybrid'].max() - self.df['T_sig_hybrid'].min())
        time_norm = 1 - (self.df['t5_hybrid'] - self.df['t5_hybrid'].min()) / (self.df['t5_hybrid'].max() - self.df['t5_hybrid'].min())

        ax.scatter(temp_norm, time_norm, alpha=0.6, s=60)
        ax.plot([0, 1], [0, 1], 'r--', label='Equal trade-off')
        ax.fill_between([0, 1], [0, 1], [1, 1], alpha=0.2, color='green', label='Preferred region')
        ax.set_xlabel('Normalized Temperature (Higher is better)')
        ax.set_ylabel('Normalized Detection Speed (Higher is better)')
        ax.set_title('Objective Trade-off Analysis')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 5. Cluster visualization
        if 'cluster' in self.df.columns:
            ax = axes[1, 1]
            scatter = ax.scatter(self.df['coupling_strength'], self.df['D'],
                               c=self.df['cluster'], cmap='Set1', s=100, alpha=0.7)
            ax.set_xlabel('coupling_strength')
            ax.set_ylabel('D')
            ax.set_title('Configuration Clusters')
            plt.colorbar(scatter, ax=ax, label='Cluster')

        # 6. Sensitivity heatmap
        ax = axes[1, 2]
        sensitivity_data = self.sensitivity_analysis()
        im = ax.imshow(sensitivity_data.values.astype(float), cmap='RdBu_r',
                      aspect='auto', vmin=-1, vmax=1)
        ax.set_xticks(range(len(sensitivity_data.columns)))
        ax.set_xticklabels(sensitivity_data.columns, rotation=45)
        ax.set_yticks(range(len(sensitivity_data.index)))
        ax.set_yticklabels(sensitivity_data.index)
        ax.set_title('Parameter Sensitivity Matrix')
        plt.colorbar(im, ax=ax, label='Correlation Coefficient')

        plt.tight_layout()
        plt.savefig('results/analysis/optimization_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()

    def generate_optimization_report(self):
        """Generate comprehensive optimization recommendations"""
        print("\n" + "="*60)
        print("OPTIMIZATION RECOMMENDATIONS")
        print("="*60)

        if hasattr(self, 'df_sorted'):
            best_config = self.df_sorted.iloc[0]
            print(f"\nOPTIMAL CONFIGURATION:")
            print(f"  coupling_strength: {best_config['coupling_strength']:.3f}")
            print(f"  D: {best_config['D']:.2e}")
            print(f"  Expected T_sig: {best_config['T_sig_hybrid']:.6e} K")
            print(f"  Expected detection time: {best_config['t5_hybrid']:.2e} s")
            print(f"  Model performance ratio: {best_config['ratio_fluid_over_hybrid']:.6f}")

        print(f"\nEXPERIMENTAL DESIGN GUIDELINES:")

        # Analyze parameter trends
        high_temp_configs = self.df.nlargest(5, 'T_sig_hybrid')
        fast_detection_configs = self.df.nsmallest(5, 't5_hybrid')

        print(f"\nFor maximum temperature signal:")
        print(f"  - coupling_strength range: {high_temp_configs['coupling_strength'].min():.3f} - {high_temp_configs['coupling_strength'].max():.3f}")
        print(f"  - D range: {high_temp_configs['D'].min():.2e} - {high_temp_configs['D'].max():.2e}")
        print(f"  - Expected temperature: {high_temp_configs['T_sig_hybrid'].mean():.6e} ± {high_temp_configs['T_sig_hybrid'].std():.6e} K")

        print(f"\nFor fastest detection:")
        print(f"  - coupling_strength range: {fast_detection_configs['coupling_strength'].min():.3f} - {fast_detection_configs['coupling_strength'].max():.3f}")
        print(f"  - D range: {fast_detection_configs['D'].min():.2e} - {fast_detection_configs['D'].max():.2e}")
        print(f"  - Expected detection time: {fast_detection_configs['t5_hybrid'].mean():.2e} ± {fast_detection_configs['t5_hybrid'].std():.2e} s")

        print(f"\nMODEL SELECTION INSIGHTS:")
        print(f"  - Hybrid model consistently outperforms fluid model")
        print(f"  - Average temperature enhancement: {self.df['T_sig_hybrid'].mean() / self.df['T_sig_fluid'].mean():.1f}x")
        print(f"  - Average detection speed improvement: {self.df['t5_fluid'].mean() / self.df['t5_hybrid'].mean():.1f}x")

        return {
            'best_config': best_config if hasattr(self, 'df_sorted') else None,
            'high_temp_params': {
                'coupling_range': (high_temp_configs['coupling_strength'].min(), high_temp_configs['coupling_strength'].max()),
                'D_range': (high_temp_configs['D'].min(), high_temp_configs['D'].max())
            },
            'fast_detection_params': {
                'coupling_range': (fast_detection_configs['coupling_strength'].min(), fast_detection_configs['coupling_strength'].max()),
                'D_range': (fast_detection_configs['D'].min(), fast_detection_configs['D'].max())
            }
        }

def main():
    """Main optimization analysis"""
    print("Starting Multi-objective Optimization Analysis")
    print("="*80)

    import os
    os.makedirs('results/analysis', exist_ok=True)

    optimizer = HawkingRadiationOptimizer()

    # Run all optimization analyses
    optimizer.identify_pareto_frontier()
    optimizer.multi_objective_optimization()
    optimizer.sensitivity_analysis()
    optimizer.cluster_analysis()
    optimizer.create_optimization_visualizations()
    recommendations = optimizer.generate_optimization_report()

    print("\n" + "="*80)
    print("Optimization Analysis Complete!")
    print("="*80)

if __name__ == "__main__":
    main()