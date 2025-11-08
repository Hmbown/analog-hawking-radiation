#!/usr/bin/env python3
"""
Tutorial 2: From Surface Gravity to Hawking Temperature

This tutorial demonstrates how surface gravity (κ) relates to Hawking temperature (T_H).
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Ensure output directory exists
output_dir = Path("results/tutorials")
output_dir.mkdir(parents=True, exist_ok=True)

def compute_kappa(v, x, horizon_idx):
    """Compute surface gravity from velocity gradient at horizon."""
    # Approximate derivative at horizon
    dv_dx = np.abs(np.gradient(v, x))
    return dv_dx[horizon_idx]

def hawking_temperature(kappa):
    """Compute Hawking temperature from surface gravity."""
    # T_H = ħκ / (2πk_B)
    hbar = 1.054571817e-34  # J·s
    k_B = 1.380649e-23      # J/K
    return (hbar * kappa) / (2 * np.pi * k_B)

def create_velocity_profile(x, v0, x0, width):
    """Create a tanh velocity profile."""
    return v0 * np.tanh((x - x0) / width)

def main():
    print("=" * 70)
    print("Tutorial 2: From Surface Gravity to Hawking Temperature")
    print("=" * 70)
    print()
    
    print("KEY FORMULA: T_H = ħκ / (2πk_B)")
    print()
    print("Where:")
    print("  • T_H = Hawking temperature (how hot the radiation is)")
    print("  • ħ = Planck's constant (quantum mechanics)")
    print("  • κ = surface gravity (how steep the horizon is)")
    print("  • k_B = Boltzmann constant (thermodynamics)")
    print()
    
    # Create spatial grid
    x = np.linspace(0, 100, 1000)  # micrometers
    x0 = 50  # horizon position
    
    print("=" * 70)
    print("Experiment: Different Velocity Gradients")
    print("=" * 70)
    print()
    
    # Test different velocity gradients
    widths = [2, 5, 10, 20]  # Different steepnesses
    v0 = 1.5  # Peak velocity
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    results = []
    
    for i, width in enumerate(widths):
        ax = axes[i]
        
        # Create velocity profile
        v = create_velocity_profile(x, v0, x0, width)
        c_s = np.ones_like(x)  # Sound speed = 1
        
        # Find horizon (where v = c_s)
        horizon_idx = np.argmin(np.abs(v - c_s))
        
        # Compute surface gravity
        kappa = compute_kappa(v, x, horizon_idx)
        T_H = hawking_temperature(kappa)
        
        results.append({
            'width': width,
            'kappa': kappa,
            'T_H': T_H
        })
        
        # Plot
        ax.plot(x, v, 'b-', linewidth=2, label='Flow velocity |v|')
        ax.plot(x, c_s, 'r--', linewidth=2, label='Sound speed c_s')
        ax.axvline(x0, color='g', linestyle=':', linewidth=2, label='Horizon')
        ax.plot(x[horizon_idx], v[horizon_idx], 'go', markersize=10, label='Crossing point')
        
        ax.set_xlabel('Position (μm)')
        ax.set_ylabel('Speed (×10⁶ m/s)')
        ax.set_title(f'Width = {width} μm\nκ = {kappa:.2e} s⁻¹, T_H = {T_H:.2e} K')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        
        # Add gradient arrow
        dx = 5
        dy = v[horizon_idx + dx] - v[horizon_idx - dx]
        ax.annotate('', xy=(x[horizon_idx + dx], v[horizon_idx + dx]), 
                   xytext=(x[horizon_idx - dx], v[horizon_idx - dx]),
                   arrowprops=dict(arrowstyle='<->', color='purple', lw=2))
        ax.text(x[horizon_idx], v[horizon_idx] + 0.2, f'Steepness\n∝ κ',
                ha='center', va='bottom', fontsize=8, color='purple')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'tutorial_02_kappa_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Summary table
    print("Results:")
    print("-" * 70)
    print(f"{'Width (μm)':<12} {'κ (s⁻¹)':<15} {'T_H (K)':<15} {'Relative T_H':<15}")
    print("-" * 70)
    
    base_T = results[0]['T_H']
    for r in results:
        rel_T = r['T_H'] / base_T
        print(f"{r['width']:<12} {r['kappa']:<15.2e} {r['T_H']:<15.2e} {rel_T:.1f}x")
    
    print()
    
    # Temperature visualization
    print("=" * 70)
    print("Visualization: Temperature vs. Gradient Steepness")
    print("=" * 70)
    print()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    widths = [r['width'] for r in results]
    kappas = [r['kappa'] for r in results]
    temps = [r['T_H'] for r in results]
    
    # Plot 1: κ vs width
    ax1.semilogy(widths, kappas, 'bo-', linewidth=2, markersize=8)
    ax1.set_xlabel('Velocity Profile Width (μm)')
    ax1.set_ylabel('Surface Gravity κ (s⁻¹)')
    ax1.set_title('κ ∝ 1/width')
    ax1.grid(True, alpha=0.3)
    
    # Add annotations
    for w, k in zip(widths, kappas):
        ax1.annotate(f'{k:.1e}', (w, k), textcoords="offset points", 
                    xytext=(0,10), ha='center', fontsize=9)
    
    # Plot 2: T_H vs κ
    ax2.loglog(kappas, temps, 'ro-', linewidth=2, markersize=8)
    ax2.set_xlabel('Surface Gravity κ (s⁻¹)')
    ax2.set_ylabel('Hawking Temperature T_H (K)')
    ax2.set_title('T_H ∝ κ (Linear Relationship)')
    ax2.grid(True, alpha=0.3)
    
    # Add formula
    ax2.text(0.95, 0.95, r'$T_H = \frac{\hbar\kappa}{2\pi k_B}$', 
             transform=ax2.transAxes, ha='right', va='top', fontsize=14,
             bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(output_dir / 'tutorial_02_scaling_relationships.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Physical interpretation
    print("=" * 70)
    print("PHYSICAL INTERPRETATION")
    print("=" * 70)
    print()
    print("1. Steeper gradient → Higher κ")
    print("   • Sharp horizon = strong gravitational effect")
    print("   • Gentle horizon = weak gravitational effect")
    print()
    print("2. Higher κ → Higher temperature")
    print("   • T_H ∝ κ (direct linear relationship)")
    print("   • Stronger gravity = hotter radiation")
    print()
    print("3. Realistic values:")
    print("   • Typical κ: 10¹¹ - 10¹³ s⁻¹")
    print("   • Typical T_H: 10⁻⁹ - 10⁻⁶ K")
    print("   • This is extremely cold!")
    print()
    
    # Comparison with everyday temperatures
    print("=" * 70)
    print("TEMPERATURE COMPARISONS")
    print("=" * 70)
    print()
    
    # Reference temperatures
    temps_ref = {
        'Room temperature': 300,
        'Human body': 310,
        'Boiling water': 373,
        'Liquid nitrogen': 77,
        'Liquid helium': 4.2,
        'Cosmic microwave background': 2.7,
        'Our Hawking radiation (typical)': 1e-7,
        'Our Hawking radiation (best case)': 1e-5
    }
    
    print("Temperature (K):")
    print("-" * 70)
    for name, temp in temps_ref.items():
        if temp >= 1:
            print(f"{name:<35} {temp:.1f}")
        else:
            print(f"{name:<35} {temp:.1e}")
    
    print()
    print("Why is detection so hard?")
    print("• Our signal (~10⁻⁷ K) is 10 million times colder than liquid helium")
    print("• Need extremely sensitive radio detectors")
    print("• Long integration times required")
    print("• Careful noise subtraction essential")
    print()
    
    print("=" * 70)
    print("KEY TAKEAWAYS")
    print("=" * 70)
    print()
    print("1. Surface gravity κ measures how 'steep' the horizon is")
    print("   • Calculated from velocity gradient at horizon")
    print("   • Units: s⁻¹ (inverse seconds)")
    print()
    print("2. Hawking temperature T_H ∝ κ")
    print("   • Linear relationship: double κ → double T_H")
    print("   • Formula: T_H = ħκ/(2πk_B)")
    print()
    print("3. Realistic temperatures are extremely low")
    print("   • 10⁻⁷ to 10⁻⁵ K for typical plasma parameters")
    print("   • Makes detection challenging but possible")
    print()
    print("4. Steeper horizons are better!")
    print("   • Sharper velocity gradients → higher κ")
    print("   • Higher κ → higher T_H → easier detection")
    print()
    
    print("=" * 70)
    print("Next: Tutorial 3 explores detection challenges!")
    print("Command: ahr tutorial 3")
    print("=" * 70)

if __name__ == "__main__":
    main()
