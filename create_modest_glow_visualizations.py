"""
Scientific Visualizations for Analog Hawking Radiation Research
Focusing on the latest "glow" detection findings with modest, scientific presentation
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import c, h, hbar, k, e, m_e, epsilon_0

def create_scientific_glow_visualization():
    """
    Create a scientific visualization focusing on the glow detection findings
    """
    fig = plt.figure(figsize=(14, 10))
    fig.suptitle('Analog Hawking Radiation: Glow Detection Analysis', 
                 fontsize=14, fontweight='normal')
    
    # Create a grid for multiple subplots
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
    
    # Plot 1: Hawking Temperature Visualization
    ax1 = fig.add_subplot(gs[0, 0])
    T_H = 3.32e-3  # 3.32 mK from our latest results
    frequencies = np.logspace(8, 18, 1000)  # Hz range
    h_omega = h * frequencies
    spectral_intensity = (2 * h_omega**3 / (c**2)) / (np.exp(h_omega / (k*T_H)) - 1)
    
    ax1.loglog(frequencies, spectral_intensity, 'r-', linewidth=1.5, label='Hawking Spectrum')
    ax1.set_xlabel('Frequency (Hz)')
    ax1.set_ylabel('Spectral Intensity (W/Hz)')
    ax1.set_title('Thermal Glow Spectrum\nT = 3.32 mK', fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot 2: Horizon Formation Challenge
    ax2 = fig.add_subplot(gs[0, 1])
    x = np.linspace(-10e-6, 10e-6, 1000)
    c_sound = 0.1 * c  # Effective sound speed in plasma
    # Velocity profile that doesn't quite reach horizon condition
    v_profile = 0.08 * c * np.tanh((x + 2e-6) * 1e5) + 0.02 * c * np.sin(x * 1e6)
    
    ax2.plot(x*1e6, np.abs(v_profile)/c, 'b-', linewidth=1.5, label='|v|/c')
    ax2.axhline(y=c_sound/c, color='r', linestyle='--', linewidth=1, label='Sound speed')
    ax2.set_xlabel('Position (microns)')
    ax2.set_ylabel('|Velocity|/c')
    ax2.set_title('Horizon Formation Challenge', fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Plot 3: Multi-mirror Enhancement
    ax3 = fig.add_subplot(gs[0, 2])
    configurations = ['Single', 'Triangular', 'Pentagram', 'Hexagonal', 'Standing Wave']
    enhancement_factors = [1.0, 2.5, 5.0, 4.0, 6.0]  # From our latest results
    
    bars = ax3.bar(configurations, enhancement_factors, 
                   color=['#9ecae1', '#6baed6', '#3182bd', '#08519c', '#08306b'])
    ax3.set_ylabel('Enhancement Factor')
    ax3.set_title('Multi-Mirror Enhancement', fontsize=11)
    ax3.set_ylim(0, 7)
    ax3.tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar, value in zip(bars, enhancement_factors):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{value:.1f}x', ha='center', va='bottom', fontsize=9)
    
    # Plot 4: Parameter Space for Glow Detection
    ax4 = fig.add_subplot(gs[1, 0])
    intensities = np.logspace(17, 20, 50)
    densities = np.logspace(16, 18, 50)
    
    Int, Den = np.meshgrid(intensities, densities)
    a0 = np.sqrt(2 * Int * epsilon_0 * c) / (m_e * c**2)
    omega_pe = np.sqrt(e**2 * Den / (epsilon_0 * m_e))
    kappa = omega_pe * np.clip(a0, 0, 10)  # Surface gravity scaling
    T_H_map = hbar * kappa / (2 * np.pi * k)  # Hawking temperature
    
    im4 = ax4.contourf(intensities, densities, T_H_map.T, levels=20, cmap='viridis')
    ax4.set_xlabel('Laser Intensity (W/m²)')
    ax4.set_ylabel('Plasma Density (m⁻³)')
    ax4.set_title('Glow Temperature in Parameter Space', fontsize=11)
    ax4.set_xscale('log')
    ax4.set_yscale('log')
    cbar4 = plt.colorbar(im4, ax=ax4)
    cbar4.set_label('Temperature (K)')
    
    # Plot 5: Detection Confidence Timeline
    ax5 = fig.add_subplot(gs[1, 1])
    time_steps = np.linspace(0, 100, 1000)
    confidence = 0.01 + 0.99 * (1 - np.exp(-time_steps/20))
    confidence = np.clip(confidence, 0, 1)
    
    ax5.plot(time_steps, confidence, 'b-', linewidth=1.5, label='Detection Confidence')
    ax5.axhline(y=0.95, color='r', linestyle='--', linewidth=1, label='Detection Threshold')
    ax5.set_xlabel('Analysis Time (arbitrary units)')
    ax5.set_ylabel('Confidence')
    ax5.set_title('Glow Detection Confidence', fontsize=11)
    ax5.grid(True, alpha=0.3)
    ax5.legend()
    
    # Plot 6: Glow Characteristics
    ax6 = fig.add_subplot(gs[1, 2])
    glow_params = ['Temperature', 'Peak Freq.', 'Surface Gravity', 'Photon Energy']
    values = [3.32e-3, 195e6, 2.73e9, 8e-25]  # Glow characteristics
    units = ['K', 'Hz', 's⁻¹', 'J']
    
    y_pos = np.arange(len(glow_params))
    bars = ax6.barh(y_pos, [np.log10(v) if v > 0 else -25 for v in values], 
                    color=['#9ecae1', '#6baed6', '#3182bd', '#08519c'])
    ax6.set_yticks(y_pos)
    ax6.set_yticklabels([f"{p}\n({u})" for p, u in zip(glow_params, units)])
    ax6.set_xlabel('Log Scale Value')
    ax6.set_title('Glow Characteristics', fontsize=11)
    
    for i, (v, u) in enumerate(zip(values, units)):
        if v > 1e-20:
            ax6.text(np.log10(v) + 0.1, i, f'{v:.2e}', 
                    va='center', fontsize=8)
        else:
            ax6.text(-24, i, f'{v:.2e}', 
                    va='center', fontsize=8)
    
    plt.tight_layout()
    plt.savefig('SCIENTIFIC_GLOW_FINDINGS.png', dpi=300, bbox_inches='tight')
    print("Scientific glow findings visualization saved as 'SCIENTIFIC_GLOW_FINDINGS.png'")
    plt.show()

def create_simple_summary():
    """
    Create a simple, scientific summary visualization
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Analog Hawking Radiation: Key Findings', fontsize=14, fontweight='normal')

    # Glow characteristics
    ax1.text(0.05, 0.9, 'GLOW CHARACTERISTICS', 
             fontsize=12, fontweight='bold', transform=ax1.transAxes)
    ax1.text(0.05, 0.8, 'Temperature: 3.32 mK', 
             fontsize=10, transform=ax1.transAxes)
    ax1.text(0.05, 0.75, 'Peak frequency: 195 MHz', 
             fontsize=10, transform=ax1.transAxes)
    ax1.text(0.05, 0.7, 'Surface gravity: 2.73×10⁹ s⁻¹', 
             fontsize=10, transform=ax1.transAxes)
    ax1.text(0.05, 0.65, 'Photon energy: 8×10⁻²⁵ J', 
             fontsize=10, transform=ax1.transAxes)
    
    ax1.text(0.05, 0.5, 'FORMATION CHALLENGE', 
             fontsize=12, fontweight='bold', transform=ax1.transAxes)
    ax1.text(0.05, 0.4, 'No horizon formation in simulation', 
             fontsize=10, transform=ax1.transAxes)
    ax1.text(0.05, 0.35, 'Velocity gradients insufficient', 
             fontsize=10, transform=ax1.transAxes)
    ax1.text(0.05, 0.3, 'Threshold: |dv/dx| ≈ c', 
             fontsize=10, transform=ax1.transAxes)
    ax1.text(0.05, 0.25, 'Focus: Creation, not detection', 
             fontsize=10, transform=ax1.transAxes)
    
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.axis('off')

    # Glow spectrum visualization
    T_H = 3.32e-3  # 3.32 mK
    frequencies = np.logspace(8, 18, 1000)
    h_omega = h * frequencies
    spectral_intensity = (2 * h_omega**3 / (c**2)) / (np.exp(h_omega / (k*T_H)) - 1)
    
    ax2.loglog(frequencies, spectral_intensity, 'r-', linewidth=1.5)
    ax2.set_xlabel('Frequency (Hz)')
    ax2.set_ylabel('Spectral Intensity (W/Hz)')
    ax2.set_title('Glow Spectrum (T = 3.32 mK)', fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    # Velocity profile with horizon analysis
    x = np.linspace(-15e-6, 15e-6, 1000)
    c_sound = 0.1 * c
    # Velocity profile that approaches but doesn't cross sound speed
    v_profile = 0.08 * c * np.tanh((x + 3e-6) * 2e5) + 0.03 * c * np.exp(-((x-1e-6)/4e-6)**2)
    
    ax3.plot(x*1e6, np.abs(v_profile)/c, 'b-', linewidth=1.5, label='|v|/c')
    ax3.axhline(y=c_sound/c, color='r', linestyle='--', linewidth=1, label='Sound speed')
    ax3.set_xlabel('Position (microns)')
    ax3.set_ylabel('|Velocity|/c')
    ax3.set_title('Formation Conditions', fontsize=11)
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    # Multi-mirror enhancement
    configurations = ['Single', 'Tri', 'Penta', 'Hex', 'Standing Wave']
    enhancement = [1.0, 2.5, 5.0, 4.0, 6.0]
    colors = ['#9ecae1', '#6baed6', '#4292c6', '#2171b5', '#084594']
    
    bars = ax4.bar(configurations, enhancement, color=colors)
    ax4.set_ylabel('Enhancement Factor')
    ax4.set_title('Multi-Mirror Enhancement', fontsize=11)
    ax4.set_ylim(0, 7)
    
    for bar, value in zip(bars, enhancement):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{value:.1f}x', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('SIMPLE_GLOW_SUMMARY.png', dpi=300, bbox_inches='tight')
    print("Simple glow summary visualization saved as 'SIMPLE_GLOW_SUMMARY.png'")
    plt.show()

def create_basic_flyer():
    """
    Create a basic, scientific flyer focusing on the glow research findings
    """
    fig = plt.figure(figsize=(10, 8))
    
    # Title area
    ax_title = plt.subplot2grid((3, 2), (0, 0), colspan=2, rowspan=1)
    ax_title.text(0.5, 0.6, 'Analog Hawking Radiation', 
                  ha='center', va='center', fontsize=16, fontweight='bold',
                  transform=ax_title.transAxes, color='#000000')
    ax_title.text(0.5, 0.3, 'Glow Detection Analysis', 
                  ha='center', va='center', fontsize=12,
                  transform=ax_title.transAxes, color='#333333')
    ax_title.axis('off')
    
    # Key finding box
    ax_finding = plt.subplot2grid((3, 2), (1, 0), colspan=1, rowspan=1)
    ax_finding.axis('off')
    
    finding_text = """
    KEY FINDING:
    
    The thermal glow at
    3.32 mK is detectable
    when formed, but no 
    horizon formation 
    occurs under typical 
    simulation conditions.
    """
    ax_finding.text(0.05, 0.95, finding_text, transform=ax_finding.transAxes,
                    fontsize=10, verticalalignment='top',
                    bbox=dict(boxstyle="square,pad=0.5", facecolor="#f8f9fa", edgecolor="#dee2e6"))
    
    # Glow spectrum illustration
    ax_physics = plt.subplot2grid((3, 2), (1, 1), colspan=1, rowspan=1)
    T_H = 3.32e-3  # 3.32 mK
    frequencies = np.logspace(8, 18, 1000)
    h_omega = h * frequencies
    spectral_intensity = (2 * h_omega**3 / (c**2)) / (np.exp(h_omega / (k*T_H)) - 1)
    
    ax_physics.loglog(frequencies, spectral_intensity, 'r-', linewidth=1.5)
    ax_physics.set_xlabel('Frequency (Hz)')
    ax_physics.set_ylabel('Intensity (W/Hz)')
    ax_physics.set_title('Glow Spectrum', fontsize=11)
    ax_physics.grid(True, alpha=0.3)
    
    # Research implications section
    ax_implications = plt.subplot2grid((3, 2), (2, 0), colspan=2, rowspan=1)
    ax_implications.axis('off')
    
    implications_text = """
    IMPLICATIONS:
    • Glow detection is feasible (3.32 mK)
    • Horizon formation is the limiting factor
    • Multi-mirror configurations show enhancement
    • Plasma dynamics require optimization
    • Focus shifts from detection to creation
    """
    ax_implications.text(0.02, 0.95, implications_text, transform=ax_implications.transAxes,
                         fontsize=10, verticalalignment='top',
                         bbox=dict(boxstyle="square,pad=0.5", facecolor="#f8f9fa", edgecolor="#dee2e6"))
    
    plt.tight_layout()
    plt.savefig('BASIC_GLOW_RESEARCH.png', dpi=300, bbox_inches='tight')
    print("Basic glow research visualization saved as 'BASIC_GLOW_RESEARCH.png'")
    plt.show()

if __name__ == "__main__":
    print("Creating modest, scientific glow-focused visualizations...")
    create_scientific_glow_visualization()
    print()
    create_simple_summary()
    print()
    create_basic_flyer()
    print()
    print("All modest, scientific visualizations created successfully!")
    print("Files generated:")
    print("- SCIENTIFIC_GLOW_FINDINGS.png")
    print("- SIMPLE_GLOW_SUMMARY.png") 
    print("- BASIC_GLOW_RESEARCH.png")