#!/usr/bin/env python3
"""
Tutorial 3: Detection Forecasts in Realistic Experiments

This tutorial demonstrates why detecting analog Hawking radiation is challenging
and how we estimate detection times.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Ensure output directory exists
output_dir = Path("results/tutorials")
output_dir.mkdir(parents=True, exist_ok=True)

def signal_power(kappa, bandwidth, eta=0.5):
    """
    Approximate signal power from Hawking radiation.
    
    Parameters
    ----------
    kappa : float
        Surface gravity (s⁻¹)
    bandwidth : float
        Detector bandwidth (Hz)
    eta : float
        Detection efficiency (0-1)
    
    Returns
    -------
    float
        Signal power (W)
    """
    hbar = 1.054571817e-34  # J·s
    T_H = (hbar * kappa) / (2 * np.pi * 1.380649e-23)  # K
    k_B = 1.380649e-23  # J/K
    
    # Signal power ~ k_B * Δν * T_H * η
    return k_B * bandwidth * T_H * eta

def noise_power(bandwidth, T_sys):
    """
    Johnson-Nyquist noise power.
    
    Parameters
    ----------
    bandwidth : float
        Detector bandwidth (Hz)
    T_sys : float
        System temperature (K)
    
    Returns
    -------
    float
        Noise power (W)
    """
    k_B = 1.380649e-23  # J/K
    return k_B * bandwidth * T_sys

def detection_time(snr_target, kappa, bandwidth, T_sys, eta=0.5):
    """
    Estimate time to achieve target SNR.
    
    Parameters
    ----------
    snr_target : float
        Target signal-to-noise ratio
    kappa : float
        Surface gravity (s⁻¹)
    bandwidth : float
        Detector bandwidth (Hz)
    T_sys : float
        System temperature (K)
    eta : float
        Detection efficiency
    
    Returns
    -------
    float
        Detection time (seconds)
    """
    P_signal = signal_power(kappa, bandwidth, eta)
    P_noise = noise_power(bandwidth, T_sys)
    
    if P_noise == 0:
        return np.inf
    
    # SNR improves as sqrt(N) where N = number of samples
    # N = time * bandwidth (Nyquist sampling)
    # So: SNR = (P_signal/P_noise) * sqrt(time * bandwidth)
    # Solve for time: time = (SNR_target * P_noise / P_signal)² / bandwidth
    
    snr_per_sample = P_signal / P_noise
    
    if snr_per_sample == 0:
        return np.inf
    
    return (snr_target / snr_per_sample)**2 / bandwidth

def main():
    print("=" * 70)
    print("Tutorial 3: Detection Forecasts in Realistic Experiments")
    print("=" * 70)
    print()
    
    print("Why is detecting analog Hawking radiation so hard?")
    print()
    print("The challenge: Our signal is EXTREMELY weak!")
    print()
    
    # Demonstrate signal vs noise
    print("=" * 70)
    print("SIGNAL vs NOISE COMPARISON")
    print("=" * 70)
    print()
    
    # Typical parameters
    kappa = 3e12  # s⁻¹ (typical surface gravity)
    bandwidth = 1e9  # Hz (1 GHz bandwidth)
    T_sys = 30  # K (system temperature)
    eta = 0.5  # Detection efficiency
    
    P_signal = signal_power(kappa, bandwidth, eta)
    P_noise = noise_power(bandwidth, T_sys)
    
    print(f"Parameters:")
    print(f"  • Surface gravity κ = {kappa:.1e} s⁻¹")
    print(f"  • Bandwidth = {bandwidth/1e9:.1f} GHz")
    print(f"  • System temperature = {T_sys} K")
    print(f"  • Detection efficiency = {eta}")
    print()
    
    print(f"Power calculations:")
    print(f"  • Signal power = {P_signal:.2e} W")
    print(f"  • Noise power  = {P_noise:.2e} W")
    print(f"  • Signal/Noise = {P_signal/P_noise:.2e}")
    print()
    
    # Visualization 1: Power comparison
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    powers = ['Signal', 'Noise']
    values = [P_signal, P_noise]
    colors = ['green', 'red']
    
    bars = ax.bar(powers, values, color=colors, alpha=0.7)
    ax.set_ylabel('Power (W)')
    ax.set_title('Signal vs Noise Power Comparison')
    ax.set_yscale('log')
    
    # Add value labels on bars
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{value:.2e} W',
                ha='center', va='bottom', fontsize=12)
    
    plt.savefig(Path("results/tutorials") / 'tutorial_03_power_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("=" * 70)
    print("DETECTION TIME CALCULATION")
    print("=" * 70)
    print()
    print("Signal-to-noise ratio (SNR) improves with integration time:")
    print("  SNR ∝ √(time)  (for Gaussian noise)")
    print()
    print("To reach SNR = 5 (5σ detection):")
    
    # Calculate detection time
    t_detect = detection_time(5.0, kappa, bandwidth, T_sys, eta)
    
    print(f"  • Required time = {t_detect:.2e} seconds")
    print(f"  •               = {t_detect*1e3:.2f} milliseconds")
    print()
    
    if t_detect < 1e-6:
        print("  ✓ This is FAST! (microseconds)")
    elif t_detect < 1e-3:
        print("  ✓ This is reasonable (sub-millisecond)")
    elif t_detect < 1e-1:
        print("  ⚠ This is challenging (hundreds of milliseconds)")
    else:
        print("  ✗ This is VERY difficult (seconds or more)")
    
    print()
    
    # Visualization 2: Detection time vs kappa
    print("=" * 70)
    print("PARAMETER SCAN: Detection Time vs Surface Gravity")
    print("=" * 70)
    print()
    
    kappas = np.logspace(11, 14, 50)  # From 1e11 to 1e14 s⁻¹
    times = [detection_time(5.0, k, bandwidth, T_sys, eta) for k in kappas]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot 1: Detection time vs kappa
    ax1.loglog(kappas, times, 'b-', linewidth=2)
    ax1.axhline(1e-6, color='g', linestyle='--', alpha=0.7, label='1 μs (fast)')
    ax1.axhline(1e-3, color='orange', linestyle='--', alpha=0.7, label='1 ms (reasonable)')
    ax1.axhline(1e-1, color='r', linestyle='--', alpha=0.7, label='100 ms (challenging)')
    ax1.set_xlabel('Surface Gravity κ (s⁻¹)')
    ax1.set_ylabel('Detection Time (s)')
    ax1.set_title('Detection Time vs Surface Gravity')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Highlight our example
    our_idx = np.argmin(np.abs(kappas - kappa))
    ax1.plot(kappas[our_idx], times[our_idx], 'ro', markersize=10, label='Our example')
    ax1.legend()
    
    # Plot 2: Detection time vs temperature
    temps = [hawking_temperature(k) for k in kappas]
    ax2.loglog(temps, times, 'r-', linewidth=2)
    ax2.set_xlabel('Hawking Temperature T_H (K)')
    ax2.set_ylabel('Detection Time (s)')
    ax2.set_title('Detection Time vs Hawking Temperature')
    ax2.grid(True, alpha=0.3)
    
    # Highlight our example
    our_temp = hawking_temperature(kappa)
    ax2.plot(our_temp, times[our_idx], 'bo', markersize=10, label='Our example')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(Path("results/tutorials") / 'tutorial_03_detection_scaling.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("Key insights from parameter scan:")
    print(f"  • Higher κ → Shorter detection time")
    print(f"  • Our κ = {kappa:.1e} s⁻¹ → t_detect = {t_detect:.2e} s")
    print(f"  • Need κ > {kappas[np.argmax(np.array(times) < 1e-6)]:.1e} s⁻¹ for microsecond detection")
    print()
    
    # Parameter sensitivity
    print("=" * 70)
    print("PARAMETER SENSITIVITY")
    print("=" * 70)
    print()
    
    # Vary each parameter
    base_params = {
        'kappa': kappa,
        'bandwidth': bandwidth,
        'T_sys': T_sys,
        'eta': eta
    }
    
    variations = {
        'kappa': [0.5, 1.0, 2.0],  # factors
        'bandwidth': [0.5, 1.0, 2.0],
        'T_sys': [0.5, 1.0, 2.0],
        'eta': [0.5, 1.0, 1.5]
    }
    
    print("How detection time changes with parameters:")
    print("(Values show factor change from baseline)")
    print()
    
    for param, factors in variations.items():
        print(f"{param}:")
        for factor in factors:
            params = base_params.copy()
            params[param] *= factor
            
            t = detection_time(5.0, params['kappa'], params['bandwidth'], 
                              params['T_sys'], params['eta'])
            
            change = t / t_detect
            print(f"  {factor:4.1f}x → {change:6.2f}x detection time")
        print()
    
    print("Insights:")
    print("  • Higher κ = dramatically better (quadratic improvement)")
    print("  • Higher bandwidth = better (linear improvement)")
    print("  • Lower T_sys = better (linear improvement)")
    print("  • Higher efficiency = better (linear improvement)")
    print()
    
    # Real experimental considerations
    print("=" * 70)
    print("REAL EXPERIMENTAL CONSIDERATIONS")
    print("=" * 70)
    print()
    
    print("Our calculation is IDEALIZED. Real experiments face:")
    print()
    print("1. Systematic Noise Sources:")
    print("   • Laser intensity fluctuations")
    print("   • Plasma density variations")
    print("   • Electromagnetic interference")
    print("   • Temperature drift")
    print("   → All increase effective T_sys")
    print()
    
    print("2. Bandwidth Limitations:")
    print("   • Electronics have finite bandwidth")
    print("   • Filters introduce phase noise")
    print("   • ADC sampling rates limited")
    print("   → Reduces usable bandwidth")
    print()
    
    print("3. Integration Time Constraints:")
    print("   • Plasma lifetime limited (~ps to ns)")
    print("   • Laser repetition rates (~10 Hz)")
    print("   • Need many shots for statistics")
    print("   → May need t_detect < 10⁻⁶ s")
    print()
    
    print("4. Calibration Challenges:")
    print("   • Need absolute temperature calibration")
    print("   • Must subtract background noise")
    print("   • Reference signals required")
    print("   → Increases complexity")
    print()
    
    # Best-case vs realistic scenarios
    print("=" * 70)
    print("BEST-CASE vs REALISTIC SCENARIOS")
    print("=" * 70)
    print()
    
    scenarios = {
        'Best case (lab)': {
            'kappa': 5e12,
            'bandwidth': 2e9,
            'T_sys': 20,
            'eta': 0.8
        },
        'Typical (ELI)': {
            'kappa': 3e12,
            'bandwidth': 1e9,
            'T_sys': 30,
            'eta': 0.5
        },
        'Challenging': {
            'kappa': 1e12,
            'bandwidth': 0.5e9,
            'T_sys': 50,
            'eta': 0.3
        }
    }
    
    print(f"{'Scenario':<20} {'κ (s⁻¹)':<12} {'T_H (K)':<12} {'t_detect':<12}")
    print("-" * 70)
    
    for name, params in scenarios.items():
        k = params['kappa']
        T_H = hawking_temperature(k)
        t = detection_time(5.0, k, params['bandwidth'], params['T_sys'], params['eta'])
        
        print(f"{name:<20} {k:.1e}     {T_H:.1e}      {t:.1e} s")
    
    print()
    
    # Summary
    print("=" * 70)
    print("KEY TAKEAWAYS")
    print("=" * 70)
    print()
    print("1. Detection is challenging but possible:")
    print("   • Signal is extremely weak (10⁻⁷ K typical)")
    print("   • Need sensitive detectors and long integration")
    print("   • Real experiments face additional noise sources")
    print()
    print("2. Parameter optimization is crucial:")
    print("   • Maximize κ (steeper horizons)")
    print("   • Maximize bandwidth")
    print("   • Minimize system temperature")
    print("   • Maximize detection efficiency")
    print()
    print("3. Our framework helps by:")
    print("   • Forecasting detection times for parameters")
    print("   • Identifying optimal experimental conditions")
    print("   • Quantifying uncertainties")
    print("   • Planning beam time allocation")
    print()
    print("4. Realistic expectations:")
    print("   • Best case: microseconds to milliseconds")
    print("   • Typical: milliseconds to hundreds of milliseconds")
    print("   • Challenging: seconds (may need many shots)")
    print()
    
    print("=" * 70)
    print("Tutorial complete!")
    print()
    print("Next steps:")
    print("  • Run: ahr pipeline --demo")
    print("  • Try: ahr sweep --gradient")
    print("  • Read: docs/ELI_Experimental_Planning_Guide.md")
    print("=" * 70)

if __name__ == "__main__":
    main()
