#!/usr/bin/env python3
"""
Tutorial 1: What is a Sonic Horizon?

This tutorial demonstrates how plasma flows can create analog black hole horizons.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Ensure output directory exists
output_dir = Path("results/tutorials")
output_dir.mkdir(parents=True, exist_ok=True)

def create_flow_profile(x, v0, x0, width):
    """Create a tanh flow profile."""
    return v0 * np.tanh((x - x0) / width)

def main():
    print("=" * 70)
    print("Tutorial 1: What is a Sonic Horizon?")
    print("=" * 70)
    print()
    
    print("Imagine you're swimming in a river...")
    print("• If you swim faster than the current, you can swim upstream")
    print("• If the current is faster than you can swim, you're stuck")
    print("• The point where current = your speed is like a 'horizon'")
    print()
    
    # Create spatial grid
    x = np.linspace(0, 100, 1000)  # micrometers
    x0 = 50  # center
    
    # Sound speed (constant for simplicity)
    c_s = np.full_like(x, 1.0)  # Speed of sound in plasma
    
    print("In our plasma:")
    print("• 'You' = sound waves (pressure disturbances)")
    print("• 'River' = plasma flow moving at velocity v")
    print("• 'Horizon' = where plasma speed equals sound speed (|v| = c_s)")
    print()
    
    # Case 1: Subsonic flow (no horizon)
    print("Case 1: Subsonic Flow (no horizon)")
    print("-" * 70)
    
    v_subsonic = 0.5 * c_s  # Flow slower than sound
    
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12))
    
    ax1.plot(x, v_subsonic, 'b-', linewidth=2, label='Flow speed |v|')
    ax1.plot(x, c_s, 'r--', linewidth=2, label='Sound speed c_s')
    ax1.set_xlabel('Position (μm)')
    ax1.set_ylabel('Speed (×10⁶ m/s)')
    ax1.set_title('Case 1: Subsonic Flow (No Horizon)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.text(0.02, 0.95, "|v| < c_s everywhere\nSound waves can propagate upstream",
             transform=ax1.transAxes, verticalalignment='top',
             bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
    
    print("Result: Sound waves can travel upstream - no horizon forms")
    print()
    
    # Case 2: Transonic flow (creates horizon)
    print("Case 2: Transonic Flow (creates horizon)")
    print("-" * 70)
    
    v_transonic = create_flow_profile(x, v0=1.5, x0=x0, width=5)
    
    ax2.plot(x, v_transonic, 'b-', linewidth=2, label='Flow speed |v|')
    ax2.plot(x, c_s, 'r--', linewidth=2, label='Sound speed c_s')
    ax2.axvspan(45, 55, alpha=0.2, color='gray', label='Supersonic region')
    ax2.axvline(50, color='g', linestyle=':', linewidth=2, label='Horizon')
    ax2.set_xlabel('Position (μm)')
    ax2.set_ylabel('Speed (×10⁶ m/s)')
    ax2.set_title('Case 2: Transonic Flow (Horizon Forms)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.text(0.02, 0.95, "|v| > c_s in gray region\nSound waves trapped - horizon!",
             transform=ax2.transAxes, verticalalignment='top',
             bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
    
    print("Result: Sound waves inside the gray region cannot escape")
    print("        This is our 'sonic horizon' - an analog black hole!")
    print()
    
    # Case 3: Multiple horizons
    print("Case 3: Flow with Multiple Horizons")
    print("-" * 70)
    
    # Create profile with two supersonic regions
    v1 = create_flow_profile(x, v0=1.2, x0=30, width=3)
    v2 = create_flow_profile(x, v0=1.2, x0=70, width=3)
    v_multiple = v1 + v2
    
    ax3.plot(x, v_multiple, 'b-', linewidth=2, label='Flow speed |v|')
    ax3.plot(x, c_s, 'r--', linewidth=2, label='Sound speed c_s')
    ax3.axvspan(25, 35, alpha=0.2, color='gray', label='Supersonic regions')
    ax3.axvspan(65, 75, alpha=0.2, color='gray')
    ax3.axvline(30, color='g', linestyle=':', linewidth=2, label='Horizons')
    ax3.axvline(70, color='g', linestyle=':', linewidth=2)
    ax3.set_xlabel('Position (μm)')
    ax3.set_ylabel('Speed (×10⁶ m/s)')
    ax3.set_title('Case 3: Multiple Supersonic Regions (Multiple Horizons)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.text(0.02, 0.95, "Two separate horizons\nSound trapped in both regions",
             transform=ax3.transAxes, verticalalignment='top',
             bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
    
    print("Result: Two separate supersonic regions create multiple horizons")
    print("        This can happen in complex plasma flows!")
    print()
    
    plt.tight_layout()
    plt.savefig(output_dir / 'tutorial_01_sonic_horizons.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Key concepts summary
    print("=" * 70)
    print("KEY CONCEPTS")
    print("=" * 70)
    print()
    print("1. Sonic Horizon: Where plasma flow speed equals sound speed")
    print("   • Analogous to black hole event horizon")
    print("   • Sound waves inside cannot escape")
    print()
    print("2. Supersonic Flow: |v| > c_s")
    print("   • Creates region where sound is trapped")
    print("   • Multiple supersonic regions → multiple horizons")
    print()
    print("3. Subsonic Flow: |v| < c_s")
    print("   • Sound can propagate in both directions")
    print("   • No horizon forms")
    print()
    print("4. Transonic Flow: Transitions between subsonic and supersonic")
    print("   • Creates well-defined horizons")
    print("   • Most interesting for analog gravity!")
    print()
    
    print("=" * 70)
    print("Next: Try Tutorial 2 to learn about surface gravity!")
    print("Command: ahr tutorial 2")
    print("=" * 70)

if __name__ == "__main__":
    main()
