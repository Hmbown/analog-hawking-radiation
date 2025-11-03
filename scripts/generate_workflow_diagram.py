#!/usr/bin/env python3
"""
Generate a professional workflow diagram showing the speculative laser-enhanced 
analog Hawking radiation pipeline.

This creates the hero image for the README showing how laser-painted plasma mirrors
could theoretically enhance fluid sonic horizons.
"""

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


def create_workflow_diagram():
    """Create a clean, professional workflow diagram showing the speculative hybrid approach."""

    fig, ax = plt.subplots(1, 1, figsize=(16, 8))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)
    ax.axis('off')

    # Define colors (professional palette)
    color_fluid = '#E8F4F8'      # Light blue - fluid background
    color_laser = '#FFE6E6'      # Light red - laser intervention  
    color_hybrid = '#F0E6FF'     # Light purple - hybrid coupling
    color_quantum = '#E8F8E8'    # Light green - quantum physics
    color_detect = '#FFF4E6'     # Light orange - detection

    edge_color = '#2C3E50'
    text_color = '#2C3E50'
    arrow_color = '#34495E'
    spec_color = '#C0392B'  # Red for speculative elements

    # Main workflow boxes
    box_width = 1.8
    box_height = 1.0
    y_main = 4.0

    main_boxes = [
        (1.5, y_main, "Fluid\nBackground", color_fluid, "Flowing medium\n(gas/liquid/plasma)"),
        (4.0, y_main, "Laser\nIonization", color_laser, "Ultra-intense pulse\ncreates plasma mirror"),
        (6.5, y_main, "Hybrid\nCoupling", color_hybrid, "Enhanced surface\ngravity κ_eff"),
        (9.0, y_main, "QFT\nSpectrum", color_quantum, "Hawking radiation\nT_H = ħκ/(2πk_B)"),
        (11.5, y_main, "Detection\nModel", color_detect, "Radio-band\nobservability")
    ]

    # Physics equations for each stage
    equations = [
        r"$|v(x)| \geq c_s(x)$" + "\n" + r"$\kappa_{fluid} = \frac{1}{2}|\frac{d}{dx}(|v|-c_s)|$",
        r"$I > 10^{18}$ W/m²" + "\n" + r"$\kappa_{mirror} = \frac{2\pi\eta_a}{D}$",
        r"$\kappa_{eff} = \kappa_{fluid} + w \cdot \kappa_{mirror}$" + "\n" + r"$w = e^{-|x-x_m|/L}$",
        r"$\frac{dP}{d\omega} \propto \frac{\omega^3}{e^{\hbar\omega/k_BT_H}-1}$",
        r"$t_{5\sigma} = 25\frac{T_{sys}^2}{T_{sig}^2 B}$"
    ]

    # Draw main workflow boxes
    for i, (x, y, label, color, desc) in enumerate(main_boxes):
        # Main box
        box = FancyBboxPatch(
            (x - box_width/2, y - box_height/2),
            box_width, box_height,
            boxstyle="round,pad=0.08",
            facecolor=color,
            edgecolor=edge_color,
            linewidth=2,
            zorder=2
        )
        ax.add_patch(box)

        # Main label
        ax.text(x, y + 0.15, label,
                ha='center', va='center',
                fontsize=11, fontweight='bold',
                color=text_color, zorder=3)
        
        # Description
        ax.text(x, y - 0.15, desc,
                ha='center', va='center',
                fontsize=8, style='italic',
                color=text_color, zorder=3)

        # Physics equation below
        ax.text(x, y - 0.7, equations[i],
                ha='center', va='center',
                fontsize=8, color=text_color, zorder=3)

    # Draw arrows between main boxes
    arrow_props = dict(
        arrowstyle='->,head_width=0.5,head_length=0.7',
        linewidth=2,
        color=arrow_color,
        zorder=1
    )

    for i in range(len(main_boxes) - 1):
        x_start = main_boxes[i][0] + box_width/2 + 0.1
        x_end = main_boxes[i+1][0] - box_width/2 - 0.1
        
        arrow = FancyArrowPatch(
            (x_start, y_main),
            (x_end, y_main),
            **arrow_props
        )
        ax.add_patch(arrow)

    # Add speculative enhancement visualization
    spec_y = 2.2
    
    # Enhanced coupling visualization
    enh_box = FancyBboxPatch(
        (5.5, spec_y - 0.4), 2.0, 0.8,
        boxstyle="round,pad=0.05",
        facecolor='#FFE6F0',
        edgecolor=spec_color,
        linewidth=2,
        linestyle='--',
        zorder=1
    )
    ax.add_patch(enh_box)

    ax.text(6.5, spec_y, "Speculative Enhancement\nκ_fluid + w·κ_mirror",
            ha='center', va='center',
            fontsize=9, fontweight='bold',
            color=spec_color, zorder=2)

    # Curved arrow showing enhancement
    enhancement_arrow = FancyArrowPatch(
        (6.5, spec_y + 0.5), (6.5, y_main - 0.6),
        arrowstyle='->,head_width=0.4,head_length=0.6',
        connectionstyle="arc3,rad=0.2",
        linewidth=2, color=spec_color,
        linestyle='--', zorder=1
    )
    ax.add_patch(enhancement_arrow)

    # Add title and subtitle
    title_text = "Speculative Laser-Enhanced Analog Hawking Radiation"
    ax.text(6.0, 5.5, title_text,
            ha='center', va='center',
            fontsize=16, fontweight='bold',
            color=text_color)

    subtitle = "Computational Exploration: Laser-Painted Plasma Mirrors + Fluid Sonic Horizons"
    ax.text(6.0, 5.1, subtitle,
            ha='center', va='center',
            fontsize=11, style='italic',
            color=arrow_color)

    # Add key insight boxes
    insight_props = dict(boxstyle='round,pad=0.3', facecolor='#FFF9E6',
                        edgecolor=edge_color, linewidth=1)

    ax.text(3.0, 0.8, "Fluid Background:\nNatural sonic horizons from flow",
            ha='center', va='center', fontsize=9,
            bbox=insight_props, color=text_color)

    ax.text(9.0, 0.8, "Laser Enhancement:\nPlasma mirrors boost local κ",
            ha='center', va='center', fontsize=9,
            bbox=insight_props, color=text_color)

    # Add disclaimer
    disclaimer = (
        "⚠️  SPECULATIVE MODEL: Hybrid coupling lacks established theoretical foundation.\n"
        "This represents computational exploration of 'what if' scenarios, not physics predictions."
    )
    ax.text(6.0, 0.2, disclaimer,
            ha='center', va='center',
            fontsize=9, style='italic', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#FFE6E6',
                     edgecolor=spec_color, linewidth=2),
            color=spec_color)

    plt.tight_layout()

    # Save to both locations
    paths = ['figures/workflow_diagram.png', 'docs/img/workflow_diagram.png']
    for path in paths:
        plt.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"✓ Workflow diagram saved to {path}")
    
    plt.close()


if __name__ == "__main__":
    import os
    os.makedirs('figures', exist_ok=True)
    os.makedirs('docs/img', exist_ok=True)
    create_workflow_diagram()
