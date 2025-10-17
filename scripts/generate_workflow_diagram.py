#!/usr/bin/env python3
"""
Generate a professional workflow diagram showing the complete pipeline
from laser configuration to radio detection.

This creates the hero image for the README.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

def create_workflow_diagram():
    """Create a clean, professional workflow diagram."""

    fig, ax = plt.subplots(1, 1, figsize=(16, 6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5)
    ax.axis('off')

    # Define colors (professional palette)
    color_input = '#E8F4F8'      # Light blue
    color_physics = '#FFF4E6'    # Light orange
    color_compute = '#F0E6FF'    # Light purple
    color_analysis = '#E8F8E8'   # Light green
    color_output = '#FFE6E6'     # Light red

    edge_color = '#2C3E50'
    text_color = '#2C3E50'
    arrow_color = '#34495E'

    # Box parameters
    box_width = 1.6
    box_height = 1.2
    y_center = 2.5

    # Positions for 5 main boxes
    positions = [
        (0.5, y_center, "Laser\nConfiguration", color_input),
        (2.5, y_center, "Plasma\nModel", color_physics),
        (4.5, y_center, "Horizon\nDetection", color_compute),
        (6.5, y_center, "QFT\nSpectrum", color_analysis),
        (8.5, y_center, "Radio\nDetection", color_output),
    ]

    # Key equations/parameters for each stage
    equations = [
        r"$I_0, \lambda, a_0$",
        r"$n_e, T_e, v(x), c_s(x)$",
        r"$|v(x)| = c_s(x)$" + "\n" + r"$\kappa = \frac{1}{2}|\frac{d}{dx}(|v|-c_s)|$",
        r"$T_H = \frac{\hbar\kappa}{2\pi k_B}$" + "\n" + r"$\frac{dP}{d\omega} \propto \frac{\omega^3}{e^{\hbar\omega/k_BT_H}-1}$",
        r"$t_{5\sigma} = 25\frac{T_{sys}^2}{T_{sig}^2 B}$"
    ]

    # Draw boxes and text
    boxes = []
    for i, (x, y, label, color) in enumerate(positions):
        # Main box
        box = FancyBboxPatch(
            (x - box_width/2, y - box_height/2),
            box_width, box_height,
            boxstyle="round,pad=0.1",
            facecolor=color,
            edgecolor=edge_color,
            linewidth=2.5,
            zorder=2
        )
        ax.add_patch(box)
        boxes.append(box)

        # Stage label
        ax.text(x, y + 0.25, label,
                ha='center', va='center',
                fontsize=13, fontweight='bold',
                color=text_color,
                zorder=3)

        # Equation text
        ax.text(x, y - 0.2, equations[i],
                ha='center', va='center',
                fontsize=9,
                color=text_color,
                zorder=3)

    # Draw arrows between boxes
    arrow_y = y_center
    arrow_props = dict(
        arrowstyle='->,head_width=0.6,head_length=0.8',
        linewidth=2.5,
        color=arrow_color,
        zorder=1
    )

    for i in range(len(positions) - 1):
        x_start = positions[i][0] + box_width/2 + 0.05
        x_end = positions[i+1][0] - box_width/2 - 0.05

        arrow = FancyArrowPatch(
            (x_start, arrow_y),
            (x_end, arrow_y),
            **arrow_props
        )
        ax.add_patch(arrow)

    # Add title
    title_text = "Analog Hawking Radiation: Complete Workflow Pipeline"
    ax.text(5, 4.3, title_text,
            ha='center', va='center',
            fontsize=16, fontweight='bold',
            color=text_color)

    # Add subtitle with key insight
    subtitle = "From Laser-Plasma Interaction to Radio-Band Detection"
    ax.text(5, 3.9, subtitle,
            ha='center', va='center',
            fontsize=11, style='italic',
            color=arrow_color)

    # Add output metric at bottom
    output_text = (
        "Key Output: Horizon Formation + Surface Gravity κ → "
        "Hawking Temperature T_H → Detection Time t_5σ"
    )
    ax.text(5, 0.8, output_text,
            ha='center', va='center',
            fontsize=10,
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#FFF9E6',
                     edgecolor=edge_color, linewidth=1.5),
            color=text_color)

    # Add hybrid coupling callout
    hybrid_box = FancyBboxPatch(
        (3.8, 0.2), 2.4, 0.4,
        boxstyle="round,pad=0.05",
        facecolor='#FFE6F0',
        edgecolor='#C0392B',
        linewidth=2,
        linestyle='--',
        zorder=1
    )
    ax.add_patch(hybrid_box)

    ax.text(5, 0.4, "Hybrid: + Plasma Mirror Coupling → 16× Detection Boost",
            ha='center', va='center',
            fontsize=9, fontweight='bold',
            color='#C0392B',
            zorder=2)

    plt.tight_layout()

    # Save
    output_path = 'figures/workflow_diagram.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Workflow diagram saved to {output_path}")
    plt.close()


if __name__ == "__main__":
    import os
    os.makedirs('figures', exist_ok=True)
    create_workflow_diagram()
