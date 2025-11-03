#!/usr/bin/env python3
"""
Quick check that calculate_hawking_spectrum() chooses radio band when T_H <= 10 K.
"""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT)
sys.path.append(os.path.join(ROOT, "src"))
from hawking_detection_experiment import calculate_hawking_spectrum


def main():
    # Choose a small kappa to yield T_H ~ 0.01 K
    # T_H = ħ κ / (2π k) => κ = 2π k T_H / ħ
    from scipy.constants import hbar, k, pi
    T_H = 0.01  # K
    kappa = 2*pi*k*T_H / hbar
    result = calculate_hawking_spectrum(kappa)
    freqs = result.get('frequencies', np.array([]))
    assert freqs.size > 0, 'No frequencies returned'
    # Expect top of band at <= 1e11 for radio/microwave path
    assert freqs.max() <= 1e11 + 1e6, f'Unexpected high frequency band: {freqs.max()} Hz'
    print('PASS: frequency gating selects radio/microwave band for low T_H')


if __name__ == '__main__':
    main()
