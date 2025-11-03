#!/usr/bin/env python3
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
from scipy.constants import c, e, epsilon_0, k, m_e, m_p


def a0_from_intensity_and_wavelength(I_Wm2: float, lambda_m: float) -> float:
    # a0 = e E0 / (m_e * omega * c), E0 = sqrt(2 I / (c epsilon_0)), omega = 2*pi*c/lambda
    E0 = np.sqrt(2.0 * I_Wm2 / (c * epsilon_0))
    omega = 2.0 * np.pi * c / float(lambda_m)
    return (e * E0) / (m_e * omega * c)


def plasma_frequency(n_m3: float) -> float:
    return np.sqrt(e**2 * n_m3 / (epsilon_0 * m_e))


def sound_speed_adiabatic(T_e_K: float, ion_mass: float = m_p, gamma: float = 5.0/3.0) -> float:
    return float(np.sqrt(max(gamma * k * T_e_K / ion_mass, 0.0)))


def validate_case(case: dict) -> dict:
    n = float(case.get('plasma_density', np.nan))
    I = float(case.get('laser_intensity', np.nan))
    lam = float(case.get('laser_wavelength', 800e-9))
    T = float(case.get('temperature_constant', np.nan))

    a0 = a0_from_intensity_and_wavelength(I, lam)
    wp = plasma_frequency(n)
    w0 = 2.0 * np.pi * c / lam
    cs = sound_speed_adiabatic(T)

    report = {
        'plasma_density_m3': n,
        'laser_intensity_Wm2': I,
        'laser_wavelength_m': lam,
        'temperature_K': T,
        'a0': a0,
        'a0_in_valid_regime': bool(0.1 < a0 < 10.0),
        'omega_p_rad_s': wp,
        'omega_0_rad_s': w0,
        'underdense_condition_wp_lt_w0': bool(wp < w0),
        'sound_speed_m_s': cs,
        'sound_speed_formula': 'c_s = sqrt(gamma k T / m_i)',
        'kappa_max': float(case.get('kappa_max', 0.0)),
        'meets_kappa_1e10': bool(case.get('meets_kappa_1e10', False)),
    }
    return report


def main() -> int:
    cases_path = Path('results') / 'horizon_success_cases.json'
    if not cases_path.exists():
        print(f"Missing {cases_path}. Run scripts/run_param_sweep.py first.")
        return 1

    with open(cases_path, 'r') as f:
        cases = json.load(f)

    validations = [validate_case(c) for c in cases]

    summary = {
        'n_cases': len(cases),
        'n_meet_kappa_threshold': int(sum(1 for v in validations if v['meets_kappa_1e10'])),
        'n_a0_in_regime': int(sum(1 for v in validations if v['a0_in_valid_regime'])),
        'n_underdense': int(sum(1 for v in validations if v['underdense_condition_wp_lt_w0'])),
    }

    out = {
        'summary': summary,
        'validations': validations,
    }

    os.makedirs('results', exist_ok=True)
    out_path = Path('results') / 'physical_validation_report.json'
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"Saved {out_path}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
