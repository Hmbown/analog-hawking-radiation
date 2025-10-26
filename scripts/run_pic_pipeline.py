#!/usr/bin/env python3
from __future__ import annotations

"""
End-to-end pipeline from a PIC/openPMD-derived 1D slice to horizons, spectrum,
and detection metrics.

Usage:
  python scripts/run_pic_pipeline.py --profile results/warpx_profile.npz \
      --graybody acoustic_wkb --kappa-method acoustic_exact
"""

import argparse
import json
import os
import subprocess
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from analog_hawking.physics_engine.horizon import find_horizons_with_uncertainty
from analog_hawking.detection.radio_snr import band_power_from_spectrum, equivalent_signal_temperature, sweep_time_for_5sigma
from scipy.constants import hbar, k, pi
from hawking_detection_experiment import calculate_hawking_spectrum
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--profile", type=str, default=None)
    p.add_argument("--config", type=str, default="configs/warpx_mock.json")
    p.add_argument("--real-warpx", action="store_true", help="Run real WarpX simulation (falls back to synthetic if not available)")
    p.add_argument("--warpx-exec", type=str, default="warpx", help="Path to WarpX executable")
    p.add_argument("--graybody", type=str, choices=["dimensionless", "wkb", "acoustic_wkb"], default="dimensionless")
    p.add_argument("--kappa-method", type=str, choices=["acoustic", "legacy", "acoustic_exact"], default="acoustic")
    p.add_argument("--alpha-gray", type=float, default=1.0)
    p.add_argument("--B", type=float, default=1e8)
    p.add_argument("--Tsys", type=float, default=30.0)
    args = p.parse_args()

    profile_path = args.profile or "results/warpx_profile.npz"
    os.makedirs(os.path.dirname(profile_path), exist_ok=True)

    if args.real_warpx:
        try:
            with open(args.config, 'r') as f:
                config = json.load(f)
            # Generate simple WarpX input file (adapt from config; for real, expand this)
            warpx_in_content = """
# Simple WarpX input adapted from config
amr.n_cell = 64 1 1
geometry.dims = 1
geometry.coord_sys = 0
boundary.left_x = periodic
boundary.right_x = periodic
boundary_lo_x = -1
boundary_hi_x = 1
max_step = 10
# Species from config
""".strip()
            # Add species
            for sp in config.get("species", []):
                warpx_in_content += f"\nparticles.species_{sp['name']}.density = 1.0e20\n"
                warpx_in_content += f"particles.species_{sp['name']}.charge = {sp['charge']}\n"
                warpx_in_content += f"particles.species_{sp['name']}.mass = {sp['mass']}\n"
            with open("warpx.in", 'w') as f:
                f.write(warpx_in_content)
            # Run WarpX
            result = subprocess.run([args.warpx_exec, "warpx.in"], capture_output=True, text=True, cwd=".")
            if result.returncode != 0:
                raise subprocess.CalledProcessError(result.returncode, [args.warpx_exec, "warpx.in"], result.stdout, result.stderr)
            # Assume output in diags/h5/small_size/ (standard WarpX openPMD path)
            h5_file = "diags/hdf5/small_size/BField_000000.h5"  # Adjust based on actual output
            if not os.path.exists(h5_file):
                raise FileNotFoundError(f"WarpX output not found at {h5_file}")
        except (FileNotFoundError, subprocess.CalledProcessError, KeyError) as e:
            print(f"WarpX run failed ({e}), falling back to synthetic data")
            # Generate synthetic
            subprocess.run(["python", "scripts/generate_synthetic_openpmd_slice.py"], check=True, cwd=".")
            h5_file = "results/synthetic_slice.h5"
        # Convert to profile npz
        subprocess.run([
            "python", "scripts/openpmd_slice_to_profile.py",
            "--in", h5_file,
            "--vel-dataset", "/vel",
            "--Te-dataset", "/Te",
            "--out", profile_path
        ], check=True, cwd=".")
    elif args.profile is None:
        # Generate synthetic if no profile provided
        subprocess.run(["python", "scripts/generate_synthetic_openpmd_slice.py"], check=True, cwd=".")
        h5_file = "results/synthetic_slice.h5"
        subprocess.run([
            "python", "scripts/openpmd_slice_to_profile.py",
            "--in", h5_file,
            "--out", profile_path
        ], check=True, cwd=".")

    npz = np.load(profile_path)
    x = npz["x"]
    v = npz["v"]
    cs = npz["c_s"]

    horizons = find_horizons_with_uncertainty(x, v, cs, kappa_method=args.kappa_method)
    positions = horizons.positions.tolist() if horizons.positions.size else []
    kappa = horizons.kappa.tolist() if horizons.kappa.size else []

    summary = {
        "horizon_positions": positions,
        "kappa": kappa,
        "kappa_err": horizons.kappa_err.tolist() if horizons.kappa_err.size else [],
    }

    if kappa:
        spec = calculate_hawking_spectrum(float(kappa[0]), graybody_profile={"x": x, "v": v, "c_s": cs}, graybody_method=args.graybody, alpha_gray=args.alpha_gray,
                                          emitting_area_m2=1e-6, solid_angle_sr=5e-2, coupling_efficiency=0.1)
        if spec.get("success"):
            f = np.asarray(spec["frequencies"])  # type: ignore[index]
            P = np.asarray(spec["power_spectrum"])  # type: ignore[index]
            peak_f = float(spec.get("peak_frequency", float(f[np.argmax(P)])))
            B = float(args.B)
            inband = band_power_from_spectrum(f, P, peak_f, B)
            Ts = equivalent_signal_temperature(inband, B)
            t = float(sweep_time_for_5sigma(np.array([args.Tsys]), np.array([B]), Ts)[0, 0]) if Ts > 0 else float("inf")
            TH = float(hbar * float(kappa[0]) / (2 * pi * k))
            summary.update({
                "spectrum_peak_frequency": peak_f,
                "inband_power_W": float(inband),
                "T_sig_K": float(Ts),
                "t5sigma_s": t,
                "T_H_K": TH,
            })
            # Figure
            try:
                plt.figure(figsize=(6, 4))
                plt.loglog(f, P)
                plt.xlabel("Frequency [Hz]")
                plt.ylabel("PSD [W/Hz]")
                os.makedirs("figures", exist_ok=True)
                plt.tight_layout()
                plt.savefig("figures/pic_pipeline_psd.png", dpi=180)
                plt.close()
            except Exception:
                pass

    os.makedirs("results", exist_ok=True)
    out = "results/pic_pipeline_summary.json"
    with open(out, "w") as fp:
        json.dump(summary, fp, indent=2)
    print(f"Saved {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

