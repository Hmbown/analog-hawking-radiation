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
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import matplotlib
from scipy.constants import hbar, k, pi

from analog_hawking.detection.radio_snr import (
    band_power_from_spectrum,
    equivalent_signal_temperature,
    sweep_time_for_5sigma,
)
from analog_hawking.physics_engine.horizon import find_horizons_with_uncertainty
from analog_hawking.physics_engine.horizon_nd import find_horizon_surface_nd
from hawking_detection_experiment import calculate_hawking_spectrum

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--profile", type=str, default=None)
    p.add_argument("--config", type=str, default="configs/warpx_mock.json")
    p.add_argument("--real-warpx", action="store_true", help="Run real WarpX simulation (falls back to synthetic if not available)")
    p.add_argument("--warpx-exec", type=str, default="warpx", help="Path to WarpX executable")
    p.add_argument("--graybody", type=str, choices=["dimensionless", "wkb", "acoustic_wkb"], default="dimensionless")
    p.add_argument("--kappa-method", type=str, choices=["acoustic", "legacy", "acoustic_exact"], default="acoustic_exact")
    p.add_argument("--alpha-gray", type=float, default=1.0)
    p.add_argument("--B", type=float, default=1e8)
    p.add_argument("--Tsys", type=float, default=30.0)
    # nD mode (experimental)
    p.add_argument("--nd-npz", type=str, default=None, help="Path to nD grid NPZ (x[,y[,z]], vx,vy[,vz], c_s)")
    p.add_argument("--nd-scan-axis", type=int, default=0, help="Axis along which to scan for horizon crossings")
    p.add_argument("--nd-patches", type=int, default=64, help="Max horizon patches to aggregate for graybody")
    # Direct HDF5/OpenPMD nD inputs (dataset paths)
    p.add_argument("--nd-h5-in", type=str, default=None, help="HDF5 file to read nD fields from")
    p.add_argument("--x-ds", type=str, default=None)
    p.add_argument("--y-ds", type=str, default=None)
    p.add_argument("--z-ds", type=str, default=None)
    p.add_argument("--vx-ds", type=str, default=None)
    p.add_argument("--vy-ds", type=str, default=None)
    p.add_argument("--vz-ds", type=str, default=None)
    p.add_argument("--cs-ds", type=str, default=None)
    args = p.parse_args()

    profile_path = args.profile or "results/warpx_profile.npz"
    # nD branch: if nd-npz or H5 provided, run nD pipeline and exit
    if args.nd_npz or args.nd_h5_in:
        if args.nd_h5_in:
            try:
                import h5py  # type: ignore
            except Exception:
                raise SystemExit("h5py is required for --nd-h5-in mode")
            with h5py.File(args.nd_h5_in, 'r') as f:
                def _rd(p):
                    return None if p is None else np.array(f[p]) if p in f else None
                x = _rd(args.x_ds)
                y = _rd(args.y_ds)
                z = _rd(args.z_ds)
                vx = _rd(args.vx_ds)
                vy = _rd(args.vy_ds)
                vz = _rd(args.vz_ds)
                cs = _rd(args.cs_ds)
            grids = [g for g in (x, y, z) if g is not None]
            dims = len(grids)
            if dims < 2:
                raise SystemExit("--nd-h5-in requires at least x and y datasets")
            comps = [c for c in (vx, vy, vz) if c is not None]
            if len(comps) != dims:
                raise SystemExit("Velocity component datasets must match dimensionality")
            v_field = np.stack(comps, axis=-1)
            if cs is None:
                raise SystemExit("--cs-ds dataset is required for sound speed")
        else:
            data = np.load(args.nd_npz)
            grids = []
            dims = 0
            for key in ("x", "y", "z"):
                if key in data:
                    grids.append(np.array(data[key]))
                    dims += 1
            if dims < 2:
                raise SystemExit("--nd-npz requires at least x and y coordinates")
            comps = []
            for comp in ("vx", "vy", "vz"):
                if comp in data:
                    comps.append(np.array(data[comp]))
            if len(comps) != dims:
                raise SystemExit("--nd-npz velocity components must match dimensionality")
            v_field = np.stack(comps, axis=-1)
            if "c_s" not in data:
                raise SystemExit("--nd-npz must include 'c_s'")
            cs = np.array(data["c_s"])

        # Horizon surface and κ summary
        surf = find_horizon_surface_nd(grids, v_field, cs, scan_axis=int(args.nd_scan_axis))
        kappa_eff = float(np.median(surf.kappa)) if surf.kappa.size else 0.0
        summary = {
            "dim": dims,
            "horizon_points": int(surf.positions.shape[0]),
            "kappa_median": kappa_eff,
            "kappa_mean": float(np.mean(surf.kappa)) if surf.kappa.size else 0.0,
            "kappa_std": float(np.std(surf.kappa)) if surf.kappa.size else 0.0,
        }

        # Aggregate graybody via reusable nD aggregator
        from analog_hawking.detection.graybody_nd import aggregate_patchwise_graybody
        agg_res = aggregate_patchwise_graybody(
            grids,
            v_field,
            cs,
            kappa_eff,
            graybody_method=str(args.graybody),
            alpha_gray=float(args.alpha_gray),
            scan_axis=int(args.nd_scan_axis),
            max_patches=int(args.nd_patches),
        )
        agg = {"success": agg_res.success}
        if agg_res.success:
            agg.update({
                "frequencies": agg_res.frequencies,
                "power_spectrum": agg_res.power_spectrum,
                "power_std": agg_res.power_std,
                "peak_frequency": agg_res.peak_frequency,
            })
        if agg.get("success"):
            f = np.asarray(agg["frequencies"])  # type: ignore[index]
            P = np.asarray(agg["power_spectrum"])  # type: ignore[index]
            peak_f = float(agg.get("peak_frequency", float(f[np.argmax(P)])))
            B = float(args.B)
            inband = band_power_from_spectrum(f, P, peak_f, B)
            Ts = equivalent_signal_temperature(inband, B)
            t = float(sweep_time_for_5sigma(np.array([args.Tsys]), np.array([B]), Ts)[0, 0]) if Ts > 0 else float("inf")
            summary.update({
                "spectrum_peak_frequency": peak_f,
                "inband_power_W": float(inband),
                "T_sig_K": float(Ts),
                "t5sigma_s": t,
            })
            # Save PSD figure
            try:
                plt.figure(figsize=(6, 4))
                plt.loglog(f, P)
                plt.xlabel("Frequency [Hz]")
                plt.ylabel("PSD [W/Hz]")
                os.makedirs("figures", exist_ok=True)
                plt.tight_layout()
                plt.savefig("figures/pic_pipeline_nd_psd.png", dpi=180)
                plt.close()
            except Exception:
                pass

        # Attach run metadata
        try:
            import datetime
            import subprocess
            commit = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
        except Exception:
            commit = "unknown"
        summary.update({
            "_run": {
                "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
                "git_commit": commit,
                "args": vars(args),
            }
        })
        os.makedirs("results", exist_ok=True)
        out = "results/pic_pipeline_nd_summary.json"
        with open(out, "w") as fp:
            json.dump(summary, fp, indent=2)
        print(f"Saved {out}")
        return 0
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
            subprocess.run([sys.executable, "scripts/generate_synthetic_openpmd_slice.py"], check=True, cwd=".")
            h5_file = "results/synthetic_slice.h5"
        # Convert to profile npz
        subprocess.run([
            sys.executable, "scripts/openpmd_slice_to_profile.py",
            "--in", h5_file,
            "--vel-dataset", "/vel",
            "--Te-dataset", "/Te",
            "--out", profile_path
        ], check=True, cwd=".")
    elif args.profile is None:
        # Generate synthetic if no profile provided
        subprocess.run([sys.executable, "scripts/generate_synthetic_openpmd_slice.py"], check=True, cwd=".")
        h5_file = "results/synthetic_slice.h5"
        subprocess.run([
            sys.executable, "scripts/openpmd_slice_to_profile.py",
            "--in", h5_file,
            "--vel-dataset", "/vel",
            "--Te-dataset", "/Te",
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
