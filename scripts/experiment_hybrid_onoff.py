#!/usr/bin/env python3
from __future__ import annotations

"""
Optional: Hybrid mirror on/off amplitude modulation at fixed flow.
Reports ΔT_sig and Δt5σ vs coupling weight using the enhanced spectrum helper.
"""

import argparse
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt

from analog_hawking.detection.hybrid_spectrum import calculate_enhanced_hawking_spectrum
from analog_hawking.detection.psd_collapse import band_temperature_and_t5sig


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="results/experiments/hybrid_onoff")
    p.add_argument("--kappa-fluid", type=float, default=1e10)
    p.add_argument("--kappa-mirror", type=float, default=5e9)
    p.add_argument("--weights", nargs="*", type=float, default=[0.0, 0.05, 0.1, 0.2, 0.4, 0.8])
    p.add_argument("--B", type=float, default=1e8)
    p.add_argument("--Tsys", type=float, default=30.0)
    args = p.parse_args()

    outdir = Path(args.out); figdir = outdir / "figures"
    outdir.mkdir(parents=True, exist_ok=True)
    figdir.mkdir(parents=True, exist_ok=True)

    deltas = []
    for w in args.weights:
        res = calculate_enhanced_hawking_spectrum(args.kappa_fluid, args.kappa_mirror, coupling_weight=float(w))
        f = res["frequencies"]; psd = res["power_spectrum"]
        T_sig, t5 = band_temperature_and_t5sig(f, psd, B=args.B, T_sys=args.Tsys)
        deltas.append({"w": float(w), "T_sig": float(T_sig), "t5sigma": float(t5)})

    # Plot ΔT_sig vs coupling weight
    ws = [d["w"] for d in deltas]
    Ts = [d["T_sig"] for d in deltas]
    t5s = [d["t5sigma"] for d in deltas]
    plt.figure(figsize=(6, 4))
    plt.plot(ws, Ts, "o-", label="T_sig [K]")
    plt.xlabel("Coupling weight")
    plt.ylabel("T_sig [K]")
    plt.tight_layout(); plt.savefig(figdir / "delta_Tsig_vs_weight.png", dpi=160); plt.close()

    plt.figure(figsize=(6, 4))
    plt.semilogy(ws, t5s, "o-", label="t_5σ [s]")
    plt.xlabel("Coupling weight")
    plt.ylabel("t_5σ [s]")
    plt.tight_layout(); plt.savefig(figdir / "t5sigma_vs_weight.png", dpi=160); plt.close()

    with open(outdir / "hybrid_onoff_summary.json", "w") as fh:
        json.dump(deltas, fh, indent=2)

    print(f"Wrote: {outdir}/hybrid_onoff_summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

