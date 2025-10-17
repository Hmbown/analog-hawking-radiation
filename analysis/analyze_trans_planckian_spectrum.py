#!/usr/bin/env python3
"""
Spectrum analysis helper for Trans-Planckian campaigns.

Given a Hawking spectrum JSON file (produced by run_trans_planckian_experiment.py),
this script generates diagnostic plots and summary statistics that compare the
measured spectrum to an ideal thermal curve.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import numpy as np
import matplotlib.pyplot as plt

try:
    from scipy.constants import c, h, k
except ImportError:  # pragma: no cover
    c = 299_792_458.0
    h = 6.62607015e-34
    k = 1.380649e-23


def load_spectrum(path: Path) -> Dict[str, np.ndarray]:
    payload = json.loads(path.read_text())
    freqs = np.asarray(payload["frequencies"], dtype=float)
    psd = np.asarray(payload["power_spectrum"], dtype=float)
    peak = float(payload.get("peak_frequency", float(freqs[np.argmax(psd)])))
    temperature = float(payload.get("temperature", 0.0))
    return {"frequencies": freqs, "power_spectrum": psd, "peak_frequency": peak, "temperature": temperature}


def planck_spectrum(freqs: np.ndarray, temperature: float) -> np.ndarray:
    if temperature <= 0:
        return np.zeros_like(freqs)
    exponent = np.exp((h * freqs) / (k * temperature))
    with np.errstate(over="ignore"):
        numerator = 2.0 * h * freqs ** 3
        denom = c ** 2 * (exponent - 1.0)
        return numerator / denom


def compute_metrics(freqs: np.ndarray, psd: np.ndarray, ref_psd: np.ndarray) -> Dict[str, float]:
    mask = (psd > 0) & (ref_psd > 0)
    if not np.any(mask):
        return {"l2_error": float("nan"), "max_ratio": float("nan")}
    l2 = float(np.sqrt(np.trapezoid((psd[mask] - ref_psd[mask]) ** 2, x=freqs[mask])))
    ratio = float(np.max(psd[mask] / ref_psd[mask]))
    return {"l2_error": l2, "max_ratio": ratio}


def plot_spectrum(freqs: np.ndarray, psd: np.ndarray, ref_psd: np.ndarray, out: Path) -> None:
    plt.figure(figsize=(8, 5))
    plt.loglog(freqs, psd, label="Measured", lw=2)
    if np.any(ref_psd > 0):
        plt.loglog(freqs, ref_psd, label="Planck reference", lw=2, ls="--")
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Power spectral density [W/Hz]")
    plt.legend()
    plt.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=200)
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze Trans-Planckian Hawking spectrum JSON.")
    parser.add_argument("--spectrum", type=Path, required=True, help="Path to hawking_spectrum.json.")
    parser.add_argument("--output-dir", type=Path, default=Path("results/trans_planckian_analysis"),
                        help="Directory for plots and metrics.")
    parser.add_argument("--temperature", type=float, help="Override temperature (Kelvin) for Planck reference.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    spec = load_spectrum(args.spectrum)
    temperature = args.temperature if args.temperature is not None else spec["temperature"]
    ref_psd = planck_spectrum(spec["frequencies"], temperature)
    metrics = compute_metrics(spec["frequencies"], spec["power_spectrum"], ref_psd)
    metrics["temperature_K"] = float(temperature)
    metrics_path = args.output_dir / "spectrum_metrics.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=2))
    plot_spectrum(spec["frequencies"], spec["power_spectrum"], ref_psd,
                  args.output_dir / "spectrum_comparison.png")
    print(f"Wrote metrics to {metrics_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
