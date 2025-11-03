#!/usr/bin/env python3
"""Validate fluctuation injector statistics in a simplified setup."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from physics_engine.plasma_models.fluctuation_injector import (
    FluctuationConfig,
    QuantumFluctuationInjector,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate fluctuation statistics")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/fluctuation_seeding.yml"),
        help="YAML configuration for the injector",
    )
    parser.add_argument("--modes", type=int, default=512, help="Number of Fourier modes to sample")
    parser.add_argument("--trials", type=int, default=200, help="Number of random draws")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/fluctuation_validation.json"),
        help="Output summary path",
    )
    return parser.parse_args()


def _load_config(path: Path) -> FluctuationConfig:
    import yaml

    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    return FluctuationConfig(**data)


def main() -> None:
    args = _parse_args()
    config = _load_config(args.config)
    injector = QuantumFluctuationInjector(config)

    band_min = float(config.band_min)
    band_max = float(config.band_max)
    k_values = np.linspace(band_min, band_max, args.modes)
    amplitudes = []
    for _ in range(args.trials):
        sample = injector.sample_fourier_modes(k_values)
        amplitudes.append(np.abs(sample))
    amplitudes = np.array(amplitudes)

    stats = {
        "mean_amplitude": float(np.mean(amplitudes)),
        "std_amplitude": float(np.std(amplitudes)),
        "target_temperature": config.target_temperature,
        "mode_cutoff": config.mode_cutoff,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
