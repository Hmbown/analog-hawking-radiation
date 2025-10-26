#!/usr/bin/env python3
"""Utility to generate and run a minimal WarpX down-ramp simulation."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
from pathlib import Path
from typing import Dict

DEFAULT_TEMPLATE = """\
# Minimal 1D WarpX down-ramp input deck (auto-generated)
max_step = {max_step}
amr.n_cell = {n_cell} 1 1
amr.max_level = 0
amr.blocking_factor = 1
geometry.dims = 1
geometry.coord_sys = 0
geometry.prob_lo = 0.0 0.0 0.0
geometry.prob_hi = {length} 0.0 0.0

algo.current_deposition = esirkepov
algo.particle_shape = 2
particles.use_fdtd_nci_corr = 1

warpx.verbose = 1
warpx.use_filter = 0
warpx.do_subcycling = 0
warpx.do_electrostatic = 0
warpx.do_moving_window = 0

particles.nspecies = 1
particles.species_names = electrons
particles.electrons.charge = -1.0
particles.electrons.mass = 1.0
particles.electrons.injection_style = NUniformPerCell
particles.electrons.n_part_per_cell_each_dim = {particles_per_cell} 1 1
particles.electrons.profile = parse_density_function
particles.electrons.density_function(x,y,z) = {density_peak} * (1.0 - min(max((x-{ramp_start})/{ramp_length},0.0),1.0)) + {density_floor}
particles.electrons.momentum_distribution_type = gaussian
particles.electrons.momentum_distribution_mean = 0.0 0.0 0.0
particles.electrons.momentum_distribution_std = {thermal_beta} {thermal_beta} {thermal_beta}

lasers.nlasers = 1
lasers.names = driver
driver.profile = Gaussian
driver.lambda = {lambda0}
driver.a0 = {a0}
driver.duration = {laser_duration}
driver.transverse_waist = {laser_waist}
driver.position = {laser_position} 0.0 0.0
driver.direction = 1.0 0.0 0.0
driver.polarization = 0.0 1.0 0.0

diagnostics.diags_names = diag1
diag1.period = {diag_period}
diag1.format = openpmd
diag1.openpmd_backend = hdf5
diag1.file_prefix = "{diag_prefix}"
"""


def _render_template(params: Dict[str, float | int | str]) -> str:
    return DEFAULT_TEMPLATE.format(**params)


def _write_template(path: Path, params: Dict[str, float | int | str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        fh.write(_render_template(params))


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate and run a WarpX down-ramp simulation.")
    parser.add_argument("--deck", type=Path, default=Path("protocols/inputs_downramp_1d.in"), help="WarpX input deck path.")
    parser.add_argument("--output", type=Path, default=Path("diags/openpmd"), help="Output directory for openPMD diagnostics.")
    parser.add_argument("--warpx-cmd", default="warpx", help="WarpX executable (or mpirun command).")
    parser.add_argument("--max-step", type=int, default=400)
    parser.add_argument("--length", type=float, default=1.0e-3)
    parser.add_argument("--n-cell", type=int, default=1024)
    parser.add_argument("--density-peak", type=float, default=1.0)
    parser.add_argument("--density-floor", type=float, default=0.05)
    parser.add_argument("--ramp-start", type=float, default=3.0e-4)
    parser.add_argument("--ramp-length", type=float, default=5.0e-4)
    parser.add_argument("--particles-per-cell", type=int, default=16)
    parser.add_argument("--thermal-beta", type=float, default=0.01)
    parser.add_argument("--lambda0", type=float, default=800e-9)
    parser.add_argument("--a0", type=float, default=0.2)
    parser.add_argument("--laser-duration", type=float, default=30e-15)
    parser.add_argument("--laser-waist", type=float, default=15e-6)
    parser.add_argument("--laser-position", type=float, default=1.0e-4)
    parser.add_argument("--diag-period", type=int, default=50)
    parser.add_argument("--dry-run", action="store_true", help="Only write the input deck without running WarpX.")
    parser.add_argument("--force-write", action="store_true", help="Overwrite an existing deck file.")
    args = parser.parse_args()

    params = {
        "max_step": args.max_step,
        "n_cell": args.n_cell,
        "length": args.length,
        "density_peak": args.density_peak,
        "density_floor": args.density_floor,
        "ramp_start": args.ramp_start,
        "ramp_length": args.ramp_length,
        "particles_per_cell": args.particles_per_cell,
        "thermal_beta": args.thermal_beta,
        "lambda0": args.lambda0,
        "a0": args.a0,
        "laser_duration": args.laser_duration,
        "laser_waist": args.laser_waist,
        "laser_position": args.laser_position,
        "diag_period": args.diag_period,
        "diag_prefix": str(args.output),
    }

    if args.deck.exists() and not args.force_write:
        print(f"Deck {args.deck} already exists (use --force-write to overwrite).")
    else:
        _write_template(args.deck, params)
        print(f"WarpX deck written to {args.deck}")

    if args.dry_run:
        return 0

    args.output.mkdir(parents=True, exist_ok=True)
    cmd = [c for c in args.warpx_cmd.split() if c]
    cmd.append(str(args.deck))
    env = os.environ.copy()
    env.setdefault("DIAG_PREFIX", str(args.output))

    print(f"Running WarpX: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True, env=env)
    except FileNotFoundError as exc:
        raise SystemExit(f"WarpX executable '{args.warpx_cmd}' not found") from exc
    except subprocess.CalledProcessError as exc:
        raise SystemExit(f"WarpX execution failed with exit code {exc.returncode}") from exc

    # Copy deck into output for provenance
    shutil.copy2(args.deck, args.output / args.deck.name)
    print(f"Simulation complete. Diagnostics available in {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
