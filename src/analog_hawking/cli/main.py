from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict

import numpy as np

from .. import __version__
from ..config.schemas import QuickstartConfig
from ..physics_engine.horizon import find_horizons_with_uncertainty, sound_speed
from ..utils.provenance import build_manifest, write_manifest


def _should_plot() -> bool:
    return os.getenv("ANALOG_HAWKING_NO_PLOTS", "").lower() not in {"1", "true", "yes", "on"}


def cmd_quickstart(args: argparse.Namespace) -> int:
    cfg = QuickstartConfig(
        nx=args.nx,
        x_min=args.x_min,
        x_max=args.x_max,
        v0=args.v0,
        x0=args.x0,
        L=args.L,
        Te=args.Te,
        results_dir=args.out,
    )

    # Build synthetic profile
    x = np.linspace(cfg.x_min, cfg.x_max, cfg.nx)
    # Snap a grid point to x0 to ensure root bracketing around the horizon center
    j = int(np.argmin(np.abs(x - cfg.x0)))
    x[j] = cfg.x0
    v = cfg.v0 * np.tanh((x - cfg.x0) / cfg.L)
    Te = np.full_like(x, cfg.Te, dtype=float)
    cs = sound_speed(Te)

    horizons = find_horizons_with_uncertainty(x, v, cs)

    results_dir = Path(cfg.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    outputs: Dict[str, object] = {
        "n_horizons": int(horizons.positions.size),
        "positions_m": horizons.positions.tolist(),
        "kappa_s_inv": horizons.kappa.tolist(),
        "kappa_err_s_inv": horizons.kappa_err.tolist(),
    }

    # Save small JSON with positions/κ
    (results_dir / "horizons.json").write_text(json.dumps(outputs, indent=2))

    # Save an optional figure
    if _should_plot():
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6, 3))
        ax.plot(x * 1e6, np.abs(v), label="|v|")
        ax.plot(x * 1e6, cs, label="c_s")
        for pos in horizons.positions:
            ax.axvline(pos * 1e6, color="r", ls="--", alpha=0.6)
        ax.set_xlabel("x [µm]")
        ax.set_ylabel("Speed [m/s]")
        ax.set_title("Quickstart: horizon(s) where |v| = c_s")
        ax.legend(loc="best")
        fig.tight_layout()
        fig.savefig(results_dir / "quickstart_profile.png", dpi=160)
        plt.close(fig)

    # Write manifest with provenance
    manifest = build_manifest(
        tool="ahr quickstart",
        config=cfg.model_dump() if hasattr(cfg, "model_dump") else cfg.__dict__,
        inputs={
            "nx": cfg.nx,
            "domain": [cfg.x_min, cfg.x_max],
            "v_profile": "tanh",
            "Te": cfg.Te,
        },
        outputs=outputs,
    )
    write_manifest(manifest, results_dir / "manifest.json")

    print(f"Quickstart complete. Results in: {results_dir}")
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    from ..physics_engine.physics_validation_framework import run_comprehensive_validation

    summary = run_comprehensive_validation()
    report_path = getattr(args, "report", None)
    if isinstance(summary, dict) and report_path:
        out = Path(report_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summary, indent=2))
        print(f"Wrote validation summary to {out}")
    return 0 if isinstance(summary, dict) and summary.get("overall_status") == "PASS" else 1


def cmd_bench(args: argparse.Namespace) -> int:
    import time

    # Tiny micro-benchmark of horizon finder
    x = np.linspace(0, 100e-6, 2000)
    v = 0.1 * 3e8 * np.tanh((x - 50e-6) / 8e-6)
    Te = np.full_like(x, 8e5)
    cs = sound_speed(Te)
    t0 = time.perf_counter()
    _ = find_horizons_with_uncertainty(x, v, cs)
    dt_ms = (time.perf_counter() - t0) * 1e3
    if getattr(args, "json", False):
        print(json.dumps({"kernel": "horizon_finder", "nx": int(x.size), "ms": dt_ms}, indent=2))
    else:
        print(f"horizon_finder: {dt_ms:.2f} ms for nx={x.size}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="ahr", description="Analog Hawking Radiation CLI")
    p.add_argument("--version", action="version", version=f"ahr {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    # quickstart
    q = sub.add_parser("quickstart", help="Run a synthetic quickstart and save results")
    q.add_argument("--nx", type=int, default=1000)
    q.add_argument("--x-min", dest="x_min", type=float, default=0.0)
    q.add_argument("--x-max", dest="x_max", type=float, default=100e-6)
    q.add_argument("--v0", type=float, default=0.1 * 3e8)
    q.add_argument("--x0", type=float, default=50e-6)
    q.add_argument("--L", type=float, default=10e-6)
    q.add_argument("--Te", type=float, default=1e6)
    q.add_argument("--out", type=str, default="results/quickstart")
    q.set_defaults(func=cmd_quickstart)

    # validate
    v = sub.add_parser("validate", help="Run comprehensive physics validation suite")
    v.add_argument(
        "--report", type=str, help="Write JSON validation summary to this path", default=None
    )
    v.set_defaults(func=cmd_validate)

    # bench
    b = sub.add_parser("bench", help="Run a small benchmark of core kernels")
    b.add_argument("--json", action="store_true", help="Emit JSON output")
    b.set_defaults(func=cmd_bench)

    # gpu-info
    gi = sub.add_parser("gpu-info", help="Show active array backend and CuPy availability")

    def _cmd_gpu_info(_args: argparse.Namespace) -> int:
        from ..utils import array_module as am

        backend = "cupy" if am.using_cupy() else "numpy"
        try:
            import cupy as _cupy  # noqa: F401

            cupy_status = "installed"
        except Exception:
            cupy_status = "not installed"
        print(
            json.dumps(
                {
                    "backend": backend,
                    "cupy": cupy_status,
                    "ANALOG_HAWKING_FORCE_CPU": os.getenv("ANALOG_HAWKING_FORCE_CPU", ""),
                    "ANALOG_HAWKING_USE_CUPY": os.getenv("ANALOG_HAWKING_USE_CUPY", ""),
                },
                indent=2,
            )
        )
        return 0

    gi.set_defaults(func=_cmd_gpu_info)

    # regress
    def _cmd_regress(_args: argparse.Namespace) -> int:
        # Compute quickstart with default params and compare to golden
        cfg = QuickstartConfig()
        x = np.linspace(cfg.x_min, cfg.x_max, cfg.nx)
        j = int(np.argmin(np.abs(x - cfg.x0)))
        x[j] = cfg.x0
        v = cfg.v0 * np.tanh((x - cfg.x0) / cfg.L)
        Te = np.full_like(x, cfg.Te, dtype=float)
        cs = sound_speed(Te)
        hr = find_horizons_with_uncertainty(x, v, cs)

        golden_path = Path("goldens/quickstart_horizons.json")
        data = json.loads(golden_path.read_text())
        ok = True
        # Count
        if int(hr.positions.size) != int(data["n_horizons"]):
            print(f"[regress] n_horizons mismatch: {hr.positions.size} != {data['n_horizons']}")
            ok = False
        # Positions absolute tolerance
        pos_tol = float(data["tolerances"].get("positions_abs", 1e-9))
        if hr.positions.size == data["n_horizons"]:
            for i, (p, g) in enumerate(zip(hr.positions.tolist(), data["positions_m"])):
                if abs(p - g) > pos_tol:
                    print(f"[regress] pos[{i}] off by {abs(p-g):.3e} > {pos_tol:.3e}")
                    ok = False
        # Kappa relative tolerance
        kap_tol = float(data["tolerances"].get("kappa_rel", 1e-3))
        if hr.kappa.size == len(data["kappa_s_inv"]):
            for i, (k, g) in enumerate(zip(hr.kappa.tolist(), data["kappa_s_inv"])):
                rel = abs(k - g) / max(abs(g), 1e-30)
                if rel > kap_tol:
                    print(f"[regress] kappa[{i}] rel diff {rel:.3e} > {kap_tol:.3e}")
                    ok = False
        print("[regress] status:", "PASS" if ok else "FAIL")
        return 0 if ok else 1

    rg = sub.add_parser("regress", help="Run golden regression checks")
    rg.set_defaults(func=_cmd_regress)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
