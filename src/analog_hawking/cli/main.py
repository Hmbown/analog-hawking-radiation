from __future__ import annotations

import argparse
import json
import os
import sys
import webbrowser
from pathlib import Path
from typing import Dict, Optional

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
    
    # Add "What just happened?" explanation
    if not args.quiet:
        print("\n" + "="*60)
        print("What just happened?")
        print("="*60)
        print(f"1. 🌊 Created a synthetic plasma flow profile")
        print(f"2. 🎯 Found {outputs['n_horizons']} sonic horizon(s) where |v| = c_s")
        if outputs['n_horizons'] > 0:
            print(f"3. ⚡ Computed surface gravity: κ ≈ {outputs['kappa_s_inv'][0]:.2e} s⁻¹")
            print(f"4. 🌡️  Equivalent Hawking temperature: T_H ≈ {1.22e-23 * outputs['kappa_s_inv'][0] / (2 * np.pi * 1.38e-23):.2e} K")
        print(f"5. 📊 Saved results to: {results_dir}/")
        print(f"6. 🖼️  Visualization: {results_dir}/quickstart_profile.png")
        print("\nNext steps:")
        print("  ahr pipeline --demo       # Run full detection pipeline")
        print("  ahr tutorial 1            # Learn about sonic horizons")
        print("  ahr docs                  # Open documentation")
        print("="*60)
    
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
    
    # Add dashboard output if requested
    if getattr(args, "dashboard", False) and isinstance(summary, dict):
        print("\n" + "="*60)
        print("Validation Dashboard")
        print("="*60)
        print(f"Overall Status: {'✅ PASS' if summary.get('overall_status') == 'PASS' else '❌ FAIL'}")
        print()
        
        if "test_categories" in summary:
            for category, results in summary["test_categories"].items():
                status = "✅" if results.get("status") == "PASS" else "⚠️" if results.get("status") == "WARN" else "❌"
                print(f"{status} {category}: {results.get('passed', 0)}/{results.get('total', 0)} tests")
                if "message" in results:
                    print(f"   {results['message']}")
        print("="*60)
    
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


def cmd_pipeline(args: argparse.Namespace) -> int:
    """Unified pipeline execution command"""
    import subprocess
    
    if args.demo:
        script = "scripts/run_full_pipeline.py"
        cmd = [sys.executable, script, "--demo"]
        
        if args.safe:
            cmd.extend(["--safe-demo", "--respect-thresholds"])
        
        if args.kappa_method:
            cmd.extend(["--kappa-method", args.kappa_method])
        
        if args.graybody:
            cmd.extend(["--graybody", args.graybody])
            
        if args.output:
            cmd.extend(["--output-dir", args.output])
            
        print(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd)
        return result.returncode
    
    print("Pipeline command - use --demo for demonstration")
    print("\nExamples:")
    print("  ahr pipeline --demo                    # Basic demo")
    print("  ahr pipeline --demo --safe            # Conservative demo")
    print("  ahr pipeline --demo --kappa-method acoustic_exact --graybody acoustic_wkb")
    return 0


def cmd_sweep(args: argparse.Namespace) -> int:
    """Parameter sweep execution"""
    import subprocess
    
    if args.gradient:
        script = "scripts/sweep_gradient_catastrophe.py"
        cmd = [sys.executable, script]
        
        if args.n_samples:
            cmd.extend(["--n-samples", str(args.n_samples)])
            
        if args.output:
            cmd.extend(["--output", args.output])
        else:
            cmd.extend(["--output", "results/gradient_catastrophe"])
            
        print(f"Running gradient catastrophe sweep: {' '.join(cmd)}")
        result = subprocess.run(cmd)
        return result.returncode
    
    print("Sweep command - use --gradient for gradient catastrophe analysis")
    print("\nExamples:")
    print("  ahr sweep --gradient                   # Default sweep")
    print("  ahr sweep --gradient --n-samples 500  # Custom sample count")
    print("  ahr sweep --gradient --output results/my_sweep")
    return 0


def cmd_analyze(args: argparse.Namespace) -> int:
    """Analysis and visualization tools"""
    import subprocess
    
    if args.correlation:
        script = "scripts/correlation_map.py"
        cmd = [sys.executable, script]
        
        if args.input:
            cmd.extend(["--series", args.input])
        if args.output:
            cmd.extend(["--output", args.output])
            
        print(f"Running correlation analysis: {' '.join(cmd)}")
        result = subprocess.run(cmd)
        return result.returncode
    
    print("Analyze command - tools for post-processing")
    print("\nExamples:")
    print("  ahr analyze --correlation              # Horizon correlation maps")
    print("  ahr analyze --kappa-inference          # Infer κ from PSDs")
    print("  ahr analyze --universality             # Spectrum collapse analysis")
    return 0


def cmd_experiment(args: argparse.Namespace) -> int:
    """Experiment planning and facility tools"""
    import subprocess
    
    if args.eli:
        script = "scripts/comprehensive_eli_facility_validator.py"
        cmd = [sys.executable, script]
        
        if args.output:
            cmd.extend(["--output-dir", args.output])
            
        print(f"Running ELI facility validation: {' '.join(cmd)}")
        result = subprocess.run(cmd)
        return result.returncode
    
    print("Experiment command - facility-specific tools")
    print("\nExamples:")
    print("  ahr experiment --eli                   # ELI facility planning")
    print("  ahr experiment --feasibility           # Detection feasibility analysis")
    print("  ahr experiment --optimize              # Parameter optimization")
    return 0


def cmd_docs(args: argparse.Namespace) -> int:
    """Open documentation in browser or show paths"""
    docs_dir = Path("docs")
    index_file = docs_dir / "index.md"
    
    if args.path:
        print(f"Documentation directory: {docs_dir.absolute()}")
        print(f"Main index: {index_file.absolute()}")
        
        # List key documentation files
        print("\nKey documentation files:")
        key_docs = [
            "index.md",
            "playbooks.md", 
            "GradientCatastropheAnalysis.md",
            "Methods.md",
            "Limitations.md",
            "Glossary.md"
        ]
        for doc in key_docs:
            doc_path = docs_dir / doc
            if doc_path.exists():
                print(f"  - {doc}")
        return 0
    
    # Try to open in browser
    readme_file = Path("README.md")
    if readme_file.exists():
        try:
            if sys.platform == "darwin":
                subprocess.run(["open", str(readme_file.absolute())])
            elif sys.platform == "win32":
                subprocess.run(["start", str(readme_file.absolute())], shell=True)
            else:
                subprocess.run(["xdg-open", str(readme_file.absolute())])
            print(f"Opening documentation: {readme_file.absolute()}")
        except Exception as e:
            print(f"Could not open browser: {e}")
            print(f"Documentation is at: {readme_file.absolute()}")
    else:
        print(f"README.md not found at: {readme_file.absolute()}")
    
    return 0


def cmd_info(args: argparse.Namespace) -> int:
    """Show system information and capabilities"""
    import platform
    
    info = {
        "system": platform.system(),
        "python_version": platform.python_version(),
        "ahr_version": __version__,
        "working_directory": str(Path.cwd().absolute()),
    }
    
    # Check for optional dependencies
    try:
        import cupy
        info["cupy"] = cupy.__version__
    except ImportError:
        info["cupy"] = "not installed"
    
    try:
        import matplotlib
        info["matplotlib"] = matplotlib.__version__
    except ImportError:
        info["matplotlib"] = "not installed"
    
    # Check environment variables
    info["env_force_cpu"] = os.getenv("ANALOG_HAWKING_FORCE_CPU", "not set")
    info["env_no_plots"] = os.getenv("ANALOG_HAWKING_NO_PLOTS", "not set")
    
    if args.json:
        print(json.dumps(info, indent=2))
    else:
        print("Analog Hawking Radiation System Information")
        print("=" * 60)
        print(f"Version:        v{info['ahr_version']}")
        print(f"Python:         {info['python_version']}")
        print(f"System:         {info['system']}")
        print(f"CuPy:           {info['cupy']}")
        print(f"Working dir:    {info['working_directory']}")
        print()
        print("Environment:")
        print(f"  ANALOG_HAWKING_FORCE_CPU: {info['env_force_cpu']}")
        print(f"  ANALOG_HAWKING_NO_PLOTS:  {info['env_no_plots']}")
        print("=" * 60)
        print("\nCapabilities:")
        print("  ✅ Horizon finding & analysis")
        print("  ✅ Graybody & detection modeling")
        print("  ✅ Parameter sweeps & optimization")
        if info["cupy"] != "not installed":
            print("  ✅ GPU acceleration available")
        else:
            print("  ⚠️  GPU acceleration not available (install cupy)")
    
    return 0


def cmd_tutorial(args: argparse.Namespace) -> int:
    """Interactive tutorial system"""
    tutorial_num = args.number
    
    tutorials = {
        1: {
            "title": "What is a Sonic Horizon?",
            "description": "Learn how plasma flows create analog black hole horizons",
            "script": "tutorials/01_sonic_horizons.py"
        },
        2: {
            "title": "From Surface Gravity to Hawking Temperature", 
            "description": "Understand κ → T_H mapping and its physical meaning",
            "script": "tutorials/02_kappa_to_temperature.py"
        },
        3: {
            "title": "Detection Forecasts in Realistic Experiments",
            "description": "How we estimate measurable signals from analog radiation",
            "script": "tutorials/03_detection_forecasts.py"
        }
    }
    
    if tutorial_num == 0 or args.list:
        print("Available Tutorials")
        print("=" * 60)
        for num, tutorial in tutorials.items():
            print(f"\n{num}. {tutorial['title']}")
            print(f"   {tutorial['description']}")
        print("\nRun: ahr tutorial <number>")
        return 0
    
    if tutorial_num not in tutorials:
        print(f"Tutorial {tutorial_num} not found. Run 'ahr tutorial --list' to see available tutorials.")
        return 1
    
    tutorial = tutorials[tutorial_num]
    print(f"\nTutorial {tutorial_num}: {tutorial['title']}")
    print("=" * 60)
    print(f"{tutorial['description']}")
    print("=" * 60)
    
    # For now, show what the tutorial would cover
    # In the future, this would run an interactive notebook or script
    print("\n🚧 Interactive tutorial system coming soon!")
    print("\nFor now, try these commands:")
    print("  ahr quickstart           # See horizons in action")
    print("  ahr pipeline --demo      # Full pipeline demonstration")
    print("  ahr docs --path          # Explore documentation")
    
    return 0


def cmd_dev(args: argparse.Namespace) -> int:
    """Development and contribution tools"""
    if args.setup:
        print("Setting up development environment...")
        print("=" * 60)
        
        # Check Python version
        import platform
        py_version = platform.python_version()
        print(f"✅ Python version: {py_version}")
        
        # Install dev dependencies
        try:
            import subprocess
            result = subprocess.run([sys.executable, "-m", "pip", "install", "-e", ".[dev]"], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                print("✅ Development dependencies installed")
            else:
                print("❌ Failed to install dev dependencies")
                print(result.stderr)
        except Exception as e:
            print(f"❌ Error installing dependencies: {e}")
        
        # Run smoke tests
        print("\nRunning smoke tests...")
        try:
            result = subprocess.run([sys.executable, "-m", "pytest", "-q", "--tb=short"], 
                                  capture_output=True, text=True, timeout=60)
            if result.returncode == 0:
                print("✅ Tests passing")
            else:
                print("⚠️  Some tests failed - check output above")
        except subprocess.TimeoutExpired:
            print("⚠️  Tests timed out")
        except Exception as e:
            print(f"⚠️  Could not run tests: {e}")
        
        print("\n" + "=" * 60)
        print("Development setup complete!")
        print("\nNext steps:")
        print("  ahr validate            # Run full validation suite")
        print("  ahr tutorial --list     # Explore tutorials")
        print("  code .                  # Open in VS Code")
        return 0
    
    print("Development tools")
    print("\nExamples:")
    print("  ahr dev setup           # Set up development environment")
    print("  ahr dev test            # Run test suite")
    print("  ahr dev lint            # Run linting")
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
    q.add_argument("--quiet", action="store_true", help="Suppress explanatory output")
    q.set_defaults(func=cmd_quickstart)

    # validate
    v = sub.add_parser("validate", help="Run comprehensive physics validation suite")
    v.add_argument(
        "--report", type=str, help="Write JSON validation summary to this path", default=None
    )
    v.add_argument(
        "--dashboard", action="store_true", help="Show visual dashboard of validation results"
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

    # pipeline (NEW)
    pipe = sub.add_parser("pipeline", help="Run analysis pipelines")
    pipe.add_argument("--demo", action="store_true", help="Run demonstration pipeline")
    pipe.add_argument("--safe", action="store_true", help="Use conservative settings")
    pipe.add_argument("--kappa-method", type=str, help="Kappa calculation method")
    pipe.add_argument("--graybody", type=str, help="Graybody model")
    pipe.add_argument("--output", type=str, help="Output directory")
    pipe.set_defaults(func=cmd_pipeline)

    # sweep (NEW)
    sweep = sub.add_parser("sweep", help="Run parameter sweeps")
    sweep.add_argument("--gradient", action="store_true", help="Run gradient catastrophe sweep")
    sweep.add_argument("--n-samples", type=int, help="Number of samples")
    sweep.add_argument("--output", type=str, help="Output directory")
    sweep.set_defaults(func=cmd_sweep)

    # analyze (NEW)
    analyze = sub.add_parser("analyze", help="Analysis and visualization tools")
    analyze.add_argument("--correlation", action="store_true", help="Run correlation analysis")
    analyze.add_argument("--input", type=str, help="Input data path")
    analyze.add_argument("--output", type=str, help="Output path")
    analyze.set_defaults(func=cmd_analyze)

    # experiment (NEW)
    exp = sub.add_parser("experiment", help="Experiment planning tools")
    exp.add_argument("--eli", action="store_true", help="ELI facility planning")
    exp.add_argument("--output", type=str, help="Output directory")
    exp.set_defaults(func=cmd_experiment)

    # docs (NEW)
    docs = sub.add_parser("docs", help="Open documentation")
    docs.add_argument("--path", action="store_true", help="Show documentation paths")
    docs.set_defaults(func=cmd_docs)

    # info (NEW)
    info = sub.add_parser("info", help="Show system information")
    info.add_argument("--json", action="store_true", help="JSON output format")
    info.set_defaults(func=cmd_info)

    # tutorial (NEW)
    tutorial = sub.add_parser("tutorial", help="Interactive tutorials")
    tutorial.add_argument("number", type=int, nargs="?", default=0, 
                         help="Tutorial number (0 for list)")
    tutorial.add_argument("--list", action="store_true", help="List available tutorials")
    tutorial.set_defaults(func=cmd_tutorial)

    # dev (NEW)
    dev = sub.add_parser("dev", help="Development tools")
    dev.add_argument("--setup", action="store_true", help="Set up development environment")
    dev.set_defaults(func=cmd_dev)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
