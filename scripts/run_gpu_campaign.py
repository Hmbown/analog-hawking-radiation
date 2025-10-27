#!/usr/bin/env python3
"""
GPU-accelerated campaign runner for analog Hawking workflows.

Designed for single-GPU workstations (e.g., RTX 3080) to launch the heaviest
parameter sweeps with CuPy enabled. The script orchestrates multiple existing
drivers (gradient catastrophe, universality collapse, detection inference) while
setting the appropriate environment so CuPy kernels are used automatically.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import textwrap
import time
from pathlib import Path
from typing import Dict, List, Sequence


def _detect_gpu() -> Dict[str, str]:
    """Return lightweight GPU metadata when CuPy can see a CUDA device."""
    try:
        import importlib

        cp = importlib.import_module("cupy")
        device_id = cp.cuda.runtime.getDevice()
        props = cp.cuda.runtime.getDeviceProperties(device_id)
        free_mem, total_mem = cp.cuda.runtime.memGetInfo()
        return {
            "name": props["name"].decode() if isinstance(props["name"], bytes) else str(props["name"]),
            "total_mem_gb": f"{total_mem / 1024**3:.1f}",
            "free_mem_gb": f"{free_mem / 1024**3:.1f}",
            "multi_processor_count": str(props["multiProcessorCount"]),
            "compute_capability": f"{props['major']}.{props['minor']}",
        }
    except Exception:
        return {}


def _run_command(cmd: Sequence[str], env: Dict[str, str], workdir: Path) -> int:
    """Run a subprocess command, streaming output live."""
    start = time.time()
    print(f"\n>>> Executing: {' '.join(cmd)}")
    proc = subprocess.Popen(cmd, cwd=workdir, env=env, stdout=sys.stdout, stderr=sys.stderr)
    ret = proc.wait()
    duration = time.time() - start
    print(f"<<< Command finished with code {ret} (elapsed {duration:.1f} s)")
    return ret


def _campaign_tasks(args: argparse.Namespace, results_dir: Path) -> List[Dict[str, Sequence[str]]]:
    """Build the list of orchestration tasks."""
    tasks: List[Dict[str, Sequence[str]]] = []

    if "gradient" in args.tasks:
        gradient_dir = results_dir / "gradient_limits_gpu"
        gradient_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
            sys.executable,
            "scripts/sweep_gradient_catastrophe.py",
            "--n-samples",
            str(args.gradient_samples),
            "--output",
            str(gradient_dir),
        ]
        tasks.append({"name": "Gradient Catastrophe Sweep", "cmd": cmd})

    if "universality" in args.tasks:
        universality_dir = results_dir / "universality_gpu"
        universality_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
            sys.executable,
            "scripts/experiment_universality_collapse.py",
            "--out",
            str(universality_dir),
            "--n",
            str(args.universality_families),
            "--alpha",
            "0.8",
            "--seed",
            str(args.seed),
        ]
        tasks.append({"name": "Universality Collapse", "cmd": cmd})

    if "detection" in args.tasks:
        detection_dir = results_dir / "detection_gpu"
        detection_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
            sys.executable,
            "scripts/experiment_kappa_inference_benchmark.py",
            "--n",
            str(args.detection_trials),
            "--families",
            "linear",
            "tanh",
            "ramp",
            "--noise",
            "0.08",
            "--out",
            str(detection_dir),
            "--seed",
            str(args.seed),
        ]
        tasks.append({"name": "Detection / κ inference benchmark", "cmd": cmd})

    return tasks


def _summarise_results(results_dir: Path) -> None:
    """Print quick summaries of artefacts produced by tasks."""
    summary_files = {
        "gradient": results_dir / "gradient_limits_gpu" / "gradient_catastrophe_sweep.json",
        "universality": results_dir / "universality_gpu" / "universality_summary.json",
        "detection": results_dir / "detection_gpu" / "kappa_inference_summary.json",
    }
    for key, path in summary_files.items():
        if path.exists():
            try:
                data = json.loads(path.read_text())
            except Exception as exc:  # pragma: no cover - diagnostic only
                print(f"[WARN] Could not parse {path}: {exc}")
                continue
            print(f"\n--- {key.upper()} SUMMARY ({path}) ---")
            print(json.dumps(data, indent=2))


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="GPU-accelerated campaign runner for analog Hawking workflows",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent(
            """\
            Tasks:
              gradient    Run the gradient catastrophe sweep with GPU acceleration.
              universality    Execute spectrum collapse experiment (CuPy-enabled).
              detection   Launch κ-inference benchmark with GPU kernels.

            Example:
              ANALOG_HAWKING_USE_CUPY=1 python scripts/run_gpu_campaign.py --tasks gradient universality --gradient-samples 1500
            """
        ),
    )

    parser.add_argument(
        "--tasks",
        nargs="+",
        default=["gradient"],
        choices=["gradient", "universality", "detection"],
        help="Subset of workloads to launch.",
    )
    parser.add_argument("--gradient-samples", type=int, default=1500, help="Samples for gradient sweep.")
    parser.add_argument("--universality-families", type=int, default=48, help="Number of flow families for universality test.")
    parser.add_argument("--detection-trials", type=int, default=96, help="Number of PSD realisations in κ inference benchmark.")
    parser.add_argument("--seed", type=int, default=13, help="Random seed forwarded to experiments where supported.")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results/gpu_campaign"),
        help="Root directory for artefacts.",
    )
    parser.add_argument(
        "--force-cpu",
        action="store_true",
        help="Disable CuPy even if available (useful for debugging).",
    )
    parser.add_argument(
        "--cuda-visible-devices",
        type=str,
        default=None,
        help="Override CUDA_VISIBLE_DEVICES before launching tasks.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    workdir = Path(__file__).resolve().parent.parent

    gpu_info = _detect_gpu() if not args.force_cpu else {}
    if gpu_info:
        print(
            f"Detected GPU: {gpu_info.get('name', 'unknown')} "
            f"(CC {gpu_info.get('compute_capability', '?')}, "
            f"{gpu_info.get('free_mem_gb', '?')}/{gpu_info.get('total_mem_gb', '?')} GiB free)"
        )
    else:
        msg = "CuPy GPU backend not detected; falling back to CPU."
        if args.force_cpu:
            msg += " (--force-cpu enabled)"
        print(msg)

    env = os.environ.copy()
    if args.cuda_visible_devices is not None:
        env["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices

    if args.force_cpu:
        env["ANALOG_HAWKING_FORCE_CPU"] = "1"
    else:
        env.setdefault("ANALOG_HAWKING_USE_CUPY", "1")
        env.pop("ANALOG_HAWKING_FORCE_CPU", None)

    args.results_dir.mkdir(parents=True, exist_ok=True)
    tasks = _campaign_tasks(args, args.results_dir)

    if not tasks:
        print("No tasks selected; exiting.")
        return 0

    for task in tasks:
        print(f"\n=== {task['name']} ===")
        code = _run_command(task["cmd"], env, workdir)
        if code != 0:
            print(f"[ERROR] Task '{task['name']}' failed with exit code {code}. Aborting campaign.")
            return code

    _summarise_results(args.results_dir)
    print("\nAll selected GPU tasks completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

