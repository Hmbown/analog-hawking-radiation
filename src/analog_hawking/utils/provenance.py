"""Provenance and results manifest utilities.

This module centralizes reproducibility metadata collection (git commit,
environment, package version) and standardized manifest writing for results.
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from .. import __version__ as package_version


def _run_cmd(args: list[str]) -> Optional[str]:
    try:
        out = subprocess.check_output(args, stderr=subprocess.DEVNULL)
        return out.decode().strip()
    except Exception:
        return None


def git_commit() -> str:
    """Return current git commit hash (short), or "unknown" if unavailable."""
    commit = _run_cmd(["git", "rev-parse", "--short", "HEAD"]) or "unknown"
    dirty = _run_cmd(["git", "status", "--porcelain"])
    if dirty:
        commit = f"{commit}+dirty"
    return commit


def python_info() -> Dict[str, str]:
    return {
        "python": sys.version.split(" (", 1)[0],
        "implementation": platform.python_implementation(),
        "executable": sys.executable,
    }


def system_info() -> Dict[str, str]:
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "release": platform.release(),
        "system": platform.system(),
    }


@dataclass
class Manifest:
    tool: str
    version: str
    git_commit: str
    created_at: str
    config: Dict[str, Any]
    inputs: Dict[str, Any]
    outputs: Dict[str, Any]
    environment: Dict[str, Any]

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)


def build_manifest(
    tool: str,
    config: Mapping[str, Any],
    inputs: Mapping[str, Any],
    outputs: Mapping[str, Any],
) -> Manifest:
    now = datetime.now(tz=timezone.utc).isoformat()
    env = {
        "package_version": package_version,
        "python": python_info(),
        "system": system_info(),
        "env": {
            k: os.environ.get(k, "")
            for k in (
                "ANALOG_HAWKING_NO_PLOTS",
                "ANALOG_HAWKING_USE_CUPY",
                "ANALOG_HAWKING_FORCE_CPU",
            )
        },
    }
    return Manifest(
        tool=tool,
        version=package_version,
        git_commit=git_commit(),
        created_at=now,
        config=dict(config),
        inputs=dict(inputs),
        outputs=dict(outputs),
        environment=env,
    )


def write_manifest(manifest: Manifest, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(manifest.to_json())
    return path


__all__ = [
    "Manifest",
    "build_manifest",
    "write_manifest",
    "git_commit",
    "python_info",
    "system_info",
]
