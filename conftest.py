"""Test configuration shared across the repository.

Ensures the ``src`` directory is importable so that compatibility shims
(such as ``physics_engine``) resolve in example and script tests as well.
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).parent.resolve()
SRC_ROOT = str(PROJECT_ROOT / "src")

if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)
