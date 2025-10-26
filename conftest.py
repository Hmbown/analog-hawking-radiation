"""Test configuration shared across the repository.

Ensures the ``src`` directory is importable so that compatibility shims
(such as ``physics_engine``) resolve in example and script tests as well.
"""

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SRC_ROOT = os.path.join(PROJECT_ROOT, "src")

if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)
