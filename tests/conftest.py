from __future__ import annotations

import sys
from pathlib import Path
import warnings

REPO_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT_STR = str(REPO_ROOT)

if REPO_ROOT_STR not in sys.path:
    sys.path.insert(0, REPO_ROOT_STR)

warnings.filterwarnings(
    "ignore",
    message=r"`torch\.jit\.script_method` is deprecated.*",
    category=DeprecationWarning,
    module=r"torch\.jit\._script",
)
