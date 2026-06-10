#!/usr/bin/env python3
"""Build CD-ViT/faceswap-compatible landmark manifests (CLI entry point).

The build logic was split out of this formerly 6.5k-line module into the
:mod:`lib.datasets.build` package: shared infrastructure (parsing, image
indexing/cropping, sample + manifest emission, pose metadata, and the generic
directory/JSON builders) lives in ``lib.datasets.build.core``; each dataset has
its own builder module (``w300``, ``helen``, ``lapa``, ``jd_landmark``, ``ffl``,
``subject_session``, ``multipie``, ``cofw``, ``wflw``, ``merl_rav``, ``video``);
and CLI dispatch lives in ``lib.datasets.build.orchestrator``.

This file is retained as the stable ``tools/build_quality_dataset.py`` entry
point. It re-exports the public and internal API (``build``, ``main``,
``_parser`` and the helpers that ``prepare_landmark_dataset.py`` and the test
suite import) so existing callers keep working unchanged.
"""

# ruff: noqa: E402, F401, F403, F405
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# Shared infrastructure + generic builders, then CLI dispatch. Each module
# defines an ``__all__`` that includes its single-underscore helpers, so these
# star imports re-export the full former surface of this module.
from lib.datasets.build.core import *  # noqa: F403
from lib.datasets.build.orchestrator import *  # noqa: F403

# Explicit re-exports for the symbols that callers/tests reference by name, so
# they remain importable even if an ``__all__`` is ever tightened.
from lib.datasets.build.core import (
    SUPPORTED_DATASETS,
    _condition_for_landmark_file,
    _dataset_condition_label,
    _draw_manifest_overlay,
    _load_landmark_file,
    _pose_metadata,
    _sample,
    _write_manifest,
    _write_visual_audit,
)
from lib.datasets.build.orchestrator import _parser, build, main

if __name__ == "__main__":
    raise SystemExit(main())
