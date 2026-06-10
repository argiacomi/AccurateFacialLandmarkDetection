"""Landmark dataset builders, split from the monolithic build tool.

``build_quality_dataset`` used to carry every dataset's build logic in one
6.5k-line module. That logic now lives here: :mod:`lib.datasets.build.core`
holds the shared infrastructure (parsing, image indexing/cropping, sample and
manifest emission, pose metadata, and the generic directory/JSON builders),
while each dataset-specific builder lives in its own sibling module. The CLI
entry point and dataset dispatch live in :mod:`lib.datasets.build.orchestrator`.
"""

from __future__ import annotations

from lib.datasets.build.core import SUPPORTED_DATASETS
from lib.datasets.build.orchestrator import build, main

__all__ = ["SUPPORTED_DATASETS", "build", "main"]
