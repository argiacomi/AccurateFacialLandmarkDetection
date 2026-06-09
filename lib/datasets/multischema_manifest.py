"""Canonical schema-aware landmark manifest dataset module.

`FS68Manifest` remains a supported `--data_name` alias for backwards
compatibility. The implementation still lives in `DatasetFS68Manifest.py` during
PR 1 to keep this rename low-risk.
"""

from lib.datasets.manifest import LandmarkDataset

__all__ = ["LandmarkDataset"]
