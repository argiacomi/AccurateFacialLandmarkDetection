"""Schema-aware landmark manifest contract and validation helpers."""

from .contract import (
    TRAINING_MANIFEST_CONTRACT,
    TRAINING_MANIFEST_VERSION,
    TRAINABLE_SCHEMA_HEADS,
    manifest_summary,
    split_safe_id_for_sample,
)
from .validator import validate_training_manifest

__all__ = [
    "TRAINING_MANIFEST_CONTRACT",
    "TRAINING_MANIFEST_VERSION",
    "TRAINABLE_SCHEMA_HEADS",
    "manifest_summary",
    "split_safe_id_for_sample",
    "validate_training_manifest",
]
