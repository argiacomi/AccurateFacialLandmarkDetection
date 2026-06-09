"""Shared aliases for schema-aware landmark manifest dataset names."""

from __future__ import annotations


LEGACY_MANIFEST_DATA_NAME = "FS68Manifest"
CANONICAL_MANIFEST_DATA_NAME = "MultiSchemaLandmarkManifest"

MANIFEST_DATA_NAME_ALIASES = (
    LEGACY_MANIFEST_DATA_NAME,
    "LandmarkManifest",
    "SchemaAwareManifest",
    CANONICAL_MANIFEST_DATA_NAME,
)

SCHEMA_AWARE_MANIFEST_DATASET_NAMES = frozenset(MANIFEST_DATA_NAME_ALIASES)


def is_schema_aware_manifest_dataset(name: str) -> bool:
    return name in SCHEMA_AWARE_MANIFEST_DATASET_NAMES


def IsSchemaAwareManifestDataset(name: str) -> bool:
    """Backward-compatible helper name used by older code/tests."""
    return is_schema_aware_manifest_dataset(name)
