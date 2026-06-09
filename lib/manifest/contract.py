#!/usr/bin/env python3
"""Shared contract for schema-aware landmark training manifests.

The canonical training manifest is a mixed-schema manifest. Each sample
keeps its native trainable schema and declares the model head that should
consume it. The legacy ``FS68Manifest`` data name remains an alias at the
dataset-registration layer.
"""

from __future__ import annotations

import hashlib
import typing as T
from collections import Counter

from lib.core.schema import canonicalize_schema, head_name_for_schema

TRAINING_MANIFEST_CONTRACT = "schema_aware_landmark_manifest_v1"
TRAINING_MANIFEST_VERSION = 2

TRAINABLE_SCHEMA_HEADS: dict[str, str] = {
    "2d_29": "landmarks_29",
    "2d_39": "profile39",
    "menpo2d_profile_39": "profile39",
    "multipie_profile_39": "profile39",
    "2d_68": "landmarks_68",
    "2d_98": "landmarks_98",
    "2d_106": "landmarks_106",
    "2d_194": "landmarks_194",
}

REQUIRED_SAMPLE_FIELDS = (
    "sample_id",
    "image",
    "landmarks",
    "dataset",
    "split",
    "source_schema",
    "target_schema",
    "landmark_count",
    "head_name",
    "split_safe_id",
)

IDENTITY_FIELDS = (
    "image_id",
    "subject_id",
    "session_id",
    "video_id",
    "archive_id",
    "split_safe_id",
)


def _label(value: T.Any, default: str = "unknown") -> str:
    label = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    while "__" in label:
        label = label.replace("__", "_")
    return label.strip("_") or default


def _nested_value(sample: T.Mapping[str, T.Any], key: str) -> T.Any:
    if key in sample and sample.get(key) not in (None, ""):
        return sample.get(key)
    source = sample.get("source") if isinstance(sample.get("source"), dict) else {}
    if source.get(key) not in (None, ""):
        return source.get(key)
    metadata = (
        sample.get("metadata") if isinstance(sample.get("metadata"), dict) else {}
    )
    return metadata.get(key)


def trainable_head_for_schema(schema: str | object) -> str:
    schema_name = canonicalize_schema(schema)
    if schema_name in TRAINABLE_SCHEMA_HEADS:
        return TRAINABLE_SCHEMA_HEADS[schema_name]
    return head_name_for_schema(schema_name)


def split_safe_id_for_sample(sample: T.Mapping[str, T.Any]) -> str:
    """Return the stable identity used for train/test leakage checks."""
    for key in (
        "split_safe_id",
        "subject_id",
        "session_id",
        "video_id",
        "archive_id",
        "image_id",
    ):
        value = _nested_value(sample, key)
        if value not in (None, ""):
            return str(value)
    source = sample.get("source") if isinstance(sample.get("source"), dict) else {}
    dataset = str(sample.get("dataset") or source.get("dataset") or "unknown")
    source_id = str(
        source.get("source_id") or sample.get("sample_id") or sample.get("image") or ""
    )
    digest = hashlib.sha256(f"{dataset}|{source_id}".encode("utf-8")).hexdigest()[:16]
    return f"{dataset}:{digest}"


def schema_counts(samples: T.Iterable[T.Mapping[str, T.Any]]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for sample in samples:
        raw = sample.get("target_schema") or sample.get("source_schema")
        if raw:
            try:
                raw = canonicalize_schema(raw)
            except ValueError:
                raw = str(raw)
        counts[str(raw or "unknown")] += 1
    return dict(sorted(counts.items()))


def manifest_summary(
    samples: T.Sequence[T.Mapping[str, T.Any]],
) -> dict[str, dict[str, int]]:
    counters: dict[str, Counter[str]] = {
        "datasets": Counter(),
        "splits": Counter(),
        "source_schemas": Counter(),
        "target_schemas": Counter(),
        "heads": Counter(),
        "hard_negative_buckets": Counter(),
        "projection_status": Counter(),
    }
    for sample in samples:
        metadata = (
            sample.get("metadata") if isinstance(sample.get("metadata"), dict) else {}
        )
        source = sample.get("source") if isinstance(sample.get("source"), dict) else {}
        mapping_audit = sample.get("mapping_audit")
        if not isinstance(mapping_audit, dict):
            mapping_audit = metadata.get("mapping_audit")
        if not isinstance(mapping_audit, dict):
            mapping_audit = {}
        projection_audit = mapping_audit.get("projection_to_68")
        if not isinstance(projection_audit, dict):
            projection_audit = {}

        dataset = (
            sample.get("dataset")
            or source.get("dataset")
            or metadata.get("dataset")
            or "unknown"
        )
        split = sample.get("split") or metadata.get("split") or "unspecified"
        source_schema = (
            sample.get("source_schema") or metadata.get("source_schema") or "unknown"
        )
        target_schema = (
            sample.get("target_schema")
            or metadata.get("target_schema")
            or source_schema
        )
        head = sample.get("head_name") or metadata.get("head_name") or "unknown"
        bucket = (
            sample.get("hard_negative_bucket")
            or metadata.get("hard_negative_bucket")
            or metadata.get("condition")
            or sample.get("condition")
            or "unknown"
        )

        for key, value in (
            ("datasets", dataset),
            ("splits", split),
            ("source_schemas", source_schema),
            ("target_schemas", target_schema),
            ("heads", head),
            ("hard_negative_buckets", bucket),
            (
                "projection_status",
                projection_audit.get("status")
                or mapping_audit.get("status")
                or "unknown",
            ),
        ):
            counters[key][_label(value)] += 1

    return {key: dict(sorted(counter.items())) for key, counter in counters.items()}
