#!/usr/bin/env python3
"""Reusable hard-negative classification helpers for CD-ViT manifest training."""

from __future__ import annotations

import typing as T
from dataclasses import dataclass

PROFILE_LABELS: frozenset[str] = frozenset(
    {
        "pose",
        "profile",
        "profile_left",
        "profile_right",
        "large_yaw_left",
        "large_yaw_right",
        "large_roll",
        "extreme_roll",
        "rolled_profile_left",
        "rolled_profile_right",
        "rolled_large_yaw_left",
        "rolled_large_yaw_right",
        "left",
        "lefthalf",
        "right",
        "righthalf",
    }
)
OCCLUSION_LABELS: frozenset[str] = frozenset(
    {
        "occlusion",
        "occluded",
        "externally_occluded",
        "self_occluded",
        "profile_occlusion",
        "rolled_profile_occlusion",
        "large_yaw_occlusion",
        "single_eye_visible",
        "mouth_or_jaw_occluded",
    }
)
ANCHOR_LABELS: frozenset[str] = frozenset(
    {"normal", "frontal", "intermediate", "default", "clean"}
)

BUCKET_PRIORITY: dict[str, int] = {
    "profile_occlusion": 1,
    "profile": 2,
    "occlusion": 3,
    "anchor": 4,
}
BUCKET_WEIGHT: dict[str, float] = {
    "profile_occlusion": 5.0,
    "profile": 3.0,
    "occlusion": 2.0,
    "anchor": 1.0,
}
# Whole-dataset fallback bucket when a sample carries no explicit profile/
# occlusion labels. Some datasets are dominated by one regime (COFW is
# occlusion-heavy, MultiPIE is profile-heavy), so their "neutral" samples are
# better bucketed by dataset than forced to anchor.
DATASET_DEFAULT_BUCKET: dict[str, str] = {
    "cofw68": "occlusion",
    "cofw6868": "occlusion",
    "cofw29": "occlusion",
    "300w": "anchor",
    "w300": "anchor",
    "production_validated": "anchor",
    "multipie": "profile",
}
DEFAULT_HARD_NEGATIVE_WEIGHT: float = 1.0
MAX_HARD_NEGATIVE_WEIGHT: float = 5.0


@dataclass(frozen=True, slots=True)
class HardNegativeClass:
    """Bucket assignment for one mined hard-negative sample."""

    bucket: str
    priority: int
    weight: float
    reasons: tuple[str, ...]


def normalize_label(value: T.Any) -> str:
    """Normalize a raw label to snake_case with collapsed separators."""
    label = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    while "__" in label:
        label = label.replace("__", "_")
    return label.strip("_")


def _marks_occlusion(key: str, value: T.Any) -> bool:
    key = normalize_label(key)
    if value is None:
        return False
    if key in {"visibility", "visible"}:
        try:
            values = list(value)
        except TypeError:
            return False
        return bool(values) and not all(bool(item) for item in values)
    if isinstance(value, str):
        raw = value.strip().lower()
        return raw not in {"", "0", "false", "none", "no", "clean"}
    if isinstance(value, (list, tuple, set)):
        text = " ".join(str(item).lower() for item in value)
        return any(
            token in text
            for token in ("occluded", "self_occluded", "externally_occluded")
        )
    return bool(value)


def _add_runtime_bucket_labels(
    labels: set[str], payload: T.Mapping[str, T.Any]
) -> None:
    """Add faceswap production resolver bucket labels from nested metadata."""
    for key in (
        "runtime_bucket",
        "bucket",
        "landmark_ensemble_runtime_bucket",
        "landmark_ensemble_bucket",
    ):
        labels.add(normalize_label(payload.get(key)))

    landmark_ensemble = payload.get("landmark_ensemble")
    if isinstance(landmark_ensemble, T.Mapping):
        labels.add(normalize_label(landmark_ensemble.get("runtime_bucket")))
        labels.add(normalize_label(landmark_ensemble.get("bucket")))
        resolver = landmark_ensemble.get("resolver")
        if isinstance(resolver, T.Mapping):
            labels.add(normalize_label(resolver.get("runtime_bucket")))
            labels.add(normalize_label(resolver.get("bucket")))


def sample_labels(sample: T.Mapping[str, T.Any]) -> set[str]:
    """Return the normalized label set derived from one manifest sample."""
    labels: set[str] = set()
    raw_conditions = sample.get("conditions") or ()
    if isinstance(raw_conditions, str):
        raw_conditions = (raw_conditions,)
    if isinstance(raw_conditions, (list, tuple, set)):
        labels.update(normalize_label(item) for item in raw_conditions)
    labels.add(normalize_label(sample.get("condition")))
    labels.add(normalize_label(sample.get("hard_slice")))
    labels.add(normalize_label(sample.get("yaw_slice")))
    _add_runtime_bucket_labels(labels, sample)

    raw_metadata = sample.get("metadata")
    metadata: dict[str, T.Any] = raw_metadata if isinstance(raw_metadata, dict) else {}
    raw_attrs = metadata.get("attributes")
    attrs: dict[str, T.Any] = raw_attrs if isinstance(raw_attrs, dict) else {}
    for key, value in attrs.items():
        if value:
            labels.add(normalize_label(key))

    _add_runtime_bucket_labels(labels, metadata)

    issue_type = normalize_label(metadata.get("issue_type"))
    if issue_type in OCCLUSION_LABELS or issue_type == "occlusion":
        labels.add("occlusion")

    for key in ("occlusion", "occlusions", "occluded", "visibility", "visible"):
        value = metadata.get(key, sample.get(key))
        if _marks_occlusion(key, value):
            labels.add("occlusion")

    return {label for label in labels if label}


def classify_hard_negative(sample: T.Mapping[str, T.Any]) -> HardNegativeClass | None:
    """Classify sample into a hard-negative bucket, or None when neutral."""
    labels = sample_labels(sample)
    is_profile = bool(labels & PROFILE_LABELS)
    is_occlusion = bool(labels & OCCLUSION_LABELS)

    if is_profile and is_occlusion:
        bucket = "profile_occlusion"
    elif is_profile:
        bucket = "profile"
    elif is_occlusion:
        bucket = "occlusion"
    elif labels & ANCHOR_LABELS:
        bucket = "anchor"
    else:
        return None

    return HardNegativeClass(
        bucket=bucket,
        priority=BUCKET_PRIORITY[bucket],
        weight=BUCKET_WEIGHT[bucket],
        reasons=tuple(sorted(labels)),
    )


def source_key(sample: T.Mapping[str, T.Any]) -> tuple[str, str]:
    """Return a (dataset, source_id) dedupe key for one sample."""
    raw_source = sample.get("source")
    source: dict[str, T.Any] = raw_source if isinstance(raw_source, dict) else {}
    dataset = str(source.get("dataset") or sample.get("dataset") or "")
    source_id = str(
        source.get("source_id")
        or source.get("image_id")
        or source.get("sample_id")
        or sample.get("image")
        or sample.get("sample_id")
        or ""
    )
    return dataset, source_id


def annotate_sample(
    sample: T.Mapping[str, T.Any], classification: HardNegativeClass
) -> dict[str, T.Any]:
    """Return a copy of sample annotated with hard-negative metadata."""
    out = dict(sample)
    raw_labels = out.get("conditions") or ()
    labels = [raw_labels] if isinstance(raw_labels, str) else list(raw_labels)
    labels.append(classification.bucket)
    if classification.bucket == "profile_occlusion":
        labels.extend(("profile", "occlusion"))
    out["condition"] = classification.bucket
    out["conditions"] = sorted(
        {normalize_label(item) for item in labels if normalize_label(item)}
    )
    metadata = (
        dict(out.get("metadata", {})) if isinstance(out.get("metadata"), dict) else {}
    )
    metadata.update(
        {
            "hard_negative_bucket": classification.bucket,
            "hard_negative_priority": classification.priority,
            "hard_negative_weight": classification.weight,
            "hard_negative_source_dataset": source_key(sample)[0],
            "hard_negative_reason": list(classification.reasons),
        }
    )
    out["metadata"] = metadata
    return out


def default_class_for_dataset(dataset: str) -> HardNegativeClass | None:
    """Default bucket for a whole dataset, or None when the dataset is unmapped."""
    label = str(dataset).strip().lower()
    bucket = DATASET_DEFAULT_BUCKET.get(label)
    if bucket is None:
        return None
    return HardNegativeClass(
        bucket=bucket,
        priority=BUCKET_PRIORITY[bucket],
        weight=BUCKET_WEIGHT[bucket],
        reasons=(f"{label}_default",),
    )


def resolve_hard_negative_class(
    sample: T.Mapping[str, T.Any],
) -> tuple[HardNegativeClass, str]:
    """Resolve a bucket for any sample: classifier, else dataset default, else anchor.

    Unlike :func:`classify_hard_negative`, this never returns ``None`` -- every
    sample gets a bucket so the training sampler's bucket dimension is never
    polluted by an overloaded ``condition`` field. Returns ``(classification,
    source)`` where ``source`` is ``classified_by_label`` / ``dataset_default``
    / ``anchor_fallback``.
    """
    classification = classify_hard_negative(sample)
    if classification is not None:
        return classification, "classified_by_label"
    default = default_class_for_dataset(source_key(sample)[0])
    if default is not None:
        return default, "dataset_default"
    anchor = HardNegativeClass(
        bucket="anchor",
        priority=BUCKET_PRIORITY["anchor"],
        weight=BUCKET_WEIGHT["anchor"],
        reasons=("anchor_fallback",),
    )
    return anchor, "anchor_fallback"


def annotate_sample_bucket_in_place(sample: dict[str, T.Any]) -> tuple[str, str]:
    """Write only hard-negative *metadata* on ``sample`` in place; return (bucket, source).

    Non-destructive: unlike :func:`annotate_sample`, it never rewrites the
    overloaded ``condition``/``conditions`` fields (which also carry dataset,
    pose, and occlusion labels consumed by evaluation slicing). It records only
    ``metadata.hard_negative_bucket`` / ``_priority`` / ``_weight`` so the
    training sampler reads an authoritative bucket.
    """
    classification, source = resolve_hard_negative_class(sample)
    metadata = sample.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}
        sample["metadata"] = metadata
    metadata["hard_negative_bucket"] = classification.bucket
    metadata["hard_negative_priority"] = classification.priority
    metadata["hard_negative_weight"] = classification.weight
    return classification.bucket, source


def clamp_hard_negative_weight(weight: float) -> float:
    """Clamp a combined hard-negative weight into the supported range."""
    if not weight or weight != weight:
        return DEFAULT_HARD_NEGATIVE_WEIGHT
    return float(
        min(max(weight, DEFAULT_HARD_NEGATIVE_WEIGHT), MAX_HARD_NEGATIVE_WEIGHT)
    )


__all__ = [
    "ANCHOR_LABELS",
    "BUCKET_PRIORITY",
    "BUCKET_WEIGHT",
    "DATASET_DEFAULT_BUCKET",
    "DEFAULT_HARD_NEGATIVE_WEIGHT",
    "MAX_HARD_NEGATIVE_WEIGHT",
    "OCCLUSION_LABELS",
    "PROFILE_LABELS",
    "HardNegativeClass",
    "annotate_sample",
    "annotate_sample_bucket_in_place",
    "clamp_hard_negative_weight",
    "classify_hard_negative",
    "default_class_for_dataset",
    "normalize_label",
    "resolve_hard_negative_class",
    "sample_labels",
    "source_key",
]
