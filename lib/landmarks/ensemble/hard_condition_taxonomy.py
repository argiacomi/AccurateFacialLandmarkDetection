#!/usr/bin/env python3
"""Hard-condition taxonomy for landmark manifest samples."""

from __future__ import annotations

import math
import typing as T
from dataclasses import dataclass

from lib.landmarks.datasets.manifest_io import LandmarkSample, coerce_conditions

PROFILE_YAW_DEGREES: float = 55.0
LARGE_YAW_DEGREES: float = 35.0
ROLLED_DEGREES: float = 30.0
EYE_VISIBLE_FRACTION: float = 0.50
EYE_OCCLUDED_FRACTION: float = 0.34

NEW_HARD_CONDITION_LABELS: tuple[str, ...] = (
    "rolled_profile_occlusion",
    "profile_occlusion",
    "large_yaw_occlusion",
    "single_eye_visible",
    "mouth_or_jaw_occluded",
)

WEIGHT_BUCKETS: tuple[str, ...] = (
    "frontal",
    "large_yaw",
    "profile",
    "rolled_profile",
    "occlusion",
    "profile_occlusion",
)
GLOBAL_WEIGHT_BUCKET: str = "global"

_LEFT_EYE = range(36, 42)
_RIGHT_EYE = range(42, 48)
_JAW = range(0, 17)
_MOUTH = range(48, 68)


@dataclass(frozen=True)
class HardConditionTaxonomy:
    """Derived hard-condition label bundle for one sample."""

    condition: str
    runtime_bucket: str
    hard_case_tags: tuple[str, ...]


def _normalized_label(value: T.Any) -> str:
    label = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    while "__" in label:
        label = label.replace("__", "_")
    return label.strip("_")


def _truthy(value: T.Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() not in {"", "0", "false", "none", "no", "clean"}
    if isinstance(value, T.Mapping):
        return any(_truthy(item) for item in value.values())
    if isinstance(value, (list, tuple, set)):
        return any(_truthy(item) for item in value)
    return bool(value)


def _metadata_conditions(metadata: T.Mapping[str, T.Any]) -> tuple[str, ...]:
    labels: list[str] = []
    labels.extend(coerce_conditions(metadata.get("conditions")))
    attributes = metadata.get("attributes")
    if isinstance(attributes, T.Mapping):
        labels.extend(_normalized_label(key) for key, value in attributes.items() if _truthy(value))
    for key in ("occlusion", "occlusions", "occluded", "pose"):
        if _truthy(metadata.get(key)):
            labels.append("occlusion" if key != "pose" else "pose")
    visibility = metadata.get("visibility")
    if isinstance(visibility, (list, tuple)):
        labels.extend(_normalized_label(item) for item in visibility if isinstance(item, str))
    return tuple(dict.fromkeys(label for label in labels if label))


def _all_condition_labels(sample: LandmarkSample) -> tuple[str, ...]:
    labels: list[str] = []
    labels.extend(sample.conditions)
    labels.extend(coerce_conditions(sample.metadata.get("conditions")))
    if sample.condition:
        labels.append(_normalized_label(sample.condition))
    labels.extend(_metadata_conditions(sample.metadata))
    return tuple(dict.fromkeys(label for label in labels if label))


def _metadata_visibility(sample: LandmarkSample) -> tuple[bool, ...] | None:
    if sample.visibility is not None:
        return sample.visibility
    visibility = sample.metadata.get("visibility")
    if not isinstance(visibility, (list, tuple)):
        visibility = sample.metadata.get("landmark_score_visibility_mask")
    if not isinstance(visibility, (list, tuple)):
        return None
    flags: list[bool] = []
    for item in visibility:
        if isinstance(item, str):
            flags.append(_normalized_label(item) == "visible")
        else:
            flags.append(bool(item))
    return tuple(flags) if flags else None


def _hidden_fraction(visibility: tuple[bool, ...] | None, indices: range) -> float:
    if visibility is None or len(visibility) <= max(indices):
        return 0.0
    values = [bool(visibility[index]) for index in indices]
    if not values:
        return 0.0
    return sum(1 for value in values if not value) / len(values)


def _finite_abs(value: float | None) -> float | None:
    if value is None:
        return None
    try:
        number = abs(float(value))
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _runtime_pose_flags(
    runtime_bucket: str,
    *,
    yaw_estimate: float | None,
    roll_estimate: float | None,
    labels: T.AbstractSet[str],
) -> tuple[bool, bool, bool]:
    bucket = _normalized_label(runtime_bucket)
    yaw = _finite_abs(yaw_estimate)
    roll = _finite_abs(roll_estimate)
    profile = (
        bucket.startswith("profile_")
        or bucket.startswith("rolled_profile_")
        or (yaw is not None and yaw >= PROFILE_YAW_DEGREES)
        or any(label.startswith("profile") for label in labels)
    )
    large_yaw = (
        profile
        or bucket.startswith("large_yaw_")
        or bucket.startswith("rolled_large_yaw_")
        or (yaw is not None and yaw >= LARGE_YAW_DEGREES)
        or any(label.startswith(("large_yaw", "yaw_")) for label in labels)
        or "pose" in labels
    )
    rolled = (
        bucket in {"large_roll", "extreme_roll"}
        or bucket.startswith("rolled_")
        or (roll is not None and roll >= ROLLED_DEGREES)
        or any("roll" in label for label in labels)
    )
    return profile, large_yaw, rolled


def derive_hard_condition_taxonomy(
    sample: LandmarkSample,
    *,
    runtime_bucket: str,
    yaw_estimate: float | None = None,
    roll_estimate: float | None = None,
) -> HardConditionTaxonomy:
    """Derive the hard-condition taxonomy for one sample."""
    labels = set(_all_condition_labels(sample))
    visibility = _metadata_visibility(sample)
    eye_left_hidden = _hidden_fraction(visibility, _LEFT_EYE)
    eye_right_hidden = _hidden_fraction(visibility, _RIGHT_EYE)
    jaw_hidden = _hidden_fraction(visibility, _JAW)
    mouth_hidden = _hidden_fraction(visibility, _MOUTH)
    single_eye_visible = (
        eye_left_hidden >= EYE_OCCLUDED_FRACTION and eye_right_hidden <= 1.0 - EYE_VISIBLE_FRACTION
    ) or (
        eye_right_hidden >= EYE_OCCLUDED_FRACTION and eye_left_hidden <= 1.0 - EYE_VISIBLE_FRACTION
    )
    mouth_or_jaw_occluded = jaw_hidden > 0.0 or mouth_hidden > 0.0
    has_visibility_occlusion = any(
        hidden > 0.0 for hidden in (eye_left_hidden, eye_right_hidden, jaw_hidden, mouth_hidden)
    )
    has_labeled_occlusion = bool(
        labels
        & {
            "occlusion",
            "occluded",
            "self_occlusion",
            "self_occluded",
            "externally_occluded",
            "external_occlusion",
        }
    ) or any(label.endswith("_occlusion") or label.endswith("_occluded") or "occlud" in label for label in labels)
    has_occlusion = has_labeled_occlusion or has_visibility_occlusion
    profile, large_yaw, rolled = _runtime_pose_flags(
        runtime_bucket,
        yaw_estimate=yaw_estimate,
        roll_estimate=roll_estimate,
        labels=labels,
    )

    tags: list[str] = []

    def add(tag: str, present: bool = True) -> None:
        if present and tag not in tags:
            tags.append(tag)

    for label in NEW_HARD_CONDITION_LABELS:
        add(label, label in labels)

    add("rolled_profile_occlusion", rolled and profile and has_occlusion)
    add("profile_occlusion", profile and has_occlusion)
    add("large_yaw_occlusion", large_yaw and has_occlusion)
    add("single_eye_visible", single_eye_visible)
    add("mouth_or_jaw_occluded", mouth_or_jaw_occluded)
    add("occlusion", has_occlusion)
    add("profile_pose", profile)
    add("large_yaw_pose", large_yaw)
    add("rolled_pose", rolled)

    condition = next((tag for tag in NEW_HARD_CONDITION_LABELS if tag in tags), "")
    if not condition:
        condition = _normalized_label(sample.condition) or next(iter(labels), "") or _normalized_label(runtime_bucket) or "unknown"

    return HardConditionTaxonomy(
        condition=condition,
        runtime_bucket=str(runtime_bucket or "unknown"),
        hard_case_tags=tuple(tags),
    )


def _pose_weight_bucket(*, profile: bool, rolled_profile: bool, large_yaw: bool, occluded: bool) -> str:
    if profile and occluded:
        return "profile_occlusion"
    if rolled_profile:
        return "rolled_profile"
    if profile:
        return "profile"
    if occluded:
        return "occlusion"
    if large_yaw:
        return "large_yaw"
    return "frontal"


def weight_bucket_from_runtime(runtime_bucket: str, *, occluded: bool = False) -> str:
    bucket = _normalized_label(runtime_bucket)
    profile = bucket.startswith("profile_") or bucket.startswith("rolled_profile_")
    rolled_profile = bucket.startswith("rolled_profile_")
    large_yaw = bucket.startswith("large_yaw_") or bucket.startswith("rolled_large_yaw_")
    return _pose_weight_bucket(
        profile=profile,
        rolled_profile=rolled_profile,
        large_yaw=large_yaw,
        occluded=occluded,
    )


def weight_bucket_from_pose(
    yaw_estimate: float | None,
    roll_estimate: float | None,
    *,
    occluded: bool = False,
) -> str:
    yaw = _finite_abs(yaw_estimate)
    roll = _finite_abs(roll_estimate)
    profile = yaw is not None and yaw >= PROFILE_YAW_DEGREES
    rolled_profile = profile and roll is not None and roll >= ROLLED_DEGREES
    large_yaw = yaw is not None and yaw >= LARGE_YAW_DEGREES
    return _pose_weight_bucket(
        profile=profile,
        rolled_profile=rolled_profile,
        large_yaw=large_yaw,
        occluded=occluded,
    )


def applicable_weight_buckets(runtime_bucket: str) -> tuple[str, ...]:
    base = weight_bucket_from_runtime(runtime_bucket, occluded=False)
    occluded = weight_bucket_from_runtime(runtime_bucket, occluded=True)
    return tuple(dict.fromkeys((base, occluded)))


__all__ = [
    "GLOBAL_WEIGHT_BUCKET",
    "HardConditionTaxonomy",
    "NEW_HARD_CONDITION_LABELS",
    "WEIGHT_BUCKETS",
    "applicable_weight_buckets",
    "derive_hard_condition_taxonomy",
    "weight_bucket_from_pose",
    "weight_bucket_from_runtime",
]
