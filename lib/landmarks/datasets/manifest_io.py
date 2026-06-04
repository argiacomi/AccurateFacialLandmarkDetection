#!/usr/bin/env python3
"""Canonical manifest, bbox, and visibility IO for landmark datasets."""

from __future__ import annotations

import json
import logging
import typing as T
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

CANONICAL_68_COMPATIBLE_SCHEMAS = frozenset({"2d_68", "2d_98"})


@dataclass(frozen=True)
class LandmarkSample:
    """One landmark manifest entry with manifest metadata attached."""

    sample_id: str
    image: str
    landmarks: str
    dataset: str = ""
    condition: str = ""
    conditions: tuple[str, ...] = ()
    source_schema: str = ""
    normalizer: float | None = None
    face_bbox: tuple[float, float, float, float] | None = None
    visibility: tuple[bool, ...] | None = None
    metadata: dict[str, T.Any] = field(default_factory=dict)


LandmarkSampleProgress = T.Callable[
    [T.Sequence[LandmarkSample], str],
    T.Iterable[LandmarkSample],
]


def coerce_bbox(value: T.Any) -> tuple[float, float, float, float] | None:
    """Coerce a manifest bbox payload to left, top, right, bottom order."""
    if value is None:
        return None
    if isinstance(value, T.Mapping):
        keys = set(value)
        if {"left", "top", "right", "bottom"}.issubset(keys):
            try:
                return tuple(float(value[key]) for key in ("left", "top", "right", "bottom"))  # type: ignore[return-value]
            except (TypeError, ValueError):
                return None
        if {"x", "y", "w", "h"}.issubset(keys):
            try:
                left = float(value["x"])
                top = float(value["y"])
                width = float(value["w"])
                height = float(value["h"])
            except (TypeError, ValueError):
                return None
            if width <= 0 or height <= 0:
                return None
            return (left, top, left + width, top + height)
        return None
    try:
        flat = np.asarray(value, dtype="float64").reshape(-1)
    except (TypeError, ValueError):
        return None
    if flat.size < 4:
        return None
    left, top, third, fourth = (float(item) for item in flat[:4])
    if third > left and fourth > top:
        return (left, top, third, fourth)
    if third > 0 and fourth > 0:
        return (left, top, left + third, top + fourth)
    return None


def coerce_visibility(value: T.Any) -> tuple[bool, ...] | None:
    """Coerce a manifest visibility payload to a bool tuple, or None."""
    if value is None:
        return None
    visible_labels = {"visible", "vis", "v", "1", "true", "yes"}
    hidden_labels = {
        "hidden",
        "invisible",
        "occluded",
        "self_occluded",
        "self_occlusion",
        "externally_occluded",
        "external_occlusion",
        "not_visible",
        "0",
        "false",
        "no",
    }
    flags: list[bool] = []
    try:
        iterator = iter(value)
    except TypeError:
        return None
    for item in iterator:
        if isinstance(item, str):
            label = item.strip().lower().replace("-", "_").replace(" ", "_")
            while "__" in label:
                label = label.replace("__", "_")
            label = label.strip("_")
            if label in visible_labels:
                flags.append(True)
            elif label in hidden_labels or "occlud" in label:
                flags.append(False)
            else:
                return None
        else:
            flags.append(bool(item))
    return tuple(flags) if flags else None


def coerce_conditions(value: T.Any, *, fallback: str = "") -> tuple[str, ...]:
    """Coerce manifest condition labels to a stable tuple."""
    raw_items: T.Iterable[T.Any]
    if isinstance(value, T.Mapping):
        explicit_labels = value.get("labels")
        if isinstance(explicit_labels, (list, tuple, set)):
            raw_items = explicit_labels
        else:
            raw_items = [
                key
                for key, present in value.items()
                if str(key).strip() and key not in {"labels", "condition", "scenario"} and present
            ]
            for key in ("condition", "scenario"):
                if value.get(key):
                    raw_items = [value[key], *raw_items]
                    break
    elif isinstance(value, (list, tuple, set)):
        raw_items = value
    elif value:
        raw_items = (value,)
    elif fallback:
        raw_items = (fallback,)
    else:
        raw_items = ()

    normalized_labels: list[str] = []
    for item in raw_items:
        label = str(item).strip().lower().replace("-", "_").replace(" ", "_")
        while "__" in label:
            label = label.replace("__", "_")
        label = label.strip("_")
        if label and label not in normalized_labels:
            normalized_labels.append(label)
    return tuple(normalized_labels)


def bbox_from_truth_fallback(truth: np.ndarray) -> tuple[float, float, float, float] | None:
    """Return the axis-aligned bbox of truth landmarks as a fallback bbox."""
    points = np.asarray(truth, dtype="float64")
    if points.ndim != 2 or points.shape[1] < 2 or points.size == 0:
        return None
    left, top = np.min(points, axis=0)[:2]
    right, bottom = np.max(points, axis=0)[:2]
    if right <= left or bottom <= top:
        return None
    return (float(left), float(top), float(right), float(bottom))


def load_manifest(path: str | Path) -> list[LandmarkSample]:
    """Load a manifest JSON file into LandmarkSample records."""
    manifest_path = Path(path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    base = manifest_path.parent
    samples: list[LandmarkSample] = []
    for entry in payload.get("samples", payload.get("scenarios", [])):
        landmarks = str(entry.get("landmarks") or entry.get("ground_truth") or "")
        if not landmarks:
            raise ValueError(f"manifest entry {entry!r} missing landmarks path")
        metadata = entry.get("metadata", {}) if isinstance(entry.get("metadata"), dict) else {}
        condition = str(entry.get("condition", entry.get("scenario", "")))
        conditions = coerce_conditions(
            entry.get("conditions", metadata.get("conditions")),
            fallback=condition,
        )
        bbox = coerce_bbox(entry.get("face_bbox", metadata.get("face_bbox")))
        if bbox is None:
            bbox = coerce_bbox(entry.get("bbox", metadata.get("bbox")))
        visibility = coerce_visibility(entry.get("visibility", metadata.get("visibility")))
        image_path = Path(str(entry.get("image", "")))
        landmarks_path = Path(landmarks)
        samples.append(
            LandmarkSample(
                sample_id=str(entry.get("sample_id") or entry.get("id") or entry.get("name")),
                image=str(image_path if image_path.is_absolute() else (base / image_path).resolve()),
                landmarks=str(landmarks_path if landmarks_path.is_absolute() else (base / landmarks_path).resolve()),
                dataset=str(entry.get("dataset", "")),
                condition=condition,
                conditions=conditions,
                source_schema=str(entry.get("source_schema", metadata.get("source_schema", ""))),
                normalizer=entry.get("normalizer", metadata.get("normalizer")),
                face_bbox=bbox,
                visibility=visibility,
                metadata=dict(metadata),
            )
        )
    return samples


def bbox_for_sample(sample: LandmarkSample, *, allow_truth_fallback: bool = True) -> tuple[float, float, float, float] | None:
    """Resolve a usable bbox for sample."""
    if sample.face_bbox is not None:
        return sample.face_bbox
    if not allow_truth_fallback:
        return None
    try:
        truth = np.load(sample.landmarks).astype("float32")
    except OSError:
        return None
    return bbox_from_truth_fallback(truth)


def sample_is_canonical_68(sample: LandmarkSample) -> bool:
    """Return True if sample ground truth can map to canonical 68 points."""
    try:
        truth = np.load(sample.landmarks)
    except OSError:
        return False
    return getattr(truth, "ndim", 0) == 2 and int(truth.shape[0]) in (68, 98)


def filter_canonical_68_samples(
    samples: T.Sequence[LandmarkSample],
    *,
    context: str = "",
    progress: LandmarkSampleProgress | None = None,
) -> list[LandmarkSample]:
    """Drop samples whose GT cannot be scored against canonical 68-point output."""
    kept: list[LandmarkSample] = []
    skipped_by_schema: dict[str, int] = {}
    progress_desc = f"Filter canonical GT [{context}]" if context else "Filter canonical GT"
    iterator = progress(samples, progress_desc) if progress is not None else samples
    for sample in iterator:
        if sample_is_canonical_68(sample):
            kept.append(sample)
        else:
            key = (sample.source_schema or "unknown").strip() or "unknown"
            skipped_by_schema[key] = skipped_by_schema.get(key, 0) + 1
    if skipped_by_schema:
        where = f" in {context}" if context else ""
        logger.warning(
            "Skipping %d non-canonical-68 ground-truth sample(s)%s (counts by schema: %s).",
            sum(skipped_by_schema.values()),
            where,
            skipped_by_schema,
        )
    return kept


__all__ = [
    "CANONICAL_68_COMPATIBLE_SCHEMAS",
    "LandmarkSample",
    "LandmarkSampleProgress",
    "bbox_for_sample",
    "bbox_from_truth_fallback",
    "coerce_bbox",
    "coerce_conditions",
    "coerce_visibility",
    "filter_canonical_68_samples",
    "load_manifest",
    "sample_is_canonical_68",
]
