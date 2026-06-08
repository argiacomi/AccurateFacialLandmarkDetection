"""Split-safe manifest evaluation and sliced metric reporting."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import typing as T
from pathlib import Path

import numpy as np

from lib.landmarks.training.loss_function import compute_fr_and_auc

EVAL_MODES = ("random_hash", "by_dataset", "leave_one_dataset_out")
SPLIT_POLICIES = ("declared_or_random_hash", "random_hash", "declared")
LEAKAGE_KEYS = {
    "image": (
        "image",
        "image_path",
        "path",
        "original_image",
        "source_image",
        "source_image_ids",
        "image_id",
        "merl_image_id",
        "frame_name",
    ),
    "landmark": (
        "landmarks",
        "ground_truth",
        "points",
        "original_landmarks",
        "source_landmarks",
    ),
    "identity": (
        "subject_id",
        "person_id",
        "identity_id",
        "source_dataset_id",
    ),
    "sequence": (
        "video_id",
        "clip_id",
        "sequence_id",
        "session_id",
        "capture_id",
    ),
    "archive": (
        "archive",
        "archive_path",
        "original_archive",
    ),
}


def normalize_label(value: T.Any) -> str:
    label = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    while "__" in label:
        label = label.replace("__", "_")
    return label.strip("_")


def normalize_dataset(value: T.Any) -> str:
    label = normalize_label(value)
    aliases = {
        "300w": "w300",
        "aflw2000_3d": "aflw2000",
        "aflw2000-3d": "aflw2000",
        "production": "production_validated",
        "prod": "production_validated",
    }
    return aliases.get(label, label)


def normalize_heldout_datasets(values: T.Iterable[str] | None) -> tuple[str, ...]:
    labels = []
    for value in values or ():
        label = normalize_dataset(value)
        if label and label not in labels:
            labels.append(label)
    return tuple(labels)


def manifest_entry_split(entry: T.Mapping[str, T.Any]) -> str:
    raw = entry.get("split")
    metadata = entry.get("metadata")
    if raw is None and isinstance(metadata, T.Mapping):
        raw = metadata.get("split")
    return normalize_label(raw)


def sample_dataset(entry: T.Mapping[str, T.Any]) -> str:
    metadata = (
        entry.get("metadata") if isinstance(entry.get("metadata"), T.Mapping) else {}
    )
    source = entry.get("source") if isinstance(entry.get("source"), T.Mapping) else {}
    return normalize_dataset(
        entry.get("dataset") or metadata.get("dataset") or source.get("dataset")
    )


def stable_random_hash_split(
    entry: T.Mapping[str, T.Any], index: int, *, test_percent: int = 5
) -> str:
    dataset = sample_dataset(entry) or "unknown"
    identity = (
        entry.get("sample_id")
        or entry.get("id")
        or entry.get("name")
        or entry.get("image")
        or entry.get("landmarks")
        or index
    )
    split_key = f"{dataset}|{identity}"
    split_hash = int(hashlib.sha256(str(split_key).encode()).hexdigest()[:8], 16)
    return "test" if (split_hash % 100) < int(test_percent) else "train"


def entry_in_eval_split(
    entry: T.Mapping[str, T.Any],
    index: int,
    *,
    split: str,
    eval_mode: str,
    heldout_datasets: T.Iterable[str] | None = None,
    has_declared_splits: bool = False,
    split_policy: str = "declared_or_random_hash",
) -> bool:
    split_label = normalize_label(split)
    mode = normalize_label(eval_mode) or "random_hash"
    heldout = set(normalize_heldout_datasets(heldout_datasets))
    dataset = sample_dataset(entry)

    if mode not in EVAL_MODES:
        raise ValueError(
            f"unknown eval mode {eval_mode!r}; expected one of {EVAL_MODES}"
        )

    if mode in {"by_dataset", "leave_one_dataset_out"}:
        if not heldout:
            raise ValueError(
                f"--eval-mode {mode} requires at least one --heldout-dataset"
            )
        if mode == "leave_one_dataset_out" and len(heldout) != 1:
            raise ValueError(
                "--eval-mode leave_one_dataset_out requires exactly one --heldout-dataset"
            )
        is_heldout = dataset in heldout
        if split_label == "train":
            return not is_heldout
        if split_label == "test":
            return is_heldout
        return False

    policy = normalize_label(split_policy) or "declared_or_random_hash"
    if policy not in SPLIT_POLICIES:
        raise ValueError(
            f"unknown split policy {split_policy!r}; expected one of {SPLIT_POLICIES}"
        )
    if policy == "declared":
        entry_split = manifest_entry_split(entry)
        return bool(entry_split) and entry_split == split_label
    if policy == "declared_or_random_hash" and has_declared_splits:
        entry_split = manifest_entry_split(entry)
        return bool(entry_split) and entry_split == split_label
    return stable_random_hash_split(entry, index) == split_label


def _flatten_sources(value: T.Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        out: list[str] = []
        for item in value:
            out.extend(_flatten_sources(item))
        return out
    if isinstance(value, T.Mapping):
        out = []
        for item in value.values():
            out.extend(_flatten_sources(item))
        return out
    text = str(value).strip()
    return [text] if text else []


def _source_values(sample: T.Mapping[str, T.Any], keys: T.Sequence[str]) -> set[str]:
    metadata = (
        sample.get("metadata") if isinstance(sample.get("metadata"), T.Mapping) else {}
    )
    source = sample.get("source") if isinstance(sample.get("source"), T.Mapping) else {}
    values: set[str] = set()
    for key in keys:
        values.update(_flatten_sources(sample.get(key)))
        values.update(_flatten_sources(metadata.get(key)))
        values.update(_flatten_sources(source.get(key)))
    return {
        str(Path(value).expanduser()) if "/" in value else value
        for value in values
        if value
    }


def image_source_ids(sample: T.Mapping[str, T.Any]) -> set[str]:
    return _source_values(sample, LEAKAGE_KEYS["image"])


def landmark_source_ids(sample: T.Mapping[str, T.Any]) -> set[str]:
    return _source_values(sample, LEAKAGE_KEYS["landmark"])


def leakage_source_ids(sample: T.Mapping[str, T.Any], category: str) -> set[str]:
    return _source_values(sample, LEAKAGE_KEYS[category])


def validate_no_train_test_leakage(
    train_samples: T.Sequence[T.Mapping[str, T.Any]],
    test_samples: T.Sequence[T.Mapping[str, T.Any]],
) -> None:
    train_sources: dict[str, dict[str, str]] = {
        category: {} for category in LEAKAGE_KEYS
    }
    for sample in train_samples:
        sample_id = str(
            sample.get("sample_id") or sample.get("image") or sample.get("landmarks")
        )
        for category in LEAKAGE_KEYS:
            for source_id in leakage_source_ids(sample, category):
                train_sources[category].setdefault(source_id, sample_id)

    duplicates: dict[str, list[tuple[str, str, str]]] = {
        category: [] for category in LEAKAGE_KEYS
    }
    for sample in test_samples:
        sample_id = str(
            sample.get("sample_id") or sample.get("image") or sample.get("landmarks")
        )
        for category in LEAKAGE_KEYS:
            for source_id in leakage_source_ids(sample, category):
                if source_id in train_sources[category]:
                    duplicates[category].append(
                        (source_id, train_sources[category][source_id], sample_id)
                    )

    if any(duplicates.values()):
        details = {
            f"duplicate_{category}_count": len(values)
            for category, values in duplicates.items()
        }
        details.update(
            {
                f"duplicate_{category}_examples": values[:10]
                for category, values in duplicates.items()
                if values
            }
        )
        raise ValueError(
            f"train/test source leakage detected: {json.dumps(details, sort_keys=True)}"
        )


def _coerce_conditions(meta: T.Mapping[str, T.Any]) -> tuple[str, ...]:
    raw = meta.get("conditions", ())
    if isinstance(raw, str):
        values = (raw,)
    elif isinstance(raw, T.Mapping):
        values = tuple(key for key, present in raw.items() if present)
    elif isinstance(raw, (list, tuple, set)):
        values = tuple(raw)
    else:
        values = ()
    condition = meta.get("condition")
    labels = []
    for item in (*values, condition):
        label = normalize_label(item)
        if label and label not in labels:
            labels.append(label)
    return tuple(labels)


def _first_label(*values: T.Any, default: str = "unknown") -> str:
    for value in values:
        label = normalize_label(value)
        if label:
            return label
    return default


def _bucket_from_conditions(conditions: T.Sequence[str]) -> str:
    labels = set(conditions)
    if "profile_occlusion" in labels or ("profile" in labels and "occlusion" in labels):
        return "profile_occlusion"
    if any(
        label in labels
        for label in ("profile", "large_yaw", "large_yaw_pose", "profile_pose")
    ):
        return "profile"
    if any("occlusion" in label or "occlud" in label for label in labels):
        return "occlusion"
    if any(label in labels for label in ("anchor", "normal", "clean", "frontal")):
        return "anchor"
    return "unknown"


def _pose_bucket(meta: T.Mapping[str, T.Any], conditions: T.Sequence[str]) -> str:
    label = _first_label(
        meta.get("pose_bucket"), meta.get("pose"), meta.get("yaw_bucket"), default=""
    )
    if label:
        return label
    labels = set(conditions)
    if any(
        label in labels for label in ("large_yaw_left", "profile_left", "left_profile")
    ):
        return "profile_left"
    if any(
        label in labels
        for label in ("large_yaw_right", "profile_right", "right_profile")
    ):
        return "profile_right"
    if any(
        label in labels
        for label in ("profile", "large_yaw", "large_yaw_pose", "profile_pose")
    ):
        return "profile"
    if any(label in labels for label in ("frontal", "normal", "clean", "anchor")):
        return "frontal"
    return "unknown"


def _occlusion_bucket(meta: T.Mapping[str, T.Any], conditions: T.Sequence[str]) -> str:
    explicit = meta.get("occlusion", meta.get("occluded"))
    if explicit is not None:
        if isinstance(explicit, str):
            label = normalize_label(explicit)
            if label in {"1", "true", "yes", "occluded", "occlusion"}:
                return "occlusion"
            if label in {"0", "false", "no", "none", "clear", "clean"}:
                return "no_occlusion"
            return "unknown"
        return "occlusion" if bool(explicit) else "no_occlusion"
    return (
        "occlusion"
        if any("occlusion" in label or "occlud" in label for label in conditions)
        else "no_occlusion"
    )


def _profile_side(meta: T.Mapping[str, T.Any], conditions: T.Sequence[str]) -> str:
    label = _first_label(
        meta.get("profile_side"), meta.get("side"), meta.get("yaw_side"), default=""
    )
    if label in {"left", "right"}:
        return label
    joined = set(conditions)
    if any("left" in label for label in joined):
        return "left"
    if any("right" in label for label in joined):
        return "right"
    if any(
        label in joined
        for label in ("profile", "large_yaw", "large_yaw_pose", "profile_pose")
    ):
        return "profile_unknown_side"
    return "not_profile"


def _bbox_value(meta: T.Mapping[str, T.Any]) -> tuple[T.Any, str]:
    for key in ("face_bbox", "bbox", "crop_bbox_xyxy"):
        value = meta.get(key)
        if value is not None:
            fmt = normalize_label(meta.get("bbox_format") or meta.get(f"{key}_format"))
            if key == "crop_bbox_xyxy" and not fmt:
                fmt = "xyxy"
            return value, fmt
    return None, ""


def _face_size_bucket(meta: T.Mapping[str, T.Any]) -> str:
    bbox, bbox_format = _bbox_value(meta)
    if bbox is None:
        return "unknown"
    try:
        if isinstance(bbox, T.Mapping):
            if {"left", "top", "right", "bottom"}.issubset(bbox):
                width = float(bbox["right"]) - float(bbox["left"])
                height = float(bbox["bottom"]) - float(bbox["top"])
            elif {"x", "y", "w", "h"}.issubset(bbox):
                width = float(bbox["w"])
                height = float(bbox["h"])
            else:
                return "unknown"
        else:
            flat = list(bbox)
            if len(flat) < 4:
                return "unknown"
            first, second, third, fourth = (float(item) for item in flat[:4])
            if bbox_format == "xyxy":
                width = third - first
                height = fourth - second
            elif bbox_format == "xywh":
                width = third
                height = fourth
            else:
                return "unknown"
    except (TypeError, ValueError):
        return "unknown"
    size = max(width, height)
    if not np.isfinite(size) or size <= 0:
        return "unknown"
    if size < 64:
        return "small"
    if size < 128:
        return "medium"
    return "large"


def slice_labels(meta: T.Mapping[str, T.Any]) -> dict[str, str]:
    conditions = _coerce_conditions(meta)
    metadata = (
        meta.get("metadata") if isinstance(meta.get("metadata"), T.Mapping) else {}
    )
    merged = {**metadata, **dict(meta)}
    dataset = normalize_dataset(merged.get("dataset")) or "unknown"
    schema = _first_label(
        merged.get("source_schema"), merged.get("schema"), merged.get("landmark_schema")
    )
    hard_bucket = _first_label(merged.get("hard_negative_bucket"), default="")
    if not hard_bucket:
        hard_bucket = _bucket_from_conditions(conditions)
    production_source = _first_label(
        merged.get("production_source"),
        merged.get("prod_source"),
        merged.get("fsa_path"),
    )
    return {
        "by_dataset": dataset,
        "by_schema": schema,
        "by_hard_negative_bucket": hard_bucket or "unknown",
        "by_pose_bucket": _pose_bucket(merged, conditions),
        "by_occlusion": _occlusion_bucket(merged, conditions),
        "by_profile_side": _profile_side(merged, conditions),
        "by_face_size": _face_size_bucket(merged),
        "by_production_source": production_source,
    }


def record_for_sample(meta: T.Mapping[str, T.Any], nme: float) -> dict[str, T.Any]:
    labels = slice_labels(meta)
    return {
        "sample_id": str(meta.get("sample_id", "")),
        "image": str(meta.get("image", "")),
        "dataset": labels["by_dataset"],
        "schema": labels["by_schema"],
        "nme": float(nme),
        "pose_bucket": labels["by_pose_bucket"],
        "hard_negative_bucket": labels["by_hard_negative_bucket"],
        "occlusion": labels["by_occlusion"],
        "profile_side": labels["by_profile_side"],
        "face_size": labels["by_face_size"],
        "production_source": labels["by_production_source"],
        **labels,
    }


def _sigmoid_if_logits(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return values.astype(np.float32)
    if float(np.nanmin(values)) < 0.0 or float(np.nanmax(values)) > 1.0:
        return (1.0 / (1.0 + np.exp(-values))).astype(np.float32)
    return values.astype(np.float32)


def _average_precision(labels: np.ndarray, scores: np.ndarray) -> float | None:
    positives = int(np.sum(labels == 1))
    if positives == 0:
        return None
    order = np.argsort(-scores)
    ranked = labels[order]
    tp = np.cumsum(ranked == 1)
    ranks = np.arange(1, ranked.size + 1)
    precision = tp / ranks
    return float(np.sum(precision[ranked == 1]) / positives)


def _f1_at_threshold(
    labels: np.ndarray, scores: np.ndarray, threshold: float = 0.5
) -> float | None:
    if labels.size == 0:
        return None
    pred = scores >= threshold
    truth = labels == 1
    tp = int(np.sum(pred & truth))
    fp = int(np.sum(pred & ~truth))
    fn = int(np.sum(~pred & truth))
    denom = (2 * tp) + fp + fn
    if denom == 0:
        return None
    return float((2 * tp) / denom)


def _roc_auc(labels: np.ndarray, scores: np.ndarray) -> float | None:
    pos_scores = scores[labels == 1]
    neg_scores = scores[labels == 0]
    if pos_scores.size == 0 or neg_scores.size == 0:
        return None
    comparisons = (pos_scores[:, None] > neg_scores[None, :]).astype(np.float32)
    comparisons += 0.5 * (pos_scores[:, None] == neg_scores[None, :]).astype(np.float32)
    return float(np.mean(comparisons))


def visibility_metrics_for_records(
    records: T.Sequence[T.Mapping[str, T.Any]],
) -> dict[str, T.Any]:
    labels: list[int] = []
    scores: list[float] = []
    prediction_skipped = 0
    for record in records:
        targets = record.get("visibility_targets")
        record_scores = record.get("visibility_scores")
        if not isinstance(targets, (list, tuple)):
            continue
        if not isinstance(record_scores, (list, tuple)):
            prediction_skipped += sum(1 for value in targets if int(value) in (0, 1))
            continue
        for target, score in zip(targets, record_scores):
            target_i = int(target)
            if target_i not in (0, 1):
                continue
            try:
                score_f = float(score)
            except (TypeError, ValueError):
                prediction_skipped += 1
                continue
            if not np.isfinite(score_f):
                prediction_skipped += 1
                continue
            labels.append(target_i)
            scores.append(score_f)

    if not labels:
        return {
            "visibility_sample_count": 0,
            "visibility_label_count": 0,
            "visibility_prediction_skipped_count": int(prediction_skipped),
            "visibility_AP": None,
            "visibility_F1@0.5": None,
            "visibility_ROC_AUC": None,
        }

    label_arr = np.asarray(labels, dtype=np.int64)
    score_arr = _sigmoid_if_logits(np.asarray(scores, dtype=np.float32))
    return {
        "visibility_sample_count": int(
            sum(
                1
                for record in records
                if record.get("visible_landmark_count") is not None
            )
        ),
        "visibility_label_count": int(label_arr.size),
        "visibility_prediction_skipped_count": int(prediction_skipped),
        "visibility_AP": _average_precision(label_arr, score_arr),
        "visibility_F1@0.5": _f1_at_threshold(label_arr, score_arr, threshold=0.5),
        "visibility_ROC_AUC": _roc_auc(label_arr, score_arr),
    }


def _nme_ci95(values: np.ndarray) -> dict[str, float | None]:
    if values.size < 2:
        return {"low": None, "high": None}
    stderr = float(np.std(values, ddof=1) / math.sqrt(values.size))
    mean = float(np.mean(values))
    return {"low": mean - 1.96 * stderr, "high": mean + 1.96 * stderr}


def metrics_for_nmes(
    values: T.Sequence[float], *, threshold: float = 0.10
) -> dict[str, T.Any]:
    finite_values = []
    for value in values:
        try:
            value_f = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(value_f):
            finite_values.append(value_f)
    arr = np.asarray(finite_values, dtype=np.float32)
    if arr.size == 0:
        return {
            "sample_count": 0,
            "nme": None,
            "NME_all": None,
            "nme_percent": None,
            "fr": None,
            "fr_percent": None,
            "auc": None,
            "nme_ci95": {"low": None, "high": None},
            "nme_percent_ci95": {"low": None, "high": None},
        }
    nme, fr, auc = compute_fr_and_auc(arr, thres=threshold, step=0.0001)
    ci = _nme_ci95(arr)
    return {
        "sample_count": int(arr.size),
        "nme": float(nme),
        "NME_all": float(nme),
        "nme_percent": float(nme * 100.0),
        "fr": float(fr),
        "fr_percent": float(fr * 100.0),
        "auc": float(auc),
        "nme_ci95": ci,
        "nme_percent_ci95": {
            "low": None if ci["low"] is None else float(ci["low"] * 100.0),
            "high": None if ci["high"] is None else float(ci["high"] * 100.0),
        },
    }


def metrics_for_records(
    records: T.Sequence[T.Mapping[str, T.Any]], *, threshold: float = 0.10
) -> dict[str, T.Any]:
    metrics = metrics_for_nmes(
        [record.get("nme", float("nan")) for record in records], threshold=threshold
    )
    visible = metrics_for_nmes(
        [record.get("nme_visible", float("nan")) for record in records],
        threshold=threshold,
    )
    occluded = metrics_for_nmes(
        [record.get("nme_occluded", float("nan")) for record in records],
        threshold=threshold,
    )
    metrics.update(
        {
            "NME_visible": visible["nme"],
            "NME_occluded": occluded["nme"],
            "visible_sample_count": visible["sample_count"],
            "occluded_sample_count": occluded["sample_count"],
            "visible_landmark_count": int(
                sum(
                    int(record.get("visible_landmark_count") or 0) for record in records
                )
            ),
            "occluded_landmark_count": int(
                sum(
                    int(record.get("occluded_landmark_count") or 0)
                    for record in records
                )
            ),
            "visibility_label_skipped_count": int(
                sum(
                    int(record.get("visibility_label_skipped_count") or 0)
                    for record in records
                )
            ),
        }
    )
    metrics.update(visibility_metrics_for_records(records))
    return metrics


def build_slice_report(
    records: T.Sequence[T.Mapping[str, T.Any]], *, threshold: float = 0.10
) -> dict[str, T.Any]:
    valid = [
        record
        for record in records
        if np.isfinite(float(record.get("nme", float("nan"))))
    ]
    report: dict[str, T.Any] = {
        "overall": metrics_for_records(valid, threshold=threshold)
    }
    for slice_name in (
        "by_dataset",
        "by_schema",
        "by_hard_negative_bucket",
        "by_pose_bucket",
        "by_occlusion",
        "by_profile_side",
        "by_face_size",
        "by_production_source",
    ):
        groups: dict[str, list[float]] = {}
        for record in valid:
            label = str(record.get(slice_name) or "unknown")
            groups.setdefault(label, []).append(float(record["nme"]))
        report[slice_name] = {
            label: metrics_for_records(
                [
                    record
                    for record in valid
                    if str(record.get(slice_name) or "unknown") == label
                ],
                threshold=threshold,
            )
            for label in sorted(groups)
        }
    return report


def write_eval_json(path: str | Path, payload: T.Mapping[str, T.Any]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def write_eval_csv(path: str | Path, payload: T.Mapping[str, T.Any]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, T.Any]] = []
    for model_key in ("model", "ema"):
        model_report = payload.get(model_key)
        if not isinstance(model_report, T.Mapping):
            continue
        for slice_name, slice_value in model_report.items():
            if slice_name == "records":
                continue
            if slice_name == "overall" and isinstance(slice_value, T.Mapping):
                rows.append(_csv_row(model_key, "overall", "overall", slice_value))
            elif isinstance(slice_value, T.Mapping):
                for label, metrics in slice_value.items():
                    if isinstance(metrics, T.Mapping):
                        rows.append(
                            _csv_row(model_key, slice_name, str(label), metrics)
                        )

    fieldnames = [
        "model",
        "slice",
        "label",
        "sample_count",
        "nme",
        "NME_all",
        "NME_visible",
        "NME_occluded",
        "nme_percent",
        "fr",
        "fr_percent",
        "auc",
        "visible_landmark_count",
        "occluded_landmark_count",
        "visibility_sample_count",
        "visibility_label_count",
        "visibility_label_skipped_count",
        "visibility_prediction_skipped_count",
        "visibility_AP",
        "visibility_F1@0.5",
        "visibility_ROC_AUC",
        "nme_ci95_low",
        "nme_ci95_high",
    ]
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_eval_records_jsonl(
    path: str | Path, records: T.Sequence[T.Mapping[str, T.Any]]
) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")


def write_eval_records_csv(
    path: str | Path, records: T.Sequence[T.Mapping[str, T.Any]]
) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "sample_id",
        "image",
        "dataset",
        "schema",
        "nme",
        "evaluation_head",
        "nme_visible",
        "nme_occluded",
        "visible_landmark_count",
        "occluded_landmark_count",
        "visibility_label_skipped_count",
        "visibility_target_source",
        "pose_bucket",
        "hard_negative_bucket",
        "occlusion",
        "profile_side",
        "face_size",
        "production_source",
    ]
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)


def _csv_row(
    model_key: str, slice_name: str, label: str, metrics: T.Mapping[str, T.Any]
) -> dict[str, T.Any]:
    ci = (
        metrics.get("nme_ci95")
        if isinstance(metrics.get("nme_ci95"), T.Mapping)
        else {}
    )
    return {
        "model": model_key,
        "slice": slice_name,
        "label": label,
        "sample_count": metrics.get("sample_count"),
        "nme": metrics.get("nme"),
        "NME_all": metrics.get("NME_all"),
        "NME_visible": metrics.get("NME_visible"),
        "NME_occluded": metrics.get("NME_occluded"),
        "nme_percent": metrics.get("nme_percent"),
        "fr": metrics.get("fr"),
        "fr_percent": metrics.get("fr_percent"),
        "auc": metrics.get("auc"),
        "visible_landmark_count": metrics.get("visible_landmark_count"),
        "occluded_landmark_count": metrics.get("occluded_landmark_count"),
        "visibility_sample_count": metrics.get("visibility_sample_count"),
        "visibility_label_count": metrics.get("visibility_label_count"),
        "visibility_label_skipped_count": metrics.get("visibility_label_skipped_count"),
        "visibility_prediction_skipped_count": metrics.get(
            "visibility_prediction_skipped_count"
        ),
        "visibility_AP": metrics.get("visibility_AP"),
        "visibility_F1@0.5": metrics.get("visibility_F1@0.5"),
        "visibility_ROC_AUC": metrics.get("visibility_ROC_AUC"),
        "nme_ci95_low": ci.get("low") if isinstance(ci, T.Mapping) else None,
        "nme_ci95_high": ci.get("high") if isinstance(ci, T.Mapping) else None,
    }
