from __future__ import annotations

from dataclasses import dataclass
import math
import typing as T

import numpy as np
import torch


AUXILIARY_CLASS_NAMES = {
    "pose_bucket": ("frontal", "profile", "profile_left", "profile_right"),
    "occlusion": ("no_occlusion", "occlusion"),
    "visibility": ("all_visible", "partially_visible"),
    "blur_quality": ("clear", "blurred"),
    "illumination_quality": ("normal", "challenging"),
    "profile_side": ("not_profile", "left", "right"),
    # Kept for checkpoint compatibility, but explicit labels are required.
    # Do not infer this from sample_weight.
    "landmark_confidence": ("normal", "low"),
}

AUXILIARY_CLASS_INDEX = {
    name: {label: index for index, label in enumerate(labels)}
    for name, labels in AUXILIARY_CLASS_NAMES.items()
}


@dataclass(frozen=True)
class AuxiliaryLabel:
    label: int
    provenance: str


def _norm(value: T.Any) -> str:
    return str(value or "").strip().lower().replace("-", "_").replace(" ", "_")


def _truthy_label(raw: T.Any, *, positive: str, negative: str) -> str | None:
    if raw is None:
        return None
    if isinstance(raw, str):
        label = _norm(raw)
        if label in {"", "unknown", "none", "missing", "null", "nan"}:
            return None
        if label in {"1", "true", "yes", "y", positive}:
            return positive
        if label in {"0", "false", "no", "n", negative}:
            return negative
        return label
    return positive if bool(raw) else negative


def _explicit_label_source(
    task: str,
    metadata: T.Mapping[str, T.Any],
    item: T.Mapping[str, T.Any],
) -> tuple[T.Any, str] | tuple[None, str]:
    """Find explicit auxiliary truth and provenance.

    Accepted locations:
    - item["auxiliary_labels"][task]
    - item[task] for task-specific manifest fields
    - metadata["auxiliary_labels"][task]
    - metadata[task]
    - metadata["attributes"][task]

    Missing labels return `(None, "missing")`.
    """

    for container_name, container in (
        ("item.auxiliary_labels", item.get("auxiliary_labels")),
        ("metadata.auxiliary_labels", metadata.get("auxiliary_labels")),
    ):
        if isinstance(container, T.Mapping) and task in container:
            return container[task], f"{container_name}.{task}"

    if task in item:
        return item[task], f"item.{task}"
    if task in metadata:
        return metadata[task], f"metadata.{task}"

    attributes = metadata.get("attributes")
    if isinstance(attributes, T.Mapping):
        if task in attributes:
            return attributes[task], f"metadata.attributes.{task}"

        # Backward-compatible explicit metadata aliases.
        attribute_aliases = {
            "blur_quality": ("blur",),
            "illumination_quality": ("illumination",),
            "pose_bucket": ("pose",),
        }
        for alias in attribute_aliases.get(task, ()):
            if alias in attributes:
                return attributes[alias], f"metadata.attributes.{alias}"

    # Profile side is often encoded explicitly in manifest condition labels.
    # Treat left/right condition labels as auditable inferred labels rather than
    # falling back to a clean/negative class.
    if task == "profile_side":
        conditions = metadata.get("conditions") or ()
        if isinstance(conditions, str):
            conditions = (conditions,)
        labels = {
            _norm(value)
            for value in conditions
        }
        condition = _norm(metadata.get("condition"))
        if condition:
            labels.add(condition)
        if "left" in labels or "profile_left" in labels:
            return "left", "metadata.conditions.profile_side"
        if "right" in labels or "profile_right" in labels:
            return "right", "metadata.conditions.profile_side"

    return None, "missing"


def _class_label_from_raw(task: str, raw: T.Any) -> str | None:
    if raw is None:
        return None

    if isinstance(raw, str):
        label = _norm(raw)
        if label in {"", "unknown", "none", "missing", "null", "nan"}:
            return None

        # Common aliases.
        if task == "occlusion":
            if label in {"1", "true", "yes", "occluded", "occlusion", "partially_occluded"}:
                return "occlusion"
            if label in {"0", "false", "no", "clear", "clean", "no_occlusion"}:
                return "no_occlusion"
        if task == "visibility":
            if label in {"all_visible", "visible", "1", "true", "yes"}:
                return "all_visible"
            if label in {"partially_visible", "partial", "occluded", "0", "false", "no"}:
                return "partially_visible"
        if task == "blur_quality":
            if label in {"1", "true", "yes", "blur", "blurred"}:
                return "blurred"
            if label in {"0", "false", "no", "clear", "sharp"}:
                return "clear"
        if task == "illumination_quality":
            if label in {"1", "true", "yes", "challenging", "harsh", "low_light"}:
                return "challenging"
            if label in {"0", "false", "no", "normal", "clear"}:
                return "normal"
        if task == "profile_side":
            if label in {"left", "right", "not_profile"}:
                return label
        if task == "pose_bucket":
            if label in {"frontal", "normal", "anchor"}:
                return "frontal"
            if label in {"profile", "large_yaw"}:
                return "profile"
            if label in {"profile_left", "large_yaw_left", "left_profile"}:
                return "profile_left"
            if label in {"profile_right", "large_yaw_right", "right_profile"}:
                return "profile_right"
        if task == "landmark_confidence":
            if label in {"normal", "low"}:
                return label

        return label

    if task == "occlusion":
        return _truthy_label(raw, positive="occlusion", negative="no_occlusion")
    if task == "visibility":
        return _truthy_label(raw, positive="partially_visible", negative="all_visible")
    if task == "blur_quality":
        return _truthy_label(raw, positive="blurred", negative="clear")
    if task == "illumination_quality":
        return _truthy_label(raw, positive="challenging", negative="normal")

    return str(raw)


def resolve_auxiliary_label(
    task: str,
    metadata: T.Mapping[str, T.Any],
    item: T.Mapping[str, T.Any],
) -> AuxiliaryLabel:
    raw, source = _explicit_label_source(task, metadata, item)
    label = _class_label_from_raw(task, raw)
    if label is None:
        return AuxiliaryLabel(-1, "missing")
    index = AUXILIARY_CLASS_INDEX[task].get(label, -1)
    if index < 0:
        return AuxiliaryLabel(-1, f"{source}:unknown_label:{label}")
    return AuxiliaryLabel(index, source)


def parse_auxiliary_loss_weights(raw: str | None) -> dict[str, float]:
    raw = str(raw or "").strip()
    if not raw:
        return {}
    out: dict[str, float] = {}
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"auxiliary loss weight item {item!r} must be task=value")
        task, value = item.split("=", 1)
        task = task.strip()
        if task not in AUXILIARY_CLASS_NAMES:
            raise ValueError(f"unknown auxiliary task in loss weights: {task!r}")
        try:
            amount = float(value)
        except ValueError as exc:
            raise ValueError(f"invalid auxiliary loss weight for {task!r}: {value!r}") from exc
        if not math.isfinite(amount) or amount < 0.0:
            raise ValueError(f"auxiliary loss weight for {task!r} must be finite and non-negative")
        out[task] = amount
    return out


def visibility_key_for_head(head_name: str) -> str:
    if head_name == "profile39":
        return "visibility_profile39"
    if head_name.startswith("landmarks_"):
        return "visibility_" + head_name.split("_", 1)[1]
    return "visibility_" + head_name


def visibility_loss_weight_for_epoch(args: T.Any) -> float:
    base = float(getattr(args, "visibility_loss_weight", 0.0))
    if base <= 0.0:
        return 0.0
    epoch = int(getattr(args, "current_epoch", 0))
    start = int(getattr(args, "visibility_loss_start_epoch", 0))
    ramp = max(int(getattr(args, "visibility_loss_ramp_epochs", 0)), 0)
    initial = float(getattr(args, "visibility_loss_initial_weight", 0.0))

    if epoch < start:
        return 0.0
    if ramp <= 0:
        return base

    progress = min(max((epoch - start + 1) / float(ramp), 0.0), 1.0)
    return initial + (base - initial) * progress


def masked_visibility_bce_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    *,
    sample_weight: torch.Tensor | None = None,
    landmark_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, int]:
    """Masked BCE for per-point visibility.

    Target convention:
      1 = visible
      0 = occluded
     -1 = unknown / do not train
    """

    target = target.to(logits.device).float()
    valid = target >= 0.0
    if landmark_mask is not None:
        valid = valid & (landmark_mask.to(logits.device).float() > 0.5)

    valid_count = int(valid.sum().item())
    if valid_count == 0:
        return logits.sum() * 0.0, 0

    loss = torch.nn.functional.binary_cross_entropy_with_logits(
        logits,
        target.clamp(0.0, 1.0),
        reduction="none",
    )

    weights = valid.to(loss.dtype)
    if sample_weight is not None:
        weights = weights * sample_weight.to(logits.device).to(loss.dtype).reshape(-1, 1)

    return (loss * weights).sum() / weights.sum().clamp_min(1.0), valid_count


def synthetic_visibility_from_occluder_mask(
    landmarks_xy: np.ndarray,
    occluder_mask: np.ndarray,
    *,
    radius: int = 2,
    overlap_threshold: float = 0.25,
    valid_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Generate pseudo per-point visibility from a binary occluder mask.

    Returns int64 targets with:
      1 = visible
      0 = pseudo-occluded
     -1 = invalid/unknown
    """

    points = np.asarray(landmarks_xy, dtype=np.float32)
    mask = np.asarray(occluder_mask).astype(bool)
    if mask.ndim != 2:
        raise ValueError("occluder_mask must be 2D")
    if points.ndim != 2 or points.shape[1] < 2:
        raise ValueError("landmarks_xy must have shape (N, 2)")

    h, w = mask.shape
    out = np.ones((points.shape[0],), dtype=np.int64)
    if valid_mask is not None:
        valid = np.asarray(valid_mask).astype(bool)
        if valid.shape[0] != points.shape[0]:
            raise ValueError("valid_mask length must match landmarks")
    else:
        valid = np.ones((points.shape[0],), dtype=bool)

    radius = max(int(radius), 0)
    for index, (x_raw, y_raw) in enumerate(points[:, :2]):
        if not valid[index] or not np.isfinite([x_raw, y_raw]).all():
            out[index] = -1
            continue

        x = int(round(float(x_raw)))
        y = int(round(float(y_raw)))
        x0 = max(0, x - radius)
        x1 = min(w, x + radius + 1)
        y0 = max(0, y - radius)
        y1 = min(h, y + radius + 1)
        if x0 >= x1 or y0 >= y1:
            out[index] = -1
            continue

        patch = mask[y0:y1, x0:x1]
        overlap = float(patch.mean()) if patch.size else 0.0
        out[index] = 0 if overlap >= float(overlap_threshold) else 1

    return out


__all__ = [
    "AUXILIARY_CLASS_NAMES",
    "AUXILIARY_CLASS_INDEX",
    "AuxiliaryLabel",
    "resolve_auxiliary_label",
    "parse_auxiliary_loss_weights",
    "visibility_key_for_head",
    "visibility_loss_weight_for_epoch",
    "masked_visibility_bce_loss",
    "synthetic_visibility_from_occluder_mask",
]
