from __future__ import annotations

import os

import numpy as np
import torch
from torch.utils.data._utils.collate import default_collate

from lib.core.schema import head_name_for_schema
from lib.evaluation.split_safe import (
    build_slice_report,
    metrics_for_nmes,
    record_for_sample,
)
from lib.logging_utils import (
    Verbosity,
    fmt_count,
    fmt_num,
    iterate_with_progress,
    log_event,
)
from lib.training.auxiliary import visibility_key_for_head


def eval_collate(batch):
    if batch and len(batch[0]) >= 4 and isinstance(batch[0][3], dict):
        data = default_collate([item[0] for item in batch])
        metadata = [item[3] for item in batch]
        head_names = []
        for item in batch:
            meta = item[3]
            head_name = str(meta.get("head_name") or "")
            if not head_name:
                schema = meta.get("source_schema") or meta.get("schema") or ""
                if schema:
                    try:
                        head_name = head_name_for_schema(schema)
                    except ValueError:
                        head_name = ""
            head_names.append(head_name or "landmarks_68")
        target_shapes = {tuple(item[1].shape) for item in batch}
        if len(target_shapes) == 1 and set(head_names) == {"landmarks_68"}:
            target = default_collate([item[1] for item in batch])
            landmark_mask = default_collate([item[2] for item in batch])
            return data, target, landmark_mask, metadata

        grouped = {}
        for index, (item, head_name) in enumerate(zip(batch, head_names)):
            grouped.setdefault(
                head_name,
                {"indices": [], "target": [], "landmark_mask": [], "metadata": []},
            )
            grouped[head_name]["indices"].append(index)
            grouped[head_name]["target"].append(item[1])
            grouped[head_name]["landmark_mask"].append(item[2])
            meta = dict(item[3])
            meta["head_name"] = head_name
            grouped[head_name]["metadata"].append(meta)
        return {
            "image": data,
            "heads": {
                head_name: {
                    "indices": torch.as_tensor(payload["indices"], dtype=torch.long),
                    "target": default_collate(payload["target"]),
                    "landmark_mask": default_collate(payload["landmark_mask"]),
                    "metadata": payload["metadata"],
                }
                for head_name, payload in grouped.items()
            },
        }
    return default_collate(batch)


def unpack_eval_batch(batch):
    data = batch[0]
    target = batch[1]
    if len(batch) >= 3:
        landmark_mask = batch[2]
    else:
        landmark_mask = torch.ones(target.shape[:2], dtype=torch.float32)
    if len(batch) >= 4:
        metadata = batch[3]
    else:
        metadata = [{} for _ in range(int(target.shape[0]))]
    return data, target, landmark_mask, metadata


def _masked_nme_values(pred_keypoints, keypoints, landmark_mask):
    pred = pred_keypoints.detach().float().cpu().numpy()
    target = keypoints.detach().float().cpu().numpy()
    mask = landmark_mask.detach().float().cpu().numpy() > 0.5

    values = []
    for pred_i, target_i, mask_i in zip(pred, target, mask):
        if mask_i.sum() <= 0:
            values.append(float("nan"))
            continue

        valid = target_i[mask_i]
        if valid.shape[0] <= 1:
            values.append(float("nan"))
            continue

        span = np.max(valid, axis=0) - np.min(valid, axis=0)
        span_norm = float(max(span[0], span[1]))

        eye_norm = None
        if mask_i.shape[0] > 45 and mask_i[36] and mask_i[45]:
            eye_norm = float(np.linalg.norm(target_i[36] - target_i[45]))

        # Prefer canonical outer-eye interocular only when it is plausible.
        #
        # MERL-RAV has some frontal samples where landmarks 36/45 are valid but
        # nearly collapsed, e.g. eye_norm ~= 0.04 in normalized coordinates.
        # That explodes NME even when the rest of the face is reasonable.
        #
        # Since targets are normalized to roughly [0, 1], 0.05 is ~12.75px on
        # a 256 crop. Also require eye_norm to be at least 15% of the visible
        # landmark span, otherwise fall back to span normalization.
        if (
            eye_norm is not None
            and np.isfinite(eye_norm)
            and eye_norm > 0.05
            and np.isfinite(span_norm)
            and span_norm > 1e-6
            and eye_norm >= 0.15 * span_norm
        ):
            normalizer = eye_norm
        else:
            normalizer = span_norm

        if not np.isfinite(normalizer) or normalizer <= 1e-6:
            values.append(float("nan"))
            continue

        dist = np.linalg.norm(target_i[mask_i] - pred_i[mask_i], axis=1)
        values.append(float(dist.mean() / normalizer))

    return np.asarray(values, dtype=np.float32)


def _normalizer_for_masked_target(target_i, mask_i):
    valid = target_i[mask_i]
    if valid.shape[0] <= 1:
        return None
    span = np.max(valid, axis=0) - np.min(valid, axis=0)
    span_norm = float(max(span[0], span[1]))
    eye_norm = None
    if mask_i.shape[0] > 45 and mask_i[36] and mask_i[45]:
        eye_norm = float(np.linalg.norm(target_i[36] - target_i[45]))
    if (
        eye_norm is not None
        and np.isfinite(eye_norm)
        and eye_norm > 0.05
        and np.isfinite(span_norm)
        and span_norm > 1e-6
        and eye_norm >= 0.15 * span_norm
    ):
        return eye_norm
    if np.isfinite(span_norm) and span_norm > 1e-6:
        return span_norm
    return None


def _visibility_target_from_meta(meta, expected_count):
    raw = meta.get("visibility_target") if isinstance(meta, dict) else None
    if raw is None:
        return np.full((expected_count,), -1, dtype=np.int64)
    arr = np.asarray(raw, dtype=np.int64).reshape(-1)
    if arr.size != expected_count:
        return np.full((expected_count,), -1, dtype=np.int64)
    return arr


def _visibility_logits_from_stage(stage_pred, head_name="landmarks_68"):
    if not isinstance(stage_pred, dict):
        return None
    point_suffix = head_name.removeprefix("landmarks_").removeprefix("profile")
    for key in (
        visibility_key_for_head(head_name),
        f"{head_name}_visibility_logits",
        f"visibility_{point_suffix}_logits",
        "visibility_logits",
    ):
        value = stage_pred.get(key)
        if torch.is_tensor(value):
            return value
    head_payload = stage_pred.get(head_name)
    if (
        isinstance(head_payload, (list, tuple))
        and len(head_payload) >= 3
        and torch.is_tensor(head_payload[2])
    ):
        return head_payload[2]
    return None


def _visibility_aware_records(
    pred_keypoints, keypoints, landmark_mask, metadata, visibility_logits=None
):
    pred = pred_keypoints.detach().float().cpu().numpy()
    target = keypoints.detach().float().cpu().numpy()
    mask = landmark_mask.detach().float().cpu().numpy() > 0.5
    logits = None
    if visibility_logits is not None:
        logits = visibility_logits.detach().float().cpu().numpy()

    records = []
    for index, (pred_i, target_i, mask_i, meta) in enumerate(
        zip(pred, target, mask, metadata)
    ):
        meta = meta if isinstance(meta, dict) else {}
        normalizer = _normalizer_for_masked_target(target_i, mask_i)
        if normalizer is None:
            continue
        point_errors = np.full((target_i.shape[0],), np.nan, dtype=np.float32)
        point_errors[mask_i] = np.linalg.norm(
            target_i[mask_i] - pred_i[mask_i], axis=1
        ) / float(normalizer)
        nme = float(np.nanmean(point_errors[mask_i]))

        visibility_target = _visibility_target_from_meta(meta, target_i.shape[0])
        visible_label_mask = visibility_target == 1
        occluded_label_mask = visibility_target == 0
        unknown_label_mask = ~np.isin(visibility_target, (0, 1))
        visible_error_mask = mask_i & visible_label_mask
        occluded_error_mask = mask_i & occluded_label_mask

        record = record_for_sample(meta, nme)
        record["evaluation_head"] = str(meta.get("head_name") or "")
        record["visibility_target_source"] = str(
            meta.get("visibility_target_source") or ""
        )
        record["visible_landmark_count"] = int(np.sum(visible_label_mask))
        record["occluded_landmark_count"] = int(np.sum(occluded_label_mask))
        record["visibility_label_skipped_count"] = int(np.sum(unknown_label_mask))
        record["nme_visible"] = (
            float(np.nanmean(point_errors[visible_error_mask]))
            if np.any(visible_error_mask)
            else None
        )
        record["nme_occluded"] = (
            float(np.nanmean(point_errors[occluded_error_mask]))
            if np.any(occluded_error_mask)
            else None
        )
        record["visibility_targets"] = visibility_target.tolist()
        if (
            logits is not None
            and index < logits.shape[0]
            and logits.shape[1] == target_i.shape[0]
        ):
            record["visibility_scores"] = [
                float(value) if not unknown_label_mask[pos] else float("nan")
                for pos, value in enumerate(logits[index].tolist())
            ]
        records.append(record)
    return records


def _masked_nme_list(pred_keypoints, keypoints, landmark_mask):
    values = _masked_nme_values(pred_keypoints, keypoints, landmark_mask)
    return values[np.isfinite(values)]


def _append_finite_nmes(target, values):
    arr = np.asarray(values, dtype=np.float32).reshape(-1)
    if arr.size == 0:
        return
    target.extend(float(value) for value in arr[np.isfinite(arr)])


def evaluate_landmark_model(
    model,
    test_dataloader,
    device,
    *,
    include_records=False,
    non_blocking=False,
    build_records=True,
    show_progress=True,
):
    """Evaluate landmarks with optional per-sample slice-report records.

    ``build_records=False`` keeps the expensive CPU/NumPy per-sample record path
    off for routine sampled evaluations. It still computes overall NME/FR/AUC via
    vectorized batch NME values. Record construction is forced on whenever callers
    request record output files.
    """
    model.eval()
    eval_device = torch.device(device)
    eval_autocast_enabled = eval_device.type == "cuda"
    build_records = bool(build_records or include_records)
    records = []
    nme_values = []

    iterator = iterate_with_progress(
        test_dataloader,
        total=len(test_dataloader) if hasattr(test_dataloader, "__len__") else None,
        description="eval",
        enabled=show_progress,
    )
    for batch_idx, batch in enumerate(iterator):
        if isinstance(batch, dict) and "heads" in batch:
            data = batch["image"].to(device, non_blocking=non_blocking)
            with torch.autocast(
                device_type=eval_device.type,
                dtype=torch.float16,
                enabled=eval_autocast_enabled,
            ):
                stage_pred = model(data)[-1]
            for head_name, payload in batch["heads"].items():
                indices = payload["indices"].to(device, non_blocking=non_blocking)
                pred_keypoints, heatmap = landmark_prediction_for_head(
                    stage_pred, head_name
                )
                pred_keypoints = pred_keypoints.index_select(0, indices)
                keypoints = payload["target"].to(device, non_blocking=non_blocking)
                landmark_mask = payload["landmark_mask"].to(
                    device, non_blocking=non_blocking
                )

                if build_records:
                    visibility_logits = _visibility_logits_from_stage(
                        stage_pred, head_name
                    )
                    if visibility_logits is not None:
                        visibility_logits = visibility_logits.index_select(0, indices)
                    records.extend(
                        _visibility_aware_records(
                            pred_keypoints,
                            keypoints,
                            landmark_mask,
                            payload["metadata"],
                            visibility_logits=visibility_logits,
                        )
                    )
                else:
                    _append_finite_nmes(
                        nme_values,
                        _masked_nme_values(pred_keypoints, keypoints, landmark_mask),
                    )
            continue

        data, target, landmark_mask, metadata = unpack_eval_batch(batch)
        data = data.to(device, non_blocking=non_blocking)
        keypoints = target.to(device, non_blocking=non_blocking)
        landmark_mask = landmark_mask.to(device, non_blocking=non_blocking)
        with torch.autocast(
            device_type=eval_device.type,
            dtype=torch.float16,
            enabled=eval_autocast_enabled,
        ):
            stage_pred = model(data)[-1]
        pred_keypoints, heatmap = landmark_prediction_for_head(
            stage_pred, "landmarks_68"
        )

        if build_records:
            records.extend(
                _visibility_aware_records(
                    pred_keypoints,
                    keypoints,
                    landmark_mask,
                    metadata,
                    visibility_logits=_visibility_logits_from_stage(
                        stage_pred, "landmarks_68"
                    ),
                )
            )
        else:
            _append_finite_nmes(
                nme_values,
                _masked_nme_values(pred_keypoints, keypoints, landmark_mask),
            )

    if build_records:
        report = build_slice_report(records)
        if include_records:
            report["records"] = records
        return report

    return {
        "overall": metrics_for_nmes(nme_values),
        "record_mode": "overall_only",
    }


def records_from_report(report):
    records = report.get("records", [])
    return records if isinstance(records, list) else []


def landmarks_68_prediction(stage_pred):
    return landmark_prediction_for_head(stage_pred, "landmarks_68")


def landmark_prediction_for_head(stage_pred, head_name):
    if isinstance(stage_pred, dict):
        if head_name not in stage_pred:
            available = ", ".join(
                sorted(key for key in stage_pred if not key.startswith("_"))
            )
            raise ValueError(
                f"model output does not include evaluation head '{head_name}' (available: {available})"
            )
        return stage_pred[head_name]
    if head_name != "landmarks_68":
        raise ValueError(
            f"legacy model output can only evaluate landmarks_68, not '{head_name}'"
        )
    return stage_pred


def print_eval_summary(title, report):
    metrics = report["overall"]
    if metrics["sample_count"] == 0:
        log_event("eval", f"{title} | no samples", level=Verbosity.QUIET)
        return
    log_event(
        "eval",
        f"{title} | "
        f"NME {fmt_num(metrics['nme_percent'])}% | "
        f"FR@0.10 {fmt_num(metrics['fr_percent'], 2)}% | "
        f"AUC@0.10 {fmt_num(metrics['auc'])} | "
        f"n={fmt_count(metrics['sample_count'])}",
        level=Verbosity.QUIET,
    )


def eval_report_json_path(args):
    if args.eval_report_json:
        return args.eval_report_json
    return os.path.join(args.ckpt_folder, "eval_report.json")


# Public trainer evaluation API.
__all__ = [
    "eval_collate",
    "unpack_eval_batch",
    "evaluate_landmark_model",
    "records_from_report",
    "landmarks_68_prediction",
    "landmark_prediction_for_head",
    "print_eval_summary",
    "eval_report_json_path",
]

# Legacy private aliases kept for TrainHeatmapStageFP16.py and older tests/tools.
_eval_collate = eval_collate
_unpack_eval_batch = unpack_eval_batch
_evaluate_landmark_model = evaluate_landmark_model
_records_from_report = records_from_report
_landmarks_68_prediction = landmarks_68_prediction
_landmark_prediction_for_head = landmark_prediction_for_head
_print_eval_summary = print_eval_summary
_eval_report_json_path = eval_report_json_path
