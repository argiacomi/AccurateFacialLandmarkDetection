from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from lib.core.schema import MAP_106_TO_68, MAP_98_TO_68
from lib.training.auxiliary import (
    masked_visibility_bce_loss,
    parse_auxiliary_loss_weights,
    visibility_key_for_head,
    visibility_loss_weight_for_epoch,
)
from loss import STARLoss_v2

CONSISTENCY_MAPS_TO_68 = {
    "landmarks_98": MAP_98_TO_68,
    "landmarks_106": MAP_106_TO_68,
}


def weighted_smooth_l1(pred_loc, target, sample_weight, landmark_mask, beta=0.001):
    per_point = F.smooth_l1_loss(pred_loc, target, beta=beta, reduction="none").mean(
        dim=2
    )
    landmark_mask = landmark_mask.to(per_point.device).float()
    per_sample = (per_point * landmark_mask).sum(dim=1) / landmark_mask.sum(
        dim=1
    ).clamp_min(1.0)
    if sample_weight is not None:
        return (per_sample * sample_weight).mean()
    return per_sample.mean()


def _schema_head_weight_map(raw):
    weights = {}
    raw = str(raw or "").strip()
    if not raw:
        return weights

    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(
                f"schema head loss weight item {item!r} must be formatted as head=value"
            )

        name, value = item.split("=", 1)
        name = name.strip()
        raw_value = value.strip()

        if not name:
            raise ValueError(
                f"schema head loss weight item {item!r} has an empty head name"
            )

        try:
            amount = float(raw_value)
        except ValueError as exc:
            raise ValueError(
                f"schema head loss weight for {name!r} has invalid value {raw_value!r}"
            ) from exc

        if not math.isfinite(amount) or amount < 0.0:
            raise ValueError(
                f"schema head loss weight for {name!r} must be a finite non-negative value, "
                f"got {raw_value!r}"
            )

        weights[name] = amount

    return weights


def _star_loss_v2_per_point(star_loss_func, pred_heatmap, target):
    # Prefer the optimized STARLoss_v2 implementation. It keeps covariance and
    # eigendecomposition on GPU when pred_heatmap is CUDA, and computes STAR
    # statistics in fp32 under AMP/autocast.
    if hasattr(star_loss_func, "per_point_loss"):
        return star_loss_func.per_point_loss(pred_heatmap, target)

    # Compatibility fallback for older STARLoss_v2-like objects.
    bs, npoints, h, w = pred_heatmap.shape
    heatmap = torch.softmax(
        pred_heatmap.float().reshape((bs, npoints, -1)),
        dim=-1,
    ).reshape((bs, npoints, h, w))
    target = target.to(device=heatmap.device, dtype=heatmap.dtype)

    means = star_loss_func.weighted_mean(heatmap)
    covars = star_loss_func.unbiased_weighted_covariance(heatmap, means)
    covars_flat = covars.reshape(bs * npoints, 2, 2)
    covars_flat = 0.5 * (covars_flat + covars_flat.transpose(-1, -2))

    # Honor the loss object's opt-in finite guard when available; older
    # STARLoss_v2-like objects without it skip the (sync-inducing) check.
    maybe_check = getattr(star_loss_func, "_maybe_check_finite", None)
    if callable(maybe_check):
        maybe_check(covars_flat)

    try:
        evalues, evectors = torch.linalg.eigh(covars_flat, UPLO="U")
    except RuntimeError:
        evalues, evectors = torch.linalg.eigh(covars_flat.cpu(), UPLO="U")
        evalues = evalues.to(covars_flat.device)
        evectors = evectors.to(covars_flat.device)

    evalues = (
        evalues.reshape(bs, npoints, 2)
        .to(heatmap)
        .clamp_min(float(getattr(star_loss_func, "EPSILON", 1e-5)))
    )
    evectors = evectors.reshape(bs, npoints, 2, 2).to(heatmap)

    loss_trans = star_loss_func.ambiguity_guided_decompose(
        target - means,
        evalues,
        evectors,
    )
    loss_eigen = star_loss_func.eigenvalue_restriction(evalues, bs, npoints)
    return loss_trans + star_loss_func.w * loss_eigen


def _weighted_star_loss_v2(
    star_loss_func, pred_heatmap, target, sample_weight, landmark_mask
):
    # STAR is only useful for valid landmarks. Avoid the expensive
    # softmax/covariance/eigh path when the active head/batch has no valid
    # landmark supervision.
    if landmark_mask is None:
        point_weights = torch.ones(
            (pred_heatmap.shape[0], pred_heatmap.shape[1]),
            device=pred_heatmap.device,
            dtype=pred_heatmap.dtype,
        )
    else:
        point_weights = landmark_mask.to(pred_heatmap.device).to(pred_heatmap.dtype)

    if sample_weight is not None:
        point_weights = point_weights * sample_weight.to(pred_heatmap.device).to(
            pred_heatmap.dtype
        ).reshape(-1, 1)

    if bool((point_weights > 0).any().item()) is False:
        return pred_heatmap.sum() * 0.0

    per_point = _star_loss_v2_per_point(star_loss_func, pred_heatmap, target)
    return (
        per_point * point_weights.to(per_point.dtype)
    ).sum() / point_weights.sum().clamp_min(1.0)


def _visibility_target_weight_for_payload(payload, args, *, device):
    base_weight = payload.get("visibility_target_weight")
    provenance = payload.get("visibility_target_provenance")

    if provenance is None:
        return base_weight

    pseudo_weight = float(getattr(args, "visibility_pseudo_loss_weight", 0.0))
    weights = []
    for source in provenance:
        source_text = str(source or "")
        weights.append(pseudo_weight if source_text.startswith("synthetic") else 1.0)

    provenance_weight = torch.as_tensor(
        weights, device=device, dtype=torch.float32
    ).reshape(-1, 1)
    if base_weight is None:
        return provenance_weight

    base_weight = base_weight.to(device).float()
    if base_weight.ndim == 1:
        base_weight = base_weight.reshape(-1, 1)
    return provenance_weight * base_weight


def schema_head_loss(
    stage_pred,
    heads,
    aux_labels,
    heatmap_loss_func,
    args,
    *,
    return_details=False,
    star_loss_func=None,
    include_auxiliary_loss=True,
    include_visibility_loss=True,
):
    loss = torch.tensor(0.0, device=next(iter(heads.values()))["target"].device)
    loss_loc = torch.tensor(0.0, device=loss.device)
    loss_heatmap = torch.tensor(0.0, device=loss.device)
    loss_aux = torch.tensor(0.0, device=loss.device)
    loss_consistency = torch.tensor(0.0, device=loss.device)
    loss_star = torch.tensor(0.0, device=loss.device)
    loss_visibility = torch.tensor(0.0, device=loss.device)
    details = {
        "head_sample_counts": {},
        "head_loss_contributions": {},
        "visibility_valid_counts": {},
        "auxiliary_valid_counts": {},
        "auxiliary_loss_contributions": {},
        "auxiliary_correct_counts": {},
        "auxiliary_accuracy": {},
        "loss_consistency": loss_consistency,
        "loss_star": loss_star,
        "loss_visibility": loss_visibility,
        "visibility_loss_weight": torch.tensor(0.0, device=loss.device),
    }
    head_weight_map = _schema_head_weight_map(
        getattr(args, "schema_head_loss_weights", "")
    )
    total_head_samples = max(
        1,
        sum(int(payload["indices"].numel()) for payload in heads.values()),
    )
    weighting = str(getattr(args, "schema_head_loss_weighting", "sample_count"))
    for head_name, payload in heads.items():
        pred_loc, pred_heatmap = stage_pred[head_name]
        indices = payload["indices"]
        pred_loc = pred_loc.index_select(0, indices)
        pred_heatmap = pred_heatmap.index_select(0, indices)
        target = payload["target"]
        heatmap = payload["heatmap"]
        sample_weight = payload["sample_weight"]
        landmark_mask = payload["landmark_mask"]
        B, C, H, W = pred_heatmap.shape
        head_loc = (
            weighted_smooth_l1(
                pred_loc, target, sample_weight, landmark_mask, beta=0.001
            )
            * args.locw
        )
        pred_prob = F.softmax(pred_heatmap.reshape((B, C, -1)), dim=2).reshape(
            (B, C, H, W)
        )
        head_heatmap = (
            heatmap_loss_func(
                pred_prob,
                heatmap,
                batch_weights=heatmap_batch_weight(
                    sample_weight, pred_heatmap, landmark_mask
                ),
            )
            * args.hw
        )
        head_star = torch.tensor(0.0, device=loss.device)
        head_visibility = torch.tensor(0.0, device=loss.device)
        # STAR is already active-head scoped because this loop iterates only
        # supervised heads present in `heads`. _weighted_star_loss_v2() also
        # skips fully masked heads before expensive STAR math.
        if float(getattr(args, "star_loss_weight", 0.0)) > 0.0:
            if star_loss_func is None:
                star_loss_func = STARLoss_v2(
                    check_finite=bool(getattr(args, "star_loss_check_finite", False)),
                    check_finite_interval=int(
                        getattr(args, "star_loss_check_finite_interval", 0)
                    ),
                )
            head_star = _weighted_star_loss_v2(
                star_loss_func,
                pred_heatmap,
                target,
                sample_weight,
                landmark_mask,
            ) * float(args.star_loss_weight)

        visibility_weight = visibility_loss_weight_for_epoch(args)
        visibility_key = visibility_key_for_head(head_name)
        if (
            include_visibility_loss
            and visibility_weight > 0.0
            and visibility_key in stage_pred
            and "visibility_target" in payload
        ):
            visibility_logits = stage_pred[visibility_key].index_select(0, indices)
            head_visibility_raw, visibility_valid = masked_visibility_bce_loss(
                visibility_logits,
                payload["visibility_target"],
                sample_weight=sample_weight,
                landmark_mask=landmark_mask,
                target_weight=_visibility_target_weight_for_payload(
                    payload, args, device=loss.device
                ),
            )
            head_visibility = head_visibility_raw * visibility_weight
            loss_visibility = loss_visibility + head_visibility.detach()
            details["visibility_valid_counts"][head_name] = visibility_valid
            details["visibility_loss_weight"] = torch.tensor(
                visibility_weight, device=loss.device
            )

        head_samples = int(indices.numel())
        head_scale = float(head_weight_map.get(head_name, 1.0))
        if weighting == "sample_count":
            head_scale *= float(head_samples) / float(total_head_samples)
        head_loss = (head_loc + head_heatmap + head_star + head_visibility) * head_scale
        loss = loss + head_loss
        loss_loc = loss_loc + head_loc.detach()
        loss_heatmap = loss_heatmap + head_heatmap.detach()
        loss_star = loss_star + head_star.detach()
        details["head_sample_counts"][head_name] = head_samples
        details["head_loss_contributions"][head_name] = head_loss.detach()

    if args.schema_consistency_weight > 0 and "landmarks_68" in stage_pred:
        consistency_terms = []
        consistency_weights = []
        for source_head, map_to_68 in CONSISTENCY_MAPS_TO_68.items():
            if source_head not in heads or source_head not in stage_pred:
                continue

            payload = heads[source_head]
            indices = payload["indices"]
            head_sample_count = int(indices.numel())
            if head_sample_count <= 0:
                continue

            pred_source = stage_pred[source_head][0].index_select(0, indices)
            pred_68 = stage_pred["landmarks_68"][0].index_select(0, indices)

            projected = pred_source[
                :, torch.as_tensor(map_to_68, device=pred_source.device), :
            ]
            consistency_terms.append(
                F.smooth_l1_loss(pred_68, projected.detach(), beta=0.001)
            )
            consistency_weights.append(
                torch.tensor(
                    float(head_sample_count),
                    device=pred_source.device,
                    dtype=pred_source.dtype,
                )
            )

        if consistency_terms:
            stacked_weights = torch.stack(consistency_weights)
            loss_consistency = (
                float(args.schema_consistency_weight)
                * (torch.stack(consistency_terms) * stacked_weights).sum()
                / stacked_weights.sum().clamp_min(1.0)
            )
            loss = loss + loss_consistency

    aux_outputs = stage_pred.get("_aux", {}) if isinstance(stage_pred, dict) else {}
    if include_auxiliary_loss and args.auxiliary_loss_weight > 0 and aux_outputs:
        task_weights = parse_auxiliary_loss_weights(
            getattr(args, "auxiliary_loss_weights", "")
        )
        valid_task_count = 0
        for task, logits in aux_outputs.items():
            labels = aux_labels.get(task)
            if labels is None:
                continue
            valid = labels >= 0
            valid_count = int(valid.sum().item())
            details["auxiliary_valid_counts"][task] = valid_count
            if valid_count <= 0:
                continue
            with torch.no_grad():
                predicted = logits[valid].argmax(dim=1)
                correct = int((predicted == labels[valid]).sum().item())
            details["auxiliary_correct_counts"][task] = correct
            details["auxiliary_accuracy"][task] = float(correct) / float(valid_count)
            task_weight = float(task_weights.get(task, 1.0))
            if task_weight <= 0.0:
                continue
            task_loss = F.cross_entropy(logits[valid], labels[valid])
            contribution = task_loss * float(args.auxiliary_loss_weight) * task_weight
            details["auxiliary_loss_contributions"][task] = contribution.detach()
            loss_aux = loss_aux + contribution
            valid_task_count += 1
        if valid_task_count > 0:
            loss_aux = loss_aux / float(valid_task_count)
            loss = loss + loss_aux

    details["loss_consistency"] = loss_consistency.detach()
    details["loss_star"] = loss_star.detach()
    details["loss_visibility"] = loss_visibility.detach()
    if return_details:
        return loss, loss_loc, loss_heatmap, loss_aux, details
    return loss, loss_loc, loss_heatmap, loss_aux


def heatmap_batch_weight(sample_weight, pred_heatmap, landmark_mask=None):
    if sample_weight is None and landmark_mask is None:
        return None
    weights = torch.ones(
        (pred_heatmap.shape[0], pred_heatmap.shape[1]),
        device=pred_heatmap.device,
        dtype=pred_heatmap.dtype,
    )
    if landmark_mask is not None:
        weights = weights * landmark_mask.to(pred_heatmap.device).to(pred_heatmap.dtype)
    if sample_weight is not None:
        weights = weights * sample_weight.to(pred_heatmap.device).to(
            pred_heatmap.dtype
        ).reshape(-1, 1)
    return weights.reshape(pred_heatmap.shape[0], pred_heatmap.shape[1], 1, 1)


# Public trainer loss API.
__all__ = [
    "weighted_smooth_l1",
    "schema_head_loss",
    "heatmap_batch_weight",
    "_weighted_star_loss_v2",
]

# Legacy private aliases kept for TrainHeatmapStageFP16.py and older tests/tools.
_weighted_smooth_l1 = weighted_smooth_l1
_schema_head_loss = schema_head_loss
_heatmap_batch_weight = heatmap_batch_weight
