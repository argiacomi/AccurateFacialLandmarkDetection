
from __future__ import annotations

import torch
import torch.nn.functional as F

from lib.landmarks.core.schema import MAP_98_TO_68


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

def schema_head_loss(stage_pred, heads, aux_labels, heatmap_loss_func, args):
    loss = torch.tensor(0.0, device=next(iter(heads.values()))["target"].device)
    loss_loc = torch.tensor(0.0, device=loss.device)
    loss_heatmap = torch.tensor(0.0, device=loss.device)
    loss_aux = torch.tensor(0.0, device=loss.device)
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
        loss = loss + head_loc + head_heatmap
        loss_loc = loss_loc + head_loc.detach()
        loss_heatmap = loss_heatmap + head_heatmap.detach()

    if (
        args.schema_consistency_weight > 0
        and "landmarks_98" in heads
        and "landmarks_68" in stage_pred
    ):
        payload = heads["landmarks_98"]
        indices = payload["indices"]
        pred_98 = stage_pred["landmarks_98"][0].index_select(0, indices)
        pred_68 = stage_pred["landmarks_68"][0].index_select(0, indices)
        projected = pred_98[:, torch.as_tensor(MAP_98_TO_68, device=pred_98.device), :]
        loss = loss + float(args.schema_consistency_weight) * F.smooth_l1_loss(
            pred_68, projected.detach(), beta=0.001
        )

    aux_outputs = stage_pred.get("_aux", {}) if isinstance(stage_pred, dict) else {}
    if args.auxiliary_loss_weight > 0 and aux_outputs:
        for task, logits in aux_outputs.items():
            labels = aux_labels.get(task)
            if labels is None:
                continue
            valid = labels >= 0
            if bool(valid.any()):
                loss_aux = loss_aux + F.cross_entropy(
                    logits[valid], labels[valid]
                ) * float(args.auxiliary_loss_weight)
        loss = loss + loss_aux

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
]

# Legacy private aliases kept for TrainHeatmapStageFP16.py and older tests/tools.
_weighted_smooth_l1 = weighted_smooth_l1
_schema_head_loss = schema_head_loss
_heatmap_batch_weight = heatmap_batch_weight
