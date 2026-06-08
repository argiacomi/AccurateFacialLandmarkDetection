from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from lib.landmarks.training.auxiliary import (
    masked_visibility_bce_loss,
    parse_auxiliary_loss_weights,
    resolve_auxiliary_label,
    synthetic_visibility_from_occluder_mask,
    visibility_loss_weight_for_epoch,
)
from lib.landmarks.training.data import _schema_aware_collate
from lib.landmarks.training.losses import schema_head_loss
from lib.landmarks.models.cdvit import VitAttnStage


def test_auxiliary_label_resolution_prefers_explicit_labels_and_keeps_provenance():
    metadata = {"auxiliary_labels": {"occlusion": "occlusion"}}
    resolved = resolve_auxiliary_label("occlusion", metadata, {})
    assert resolved.label == 1
    assert resolved.provenance == "metadata.auxiliary_labels.occlusion"


def test_auxiliary_label_resolution_leaves_unknown_missing_instead_of_clean_negative():
    assert resolve_auxiliary_label("occlusion", {}, {}).label == -1
    assert (
        resolve_auxiliary_label(
            "visibility", {}, {"landmark_mask": torch.ones(68)}
        ).label
        == -1
    )
    assert (
        resolve_auxiliary_label(
            "landmark_confidence", {}, {"sample_weight": torch.tensor(5.0)}
        ).label
        == -1
    )


def test_schema_aware_collate_batches_auxiliary_provenance_and_visibility_targets():
    item = {
        "image": torch.zeros(3, 256, 256),
        "target": torch.zeros(68, 2),
        "heatmap": torch.zeros(68, 8, 8),
        "landmark_mask": torch.ones(68),
        "visibility_target": torch.tensor([1, 0, *([-1] * 66)], dtype=torch.float32),
        "sample_weight": torch.tensor(1.0),
        "schema": "2d_68",
        "head_name": "landmarks_68",
        "metadata": {
            "dataset": "w300",
            "source_schema": "2d_68",
            "auxiliary_labels": {"occlusion": "occlusion"},
        },
    }

    batch = _schema_aware_collate([item])

    assert batch["aux_labels"]["occlusion"].tolist() == [1]
    assert batch["aux_provenance"]["occlusion"] == [
        "metadata.auxiliary_labels.occlusion"
    ]
    assert batch["heads"]["landmarks_68"]["visibility_target"].shape == (1, 68)


def test_auxiliary_loss_is_normalized_by_valid_task_count_and_skips_missing_labels():
    args = SimpleNamespace(
        locw=0.0,
        hw=0.0,
        schema_consistency_weight=0.0,
        auxiliary_loss_weight=0.5,
        auxiliary_loss_weights="occlusion=1.0,blur_quality=0.0",
        schema_head_loss_weighting="sample_count",
        schema_head_loss_weights="",
        star_loss_weight=0.0,
        visibility_loss_weight=0.0,
        current_epoch=0,
    )
    stage_pred = {
        "landmarks_68": (
            torch.zeros(2, 68, 2, requires_grad=True),
            torch.zeros(2, 68, 4, 4, requires_grad=True),
        ),
        "_aux": {
            "occlusion": torch.tensor([[0.0, 2.0], [2.0, 0.0]], requires_grad=True),
            "blur_quality": torch.randn(2, 2, requires_grad=True),
            "visibility": torch.randn(2, 2, requires_grad=True),
        },
    }
    heads = {
        "landmarks_68": {
            "indices": torch.tensor([0, 1]),
            "target": torch.zeros(2, 68, 2),
            "heatmap": torch.zeros(2, 68, 4, 4),
            "landmark_mask": torch.ones(2, 68),
            "sample_weight": torch.ones(2),
            "visibility_target": torch.full((2, 68), -1.0),
        }
    }
    aux_labels = {
        "occlusion": torch.tensor([1, -1]),
        "blur_quality": torch.tensor([0, 1]),
        "visibility": torch.tensor([-1, -1]),
    }

    loss, *_rest, details = schema_head_loss(
        stage_pred,
        heads,
        aux_labels,
        lambda *args, **kwargs: torch.tensor(0.0),
        args,
        return_details=True,
    )
    loss.backward()

    assert details["auxiliary_valid_counts"]["occlusion"] == 1
    assert details["auxiliary_valid_counts"]["blur_quality"] == 2
    assert details["auxiliary_valid_counts"]["visibility"] == 0
    assert "occlusion" in details["auxiliary_loss_contributions"]
    assert "blur_quality" not in details["auxiliary_loss_contributions"]


def test_visibility_head_and_loss_use_masked_targets():
    args = SimpleNamespace(
        locw=0.0,
        hw=0.0,
        schema_consistency_weight=0.0,
        auxiliary_loss_weight=0.0,
        auxiliary_loss_weights="",
        schema_head_loss_weighting="sample_count",
        schema_head_loss_weights="",
        star_loss_weight=0.0,
        visibility_loss_weight=0.25,
        visibility_loss_initial_weight=0.0,
        visibility_loss_start_epoch=0,
        visibility_loss_ramp_epochs=0,
        current_epoch=0,
    )
    stage_pred = {
        "landmarks_68": (
            torch.zeros(1, 68, 2, requires_grad=True),
            torch.zeros(1, 68, 4, 4, requires_grad=True),
        ),
        "visibility_68": torch.zeros(1, 68, requires_grad=True),
    }
    heads = {
        "landmarks_68": {
            "indices": torch.tensor([0]),
            "target": torch.zeros(1, 68, 2),
            "heatmap": torch.zeros(1, 68, 4, 4),
            "landmark_mask": torch.ones(1, 68),
            "sample_weight": torch.ones(1),
            "visibility_target": torch.tensor([[1.0, 0.0, *([-1.0] * 66)]]),
        }
    }

    loss, *_rest, details = schema_head_loss(
        stage_pred,
        heads,
        {},
        lambda *args, **kwargs: torch.tensor(0.0),
        args,
        return_details=True,
    )
    loss.backward()

    assert details["visibility_valid_counts"]["landmarks_68"] == 2
    assert details["loss_visibility"].item() > 0.0
    assert stage_pred["visibility_68"].grad[:, :2].abs().sum().item() > 0.0
    assert stage_pred["visibility_68"].grad[:, 2:].abs().sum().item() == pytest.approx(
        0.0
    )


def test_visibility_loss_schedule_warm_start_and_ramp():
    args = SimpleNamespace(
        visibility_loss_weight=0.2,
        visibility_loss_initial_weight=0.0,
        visibility_loss_start_epoch=2,
        visibility_loss_ramp_epochs=4,
        current_epoch=1,
    )
    assert visibility_loss_weight_for_epoch(args) == 0.0
    args.current_epoch = 2
    assert visibility_loss_weight_for_epoch(args) == pytest.approx(0.05)
    args.current_epoch = 5
    assert visibility_loss_weight_for_epoch(args) == pytest.approx(0.2)


def test_synthetic_occlusion_generates_pseudo_visibility_targets():
    points = np.asarray([[5, 5], [10, 10], [50, 50]], dtype=np.float32)
    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[8:13, 8:13] = 1

    visibility = synthetic_visibility_from_occluder_mask(
        points,
        mask,
        radius=2,
        overlap_threshold=0.25,
    )

    assert visibility.tolist() == [1, 0, 1]


def test_cdvit_emits_schema_visibility_heads():
    model = VitAttnStage(
        lmk_num=68,
        nstack=1,
        heatmap_size=8,
        max_depth=16,
        backbone_net=lambda max_depth: torch.nn.Sequential(
            torch.nn.Conv2d(3, max_depth, kernel_size=3, padding=1),
            torch.nn.AdaptiveAvgPool2d((8, 8)),
        ),
        Attn=lambda: torch.nn.Identity(),
        num_dvit_per_pred_blk=1,
        schema_heads={"landmarks_68": 68, "landmarks_98": 98, "profile39": 39},
        auxiliary_heads={"occlusion": 2},
        visibility_heads=True,
    )

    outputs = model(torch.zeros(2, 3, 32, 32))[-1]

    assert outputs["visibility_68"].shape == (2, 68)
    assert outputs["visibility_98"].shape == (2, 98)
    assert outputs["visibility_profile39"].shape == (2, 39)
    assert outputs["_aux"]["occlusion"].shape == (2, 2)


def test_masked_visibility_bce_returns_zero_for_all_unknown_targets():
    logits = torch.zeros(1, 3, requires_grad=True)
    loss, valid_count = masked_visibility_bce_loss(logits, torch.full((1, 3), -1.0))
    loss.backward()
    assert valid_count == 0
    assert loss.item() == pytest.approx(0.0)
    assert logits.grad.abs().sum().item() == pytest.approx(0.0)


def test_parse_auxiliary_loss_weights_rejects_bad_values():
    assert parse_auxiliary_loss_weights("occlusion=1,blur_quality=0") == {
        "occlusion": 1.0,
        "blur_quality": 0.0,
    }
    with pytest.raises(ValueError):
        parse_auxiliary_loss_weights("unknown=1")
    with pytest.raises(ValueError):
        parse_auxiliary_loss_weights("occlusion=nan")
