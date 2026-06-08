from __future__ import annotations

import json
import sys
import types
from types import SimpleNamespace

import numpy as np
import pytest
import torch

augmentation_stub = types.ModuleType("ImageAugmentation")
augmentation_stub.GetAugTransform = lambda: None
sys.modules.setdefault("ImageAugmentation", augmentation_stub)

cv2 = pytest.importorskip("cv2")

from lib.landmarks.core.schema import flip_map_for_schema
from lib.landmarks.datasets.manifest import LandmarkDataset
from lib.landmarks.training.losses import schema_head_loss


def _args(**overrides):
    base = dict(
        locw=0.0,
        hw=0.0,
        schema_consistency_weight=0.0,
        auxiliary_loss_weight=0.5,
        auxiliary_loss_weights="",
        schema_head_loss_weighting="sample_count",
        schema_head_loss_weights="",
        star_loss_weight=0.0,
        visibility_loss_weight=0.25,
        visibility_loss_initial_weight=0.0,
        visibility_loss_start_epoch=0,
        visibility_loss_ramp_epochs=0,
        visibility_pseudo_loss_weight=0.0,
        current_epoch=0,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def _stage_and_heads():
    stage_pred = {
        "landmarks_68": (
            torch.zeros(1, 68, 2, requires_grad=True),
            torch.zeros(1, 68, 4, 4, requires_grad=True),
        ),
        "visibility_68": torch.zeros(1, 68, requires_grad=True),
        "_aux": {
            "occlusion": torch.tensor([[0.0, 2.0]], requires_grad=True),
        },
    }
    heads = {
        "landmarks_68": {
            "indices": torch.tensor([0]),
            "target": torch.zeros(1, 68, 2),
            "heatmap": torch.zeros(1, 68, 4, 4),
            "landmark_mask": torch.ones(1, 68),
            "sample_weight": torch.ones(1),
            "visibility_target": torch.tensor([[1.0, 0.0, *([-1.0] * 66)]]),
            "visibility_target_provenance": ["entry.visibility_target"],
        }
    }
    aux_labels = {"occlusion": torch.tensor([1])}
    return stage_pred, heads, aux_labels


def test_schema_head_loss_can_skip_auxiliary_and_visibility_for_nonfinal_stages():
    stage_pred, heads, aux_labels = _stage_and_heads()

    loss, *_rest, details = schema_head_loss(
        stage_pred,
        heads,
        aux_labels,
        lambda *args, **kwargs: torch.tensor(0.0),
        _args(),
        return_details=True,
        include_auxiliary_loss=False,
        include_visibility_loss=False,
    )
    loss.backward()

    assert details["loss_visibility"].item() == pytest.approx(0.0)
    assert details["auxiliary_loss_contributions"] == {}
    assert stage_pred["visibility_68"].grad is None or stage_pred["visibility_68"].grad.abs().sum().item() == pytest.approx(0.0)
    assert stage_pred["_aux"]["occlusion"].grad is None or stage_pred["_aux"]["occlusion"].grad.abs().sum().item() == pytest.approx(0.0)


def test_synthetic_visibility_targets_are_weighted_separately_from_explicit_targets():
    stage_pred, heads, aux_labels = _stage_and_heads()
    heads["landmarks_68"]["visibility_target_provenance"] = ["synthetic_occluder_mask"]

    _, *_rest, details = schema_head_loss(
        stage_pred,
        heads,
        aux_labels,
        lambda *args, **kwargs: torch.tensor(0.0),
        _args(visibility_pseudo_loss_weight=0.0),
        return_details=True,
    )
    assert details["loss_visibility"].item() == pytest.approx(0.0)

    _, *_rest, details = schema_head_loss(
        stage_pred,
        heads,
        aux_labels,
        lambda *args, **kwargs: torch.tensor(0.0),
        _args(visibility_pseudo_loss_weight=1.0),
        return_details=True,
    )
    assert details["loss_visibility"].item() > 0.0


def test_visibility_target_is_reindexed_when_schema_sample_is_flipped(tmp_path, monkeypatch):
    import lib.landmarks.datasets.manifest as manifest_module

    class IdentityAug:
        def __call__(self, *, image, keypoints):
            return {"image": image, "keypoints": keypoints}

    monkeypatch.setattr(manifest_module, "GetAugTransform", lambda: IdentityAug())
    monkeypatch.setattr(np.random, "random", lambda: 0.0)

    image = np.full((256, 256, 3), 127, dtype=np.uint8)
    image_path = tmp_path / "image.jpg"
    cv2.imwrite(str(image_path), image)

    points = np.stack(
        [np.linspace(32, 224, 68), np.linspace(40, 216, 68)],
        axis=1,
    ).astype(np.float32)
    points_path = tmp_path / "points.npy"
    np.save(points_path, points)

    visibility = np.asarray(([1, 0, -1] * 23)[:68], dtype=np.float32)
    manifest = {
        "samples": [
            {
                "sample_id": "sample",
                "dataset": "w300",
                "split": "train",
                "image": str(image_path),
                "landmarks": str(points_path),
                "source_schema": "2d_68",
                "visibility_target": visibility.tolist(),
            }
        ]
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    dataset = LandmarkDataset(
        manifest_path=str(manifest_path),
        split="train",
        preload=False,
        aug=True,
        heatmap_size=8,
        schema_aware_training=True,
    )

    item = dataset[0]
    expected = visibility[flip_map_for_schema("2d_68")]

    assert item["visibility_target"].numpy().tolist() == pytest.approx(expected.tolist())


def test_manifest_occluder_mask_generates_synthetic_visibility_targets(tmp_path):
    image = np.full((256, 256, 3), 127, dtype=np.uint8)
    image_path = tmp_path / "image.jpg"
    cv2.imwrite(str(image_path), image)

    points = np.zeros((68, 2), dtype=np.float32)
    points[:, 0] = np.linspace(20, 220, 68)
    points[:, 1] = np.linspace(20, 220, 68)
    points[0] = [10, 10]
    points[1] = [30, 30]
    points_path = tmp_path / "points.npy"
    np.save(points_path, points)

    mask = np.zeros((256, 256), dtype=np.uint8)
    mask[8:13, 8:13] = 255
    mask_path = tmp_path / "mask.png"
    cv2.imwrite(str(mask_path), mask)

    manifest = {
        "samples": [
            {
                "sample_id": "sample",
                "dataset": "w300",
                "split": "train",
                "image": str(image_path),
                "landmarks": str(points_path),
                "source_schema": "2d_68",
                "synthetic_occluder_mask": str(mask_path),
            }
        ]
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    dataset = LandmarkDataset(
        manifest_path=str(manifest_path),
        split="train",
        preload=False,
        aug=False,
        heatmap_size=8,
        schema_aware_training=True,
    )

    item = dataset[0]

    assert item["visibility_target"][0].item() == 0.0
    assert item["visibility_target"][1].item() == 1.0
    assert item["visibility_target_provenance"] == "synthetic_occluder_mask"
