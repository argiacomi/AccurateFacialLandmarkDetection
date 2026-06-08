import json
import sys
import types
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import torch

augmentation_stub = types.ModuleType("ImageAugmentation")
augmentation_stub.GetAugTransform = lambda: None
sys.modules.setdefault("ImageAugmentation", augmentation_stub)

from DatasetFS68Manifest import LandmarkDataset
from lib.landmarks.training.data import _schema_aware_collate
from lib.landmarks.training.evaluator import _eval_collate, _evaluate_landmark_model
from lib.landmarks.core.schema import DEFAULT_SCHEMA_HEADS, flip_map_for_schema, head_name_for_schema
from tools.landmarks.evaluate_cdvit_manifest import _dataset as standalone_eval_dataset


def _write_manifest(tmp_path, points, *, schema, extra_sample=None):
    image = np.full((256, 256, 3), 127, dtype=np.uint8)
    image_path = tmp_path / "image.jpg"
    landmark_path = tmp_path / "points.npy"
    cv2.imwrite(str(image_path), image)
    np.save(landmark_path, points.astype(np.float32))
    sample = {
        "sample_id": "sample",
        "dataset": "multipie",
        "split": "train",
        "image": str(image_path),
        "landmarks": str(landmark_path),
        "source_schema": schema,
        "metadata": {"source_schema": schema, "hard_negative_bucket": "profile"},
    }
    if extra_sample:
        sample.update(extra_sample)
    manifest = {
        "samples": [
            sample
        ]
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest_path


def test_schema_registry_has_required_training_heads_and_flip_maps():
    assert head_name_for_schema("2d_68") == "landmarks_68"
    assert head_name_for_schema("2d_98") == "landmarks_98"
    assert head_name_for_schema("2d_106") == "landmarks_106"
    assert head_name_for_schema("2d_194") == "landmarks_194"
    assert head_name_for_schema("2d_29") == "landmarks_29"
    assert head_name_for_schema("multipie_profile_39") == "profile39"
    assert DEFAULT_SCHEMA_HEADS["landmarks_106"] == 106
    assert DEFAULT_SCHEMA_HEADS["landmarks_194"] == 194
    assert DEFAULT_SCHEMA_HEADS["landmarks_29"] == 29
    assert flip_map_for_schema("2d_68").shape == (68,)
    assert flip_map_for_schema("2d_98").shape == (98,)
    assert flip_map_for_schema("multipie_profile_39").shape == (39,)


def test_fs68_schema_aware_loader_accepts_profile39(tmp_path):
    points = np.stack([np.linspace(32, 224, 39), np.linspace(40, 216, 39)], axis=1)
    manifest_path = _write_manifest(tmp_path, points, schema="multipie_profile_39")

    dataset = LandmarkDataset(
        manifest_path=str(manifest_path),
        split="train",
        preload=False,
        aug=False,
        heatmap_size=8,
        schema_aware_training=True,
    )

    item = dataset[0]
    assert item["head_name"] == "profile39"
    assert item["target"].shape == (39, 2)
    assert item["heatmap"].shape == (39, 8, 8)
    assert item["landmark_mask"].shape == (39,)


def test_fs68_legacy_loader_skips_profile39(tmp_path):
    points = np.stack([np.linspace(32, 224, 39), np.linspace(40, 216, 39)], axis=1)
    manifest_path = _write_manifest(tmp_path, points, schema="multipie_profile_39")

    try:
        LandmarkDataset(
            manifest_path=str(manifest_path),
            split="train",
            preload=False,
            aug=False,
            heatmap_size=8,
            schema_aware_training=False,
        )
    except ValueError as err:
        assert "no trainable schema-aware samples found" in str(err)
    else:
        raise AssertionError("legacy loader should skip profile39 samples")


def test_schema_aware_collate_extracts_optional_auxiliary_labels():
    item = {
        "image": torch.zeros(3, 256, 256),
        "target": torch.zeros(39, 2),
        "heatmap": torch.zeros(39, 8, 8),
        "landmark_mask": torch.ones(39),
        "sample_weight": torch.tensor(3.0),
        "schema": "multipie_profile_39",
        "head_name": "profile39",
        "metadata": {
            "dataset": "multipie",
            "condition": "profile",
            "conditions": ["profile", "left"],
            "source_schema": "multipie_profile_39",
            "hard_negative_bucket": "profile",
            "attributes": {"occlusion": 1, "blur": 0},
        },
    }

    batch = _schema_aware_collate([item])

    assert batch["aux_labels"]["occlusion"].tolist() == [1]
    assert batch["aux_labels"]["blur_quality"].tolist() == [0]
    assert batch["aux_labels"]["profile_side"].tolist() == [1]
    assert batch["aux_labels"]["illumination_quality"].tolist() == [-1]


def test_fs68_eval_metadata_preserves_optional_visibility_targets(tmp_path):
    points = np.stack([np.linspace(32, 224, 68), np.linspace(40, 216, 68)], axis=1)
    visibility = [1, 0, "visible", "occluded", None, "unknown", *([1] * 62)]
    manifest_path = _write_manifest(
        tmp_path,
        points,
        schema="2d_68",
        extra_sample={"visibility": visibility},
    )

    dataset = LandmarkDataset(
        manifest_path=str(manifest_path),
        split="train",
        preload=False,
        aug=False,
        heatmap_size=0,
        include_metadata=True,
    )

    _, _, _, metadata = dataset[0]
    assert metadata["visibility_target"][:6] == [1, 0, 1, 0, -1, -1]
    assert metadata["visibility_target_source"] == "entry.visibility"


def test_standalone_eval_dataset_keeps_native_schema_when_enabled(tmp_path):
    points = np.stack([np.linspace(32, 224, 98), np.linspace(40, 216, 98)], axis=1)
    manifest_path = _write_manifest(
        tmp_path,
        points,
        schema="2d_98",
        extra_sample={"split": "test", "visibility_target": [1, 0, *([-1] * 96)]},
    )
    args = SimpleNamespace(
        manifest=str(manifest_path),
        preload=0,
        eval_mode="random_hash",
        heldout_dataset=[],
        split_policy="declared",
        schema_aware_eval=True,
    )

    dataset = standalone_eval_dataset(args, "test")

    _, target, _, metadata = dataset[0]
    assert target.shape == (98, 2)
    assert metadata["head_name"] == "landmarks_98"
    assert metadata["visibility_target"][:2] == [1, 0]


def test_eval_collate_routes_mixed_schema_samples_to_native_heads():
    points_68 = torch.stack((torch.linspace(0.1, 0.9, 68), torch.linspace(0.2, 0.8, 68)), dim=1)
    points_98 = torch.stack((torch.linspace(0.1, 0.9, 98), torch.linspace(0.2, 0.8, 98)), dim=1)
    batch = _eval_collate(
        [
            (
                torch.zeros(3, 256, 256),
                points_68,
                torch.ones(68),
                {
                    "sample_id": "sample-68",
                    "source_schema": "2d_68",
                    "head_name": "landmarks_68",
                    "visibility_target": [1, 0, *([-1] * 66)],
                    "visibility_target_source": "entry.visibility_target",
                },
            ),
            (
                torch.zeros(3, 256, 256),
                points_98,
                torch.ones(98),
                {
                    "sample_id": "sample-98",
                    "source_schema": "2d_98",
                    "head_name": "landmarks_98",
                    "visibility_target": [1, 1, 0, *([-1] * 95)],
                    "visibility_target_source": "metadata.landmark_visibility",
                },
            ),
        ]
    )

    class NativeHeadModel(torch.nn.Module):
        def forward(self, data):
            pred_68 = torch.zeros(data.shape[0], 68, 2)
            pred_98 = torch.zeros(data.shape[0], 98, 2)
            pred_68[0] = points_68
            pred_98[1] = points_98
            return [
                {
                    "landmarks_68": (pred_68, torch.zeros(data.shape[0], 68, 8, 8)),
                    "landmarks_98": (pred_98, torch.zeros(data.shape[0], 98, 8, 8)),
                }
            ]

    report = _evaluate_landmark_model(NativeHeadModel(), [batch], torch.device("cpu"), include_records=True)

    assert report["overall"]["sample_count"] == 2
    assert report["overall"]["NME_all"] == pytest.approx(0.0)
    assert report["overall"]["visible_landmark_count"] == 3
    assert report["overall"]["occluded_landmark_count"] == 2
    assert report["by_schema"]["2d_98"]["sample_count"] == 1
    records = {record["sample_id"]: record for record in report["records"]}
    assert records["sample-68"]["evaluation_head"] == "landmarks_68"
    assert records["sample-98"]["evaluation_head"] == "landmarks_98"
    assert records["sample-98"]["visibility_target_source"] == "metadata.landmark_visibility"
