import json
import sys
import types

import cv2
import numpy as np
import torch

augmentation_stub = types.ModuleType("ImageAugmentation")
augmentation_stub.GetAugTransform = lambda: None
sys.modules.setdefault("ImageAugmentation", augmentation_stub)

from DatasetFS68Manifest import LandmarkDataset
from TrainHeatmapStageFP16 import _schema_aware_collate
from lib.landmarks.core.schema import flip_map_for_schema, head_name_for_schema


def _write_manifest(tmp_path, points, *, schema):
    image = np.full((256, 256, 3), 127, dtype=np.uint8)
    image_path = tmp_path / "image.jpg"
    landmark_path = tmp_path / "points.npy"
    cv2.imwrite(str(image_path), image)
    np.save(landmark_path, points.astype(np.float32))
    manifest = {
        "samples": [
            {
                "sample_id": "sample",
                "dataset": "multipie",
                "split": "train",
                "image": str(image_path),
                "landmarks": str(landmark_path),
                "source_schema": schema,
                "metadata": {"source_schema": schema, "hard_negative_bucket": "profile"},
            }
        ]
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest_path


def test_schema_registry_has_required_training_heads_and_flip_maps():
    assert head_name_for_schema("2d_68") == "landmarks_68"
    assert head_name_for_schema("2d_98") == "landmarks_98"
    assert head_name_for_schema("multipie_profile_39") == "profile39"
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
        assert "no 68-point samples found" in str(err)
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
