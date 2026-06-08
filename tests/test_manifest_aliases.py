import json
import sys
import types

import cv2
import numpy as np
import pytest

augmentation_stub = types.ModuleType("ImageAugmentation")
augmentation_stub.GetAugTransform = lambda: None
sys.modules.setdefault("ImageAugmentation", augmentation_stub)

from DatasetAll import GetDataset, IsSchemaAwareManifestDataset
from DatasetMultiSchemaLandmarkManifest import LandmarkDataset as MultiSchemaDataset
from DatasetFS68Manifest import LandmarkDataset as LegacyFS68Dataset


MANIFEST_ALIASES = [
    "FS68Manifest",
    "LandmarkManifest",
    "SchemaAwareManifest",
    "MultiSchemaLandmarkManifest",
]


@pytest.fixture()
def manifest_68(tmp_path):
    image = np.full((256, 256, 3), 127, dtype=np.uint8)
    image_path = tmp_path / "image.jpg"
    landmarks_path = tmp_path / "landmarks.npy"

    cv2.imwrite(str(image_path), image)

    points = np.stack(
        [np.linspace(32, 224, 68), np.linspace(40, 216, 68)],
        axis=1,
    ).astype(np.float32)
    np.save(landmarks_path, points)

    manifest = {
        "samples": [
            {
                "sample_id": "sample-68",
                "dataset": "unit",
                "split": "train",
                "image": str(image_path),
                "landmarks": str(landmarks_path),
                "source_schema": "2d_68",
                "metadata": {
                    "source_schema": "2d_68",
                    "hard_negative_bucket": "anchor",
                },
            }
        ]
    }

    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest_path


@pytest.mark.parametrize("name", MANIFEST_ALIASES)
def test_schema_aware_manifest_aliases_route_through_getdataset(name, manifest_68):
    assert IsSchemaAwareManifestDataset(name)

    dataset = GetDataset(
        name=name,
        data_root="",
        split="train",
        preload=False,
        aug=False,
        heatmap_size=8,
        manifest_path=str(manifest_68),
        schema_aware_training=True,
    )

    assert isinstance(dataset, LegacyFS68Dataset)
    assert len(dataset) == 1

    item = dataset[0]
    assert item["schema"] == "2d_68"
    assert item["head_name"] == "landmarks_68"
    assert item["target"].shape == (68, 2)
    assert item["heatmap"].shape == (68, 8, 8)


def test_canonical_module_is_backward_compatible_wrapper():
    assert MultiSchemaDataset is LegacyFS68Dataset


def test_unknown_dataset_name_still_fails(manifest_68):
    with pytest.raises(ValueError, match="unknown dataset name"):
        GetDataset(
            name="NotARealDataset",
            data_root="",
            split="train",
            preload=False,
            aug=False,
            manifest_path=str(manifest_68),
        )
