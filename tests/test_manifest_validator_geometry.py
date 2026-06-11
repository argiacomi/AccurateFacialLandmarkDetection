import json

import cv2
import numpy as np

from lib.manifest.validator import validate_training_manifest


def test_validator_reports_unreasonable_loader_padding(tmp_path):
    img = tmp_path / "image.png"
    assert cv2.imwrite(str(img), np.zeros((256, 256, 3), dtype=np.uint8))

    pts = np.stack(
        [np.linspace(75, 280, 106), np.linspace(344, 1163, 106)],
        axis=1,
    ).astype(np.float32)
    lmk = tmp_path / "bad.npy"
    np.save(lmk, pts)

    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "manifest_contract": "landmark_training_manifest",
                "version": 1,
                "samples": [
                    {
                        "sample_id": "bad",
                        "dataset": "jd-landmark",
                        "split": "train",
                        "image": str(img),
                        "landmarks": str(lmk),
                        "source_schema": "2d_106",
                        "target_schema": "2d_106",
                        "landmark_count": 106,
                        "head_name": "landmarks_2d_106",
                        "split_safe_id": "bad",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    report = validate_training_manifest(
        manifest,
        allow_legacy_missing_contract_fields=True,
    )

    assert report["geometry"]["checked_samples"] == 1
    assert report["geometry"]["unreasonable_loader_padding"] == 1
    assert report["invalid_samples"] == 1
    assert report["ok"] is False


def test_validator_falls_back_when_prepared_crop_is_missing(tmp_path):
    img = tmp_path / "image.png"
    assert cv2.imwrite(str(img), np.zeros((256, 256, 3), dtype=np.uint8))

    pts = np.stack(
        [np.linspace(75, 280, 106), np.linspace(344, 1163, 106)],
        axis=1,
    ).astype(np.float32)
    lmk = tmp_path / "bad.npy"
    np.save(lmk, pts)

    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "manifest_contract": "landmark_training_manifest",
                "version": 1,
                "samples": [
                    {
                        "sample_id": "bad",
                        "dataset": "jd-landmark",
                        "split": "train",
                        "image": str(img),
                        "prepared_image": str(tmp_path / "missing_prepared.png"),
                        "prepared_image_orig_hw": [4000, 4000],
                        "landmarks": str(lmk),
                        "source_schema": "2d_106",
                        "target_schema": "2d_106",
                        "landmark_count": 106,
                        "head_name": "landmarks_2d_106",
                        "split_safe_id": "bad",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    report = validate_training_manifest(
        manifest,
        allow_legacy_missing_contract_fields=True,
    )

    assert report["geometry"]["checked_samples"] == 1
    assert report["geometry"]["unreasonable_loader_padding"] == 1
    assert report["invalid_samples"] == 1
