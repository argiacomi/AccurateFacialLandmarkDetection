from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

import lib.datasets.build.jd_landmark as jd_mod
import lib.manifest.validator as validator
import tools.build_quality_dataset as builder
import tools.stage_prepared_crops as stage_mod


def _write_image(path: Path, *, size: tuple[int, int] = (256, 256)) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.full((size[1], size[0], 3), 127, dtype=np.uint8)
    assert cv2.imwrite(str(path), image)
    return path


def _points(count: int) -> np.ndarray:
    return np.stack(
        [np.linspace(32, 180, count), np.linspace(40, 190, count)],
        axis=1,
    ).astype(np.float32)


def _write_counted_txt(path: Path, points: np.ndarray) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"{points.shape[0]}\n"
        + "\n".join(f"{float(x)} {float(y)}" for x, y in points)
        + "\n",
        encoding="utf-8",
    )
    return path


def test_stage_crops_drop_invalid_geometry_removes_suspicious_samples(
    tmp_path, monkeypatch
):
    image = _write_image(tmp_path / "image.jpg")
    landmarks = tmp_path / "landmarks.npy"
    np.save(landmarks, _points(106))

    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "version": 1,
                "samples": [
                    {
                        "sample_id": "suspicious",
                        "dataset": "jd-landmark",
                        "image": str(image),
                        "landmarks": str(landmarks),
                        "source_schema": "2d_106",
                        "target_schema": "2d_106",
                        "landmark_count": 106,
                        "head_name": "landmarks_106",
                        "normalizer": 100.0,
                        "split": "train",
                        "split_safe_id": "suspicious",
                        "condition": "jd_landmark",
                        "conditions": ["jd_landmark", "trainset"],
                        "source": {"dataset": "jd-landmark", "source_id": "suspicious"},
                        "metadata": {},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        stage_mod,
        "_sample_loader_geometry",
        lambda entry, *, base_dir: {
            "ok": True,
            "suspicious": True,
            "reason": "suspicious_loader_padding",
            "padding": 96.0,
            "source_image_hw": [256, 256],
        },
    )

    stats = stage_mod.stage_crops(
        manifest,
        out_manifest=manifest,
        validate_geometry=True,
        drop_invalid_geometry=True,
        workers=1,
        max_geometry_overlays=0,
    )

    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert payload["samples"] == []
    assert stats["geometry_dropped"] == 1
    assert len(stats["suspicious_geometry"]) == 1


def test_validator_fails_suspicious_geometry_by_default(tmp_path, monkeypatch):
    image = _write_image(tmp_path / "image.jpg")
    landmarks = tmp_path / "landmarks.npy"
    np.save(landmarks, _points(106))

    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "version": validator.TRAINING_MANIFEST_VERSION,
                "manifest_contract": validator.TRAINING_MANIFEST_CONTRACT,
                "samples": [
                    {
                        "sample_id": "suspicious",
                        "dataset": "jd-landmark",
                        "image": str(image),
                        "landmarks": str(landmarks),
                        "source_schema": "2d_106",
                        "target_schema": "2d_106",
                        "landmark_count": 106,
                        "head_name": "landmarks_106",
                        "normalizer": 100.0,
                        "split": "train",
                        "split_safe_id": "suspicious",
                        "condition": "jd_landmark",
                        "conditions": ["jd_landmark", "trainset"],
                        "source": {"dataset": "jd-landmark", "source_id": "suspicious"},
                        "metadata": {},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        validator,
        "resolve_loader_source_hw",
        lambda sample, *, base_dir: ((256, 256), "image", None),
    )
    monkeypatch.setattr(
        validator,
        "simulate_loader_geometry",
        lambda points, hw, *, landmark_mask=None: {
            "ok": True,
            "suspicious": True,
            "reason": "suspicious_loader_padding",
            "padding": 96.0,
            "source_image_hw": [256, 256],
        },
    )

    report = validator.validate_training_manifest(manifest)

    assert not report["ok"]
    assert report["invalid_samples"] == 1
    assert report["geometry"]["suspicious_loader_padding"] == 1
    assert "suspicious_loader_padding" in report["examples"]["invalid"][0]["errors"]


def test_jd_corrected_suspicious_geometry_does_not_use_bbox_fallback(
    tmp_path, monkeypatch
):
    source = tmp_path / "source"
    output = tmp_path / "out"
    name = "AFW_134212_1_0.jpg"

    _write_image(source / "Training_data" / "AFW" / "picture" / name)
    _write_counted_txt(
        source / "Training_data" / "AFW" / "landmark" / f"{name}.txt",
        _points(106),
    )
    _write_counted_txt(source / "Corrected_landmark" / f"{name}.txt", _points(106))

    bbox_dir = (
        source
        / "training_dataset_face_detection_bounding_box_v1"
        / "training_dataset_face_detection_bounding_box"
    )
    bbox_dir.mkdir(parents=True)
    (bbox_dir / f"{name}.rect").write_text("10 20 210 220\n", encoding="utf-8")

    monkeypatch.setattr(
        jd_mod,
        "_simulate_loader_geometry",
        lambda image_path, points, *, landmark_mask=None: {
            "ok": True,
            "suspicious": True,
            "reason": "suspicious_loader_padding",
            "padding": 96.0,
            "source_image_hw": [256, 256],
        },
    )

    crop_called = False

    def fake_crop(*args, **kwargs):
        nonlocal crop_called
        crop_called = True
        raise AssertionError("corrected suspicious geometry must not use bbox fallback")

    monkeypatch.setattr(jd_mod, "_crop_sample_image", fake_crop)

    with pytest.raises(ValueError, match="no JD-landmark native release samples built"):
        builder.build(
            builder._parser().parse_args(
                [
                    "--dataset",
                    "jd-landmark",
                    "--source-dir",
                    str(source),
                    "--output-dir",
                    str(output),
                ]
            )
        )

    assert crop_called is False
