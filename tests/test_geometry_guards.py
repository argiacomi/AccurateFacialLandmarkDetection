"""Coordinate-frame guards: suspicious threshold, overlays, and builder fixes.

Covers the stricter-than-crash-guard geometry policy: samples whose loader
padding exceeds ``SUSPICIOUS_LOADER_PADDING`` are quarantined with review
overlays instead of training on a likely-wrong coordinate frame.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from lib.datasets.build import core as build_core
from lib.datasets.build import video as build_video
from lib.datasets.loader_geometry import (
    SUSPICIOUS_LOADER_PADDING,
    simulate_loader_geometry,
    write_geometry_overlay,
)
from lib.manifest.validator import validate_training_manifest
from tools import build_quality_dataset as builder
from tools.stage_prepared_crops import stage_crops


def _points(count: int, low: float = 16.0, high: float = 220.0) -> np.ndarray:
    return np.stack(
        [np.linspace(low, high, count), np.linspace(low + 8, high + 4, count)],
        axis=1,
    ).astype(np.float32)


def _write_image(path: Path, *, size: tuple[int, int] = (256, 256)) -> Path:
    image = np.full((size[1], size[0], 3), 127, dtype=np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    assert cv2.imwrite(str(path), image)
    return path


def _write_counted_txt(path: Path, points: np.ndarray) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"{points.shape[0]}\n" + "\n".join(f"{x} {y}" for x, y in points) + "\n",
        encoding="utf-8",
    )
    return path


# ---------------------------------------------------------------------------
# Suspicious-threshold classification and overlay writer.
# ---------------------------------------------------------------------------


def test_simulate_loader_geometry_flags_suspicious_but_trainable_overflow():
    points = _points(68)
    in_frame = simulate_loader_geometry(points, (256, 256))
    assert in_frame["ok"] and not in_frame["suspicious"]

    # Overflow beyond the suspicious threshold but far below the 2048 guard.
    shifted = points.copy()
    shifted[:, 1] += 150.0
    suspicious = simulate_loader_geometry(shifted, (256, 256))
    assert suspicious["ok"] is True
    assert suspicious["suspicious"] is True
    assert suspicious["reason"] == "suspicious_loader_padding"
    assert suspicious["padding"] > SUSPICIOUS_LOADER_PADDING

    # Mild overflow (a chin slightly past the crop) stays unflagged.
    mild = points.copy()
    mild[:, 1] += 20.0
    benign = simulate_loader_geometry(mild, (256, 256))
    assert benign["ok"] is True and benign["suspicious"] is False


def test_write_geometry_overlay_writes_review_png(tmp_path):
    image = _write_image(tmp_path / "img.png", size=(128, 96))
    points = _points(68)
    points[:, 1] += 200.0
    diag = simulate_loader_geometry(points, (96, 128))
    out = write_geometry_overlay(
        tmp_path / "review" / "sample.png", image, points, (96, 128), diag=diag
    )
    assert out is not None and out.is_file()
    canvas = cv2.imread(str(out))
    assert canvas is not None
    # Canvas is the padded frame: strictly larger than the 256 crop.
    assert canvas.shape[0] > 256 and canvas.shape[1] > 256


# ---------------------------------------------------------------------------
# Validator: quarantine counting + overlays, loader-parity masks.
# ---------------------------------------------------------------------------


def _helen_source(tmp_path, entry):
    """Create a tiny HELEN annotations.json source plus matching 300W cache root."""
    source = tmp_path / "helen_source"
    cache = tmp_path / "300w_cache"
    output = tmp_path / "out"

    source.mkdir(parents=True, exist_ok=True)
    (source / "annotations.json").write_text(
        json.dumps([entry]),
        encoding="utf-8",
    )

    return source, cache, output


def _manifest_sample(tmp_path: Path, points: np.ndarray, **extra) -> dict:
    image = _write_image(tmp_path / "images" / f"{extra.get('sample_id', 's')}.png")
    lmk_path = tmp_path / "landmarks" / f"{extra.get('sample_id', 's')}.npy"
    lmk_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(lmk_path, points)
    sample = {
        "sample_id": "s",
        "dataset": "unittest",
        "split": "train",
        "image": str(image),
        "landmarks": str(lmk_path),
        "source_schema": "2d_68",
        "target_schema": "2d_68",
        "landmark_count": 68,
        "head_name": "landmarks_68",
        "split_safe_id": "s",
        "image_id": extra.get("sample_id", "s"),
        "metadata": {},
    }
    sample.update(extra)
    return sample


def test_validator_quarantines_suspicious_geometry_with_overlay(tmp_path):
    bad = _points(68)
    bad[:, 1] += 150.0
    manifest = {
        "samples": [
            _manifest_sample(tmp_path, _points(68), sample_id="good"),
            _manifest_sample(tmp_path, bad, sample_id="bad"),
        ]
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    overlay_dir = tmp_path / "overlays"

    report = validate_training_manifest(
        manifest_path,
        allow_legacy_missing_contract_fields=True,
        allow_missing_projection_audit=True,
        allow_suspicious_geometry=True,
        geometry_overlay_dir=overlay_dir,
    )

    assert report["geometry"]["checked_samples"] == 2
    assert report["geometry"]["suspicious_loader_padding"] == 1
    assert report["geometry"]["unreasonable_loader_padding"] == 0
    # Explicit allow mode: suspicious geometry is reported and overlaid, but not invalid.
    assert report["invalid_samples"] == 0
    assert report["geometry"]["overlays_written"] == 1
    written = list(overlay_dir.rglob("*.png"))
    assert len(written) == 1 and "bad" in written[0].name


def test_validator_uses_loader_parity_landmark_mask(tmp_path):
    # MERL-RAV style: self-occluded points zeroed by the builder and masked
    # out via landmark_coordinate_valid_mask. The validator must not report
    # the sentinel (0, 0) coordinates as out-of-frame landmarks.
    points = _points(68, low=120.0, high=200.0)
    points[:5] = 0.0
    mask = [False] * 5 + [True] * 63
    sample = _manifest_sample(tmp_path, points, sample_id="masked")
    sample["metadata"]["landmark_coordinate_valid_mask"] = mask

    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps({"samples": [sample]}), encoding="utf-8")

    report = validate_training_manifest(
        manifest_path,
        allow_legacy_missing_contract_fields=True,
        allow_missing_projection_audit=True,
    )

    assert report["geometry"]["checked_samples"] == 1
    assert report["geometry"]["landmarks_outside_image"] == 0
    assert report["invalid_samples"] == 0


def test_validator_flags_normalized_landmarks_on_non_256_source(tmp_path):
    normalized = (_points(68) / 256.0).astype(np.float32)
    sample = _manifest_sample(tmp_path, normalized, sample_id="norm")
    image = _write_image(tmp_path / "images" / "norm_big.png", size=(512, 512))
    sample["image"] = str(image)

    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps({"samples": [sample]}), encoding="utf-8")

    report = validate_training_manifest(
        manifest_path,
        allow_legacy_missing_contract_fields=True,
        allow_missing_projection_audit=True,
    )

    assert report["geometry"]["normalized_landmarks_non_256_source"] == 1


# ---------------------------------------------------------------------------
# stage-crops: suspicious samples kept, listed, and given overlays.
# ---------------------------------------------------------------------------


def test_stage_crops_keeps_suspicious_with_overlay_by_default(tmp_path):
    image = _write_image(tmp_path / "native.png", size=(512, 512))
    good_lmk = tmp_path / "good.npy"
    np.save(good_lmk, _points(68) * 2.0)  # in the 512 frame
    bad_lmk = tmp_path / "bad.npy"
    bad = _points(68) * 2.0
    bad[:, 1] += 400.0  # ~200px overflow after the 256 rescale
    np.save(bad_lmk, bad)

    manifest = {
        "samples": [
            {
                "sample_id": "good",
                "dataset": "unittest",
                "image": str(image),
                "landmarks": str(good_lmk),
            },
            {
                "sample_id": "bad",
                "dataset": "unittest",
                "image": str(image),
                "landmarks": str(bad_lmk),
            },
        ]
    }
    manifest_path = tmp_path / "manifest.json"
    staged_path = tmp_path / "staged.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    stats = stage_crops(
        manifest_path,
        out_manifest=staged_path,
        validate_geometry=True,
        drop_invalid_geometry=True,
    )

    assert stats["geometry_dropped"] == 0
    assert len(stats["suspicious_geometry"]) == 1
    assert stats["suspicious_geometry"][0]["sample_id"] == "bad"

    staged = json.loads(staged_path.read_text(encoding="utf-8"))
    assert [sample["sample_id"] for sample in staged["samples"]] == ["good", "bad"]

    overlay_dir = tmp_path / "geometry_review"
    overlays = list(overlay_dir.rglob("*.png")) + list(overlay_dir.rglob("*.jpg"))
    assert overlays


def test_stage_crops_drops_suspicious_when_explicitly_requested(tmp_path):
    image = _write_image(tmp_path / "native.png", size=(512, 512))
    good_lmk = tmp_path / "good.npy"
    np.save(good_lmk, _points(68) * 2.0)  # in the 512 frame
    bad_lmk = tmp_path / "bad.npy"
    bad = _points(68) * 2.0
    bad[:, 1] += 400.0  # ~200px overflow after the 256 rescale
    np.save(bad_lmk, bad)

    manifest = {
        "samples": [
            {
                "sample_id": "good",
                "dataset": "unittest",
                "image": str(image),
                "landmarks": str(good_lmk),
            },
            {
                "sample_id": "bad",
                "dataset": "unittest",
                "image": str(image),
                "landmarks": str(bad_lmk),
            },
        ]
    }
    manifest_path = tmp_path / "manifest.json"
    staged_path = tmp_path / "staged.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    stats = stage_crops(
        manifest_path,
        out_manifest=staged_path,
        validate_geometry=True,
        drop_invalid_geometry=True,
        drop_suspicious_geometry=True,
    )

    assert stats["geometry_dropped"] == 1
    assert len(stats["suspicious_geometry"]) == 1
    assert stats["suspicious_geometry"][0]["sample_id"] == "bad"

    staged = json.loads(staged_path.read_text(encoding="utf-8"))
    assert [sample["sample_id"] for sample in staged["samples"]] == ["good"]

    overlay_dir = tmp_path / "geometry_review"
    overlays = list(overlay_dir.rglob("*.png")) + list(overlay_dir.rglob("*.jpg"))
    assert overlays


def test_helen_rescales_landmarks_from_declared_dims(tmp_path):
    # Annotated on a 512x512 original; resolved cache image is 256x256.
    points = (_points(194) * 2.0).astype(float)
    entry = {
        "image": "12345_1.jpg",
        "landmarks": points.tolist(),
        "width": 512,
        "height": 512,
    }
    source, cache, output = _helen_source(tmp_path, entry)
    _write_image(cache / "helen" / "12345_1.jpg", size=(256, 256))

    manifest_path = builder.build(
        builder._parser().parse_args(
            [
                "--dataset",
                "helen",
                "--source-dir",
                str(source),
                "--image-root",
                str(cache),
                "--output-dir",
                str(output),
            ]
        )
    )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    sample = manifest["samples"][0]
    saved = np.load(output / sample["landmarks"])
    assert sample["metadata"]["landmarks_rescaled_from_declared_dims"] is True
    assert sample["metadata"]["landmarks_rescale_factors"] == [0.5, 0.5]
    assert np.allclose(saved, points * 0.5, atol=1e-4)


def test_helen_quarantines_wrong_frame_annotation(tmp_path):
    # No declared dims; landmarks extend ~256px past the 256 cache image.
    points = (_points(194) + 250.0).astype(float)
    entry = {"image": "99999_1.jpg", "landmarks": points.tolist()}
    source, cache, output = _helen_source(tmp_path, entry)
    _write_image(cache / "helen" / "99999_1.jpg", size=(256, 256))

    with pytest.raises(ValueError, match="no HELEN annotation samples built"):
        builder.build(
            builder._parser().parse_args(
                [
                    "--dataset",
                    "helen",
                    "--source-dir",
                    str(source),
                    "--image-root",
                    str(cache),
                    "--output-dir",
                    str(output),
                ]
            )
        )
    overlays = list((output / "geometry_review" / "helen").rglob("*.png"))
    assert len(overlays) == 1


# ---------------------------------------------------------------------------
# JD: suspicious-but-trainable native geometry must not pass path A.
# ---------------------------------------------------------------------------


def test_jd_quarantines_suspicious_native_geometry_without_bbox(tmp_path):
    source = tmp_path / "source"
    output = tmp_path / "out"
    name = "AFW_55555_1_0.jpg"
    _write_image(source / "Training_data" / "AFW" / "picture" / name, size=(256, 256))
    # Wrong-frame style: y far beyond the 256 image but under the 2048 guard.
    points = _points(106)
    points[:, 1] += 300.0
    _write_counted_txt(
        source / "Training_data" / "AFW" / "landmark" / f"{name}.txt", points
    )

    with pytest.raises(ValueError, match="no JD-landmark native release samples"):
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
    overlays = list((output / "geometry_review").rglob("*.png"))
    assert len(overlays) == 1


# ---------------------------------------------------------------------------
# 300VW-style frame numbering base detection.
# ---------------------------------------------------------------------------


def test_frame_landmark_index_commits_to_one_based_directories(tmp_path):
    annot = tmp_path / "001" / "annot"
    annot.mkdir(parents=True)
    points = _points(68)
    for number in (1, 2, 3):
        path = annot / f"{number:06d}.pts"
        path.write_text(
            "version: 1\nn_points: 68\n{\n"
            + "\n".join(f"{x} {y}" for x, y in points)
            + "\n}\n",
            encoding="utf-8",
        )

    index = build_video._build_frame_landmark_index(tmp_path)
    video_id = "001/vid"
    # 1-based release: extracted zero-based frame i maps to file i+1.
    assert index[(video_id, 0)].name == "000001.pts"
    assert index[(video_id, 1)].name == "000002.pts"
    assert index[(video_id, 2)].name == "000003.pts"
    assert (video_id, 3) not in index


def test_frame_landmark_index_keeps_zero_based_directories_identity(tmp_path):
    annot = tmp_path / "002" / "annot"
    annot.mkdir(parents=True)
    points = _points(68)
    for number in (0, 1, 2):
        path = annot / f"{number:06d}.pts"
        path.write_text(
            "version: 1\nn_points: 68\n{\n"
            + "\n".join(f"{x} {y}" for x, y in points)
            + "\n}\n",
            encoding="utf-8",
        )

    index = build_video._build_frame_landmark_index(tmp_path)
    video_id = "002/vid"
    assert index[(video_id, 0)].name == "000000.pts"
    assert index[(video_id, 1)].name == "000001.pts"
    assert index[(video_id, 2)].name == "000002.pts"


# ---------------------------------------------------------------------------
# Ambiguous stem-index image resolution must raise, not silently pick.
# ---------------------------------------------------------------------------


def test_find_named_image_raises_on_ambiguous_index_matches(tmp_path):
    first = _write_image(tmp_path / "a" / "face.png")
    second = _write_image(tmp_path / "b" / "face.png")
    index = {"face": [first, second]}

    with pytest.raises(ValueError, match="ambiguous image match"):
        build_core._find_named_image((), "face", image_index=index)

    # A single match (or duplicates of one file) still resolves.
    assert (
        build_core._find_named_image((), "face", image_index={"face": [first]}) == first
    )


def test_matching_image_raises_on_ambiguous_index_matches(tmp_path):
    first = _write_image(tmp_path / "a" / "sample.png")
    second = _write_image(tmp_path / "b" / "sample.png")
    landmarks = tmp_path / "annotations" / "sample.pts"
    landmarks.parent.mkdir(parents=True)
    landmarks.write_text("version: 1\nn_points: 2\n{\n1 1\n2 2\n}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="ambiguous image match"):
        build_core._matching_image(landmarks, image_index={"sample": [first, second]})


# ---------------------------------------------------------------------------
# cofw29: annotation rows beyond the MAT image list are skipped.
# ---------------------------------------------------------------------------


def test_cofw29_skips_annotation_rows_without_images(tmp_path):
    scipy = pytest.importorskip("scipy.io")
    source = tmp_path / "source"
    source.mkdir()
    output = tmp_path / "out"
    # Two annotation rows, one image: the overflow row must be skipped instead
    # of being paired with a stem-matched (wrong) image.
    points = np.concatenate(
        [_points(29).reshape(1, 58), (_points(29) + 4).reshape(1, 58)], axis=0
    )
    images = np.full((1, 256, 256, 3), 127, dtype=np.uint8)
    scipy.savemat(
        source / "COFW_train_color.mat",
        {"phisTr": points, "IsTr": images},
    )

    manifest_path = builder.build(
        builder._parser().parse_args(
            [
                "--dataset",
                "cofw29",
                "--source-dir",
                str(source),
                "--output-dir",
                str(output),
            ]
        )
    )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert len(manifest["samples"]) == 1
    audit = json.loads(
        (manifest_path.parent / "dataset_audit.json").read_text(encoding="utf-8")
    )
    assert any(
        "no corresponding image" in skip["reason"]
        for skip in audit.get("skipped_examples", [])
    )


# ---------------------------------------------------------------------------
# Crop remap: points stay aligned with the actually-assembled crop.
# ---------------------------------------------------------------------------


def test_crop_remap_uses_actual_assembled_frame(tmp_path):
    # Gradient image so pixel content is position-dependent.
    image = np.zeros((200, 200, 3), dtype=np.uint8)
    image[:, :, 0] = np.tile(np.arange(200, dtype=np.uint8), (200, 1))
    image_path = tmp_path / "img.png"
    assert cv2.imwrite(str(image_path), image)

    points = np.asarray([[60.0, 60.0], [140.0, 140.0]], dtype=np.float32)
    crop, remapped, crop_bbox = build_core._crop_image_and_remap_points(
        image_path, points, [60.0, 60.0, 140.0, 140.0], pad_ratio=0.25
    )

    left, top, right, bottom = crop_bbox
    # The returned bbox is the actual assembled frame: integer-aligned and
    # consistent with the crop's pixel dimensions.
    assert float(right - left) == pytest.approx(crop.shape[1] / 256 * (right - left))
    scale_x = 256.0 / (right - left)
    scale_y = 256.0 / (bottom - top)
    expected = (points - np.asarray([[left, top]], dtype=np.float32)) * np.asarray(
        [[scale_x, scale_y]], dtype=np.float32
    )
    assert np.allclose(remapped, expected, atol=1e-4)
    # Remapped points stay inside the 256 crop for an in-image bbox.
    assert remapped.min() >= 0.0 and remapped.max() <= 256.0
