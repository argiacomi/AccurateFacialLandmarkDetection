"""Prepared-crop staging must be a byte-for-byte acceleration of native decode."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import cv2
import numpy as np

from lib.datasets.manifest import LandmarkDataset
from tools import prepare_landmark_dataset as prepare
from tools import stage_prepared_crops as staging
from tools.stage_prepared_crops import build_arg_parser, stage_manifest


def _write_sample(src_dir: Path, name: str, h: int, w: int, seed: int):
    rng = np.random.default_rng(seed)
    image = rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)
    image_path = src_dir / f"{name}.png"  # PNG keeps the native decode deterministic
    assert cv2.imwrite(str(image_path), image)
    landmarks = rng.uniform(1.6, min(h, w) - 1.0, size=(68, 2)).astype(np.float32)
    np.save(src_dir / f"{name}.npy", landmarks)
    return image_path


def _manifest(src_dir: Path, names, *, dataset: str = "unit-test") -> dict:
    return {
        "manifest_contract": "schema_aware_landmark_manifest_v1",
        "version": 2,
        "samples": [
            {
                "image": f"src/{name}.png",
                "landmarks": f"src/{name}.npy",
                "dataset": dataset,
                "split": "train",
                "source_schema": "2d_68",
                "target_schema": "2d_68",
                "head_name": "landmarks_68",
            }
            for name in names
        ],
    }


def _dataset(manifest_path: Path) -> LandmarkDataset:
    return LandmarkDataset(
        str(manifest_path),
        split="train",
        preload=False,
        aug=False,
        heatmap_size=64,
        schema_aware_training=True,
        split_policy="declared",
    )


def _stage(manifest_path: Path, out_path: Path):
    args = build_arg_parser().parse_args(
        ["--manifest", str(manifest_path), "--out-manifest", str(out_path), "--strict"]
    )
    return stage_manifest(args)


def test_prepared_crop_matches_native_decode(tmp_path: Path):
    src = tmp_path / "src"
    src.mkdir()
    # Mix of native sizes; none already 256x256 so all require resizing.
    _write_sample(src, "a", 333, 500, seed=1)
    _write_sample(src, "b", 200, 120, seed=2)
    _write_sample(src, "c", 640, 480, seed=3)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(_manifest(src, ["a", "b", "c"])))

    staged_path = tmp_path / "manifest.staged.json"
    assert _stage(manifest_path, staged_path) == 0

    native_ds = _dataset(manifest_path)
    staged_ds = _dataset(staged_path)
    assert len(staged_ds.samples) == 3
    assert all(s.get("prepared_image") for s in staged_ds.samples)

    by_id = {s["sample_id"]: s for s in native_ds.samples}
    for staged in staged_ds.samples:
        native = by_id[staged["sample_id"]]
        img_fast, lmk_fast = staged_ds._load_image_and_landmarks(staged)
        img_native, lmk_native = native_ds._load_image_and_landmarks(native)
        assert np.array_equal(img_fast, img_native)
        assert np.array_equal(lmk_fast, lmk_native)
        assert img_fast.shape == (256, 256, 3)


def test_prepared_crop_falls_back_when_missing(tmp_path: Path):
    src = tmp_path / "src"
    src.mkdir()
    _write_sample(src, "a", 333, 500, seed=7)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(_manifest(src, ["a"])))
    staged_path = tmp_path / "manifest.staged.json"
    assert _stage(manifest_path, staged_path) == 0

    staged_ds = _dataset(staged_path)
    native_ds = _dataset(manifest_path)
    sample = staged_ds.samples[0]
    expected_img, expected_lmk = native_ds._load_image_and_landmarks(
        native_ds.samples[0]
    )

    # Delete the staged crop: the loader must silently fall back to native decode.
    Path(sample["prepared_image"]).unlink()
    img, lmk = staged_ds._load_image_and_landmarks(sample)
    assert np.array_equal(img, expected_img)
    assert np.array_equal(lmk, expected_lmk)


def test_prepared_crop_skips_native_256(tmp_path: Path):
    src = tmp_path / "src"
    src.mkdir()
    _write_sample(src, "already256", 256, 256, seed=9)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(_manifest(src, ["already256"])))
    staged_path = tmp_path / "manifest.staged.json"
    assert _stage(manifest_path, staged_path) == 0

    # A native 256x256 image needs no resize, so no crop is staged for it.
    staged_ds = _dataset(staged_path)
    assert not staged_ds.samples[0].get("prepared_image")


def test_prepare_orchestrator_stage_crops_in_place(tmp_path: Path):
    import types

    from tools import prepare_landmark_dataset as prep

    # --stage-crops is wired into the orchestrator's argument parser.
    args = prep._parser().parse_args(["--datasets", "merl-rav", "--stage-crops"])
    assert args.stage_crops is True
    assert args.stage_crops_subdir == "images"

    src = tmp_path / "src"
    src.mkdir()
    _write_sample(src, "a", 333, 500, seed=11)
    manifest_path = tmp_path / "manifest.json"
    payload = _manifest(src, ["a"])
    manifest_path.write_text(json.dumps(payload))

    ns = types.SimpleNamespace(stage_crops_subdir="images", force_stage_crops=False)
    new_payload = prep._stage_combined_crops(manifest_path, ns, payload)

    # Manifest is augmented in place and crops land under <output>/images.
    augmented = json.loads(manifest_path.read_text())["samples"][0]
    assert augmented["prepared_image"]
    assert augmented["prepared_image_orig_hw"] == [333, 500]
    assert (tmp_path / augmented["prepared_image"]).is_file()
    assert new_payload["samples"][0]["prepared_image"] == augmented["prepared_image"]

    # The staged crop reproduces the native decode exactly through the loader.
    staged_ds = _dataset(manifest_path)
    native_sample = {
        **staged_ds.samples[0],
        "prepared_image": "",
        "prepared_image_orig_hw": None,
    }
    img_fast, lmk_fast = staged_ds._load_image_and_landmarks(staged_ds.samples[0])
    img_native, lmk_native = staged_ds._load_image_and_landmarks(native_sample)
    assert np.array_equal(img_fast, img_native)
    assert np.array_equal(lmk_fast, lmk_native)


def test_stage_crops_reuses_existing_prepared_crop(tmp_path: Path, monkeypatch):
    src = tmp_path / "src"
    src.mkdir()
    _write_sample(src, "a", 333, 500, seed=12)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(_manifest(src, ["a"])))

    first = staging.stage_crops(manifest_path, out_manifest=manifest_path, strict=True)
    assert first["staged"] == 1

    def unexpected_native_decode(*args, **kwargs):
        raise AssertionError("existing prepared crop should skip native decode")

    monkeypatch.setattr(
        staging, "_native_image_and_landmarks", unexpected_native_decode
    )
    second = staging.stage_crops(manifest_path, out_manifest=manifest_path, strict=True)

    assert second["staged"] == 0
    assert second["existing"] == 1


def test_stage_crops_reuses_and_repairs_tight_face_crop(tmp_path: Path, monkeypatch):
    src = tmp_path / "src"
    src.mkdir()
    _write_sample(src, "face", 400, 500, seed=13)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(_manifest(src, ["face"], dataset="300vw")))

    first = staging.stage_crops(manifest_path, out_manifest=manifest_path, strict=True)
    assert first["face_cropped"] == 1
    staged_payload = json.loads(manifest_path.read_text())
    prepared_path = tmp_path / staged_payload["samples"][0]["prepared_image"]

    original_loader = staging._load_landmark_array

    def unexpected_landmark_load(*args, **kwargs):
        raise AssertionError("existing tight face crop should skip source loading")

    monkeypatch.setattr(staging, "_load_landmark_array", unexpected_landmark_load)
    second = staging.stage_crops(manifest_path, out_manifest=manifest_path, strict=True)
    assert second["face_cropped"] == 0
    assert second["face_crop_existing"] == 1

    monkeypatch.setattr(staging, "_load_landmark_array", original_loader)
    prepared_path.unlink()
    repaired = staging.stage_crops(
        manifest_path, out_manifest=manifest_path, strict=True
    )
    assert repaired["face_cropped"] == 1
    assert prepared_path.is_file()


def test_prepare_restores_existing_staged_crops_after_manifest_rebuild(
    tmp_path: Path,
):
    src = tmp_path / "src"
    src.mkdir()
    _write_sample(src, "prepared", 333, 500, seed=14)
    _write_sample(src, "tight", 400, 500, seed=15)
    original = _manifest(src, ["prepared", "tight"])
    for index, sample in enumerate(original["samples"]):
        sample["sample_id"] = f"sample-{index}"
        sample["source"] = {
            "dataset": sample["dataset"],
            "source_id": f"source-{index}",
        }
    original["samples"][1]["dataset"] = "300vw"
    original["samples"][1]["source"]["dataset"] = "300vw"

    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(original))
    staging.stage_crops(manifest_path, out_manifest=manifest_path, strict=True)

    cached = prepare._load_existing_staged_crops(manifest_path)
    assert {snapshot["kind"] for snapshot in cached.values()} == {
        "prepared",
        "tight",
    }

    rebuilt = copy.deepcopy(original)
    manifest_path.write_text(json.dumps(rebuilt))
    restored_prepared, restored_tight = prepare._restore_existing_staged_crops(
        manifest_path,
        rebuilt,
        cached,
    )

    assert (restored_prepared, restored_tight) == (1, 1)
    restored = json.loads(manifest_path.read_text())["samples"]
    assert restored[0]["prepared_image"]
    assert restored[1]["metadata"]["stage_face_crop"]["enabled"] is True
    assert (tmp_path / restored[1]["image"]).is_file()
    assert (tmp_path / restored[1]["landmarks"]).is_file()
