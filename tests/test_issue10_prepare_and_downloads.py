from __future__ import annotations

import argparse
import contextlib
import io
import json
import pickle
import tarfile
import zipfile
import zlib
from pathlib import Path

import cv2
import numpy as np
import pytest

from lib.datasets.progress import track
from tools import build_production_validated_manifest as production_builder
from tools import download_landmark_datasets as downloader
from tools import prepare_landmark_dataset as prepare


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------
def _points68() -> list[list[float]]:
    xs = np.linspace(40, 210, 68)
    ys = np.linspace(40, 210, 68)
    return [[float(x), float(y)] for x, y in zip(xs, ys)]


def _write_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.full((256, 256, 3), 127, dtype=np.uint8)
    assert cv2.imwrite(str(path), image)


def _make_json_source(extracted: Path, *, dataset: str, count: int = 3) -> None:
    """Write a tiny JSON source layout the builder can consume for any dataset."""
    extracted.mkdir(parents=True, exist_ok=True)
    samples = []
    for idx in range(count):
        image_name = f"{dataset}_{idx}.jpg"
        _write_image(extracted / image_name)
        samples.append(
            {
                "sample_id": f"{dataset}/sample_{idx}",
                "image": image_name,
                "landmarks": _points68(),
                "source_schema": "2d_68",
                "video_id": f"{dataset}_clip_{idx}",
            }
        )
    (extracted / "samples.json").write_text(
        json.dumps({"samples": samples}), encoding="utf-8"
    )


def _make_production_source(
    root: Path,
    *,
    points: list[list[float]] | None = None,
    runtime_bucket: str | None = "normal",
) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    image_name = "production_frame.jpg"
    _write_image(root / image_name)
    metadata = {"runtime_bucket": runtime_bucket} if runtime_bucket else {}
    face = {
        "landmarks_xy": points or _points68(),
        "x": 20,
        "y": 20,
        "w": 216,
        "h": 216,
        "metadata": metadata,
    }
    payload = {"__data__": {image_name: {"faces": [face]}}}
    (root / "alignments.fsa").write_bytes(zlib.compress(pickle.dumps(payload)))
    return root


def _production_pose_points68() -> list[list[float]]:
    points = np.zeros((68, 2), dtype=np.float64)
    points[0] = [-1.0, 1.0]
    points[16] = [1.0, 1.0]
    points[8] = [0.0, 1.6]
    for index in range(36, 42):
        points[index] = [-0.5 + 0.04 * (index - 36), 0.0]
    for index in range(42, 48):
        points[index] = [0.3 + 0.04 * (index - 42), 0.0]
    points[30] = [-0.5, 0.7]
    for index in range(48, 68):
        points[index] = [-0.2 + 0.02 * (index - 48), 0.9]
    points = (points + 2.0) * 50.0
    return points.tolist()


# ---------------------------------------------------------------------------
# Progress tracking
# ---------------------------------------------------------------------------
def test_progress_track_disables_on_non_tty():
    buf = io.StringIO()
    with contextlib.redirect_stderr(buf):
        collected = list(track(range(5), desc="x"))
    assert collected == list(range(5))
    assert buf.getvalue() == ""


def test_progress_track_renders_when_forced():
    buf = io.StringIO()
    with contextlib.redirect_stderr(buf):
        list(track(range(3), desc="loud", disable=False))
    assert "loud" in buf.getvalue()


def test_progress_track_manual_bar_updates():
    buf = io.StringIO()
    with contextlib.redirect_stderr(buf):
        bar = track(desc="dl", total=4, unit="B", disable=False)
        with bar:
            bar.update(4)
    assert bar.n == 4


def test_extract_zip_and_tar_roundtrip(tmp_path):
    zip_path = tmp_path / "a.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("dir/one.txt", "1")
        zf.writestr("two.txt", "2")
    zip_out = tmp_path / "zip_out"
    downloader._extract_zip(zip_path, zip_out)
    assert (zip_out / "dir" / "one.txt").read_text() == "1"
    assert (zip_out / "two.txt").read_text() == "2"

    tar_path = tmp_path / "a.tar.gz"
    payload = tmp_path / "payload.txt"
    payload.write_text("hello")
    with tarfile.open(tar_path, "w:gz") as tf:
        tf.add(payload, arcname="nested/hello.txt")
    tar_out = tmp_path / "tar_out"
    downloader._extract_tar(tar_path, tar_out)
    assert (tar_out / "nested" / "hello.txt").read_text() == "hello"


# ---------------------------------------------------------------------------
# Keyboard interrupt handling
# ---------------------------------------------------------------------------
def test_download_url_cleans_partial_file_on_interrupt(tmp_path, monkeypatch):
    def boom(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(downloader.urllib.request, "urlopen", boom)
    destination = tmp_path / "archives" / "file.zip"

    with pytest.raises(KeyboardInterrupt):
        downloader._download_url(
            "http://example.com/file.zip", destination, force=False
        )

    assert not destination.exists()
    assert list(destination.parent.glob("*.part")) == []


def test_downloader_main_returns_130_on_interrupt(tmp_path, monkeypatch, capsys):
    def boom(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(downloader, "_process_asset", boom)
    rc = downloader.main(["--datasets", "helen", "--output-root", str(tmp_path)])
    assert rc == 130
    assert "Ctrl-C" in capsys.readouterr().err


def test_prepare_main_returns_130_on_interrupt(tmp_path, monkeypatch, capsys):
    def boom(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(prepare, "_build_dataset", boom)
    rc = prepare.main(
        [
            "--datasets",
            "wflw-v",
            "--skip-download",
            "--data-root",
            str(tmp_path / "data"),
            "--output-root",
            str(tmp_path / "out"),
        ]
    )
    assert rc == 130
    assert "Ctrl-C" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# --datasets parsing
# ---------------------------------------------------------------------------
def test_normalize_datasets_space_and_comma():
    assert downloader.normalize_datasets(["wflw-v", "300vw,cofw29"]) == [
        "wflw-v",
        "300vw",
        "cofw29",
    ]


def test_normalize_datasets_aliases_and_dedup():
    assert downloader.normalize_datasets(["wflwv", "jd,jd-landmark", "wflw-v"]) == [
        "wflw-v",
        "jd-landmark",
    ]


def test_normalize_datasets_all_expands():
    assert downloader.normalize_datasets(["all"]) == list(downloader.ALL_DATASETS)


def test_normalize_prepare_datasets_accepts_prod_aliases():
    assert prepare._normalize_prepare_datasets(
        ["prod,production_validated", "production"]
    ) == ["production_validated"]


def test_normalize_datasets_accepts_prod_aliases():
    assert downloader.normalize_datasets(
        ["prod,production_validated", "production"]
    ) == ["production_validated"]


def test_production_validated_source_uses_requested_drive_id():
    source = next(
        asset for asset in downloader.SOURCES if asset.dataset == "production_validated"
    )

    assert (
        source.google_drive_file_id
        == downloader.PRODUCTION_VALIDATED_GOOGLE_DRIVE_FILE_ID
        == "1XFW3_xx9t6gnyAIRY6g71keDzzHFRWRg"
    )
    assert source.filename == "production_validated.zip"


def test_unknown_dataset_raises():
    with pytest.raises(ValueError, match="unknown dataset"):
        downloader._selected_sources(["not-a-dataset"], include_alternates=False)


# ---------------------------------------------------------------------------
# Registry creation / reuse
# ---------------------------------------------------------------------------
def _fake_results(dataset: str, extracted: Path) -> list[dict]:
    return [
        {
            "dataset": dataset,
            "name": f"{dataset} archive",
            "filename": f"{dataset}.zip",
            "status": "downloaded",
            "archive": str(extracted.parent / "archives" / f"{dataset}.zip"),
            "extracted": str(extracted),
            "source_kind": "url",
            "source": "https://example.com/x.zip",
            "checksum_status": "none",
            "required_for_builder": True,
            "alternate": False,
        }
    ]


def test_registry_creation_and_resolution(tmp_path):
    data_root = tmp_path / "data"
    extracted = data_root / "wflw-v" / "extracted"
    extracted.mkdir(parents=True)

    path = downloader.write_registry(_fake_results("wflw-v", extracted), data_root)
    assert path == downloader.registry_path(data_root)

    registry = downloader.load_registry(data_root)
    assert registry["version"] == downloader.REGISTRY_VERSION
    assert "wflw-v" in registry["datasets"]
    asset = registry["datasets"]["wflw-v"]["assets"][0]
    assert asset["extracted"] == str(extracted)
    assert asset["checksum_status"] == "none"
    assert asset["manual"] is False

    resolved = downloader.resolve_source_dir(registry, "wflw-v", data_root)
    assert resolved == extracted


def test_registry_reuse_merges_new_datasets(tmp_path):
    data_root = tmp_path / "data"
    ext_a = data_root / "wflw-v" / "extracted"
    ext_b = data_root / "300vw" / "extracted"
    ext_a.mkdir(parents=True)
    ext_b.mkdir(parents=True)

    downloader.write_registry(_fake_results("wflw-v", ext_a), data_root)
    downloader.write_registry(_fake_results("300vw", ext_b), data_root)

    registry = downloader.load_registry(data_root)
    assert set(registry["datasets"]) == {"wflw-v", "300vw"}


def test_resolve_source_dir_falls_back_without_registry(tmp_path):
    data_root = tmp_path / "data"
    extracted = data_root / "cofw29" / "extracted"
    extracted.mkdir(parents=True)
    assert downloader.resolve_source_dir({}, "cofw29", data_root) == extracted


# ---------------------------------------------------------------------------
# JD-landmark source artifacts
# ---------------------------------------------------------------------------
def test_jd_landmark_source_artifacts_configured():
    jd = [s for s in downloader.SOURCES if s.dataset == "jd-landmark"]
    names = {s.name for s in jd}
    assert "JD-landmark Training_data" in names
    assert "JD-landmark Test_data1" in names
    assert "JD-landmark corrected landmarks" in names
    assert "JD-landmark training bbox" in names

    by_name = {s.name: s for s in jd}
    assert (
        by_name["JD-landmark Training_data"].google_drive_file_id
        == "1gD4xcUUKQo6-70KgBUbODSdQtb_tnuvu"
    )
    assert by_name["JD-landmark Training_data"].filename == "Training_data.zip"
    assert (
        by_name["JD-landmark Test_data1"].google_drive_file_id
        == "12wRlDARRKe0u-lzFPRw-klG2MUa_JBQm"
    )
    assert by_name["JD-landmark corrected landmarks"].url.endswith(
        "Corrected_landmark.zip"
    )
    assert by_name["JD-landmark training bbox"].url.endswith(
        "training_dataset_face_detection_bounding_box_v1.zip"
    )


def test_jd_landmark_registry_entries_without_network(tmp_path, monkeypatch):
    data_root = tmp_path / "data"

    def fake_download_google_drive(file_id, destination, *, force=False):
        destination.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(destination, "w") as zf:
            zf.writestr("placeholder.txt", "fake jd landmark content")
        return destination

    monkeypatch.setattr(
        downloader, "_download_google_drive", fake_download_google_drive
    )

    jd_assets = [s for s in downloader.SOURCES if s.dataset == "jd-landmark"]
    args = argparse.Namespace(
        output_root=data_root,
        extract=False,
        force=False,
        skip_checksum=True,
        keep_going=True,
    )

    results = [
        downloader._process_asset(asset, args)
        for asset in jd_assets
        if asset.google_drive_file_id
    ]

    downloader.write_registry(results, data_root)
    registry = downloader.load_registry(data_root)

    assert "jd-landmark" in registry["datasets"]

    assets = registry["datasets"]["jd-landmark"]["assets"]
    by_name = {a["name"]: a for a in assets}

    assert "JD-landmark Test_data1" in by_name
    assert by_name["JD-landmark Test_data1"]["manual"] is False
    assert by_name["JD-landmark Test_data1"]["status"] in {"downloaded", "reused"}

    assert all(not a["manual"] for a in assets)
    assert all(a["status"] != "manual_google_drive" for a in assets)


def test_stage_jd_landmark_links_artifacts(tmp_path):
    data_root = tmp_path / "data"
    extracted = data_root / "jd-landmark" / "extracted"
    # Each artifact lands in its own archive-named subfolder, as the downloader extracts them.
    for subset in ("AFW", "HELEN", "IBUG", "LFPW"):
        (extracted / "Training_data.zip" / subset / "landmark").mkdir(parents=True)
        (extracted / "Training_data.zip" / subset / "picture").mkdir(parents=True)
    (extracted / "Test_data1.zip" / "Test_data1" / "landmark").mkdir(parents=True)
    (extracted / "Test_data1.zip" / "Test_data1" / "picture").mkdir(parents=True)
    (extracted / "Corrected_landmark.zip" / "Corrected_landmark").mkdir(parents=True)
    bbox = extracted / "bbox.zip" / "training_dataset_face_detection_bounding_box"
    bbox.mkdir(parents=True)

    staged = prepare._stage_jd_landmark(data_root, None)
    assert staged is not None
    assert (staged / "Training_data" / "AFW" / "landmark").is_dir()
    assert (staged / "Training_data" / "LFPW" / "picture").is_dir()
    assert (staged / "Test_data1").is_dir()
    assert (staged / "Test_data1" / "landmark").is_dir()
    assert (staged / "Corrected_landmark").is_dir()
    assert (staged / "training_dataset_face_detection_bounding_box").is_dir()


# ---------------------------------------------------------------------------
# --list table
# ---------------------------------------------------------------------------
def test_list_table_alignment():
    sources = downloader._selected_sources(None, include_alternates=False)
    table = downloader.format_list_table(sources)
    lines = table.split("\n")

    header = lines[0]
    for column in ("dataset", "asset", "source", "checksum", "kind"):
        assert column in header
    assert set(lines[1]) <= {"-", " "}

    # No trailing whitespace on any row.
    assert all(line == line.rstrip() for line in lines)

    # The asset column starts at a stable offset across every data row.
    asset_offset = header.index("asset")
    for line in lines[2:]:
        assert line[asset_offset - 1] == " "
        assert line[asset_offset] != " "


def test_list_table_marks_manual_and_gdrive():
    sources = downloader._selected_sources(["jd-landmark"], include_alternates=False)
    table = downloader.format_list_table(sources)
    assert "gdrive:12wRlDARRKe0u-lzFPRw-klG2MUa_JBQm" in table


def test_buildable_datasets_excludes_download_only():
    # aflw stays downloadable (a source layer for merl-rav) but is not buildable.
    assert "aflw" in downloader.DOWNLOAD_ONLY_DATASETS
    assert "aflw" in downloader.ALL_DATASETS
    assert "aflw" not in downloader.BUILDABLE_DATASETS
    assert set(downloader.BUILDABLE_DATASETS) == set(downloader.ALL_DATASETS) - {"aflw"}


def test_list_table_marks_download_only_aflw():
    sources = downloader._selected_sources(["aflw", "wflw-v"], include_alternates=False)
    table = downloader.format_list_table(sources)
    aflw_rows = [line for line in table.splitlines() if line.startswith("aflw")]
    assert aflw_rows and all("download-only" in line for line in aflw_rows)
    # A buildable dataset keeps its normal kind marker.
    assert any(
        line.startswith("wflw-v") and "download-only" not in line
        for line in table.splitlines()
    )


def _fake_full_result(asset, args):
    extracted = Path(args.output_root) / asset.dataset / "extracted"
    extracted.mkdir(parents=True, exist_ok=True)
    return {
        "dataset": asset.dataset,
        "name": asset.name,
        "filename": asset.filename,
        "status": "downloaded",
        "archive": str(extracted.parent / "archives" / asset.filename),
        "extracted": str(extracted),
        "source_kind": asset.source_kind,
        "source": asset.source_display,
        "checksum_status": "none",
        "required_for_builder": asset.required_for_builder,
        "alternate": asset.alternate,
    }


def test_process_assets_parallel_preserves_source_order(monkeypatch):
    sources = downloader._selected_sources(None, include_alternates=False)
    monkeypatch.setattr(downloader, "_process_asset", _fake_full_result)
    args = argparse.Namespace(output_root=Path("/tmp/unused"), extract=False)

    serial = downloader._process_assets_with_status(sources, args, workers=1)
    parallel = downloader._process_assets_with_status(sources, args, workers=4)

    assert len(parallel) == len(sources)
    assert [r["name"] for r in parallel] == [r["name"] for r in serial]


def test_download_datasets_accepts_workers(tmp_path, monkeypatch):
    monkeypatch.setattr(downloader, "_process_asset", _fake_full_result)
    results, registry = downloader.download_datasets(
        ["wflw-v", "300vw"], output_root=tmp_path, extract=False, workers=2
    )
    assert {r["dataset"] for r in results} == {"wflw-v", "300vw"}
    assert set(registry["datasets"]) >= {"wflw-v", "300vw"}


# ---------------------------------------------------------------------------
# prepare flow (WFLW-V via tiny fake source layout)
# ---------------------------------------------------------------------------
def _prepare_args(**overrides) -> argparse.Namespace:
    base = dict(
        datasets=["wflw-v"],
        data_root=overrides.get("data_root"),
        output_root=overrides.get("output_root"),
        prod_dir=None,
        image_root=None,
        manifest_mode="replace",
        allow_overlap=False,
        write_overlays=False,
        audit_overlay_limit=50,
        frame_stride=1,
        max_frames_per_video=None,
        workers=1,
        force=False,
        skip_checksum=True,
        skip_download=True,
        skip_build=False,
        skip_validate=False,
        skip_image_exists_check=False,
        keep_going=True,
        samples_per_scenario=None,
        dataset_workers=1,
    )
    base.update(overrides)
    return argparse.Namespace(**base)


def test_prepare_wflwv_flow(tmp_path, capsys):
    data_root = tmp_path / "data"
    output_root = tmp_path / "out"
    _make_json_source(data_root / "wflw-v" / "extracted", dataset="wflw-v", count=3)

    args = _prepare_args(
        datasets=["wflw-v"], data_root=data_root, output_root=output_root
    )
    rc = prepare.prepare(args)
    assert rc == 0

    manifest = output_root / "manifest.json"
    assert manifest.is_file()
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert payload["metadata"]["sample_count"] == 3

    out = capsys.readouterr().out
    assert "Per-dataset summary" in out
    assert "Combined manifest summary" in out
    assert "run_cdvit_manifest_training_pipeline.py --manifest" in out


def test_prepare_prod_skips_download_and_uses_production_builder(tmp_path, monkeypatch):
    prod_dir = _make_production_source(tmp_path / "prod")
    output_root = tmp_path / "out"

    def unexpected_download(*args, **kwargs):
        raise AssertionError("prod must not enter the downloader")

    monkeypatch.setattr(downloader, "download_datasets", unexpected_download)
    args = _prepare_args(
        datasets=["prod"],
        data_root=tmp_path / "data",
        output_root=output_root,
        prod_dir=prod_dir,
        skip_download=False,
    )

    assert prepare.prepare(args) == 0

    payload = json.loads((output_root / "manifest.json").read_text("utf-8"))
    assert payload["metadata"]["dataset"] == "production_validated"
    assert [sample["dataset"] for sample in payload["samples"]] == [
        "production_validated"
    ]
    landmark_path = output_root / payload["samples"][0]["landmarks"]
    assert landmark_path.is_file()


def test_prepare_prod_downloads_default_when_prod_dir_missing(tmp_path, monkeypatch):
    data_root = tmp_path / "data"
    output_root = tmp_path / "out"

    def fake_download(datasets, *, output_root, **kwargs):
        assert datasets == ["production_validated"]
        assert kwargs["extract"] is True
        source = _make_production_source(
            Path(output_root) / "production_validated" / "extracted"
        )
        results = _fake_results("production_validated", source)
        return results, downloader.build_registry(results, Path(output_root))

    monkeypatch.setattr(downloader, "download_datasets", fake_download)

    args = _prepare_args(
        datasets=["prod"],
        data_root=data_root,
        output_root=output_root,
        prod_dir=None,
        skip_download=False,
    )

    assert prepare.prepare(args) == 0

    payload = json.loads((output_root / "manifest.json").read_text("utf-8"))
    assert payload["metadata"]["dataset"] == "production_validated"
    assert payload["metadata"]["sample_count"] == 1


@pytest.mark.parametrize(
    ("runtime_bucket", "expected_primary"),
    [("normal", "normal"), (None, "unknown")],
)
def test_production_manifest_appends_landmark_geometry_pose_conditions(
    tmp_path, runtime_bucket, expected_primary
):
    prod_dir = _make_production_source(
        tmp_path / "prod",
        points=_production_pose_points68(),
        runtime_bucket=runtime_bucket,
    )
    output_dir = tmp_path / "out"

    production_builder.build_manifest(prod_dir, output_dir)

    sample = json.loads((output_dir / "manifest.json").read_text("utf-8"))["samples"][0]
    assert sample["condition"] == expected_primary
    assert sample["conditions"] == [
        expected_primary,
        "pose_left_profile",
        "pitch_down",
        "pose_side_left",
    ]
    assert sample["metadata"]["pose_source"] == "landmark_geometry"
    assert sample["metadata"]["pose_bucket"] == "left_profile"
    assert sample["metadata"]["pitch_bucket"] == "down"
    assert sample["metadata"]["pose_side"] == "left"


@pytest.mark.parametrize(
    ("datasets", "dataset_workers"),
    [
        (["wflw-v", "prod"], 1),
        (["prod", "wflw-v"], 1),
        (["wflw-v", "prod"], 2),
    ],
)
def test_prepare_prod_merges_with_generic_dataset(
    tmp_path, monkeypatch, datasets, dataset_workers
):
    monkeypatch.setattr(prepare.os, "cpu_count", lambda: 4)
    data_root = tmp_path / "data"
    output_root = tmp_path / "out"
    prod_dir = _make_production_source(tmp_path / "prod")
    _make_json_source(data_root / "wflw-v" / "extracted", dataset="wflw-v", count=2)

    args = _prepare_args(
        datasets=datasets,
        data_root=data_root,
        output_root=output_root,
        prod_dir=prod_dir,
        dataset_workers=dataset_workers,
    )

    assert prepare.prepare(args) == 0

    payload = json.loads((output_root / "manifest.json").read_text("utf-8"))
    assert {sample["dataset"] for sample in payload["samples"]} == {
        "wflw-v",
        "production_validated",
    }
    assert payload["metadata"]["sample_count"] == 3


def test_production_manifest_downloads_default_without_prod_dir(tmp_path, monkeypatch):
    output_dir = tmp_path / "out"

    def fake_download(datasets, *, output_root, **kwargs):
        assert datasets == ["production_validated"]
        assert kwargs["extract"] is True
        source = _make_production_source(
            Path(output_root) / "production_validated" / "extracted"
        )
        results = _fake_results("production_validated", source)
        return results, downloader.build_registry(results, Path(output_root))

    monkeypatch.setattr(downloader, "download_datasets", fake_download)

    assert production_builder.main(["--output-dir", str(output_dir)]) == 0

    payload = json.loads((output_dir / "manifest.json").read_text("utf-8"))
    assert payload["metadata"]["sample_count"] == 1


def _write_fsa_source_with_faces(
    root: Path, *, frame_hw: tuple[int, int], faces: list[dict]
) -> Path:
    """Write a production source with a custom frame size and explicit faces."""
    root.mkdir(parents=True, exist_ok=True)
    image_name = "frame.jpg"
    image = np.full((frame_hw[0], frame_hw[1], 3), 110, dtype=np.uint8)
    assert cv2.imwrite(str(root / image_name), image)
    payload = {"__data__": {image_name: {"faces": faces}}}
    (root / "alignments.fsa").write_bytes(zlib.compress(pickle.dumps(payload)))
    return root


def test_production_manifest_crops_small_face_to_256(tmp_path):
    # A small face inside a large native frame: the builder must crop+remap, not
    # leave whole-frame landmarks for the loader to squash. Regression for the
    # uncropped production_validated path that drove a ~28% eval NME.
    xs = np.linspace(840, 1040, 68)
    ys = np.linspace(430, 660, 68)
    native_pts = np.stack([xs, ys], axis=1)
    face = {
        "landmarks_xy": native_pts.tolist(),
        "x": 820,
        "y": 410,
        "w": 240,
        "h": 270,
    }
    prod_dir = _write_fsa_source_with_faces(
        tmp_path / "prod", frame_hw=(1080, 1920), faces=[face]
    )
    output_dir = tmp_path / "out"

    production_builder.build_manifest(prod_dir, output_dir)
    sample = json.loads((output_dir / "manifest.json").read_text("utf-8"))["samples"][0]

    crop = cv2.imread(sample["image"])
    assert crop is not None and crop.shape[:2] == (256, 256)

    lmk = np.load(output_dir / sample["landmarks"])
    assert lmk.shape == (68, 2)
    # Landmarks were remapped into the 256 crop frame, not left at native scale.
    assert lmk.max() <= 256.0 and lmk.min() >= -8.0
    assert lmk.max() > 64.0  # face fills the crop instead of a few native pixels
    # Crop provenance points back at the native frame for split-safe grouping.
    assert sample["metadata"]["original_image"].endswith("frame.jpg")
    assert sample["normalizer"] > 1.0


def test_production_manifest_preserves_mixed_68_and_98_source_schema(tmp_path):
    pts68 = np.stack(
        [np.linspace(820, 1010, 68), np.linspace(410, 640, 68)], axis=1
    )
    pts98 = np.stack(
        [np.linspace(800, 1000, 98), np.linspace(400, 650, 98)], axis=1
    )
    faces = [
        {"landmarks_xy": pts68.tolist(), "x": 800, "y": 390, "w": 230, "h": 270},
        {"landmarks_xy": pts98.tolist(), "x": 790, "y": 380, "w": 230, "h": 280},
    ]
    prod_dir = _write_fsa_source_with_faces(
        tmp_path / "prod", frame_hw=(1080, 1920), faces=faces
    )
    output_dir = tmp_path / "out"

    metadata = production_builder.build_manifest(prod_dir, output_dir)
    assert metadata["sample_count"] == 2
    samples = json.loads((output_dir / "manifest.json").read_text("utf-8"))["samples"]
    by_schema = {s["source_schema"]: s for s in samples}

    assert set(by_schema) == {"2d_68", "2d_98"}
    # Each face is kept in its native schema so 98-point faces train landmarks_98.
    for source_schema, sample in by_schema.items():
        assert sample["target_schema"] == source_schema
        assert sample["head_name"] == (
            "landmarks_98" if source_schema == "2d_98" else "landmarks_68"
        )
        expected_count = 98 if source_schema == "2d_98" else 68
        assert sample["landmark_count"] == expected_count
        assert np.load(output_dir / sample["landmarks"]).shape == (expected_count, 2)
        # Native schema (source == target) records a native, validator-accepted audit.
        assert sample["mapping_audit"]["status"] == "native"

    from lib.manifest.validator import _validate_projection_audit

    for sample in samples:
        assert (
            _validate_projection_audit(
                sample,
                source_schema=sample["source_schema"],
                target_schema=sample["target_schema"],
                allow_missing_projection_audit=False,
            )
            is None
        )


def test_prepare_prod_validates_98_point_faces(tmp_path):
    # End-to-end guard: a 98-point production face survives full prepare
    # validation as a NATIVE landmarks_98 sample (not collapsed to 68).
    pts98 = np.stack(
        [np.linspace(60, 200, 98), np.linspace(70, 205, 98)], axis=1
    ).tolist()
    prod_dir = _make_production_source(tmp_path / "prod", points=pts98)
    args = _prepare_args(
        datasets=["prod"],
        data_root=tmp_path / "data",
        output_root=tmp_path / "out",
        prod_dir=prod_dir,
    )

    assert prepare.prepare(args) == 0

    payload = json.loads((tmp_path / "out" / "manifest.json").read_text("utf-8"))
    sample = payload["samples"][0]
    assert sample["source_schema"] == "2d_98"
    assert sample["target_schema"] == "2d_98"
    assert sample["head_name"] == "landmarks_98"
    assert sample["landmark_count"] == 98
    assert sample["mapping_audit"]["status"] == "native"
    assert np.load(tmp_path / "out" / sample["landmarks"]).shape == (98, 2)


def test_prepare_stage_crops_runs_when_validation_not_ok(tmp_path, monkeypatch):
    # Regression: --stage-crops used to be silently skipped whenever the combined
    # manifest validated ok=False (which is normal when some images are corrupt).
    # It must now stage the valid samples and emit a visible warning instead.
    data_root = tmp_path / "data"
    output_root = tmp_path / "out"
    _make_json_source(data_root / "wflw-v" / "extracted", dataset="wflw-v", count=2)

    real_validate = prepare._validate

    def fake_validate(manifest, **kwargs):
        report = real_validate(manifest, **kwargs)
        report["ok"] = False
        report["invalid_samples"] = max(int(report.get("invalid_samples", 0)), 1)
        report["valid_samples"] = max(int(report.get("valid_samples", 0)), 1)
        return report

    monkeypatch.setattr(prepare, "_validate", fake_validate)

    staged = {"called": False}
    real_stage = prepare._stage_combined_crops

    def fake_stage(manifest, args, payload):
        staged["called"] = True
        return real_stage(manifest, args, payload)

    monkeypatch.setattr(prepare, "_stage_combined_crops", fake_stage)

    args = _prepare_args(
        datasets=["wflw-v"],
        data_root=data_root,
        output_root=output_root,
        stage_crops=True,
    )
    prepare.prepare(args)

    assert staged["called"] is True


def _configure_info_logging():
    from lib.logging_utils import configure_console_logging, verbosity_from_name

    configure_console_logging(verbosity_from_name("info"), "human")


def test_log_build_errors_lists_failed_datasets(capsys):
    _configure_info_logging()
    prepare._log_build_errors(
        [
            {"dataset": "wflw", "status": "ok"},
            {"dataset": "cofw68", "status": "error", "error": "boom while parsing"},
        ]
    )
    out = capsys.readouterr().out
    assert "build FAILED for 1 dataset(s): cofw68" in out
    assert "cofw68: boom while parsing" in out


def test_log_validation_failures_reports_actionable_breakdown(capsys):
    _configure_info_logging()
    report = {
        "manifest": "data/prepared/manifest.json",
        "ok": False,
        "total_samples": 94384,
        "valid_samples": 70911,
        "invalid_samples": 23473,
        "missing_required_fields": {"split_safe_id": 12},
        "invalid_reasons": {
            "missing_image": 12000,
            "invalid_mapping_or_projection_audit_status": 9000,
        },
        "invalid_by_dataset": {"multipie": 9000, "fll3": 6473},
        "geometry": {"suspicious_loader_padding": 2473},
        "leakage": {
            "violations": [
                {"field": "image", "value": "/x/f.jpg", "splits": ["test", "train"]}
            ]
        },
        "examples": {
            "invalid": [
                {
                    "dataset": "multipie",
                    "sample_id": "s_12",
                    "split": "train",
                    "errors": ["invalid_mapping_or_projection_audit_status:audited"],
                }
            ]
        },
    }
    prepare._log_validation_failures(report)
    out = capsys.readouterr().out
    assert "validation FAILED" in out
    assert "invalid 23,473/94,384" in out
    assert "missing_image=12,000" in out
    assert "by dataset | multipie=9,000" in out
    assert "leakage | 1 split-safe violation(s)" in out
    assert "multipie/s_12 [train]: invalid_mapping_or_projection_audit_status:audited" in out


def test_log_validation_failures_silent_when_ok(capsys):
    _configure_info_logging()
    prepare._log_validation_failures({"ok": True})
    assert "validation FAILED" not in capsys.readouterr().out


def test_prepare_prints_validation_breakdown_on_failure(tmp_path, monkeypatch, capsys):
    data_root = tmp_path / "data"
    output_root = tmp_path / "out"
    _make_json_source(data_root / "wflw-v" / "extracted", dataset="wflw-v", count=2)

    real_validate = prepare._validate

    def fake_validate(manifest, **kwargs):
        report = real_validate(manifest, **kwargs)
        report["ok"] = False
        report["invalid_samples"] = 1
        report["invalid_reasons"] = {"missing_image": 1}
        report["invalid_by_dataset"] = {"wflw-v": 1}
        report.setdefault("examples", {})["invalid"] = [
            {
                "dataset": "wflw-v",
                "sample_id": "s0",
                "split": "train",
                "errors": ["missing_image:/x.jpg"],
            }
        ]
        return report

    monkeypatch.setattr(prepare, "_validate", fake_validate)
    args = _prepare_args(
        datasets=["wflw-v"], data_root=data_root, output_root=output_root
    )
    prepare.prepare(args)

    out = capsys.readouterr().out
    assert "validation FAILED" in out
    assert "missing_image=1" in out
    assert "wflw-v/s0" in out
    # A real per-sample validation report is written and referenced in the log.
    report_path = output_root / "validation_report.json"
    assert report_path.is_file()
    assert str(report_path) in out


def test_resolve_parallel_budget_priority_and_cap(monkeypatch):
    monkeypatch.setattr(prepare.os, "cpu_count", lambda: 8)

    # --dataset-workers has priority: 4 outer leaves floor(8/4)=2 inner even
    # though 16 were requested, and outer * inner == 8 never exceeds the budget.
    assert prepare._resolve_parallel_budget(4, 16, 4) == (4, 2)
    # outer is clamped to the dataset count, freeing more inner workers.
    assert prepare._resolve_parallel_budget(4, 16, 2) == (2, 4)
    # inner <= 0 means "all", still capped to the per-dataset budget.
    assert prepare._resolve_parallel_budget(2, 0, 5) == (2, 4)
    # outer <= 0 means "all CPUs" but is clamped to the dataset count.
    assert prepare._resolve_parallel_budget(0, 16, 3) == (3, 2)
    # --dataset-workers 1 keeps outer == 1 (serial), inner unchanged within budget.
    assert prepare._resolve_parallel_budget(1, 16, 4) == (1, 8)


def test_prepare_dataset_workers_parallel_matches_serial(tmp_path, monkeypatch):
    monkeypatch.setattr(prepare.os, "cpu_count", lambda: 4)

    def _run(out_name: str, dataset_workers: int) -> dict:
        data_root = tmp_path / f"data_{out_name}"
        output_root = tmp_path / out_name
        _make_json_source(data_root / "wflw-v" / "extracted", dataset="wflw-v", count=2)
        _make_json_source(data_root / "300vw" / "extracted", dataset="300vw", count=3)
        args = _prepare_args(
            datasets=["wflw-v", "300vw"],
            data_root=data_root,
            output_root=output_root,
            dataset_workers=dataset_workers,
        )
        assert prepare.prepare(args) == 0
        return json.loads((output_root / "manifest.json").read_text(encoding="utf-8"))

    serial = _run("serial", 1)
    parallel = _run("parallel", 2)

    assert (
        parallel["metadata"]["sample_count"] == serial["metadata"]["sample_count"] == 5
    )
    assert (
        {s["dataset"] for s in parallel["samples"]}
        == {s["dataset"] for s in serial["samples"]}
        == {"wflw-v", "300vw"}
    )
    # Every artifact in the merged manifest resolves relative to its directory.
    base = tmp_path / "parallel"
    for sample in parallel["samples"]:
        for key in ("image", "landmarks"):
            path = Path(sample[key])
            resolved = path if path.is_absolute() else base / path
            assert resolved.exists(), f"missing {key}: {sample[key]}"


def test_prepare_skip_build_remerges_cached_datasets_and_continues_pipeline(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(prepare.os, "cpu_count", lambda: 4)
    data_root = tmp_path / "data"
    output_root = tmp_path / "out"
    _make_json_source(data_root / "wflw-v" / "extracted", dataset="wflw-v", count=2)
    _make_json_source(data_root / "300vw" / "extracted", dataset="300vw", count=3)

    initial_args = _prepare_args(
        datasets=["wflw-v", "300vw"],
        data_root=data_root,
        output_root=output_root,
        dataset_workers=2,
    )
    assert prepare.prepare(initial_args) == 0
    (output_root / "manifest.json").unlink()

    def unexpected_call(*args, **kwargs):
        raise AssertionError("--skip-build must not download or build")

    monkeypatch.setattr(downloader, "download_datasets", unexpected_call)
    monkeypatch.setattr(prepare, "_build_dataset", unexpected_call)

    cached_args = _prepare_args(
        datasets=["300vw", "wflw-v"],
        data_root=data_root,
        output_root=output_root,
        dataset_workers=1,
        skip_download=False,
        skip_build=True,
    )
    assert prepare.prepare(cached_args) == 0

    payload = json.loads((output_root / "manifest.json").read_text("utf-8"))
    assert payload["metadata"]["sample_count"] == 5
    assert {sample["dataset"] for sample in payload["samples"]} == {
        "wflw-v",
        "300vw",
    }
    assert all(
        sample.get("metadata", {}).get("hard_negative_bucket")
        for sample in payload["samples"]
    )

    cached_args.manifest_mode = "merge"
    assert prepare.prepare(cached_args) == 0
    merged = json.loads((output_root / "manifest.json").read_text("utf-8"))
    assert merged["metadata"]["sample_count"] == 5


def test_prepare_skip_build_requires_requested_cached_manifest(tmp_path):
    args = _prepare_args(
        datasets=["wflw-v"],
        data_root=tmp_path / "data",
        output_root=tmp_path / "out",
        skip_build=True,
    )

    assert prepare.prepare(args) == 1


def test_prepare_skip_build_reuses_cached_prod_without_prod_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(prepare.os, "cpu_count", lambda: 4)
    data_root = tmp_path / "data"
    output_root = tmp_path / "out"
    prod_dir = _make_production_source(tmp_path / "prod")
    _make_json_source(data_root / "wflw-v" / "extracted", dataset="wflw-v", count=1)

    initial_args = _prepare_args(
        datasets=["wflw-v", "prod"],
        data_root=data_root,
        output_root=output_root,
        prod_dir=prod_dir,
        dataset_workers=2,
    )
    assert prepare.prepare(initial_args) == 0
    (output_root / "manifest.json").unlink()

    cached_args = _prepare_args(
        datasets=["prod", "wflw-v"],
        data_root=data_root,
        output_root=output_root,
        prod_dir=None,
        skip_build=True,
    )
    assert prepare.prepare(cached_args) == 0

    payload = json.loads((output_root / "manifest.json").read_text("utf-8"))
    assert {sample["dataset"] for sample in payload["samples"]} == {
        "wflw-v",
        "production_validated",
    }


def test_prepare_dataset_workers_merge_does_not_double(tmp_path, monkeypatch):
    monkeypatch.setattr(prepare.os, "cpu_count", lambda: 4)
    data_root = tmp_path / "data"
    output_root = tmp_path / "out"
    _make_json_source(data_root / "wflw-v" / "extracted", dataset="wflw-v", count=2)
    _make_json_source(data_root / "300vw" / "extracted", dataset="300vw", count=3)

    def _args(mode: str, dataset_workers: int) -> argparse.Namespace:
        return _prepare_args(
            datasets=["wflw-v", "300vw"],
            data_root=data_root,
            output_root=output_root,
            dataset_workers=dataset_workers,
            manifest_mode=mode,
        )

    def _sample_count() -> int:
        payload = json.loads((output_root / "manifest.json").read_text("utf-8"))
        ids = [s.get("sample_id") for s in payload["samples"]]
        assert len(ids) == len(set(ids)), "merge produced duplicate sample_ids"
        assert payload["metadata"]["sample_count"] == len(payload["samples"])
        return len(payload["samples"])

    # Serial replace writes absolute crop paths; a parallel merge rebuilds the
    # same logical samples under _datasets/.. (different image strings). Without
    # source-identity dedupe this would double to 10.
    assert prepare.prepare(_args("replace", 1)) == 0
    assert _sample_count() == 5
    assert prepare.prepare(_args("merge", 2)) == 0
    assert _sample_count() == 5

    # Repeated parallel merge must also stay stable.
    assert prepare.prepare(_args("merge", 2)) == 0
    assert _sample_count() == 5


def test_build_dataset_rejects_non_buildable_id():
    # aflw is an image-only base cache, not a builder --dataset choice. The guard
    # must raise a clean ValueError instead of letting argparse SystemExit.
    with pytest.raises(ValueError, match="not buildable"):
        prepare._build_dataset(
            "aflw",
            None,
            None,
            Path("/tmp/unused"),
            mode="replace",
            args=argparse.Namespace(),
        )


def test_prepare_skips_non_buildable_source_cache(tmp_path, capsys):
    data_root = tmp_path / "data"
    output_root = tmp_path / "out"
    _make_json_source(data_root / "wflw-v" / "extracted", dataset="wflw-v", count=2)

    # aflw is requested alongside a buildable dataset; it must be skipped (not fed
    # to the builder) while wflw-v still builds and the run succeeds.
    args = _prepare_args(
        datasets=["aflw", "wflw-v"], data_root=data_root, output_root=output_root
    )
    assert prepare.prepare(args) == 0

    payload = json.loads((output_root / "manifest.json").read_text("utf-8"))
    assert {s["dataset"] for s in payload["samples"]} == {"wflw-v"}
    assert "skipping non-buildable source caches" in capsys.readouterr().out


def test_prepare_only_non_buildable_returns_2(tmp_path):
    args = _prepare_args(
        datasets=["aflw"],
        data_root=tmp_path / "data",
        output_root=tmp_path / "out",
    )
    assert prepare.prepare(args) == 2


def test_prepare_multi_dataset_merge(tmp_path, capsys):
    data_root = tmp_path / "data"
    output_root = tmp_path / "out"
    _make_json_source(data_root / "wflw-v" / "extracted", dataset="wflw-v", count=2)
    _make_json_source(data_root / "300vw" / "extracted", dataset="300vw", count=3)

    args = _prepare_args(
        datasets=["wflw-v", "300vw"], data_root=data_root, output_root=output_root
    )
    rc = prepare.prepare(args)
    assert rc == 0

    payload = json.loads((output_root / "manifest.json").read_text(encoding="utf-8"))
    datasets = {sample["dataset"] for sample in payload["samples"]}
    assert datasets == {"wflw-v", "300vw"}
    assert payload["metadata"]["sample_count"] == 5

    out = capsys.readouterr().out
    assert "wflw-v" in out and "300vw" in out


def test_prepare_merge_mode_appends_to_existing(tmp_path):
    data_root = tmp_path / "data"
    output_root = tmp_path / "out"
    _make_json_source(data_root / "wflw-v" / "extracted", dataset="wflw-v", count=2)
    _make_json_source(data_root / "300vw" / "extracted", dataset="300vw", count=2)

    rc = prepare.prepare(
        _prepare_args(datasets=["wflw-v"], data_root=data_root, output_root=output_root)
    )
    assert rc == 0
    first = json.loads((output_root / "manifest.json").read_text(encoding="utf-8"))
    assert first["metadata"]["sample_count"] == 2

    rc = prepare.prepare(
        _prepare_args(
            datasets=["300vw"],
            data_root=data_root,
            output_root=output_root,
            manifest_mode="merge",
        )
    )
    assert rc == 0
    merged = json.loads((output_root / "manifest.json").read_text(encoding="utf-8"))
    assert merged["metadata"]["sample_count"] == 4
    assert {s["dataset"] for s in merged["samples"]} == {"wflw-v", "300vw"}
