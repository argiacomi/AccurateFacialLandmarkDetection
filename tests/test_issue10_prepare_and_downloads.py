from __future__ import annotations

import argparse
import contextlib
import io
import json
import tarfile
import zipfile
from pathlib import Path

import cv2
import numpy as np
import pytest

from lib.datasets.progress import track
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
    assert "JD-landmark Test_data1" in names
    assert "JD-landmark corrected landmarks" in names
    assert "JD-landmark training bbox" in names

    by_name = {s.name: s for s in jd}
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
    (extracted / "Test_data1.zip" / "Test_data1" / "landmark").mkdir(parents=True)
    (extracted / "Test_data1.zip" / "Test_data1" / "picture").mkdir(parents=True)
    (extracted / "Corrected_landmark.zip" / "Corrected_landmark").mkdir(parents=True)
    bbox = extracted / "bbox.zip" / "training_dataset_face_detection_bounding_box"
    bbox.mkdir(parents=True)

    staged = prepare._stage_jd_landmark(data_root, None)
    assert staged is not None
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


# ---------------------------------------------------------------------------
# prepare flow (WFLW-V via tiny fake source layout)
# ---------------------------------------------------------------------------
def _prepare_args(**overrides) -> argparse.Namespace:
    base = dict(
        datasets=["wflw-v"],
        data_root=overrides.get("data_root"),
        output_root=overrides.get("output_root"),
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
        skip_validate=False,
        skip_image_exists_check=False,
        keep_going=True,
        samples_per_scenario=None,
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
