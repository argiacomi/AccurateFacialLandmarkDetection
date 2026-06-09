from __future__ import annotations

import argparse
import json
import tarfile
import zipfile
from pathlib import Path

import cv2
import numpy as np
import pytest

from tools import build_quality_dataset as builder
from tools import download_landmark_datasets as downloader
from tools import prepare_landmark_dataset as prepare


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _valid_zip(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("content.txt", "payload")
    return path


def _valid_targz(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    inner = path.parent / "content.txt"
    inner.write_text("payload")
    with tarfile.open(path, "w:gz") as tf:
        tf.add(inner, arcname="content.txt")
    inner.unlink()
    return path


def _args(
    output_root: Path,
    *,
    extract: bool = True,
    force: bool = False,
    keep_going: bool = False,
):
    return argparse.Namespace(
        output_root=output_root,
        extract=extract,
        force=force,
        skip_checksum=True,
        keep_going=keep_going,
    )


def _asset(
    dataset: str, *, gdrive: bool = False, name: str | None = None
) -> downloader.SourceAsset:
    for a in downloader.SOURCES:
        if a.dataset != dataset:
            continue
        if name is not None and a.name != name:
            continue
        if gdrive and not a.google_drive_file_id:
            continue
        if (
            not gdrive
            and gdrive is False
            and name is None
            and a.url is None
            and a.google_drive_file_id is None
        ):
            continue
        return a
    raise AssertionError(
        f"no source asset for {dataset!r} name={name!r} gdrive={gdrive}"
    )


# ---------------------------------------------------------------------------
# WFLW local archive reuse (configured + alternate filename)
# ---------------------------------------------------------------------------
def test_wflw_images_reuse_configured_tarball_skips_gdrive(tmp_path, monkeypatch):
    output_root = tmp_path / "data"
    wflw_images = _asset("wflw", gdrive=True)
    _valid_targz(output_root / "wflw" / "archives" / "WFLW_images.tar.gz")

    def fail(*args, **kwargs):  # pragma: no cover - must not be called
        raise AssertionError(
            "_download_google_drive should not be called when a tarball exists"
        )

    monkeypatch.setattr(downloader, "_download_google_drive", fail)
    result = downloader._process_asset(wflw_images, _args(output_root, extract=False))

    assert result["status"] == "reused"
    assert result["reused_from"].endswith("WFLW_images.tar.gz")


def test_wflw_images_reuse_alternate_zip(tmp_path, monkeypatch):
    output_root = tmp_path / "data"
    wflw_images = _asset("wflw", gdrive=True)
    # Only an alternate-named archive exists (configured is .tar.gz).
    _valid_zip(output_root / "wflw" / "archives" / "WFLW_images.zip")

    monkeypatch.setattr(
        downloader,
        "_download_google_drive",
        lambda *a, **k: (_ for _ in ()).throw(
            AssertionError("should reuse, not download")
        ),
    )
    result = downloader._process_asset(wflw_images, _args(output_root, extract=False))
    assert result["status"] == "reused"
    assert Path(result["reused_from"]).name == "WFLW_images.zip"


# ---------------------------------------------------------------------------
# Shared COFW image archive reuse (both orders) + separate annotations
# ---------------------------------------------------------------------------
def _cofw_color(dataset: str) -> downloader.SourceAsset:
    return _asset(
        dataset,
        name="cofw68 color images"
        if dataset == "cofw68"
        else "cofw29 original color images",
    )


def test_cofw68_then_cofw29_reuses_shared_image_archive(tmp_path, monkeypatch):
    output_root = tmp_path / "data"

    def fake_url(url, destination, *, force=False):
        return _valid_zip(destination)

    monkeypatch.setattr(downloader, "_download_url", fake_url)
    first = downloader._process_asset(_cofw_color("cofw68"), _args(output_root))
    assert first["status"] == "downloaded"

    # cofw29 must reuse the COFW color archive cofw68 already fetched.
    monkeypatch.setattr(
        downloader,
        "_download_url",
        lambda *a, **k: (_ for _ in ()).throw(
            AssertionError("should reuse cofw68 archive")
        ),
    )
    second = downloader._process_asset(_cofw_color("cofw29"), _args(output_root))
    assert second["status"] == "reused_shared"
    assert "cofw68" in second["reused_from"]
    assert Path(second["extracted"]).is_dir()


def test_cofw29_then_cofw68_reuses_shared_image_archive(tmp_path, monkeypatch):
    output_root = tmp_path / "data"
    monkeypatch.setattr(
        downloader,
        "_download_url",
        lambda url, destination, *, force=False: _valid_zip(destination),
    )
    first = downloader._process_asset(_cofw_color("cofw29"), _args(output_root))
    assert first["status"] == "downloaded"

    monkeypatch.setattr(
        downloader,
        "_download_url",
        lambda *a, **k: (_ for _ in ()).throw(
            AssertionError("should reuse cofw29 archive")
        ),
    )
    second = downloader._process_asset(_cofw_color("cofw68"), _args(output_root))
    assert second["status"] == "reused_shared"
    assert "cofw29" in second["reused_from"]


def test_cofw_annotation_assets_stay_separate(tmp_path, monkeypatch):
    output_root = tmp_path / "data"
    monkeypatch.setattr(
        downloader,
        "_download_url",
        lambda url, destination, *, force=False: _valid_zip(destination),
    )
    results = [
        downloader._process_asset(a, _args(output_root))
        for a in downloader.SOURCES
        if a.dataset in {"cofw68", "cofw29"}
    ]
    downloader.write_registry(results, output_root)
    registry = downloader.load_registry(output_root)

    cofw68 = {a["name"] for a in registry["datasets"]["cofw68"]["assets"]}
    cofw29 = {a["name"] for a in registry["datasets"]["cofw29"]["assets"]}
    assert (
        "cofw6868 benchmark annotations" in cofw68
    )  # annotation asset distinct to cofw68
    assert "cofw6868 benchmark annotations" not in cofw29
    # Both datasets still have their own color-image registry entry.
    assert "cofw68 color images" in cofw68
    assert "cofw29 original color images" in cofw29


# ---------------------------------------------------------------------------
# Invalid archive detection
# ---------------------------------------------------------------------------
def test_invalid_archive_is_rejected_and_removed(tmp_path, monkeypatch):
    output_root = tmp_path / "data"

    def fake_html(url, destination, *, force=False):
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"<html><body>Download denied</body></html>")
        return destination

    monkeypatch.setattr(downloader, "_download_url", fake_html)
    with pytest.raises(ValueError, match="not a valid zip/tar archive"):
        downloader._process_asset(
            _cofw_color("cofw68"), _args(output_root, keep_going=False)
        )

    # The bogus archive must not be left where a later run would reuse it.
    assert not (output_root / "cofw68" / "archives" / "COFW_color.zip").exists()


def test_invalid_archive_is_not_reused(tmp_path):
    output_root = tmp_path / "data"
    bogus = output_root / "wflw" / "archives" / "WFLW_images.tar.gz"
    bogus.parent.mkdir(parents=True)
    bogus.write_bytes(b"<html>not a tarball</html>")
    assert (
        downloader._find_reusable_archive(_asset("wflw", gdrive=True), output_root)
        is None
    )


# ---------------------------------------------------------------------------
# Reuse audit: idempotent re-runs
# ---------------------------------------------------------------------------
def test_rerun_reuses_existing_url_archive(tmp_path, monkeypatch):
    output_root = tmp_path / "data"
    monkeypatch.setattr(
        downloader,
        "_download_url",
        lambda url, destination, *, force=False: _valid_zip(destination),
    )
    first = downloader._process_asset(_cofw_color("cofw68"), _args(output_root))
    assert first["status"] == "downloaded"

    monkeypatch.setattr(
        downloader,
        "_download_url",
        lambda *a, **k: (_ for _ in ()).throw(
            AssertionError("rerun must reuse, not re-download")
        ),
    )
    second = downloader._process_asset(_cofw_color("cofw68"), _args(output_root))
    assert second["status"] == "reused"


def test_force_redownloads_even_when_archive_present(tmp_path, monkeypatch):
    output_root = tmp_path / "data"
    _valid_zip(output_root / "cofw68" / "archives" / "COFW_color.zip")
    calls = {"n": 0}

    def counting(url, destination, *, force=False):
        calls["n"] += 1
        return _valid_zip(destination)

    monkeypatch.setattr(downloader, "_download_url", counting)
    result = downloader._process_asset(
        _cofw_color("cofw68"), _args(output_root, force=True)
    )
    assert result["status"] == "downloaded"
    assert calls["n"] == 1  # --force bypasses reuse


# ---------------------------------------------------------------------------
# HELEN automated annotation download + resolution
# ---------------------------------------------------------------------------
def test_helen_annotation_download_and_source_resolution(tmp_path, monkeypatch):
    output_root = tmp_path / "data"
    helen = _asset("helen")
    assert helen.url.endswith("annotations.json")
    assert helen.extract is False

    def fake_json(url, destination, *, force=False):
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps([[["sample.jpg", 256, 256], [[1.0, 2.0]] * 194]])
        )
        return destination

    monkeypatch.setattr(downloader, "_download_url", fake_json)
    result = downloader._process_asset(helen, _args(output_root, extract=True))
    assert result["status"] == "downloaded"
    assert "extracted" not in result  # single-file asset, not an archive

    downloader.write_registry([result], output_root)
    registry = downloader.load_registry(output_root)
    resolved = downloader.resolve_source_dir(registry, "helen", output_root)
    assert resolved == output_root / "helen" / "archives"
    assert (resolved / "annotations.json").is_file()


# ---------------------------------------------------------------------------
# Per-dataset audit overlay limits
# ---------------------------------------------------------------------------
def _make_json_source(extracted: Path, *, dataset: str, count: int) -> None:
    extracted.mkdir(parents=True, exist_ok=True)
    pts = [[float(x), float(x)] for x in np.linspace(40, 200, 68)]
    samples = []
    for idx in range(count):
        name = f"{dataset}_{idx}.jpg"
        assert cv2.imwrite(
            str(extracted / name), np.full((96, 96, 3), 100, dtype=np.uint8)
        )
        samples.append(
            {
                "sample_id": f"{dataset}/s{idx}",
                "image": name,
                "landmarks": pts,
                "source_schema": "2d_68",
            }
        )
    (extracted / "samples.json").write_text(
        json.dumps({"samples": samples}), encoding="utf-8"
    )


def test_audit_overlay_limit_applies_per_dataset(tmp_path):
    out = tmp_path / "out"
    _make_json_source(tmp_path / "wflwv", dataset="wflw-v", count=4)
    _make_json_source(tmp_path / "vw", dataset="300vw", count=4)

    builder.build(
        builder._parser().parse_args(
            [
                "--dataset",
                "wflw-v",
                "--source-dir",
                str(tmp_path / "wflwv"),
                "--output-dir",
                str(out),
            ]
        )
    )
    manifest = builder.build(
        builder._parser().parse_args(
            [
                "--dataset",
                "300vw",
                "--source-dir",
                str(tmp_path / "vw"),
                "--output-dir",
                str(out),
                "--manifest-mode",
                "merge",
            ]
        )
    )

    report_path = builder._write_visual_audit(manifest, out, limit=2)
    report = json.loads(report_path.read_text(encoding="utf-8"))

    per_dataset: dict[str, int] = {}
    for overlay in report["overlays"]:
        per_dataset[overlay["dataset"]] = per_dataset.get(overlay["dataset"], 0) + 1
        # Overlays are organized by dataset/schema on disk.
        assert f"/overlays/{overlay['dataset']}/" in overlay["overlay"].replace(
            "\\", "/"
        )
    assert per_dataset == {
        "wflw-v": 2,
        "300vw": 2,
    }  # up to N=2 per dataset, not 2 total


# ---------------------------------------------------------------------------
# HELEN end-to-end prepare flow (downloaded annotations + auto 300W cache)
# ---------------------------------------------------------------------------
def test_prepare_helen_resolves_annotations_and_300w_cache(tmp_path):
    data_root = tmp_path / "data" / "landmarks"
    output_root = tmp_path / "out"

    # Annotation asset as it would land after download (helen/archives/annotations.json).
    points = np.stack([np.linspace(16, 220, 194), np.linspace(24, 224, 194)], axis=1)
    annotations = data_root / "helen" / "archives" / "annotations.json"
    annotations.parent.mkdir(parents=True)
    annotations.write_text(
        json.dumps([[["sample.jpg", 256, 256], points.tolist()]]), encoding="utf-8"
    )

    # Existing 300W Helen image cache that prepare must resolve automatically.
    cache = data_root / "300w" / "extracted"
    img = cache / "helen" / "trainset" / "sample.jpg"
    img.parent.mkdir(parents=True)
    assert cv2.imwrite(str(img), np.full((256, 256, 3), 127, dtype=np.uint8))

    rc = prepare.main(
        [
            "--datasets",
            "helen",
            "--skip-download",
            "--data-root",
            str(data_root),
            "--output-root",
            str(output_root),
        ]
    )
    assert rc == 0

    manifest = json.loads((output_root / "manifest.json").read_text(encoding="utf-8"))
    sample = manifest["samples"][0]
    assert sample["dataset"] == "helen"
    assert sample["source_schema"] == "2d_194"
    assert Path(sample["image"]).is_file()
