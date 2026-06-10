"""Regression tests for the 300W build after the build_quality_dataset split.

Covers the two things the refactor changed: that the dataset-specific build
logic now lives in the :mod:`lib.datasets.build` package (and is still reachable
through the ``tools.build_quality_dataset`` shim), and that the threaded crop
loop added to ``_build_directory`` produces output byte-identical to the serial
path while skipping the recursive image index for co-located 300W sources.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

import lib.datasets.build.core as core
from tools import build_quality_dataset as builder


def _write_pts(path: Path, points: np.ndarray) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"version: 1\nn_points: {points.shape[0]}\n{{\n"
        + "\n".join(f"{x:.3f} {y:.3f}" for x, y in points)
        + "\n}\n",
        encoding="utf-8",
    )
    return path


def _make_w300_source(root: Path, count: int, *, seed: int = 11) -> None:
    """Write ``count`` synthetic 300W samples (.pts next to its .jpg)."""
    rng = np.random.default_rng(seed)
    subset = root / "afw"
    subset.mkdir(parents=True, exist_ok=True)
    for i in range(count):
        height, width = 300 + (i % 3) * 32, 320 + (i % 2) * 24
        image = rng.integers(0, 255, (height, width, 3), dtype=np.uint8)
        assert cv2.imwrite(str(subset / f"image_{i:04d}.jpg"), image)
        points = np.stack(
            [rng.uniform(20, width - 20, 68), rng.uniform(20, height - 20, 68)],
            axis=1,
        ).astype(np.float32)
        _write_pts(subset / f"image_{i:04d}.pts", points)


def _build(source: Path, output: Path, *, workers: int) -> Path:
    args = builder._parser().parse_args(
        [
            "--dataset",
            "300w",
            "--source-dir",
            str(source),
            "--output-dir",
            str(output),
            "--workers",
            str(workers),
            "--log-level",
            "quiet",
        ]
    )
    return builder.build(args)


def _normalized_samples(manifest_path: Path) -> list[dict]:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    samples = sorted(payload["samples"], key=lambda s: s["sample_id"])
    normalized: list[dict] = []
    for sample in samples:
        sample = dict(sample)
        # Drop fields that legitimately encode the (per-run) output directory.
        sample.pop("image", None)
        sample.pop("landmarks", None)
        metadata = dict(sample.get("metadata", {}))
        for key in ("original_image", "source_landmarks", "source_file"):
            metadata.pop(key, None)
        sample["metadata"] = metadata
        normalized.append(sample)
    return normalized


def _artifact_hashes(output: Path, kind: str) -> dict[str, str]:
    directory = output / kind / "300w"
    return {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(directory.rglob("*"))
        if path.is_file()
    }


def test_build_package_layout_is_importable_via_shim():
    # Dataset-specific builders live in their own modules now.
    import lib.datasets.build.orchestrator  # noqa: F401
    import lib.datasets.build.w300  # noqa: F401
    from lib.datasets.build import wflw

    assert hasattr(wflw, "_build_wflw")
    # The shim must still expose the public + internal API callers depend on.
    for name in (
        "build",
        "main",
        "_parser",
        "SUPPORTED_DATASETS",
        "_write_manifest",
        "_load_landmark_file",
        "_sample",
        "_build_directory",
        "_build_helen",
    ):
        assert hasattr(builder, name), name


def test_parallel_300w_build_matches_serial(tmp_path):
    source = tmp_path / "src"
    _make_w300_source(source, count=12)

    serial_out = tmp_path / "serial"
    parallel_out = tmp_path / "parallel"
    serial_manifest = _build(source, serial_out, workers=1)
    parallel_manifest = _build(source, parallel_out, workers=4)

    assert _normalized_samples(serial_manifest) == _normalized_samples(
        parallel_manifest
    )
    assert _artifact_hashes(serial_out, "images") == _artifact_hashes(
        parallel_out, "images"
    )
    assert _artifact_hashes(serial_out, "landmarks") == _artifact_hashes(
        parallel_out, "landmarks"
    )
    assert len(json.loads(serial_manifest.read_text())["samples"]) == 12


def test_300w_build_skips_recursive_image_index(tmp_path, monkeypatch):
    source = tmp_path / "src"
    _make_w300_source(source, count=6)

    calls = {"n": 0}
    original = core._build_image_index

    def _counting(root):
        calls["n"] += 1
        return original(root)

    monkeypatch.setattr(core, "_build_image_index", _counting)
    _build(source, tmp_path / "out", workers=1)
    # 300W keeps the .pts beside its image, so the co-located lookup always hits
    # and the recursive index is never built.
    assert calls["n"] == 0
