from __future__ import annotations

import json
import time
from pathlib import Path

import cv2
import numpy as np
import pytest

from lib.datasets.parallel import parallel_map, resolve_worker_count
from tools import build_quality_dataset as builder


# ---------------------------------------------------------------------------
# parallel_map helper
# ---------------------------------------------------------------------------
def test_resolve_worker_count_clamps():
    assert resolve_worker_count(1, 10) == 1
    assert resolve_worker_count(8, 3) == 3  # never exceeds item count
    assert resolve_worker_count(4, 10) == 4
    assert resolve_worker_count(0, 10) >= 1  # <=0 means "all CPUs"
    assert resolve_worker_count(None, 0) == 1


def test_parallel_map_matches_sequential_and_preserves_order():
    items = list(range(20))
    sequential = parallel_map(lambda x: x * x, items, workers=1, desc="seq")
    parallel = parallel_map(lambda x: x * x, items, workers=4, desc="par")
    assert sequential == [x * x for x in items]
    assert parallel == sequential  # order preserved regardless of worker count


def test_parallel_map_preserves_order_when_tasks_finish_out_of_order():
    # Earlier items sleep longer, so completion order != submission order.
    def work(x: int) -> int:
        time.sleep((10 - x) * 0.01)
        return x

    assert parallel_map(work, list(range(10)), workers=5, desc="ooo") == list(range(10))


def test_parallel_map_propagates_exceptions():
    def boom(x: int) -> int:
        if x == 3:
            raise ValueError("kaboom")
        return x

    with pytest.raises(ValueError, match="kaboom"):
        parallel_map(boom, list(range(6)), workers=4, desc="err")


# ---------------------------------------------------------------------------
# Video frame extraction parallelism (the named bottleneck)
# ---------------------------------------------------------------------------
def _points68() -> np.ndarray:
    return np.stack([np.linspace(8, 56, 68), np.linspace(8, 56, 68)], axis=1).astype(
        np.float32
    )


def _write_pts(path: Path, points: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"version: 1\nn_points: {points.shape[0]}\n{{\n"
        + "\n".join(f"{x} {y}" for x, y in points)
        + "\n}\n",
        encoding="utf-8",
    )


def _write_video(path: Path, *, frames: int = 4, size: int = 64) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"MJPG"), 5.0, (size, size)
    )
    if not writer.isOpened():
        return False
    try:
        for idx in range(frames):
            frame = np.full((size, size, 3), (idx * 30) % 255, dtype=np.uint8)
            writer.write(frame)
    finally:
        writer.release()
    return True


def _build_video_root(tmp_path: Path) -> Path:
    root = tmp_path / "videos"
    points = _points68()
    for clip in ("clipA", "clipB"):
        if not _write_video(root / f"{clip}.avi"):
            pytest.skip("MJPG VideoWriter unavailable in this OpenCV build")
        for frame_index in range(4):
            _write_pts(root / clip / f"{frame_index:06d}.pts", points)
    return root


def _manifest_fingerprint(manifest_path: Path) -> list[tuple]:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = [
        (
            s["sample_id"],
            s["dataset"],
            s["split"],
            s["target_schema"],
            s["landmark_count"],
            Path(s["image"]).name,
        )
        for s in payload["samples"]
    ]
    return sorted(rows)


def test_video_build_is_deterministic_across_worker_counts(tmp_path):
    root = _build_video_root(tmp_path)

    def build(out: str, workers: int) -> Path:
        return builder.build(
            builder._parser().parse_args(
                [
                    "--dataset",
                    "wflw-v",
                    "--source-dir",
                    str(root),
                    "--output-dir",
                    str(tmp_path / out),
                    "--workers",
                    str(workers),
                ]
            )
        )

    one = _manifest_fingerprint(build("out1", 1))
    many = _manifest_fingerprint(build("out4", 4))
    assert one == many
    assert one  # frames were actually extracted and matched to landmarks


# ---------------------------------------------------------------------------
# Overlay rendering parallelism
# ---------------------------------------------------------------------------
def _build_overlay_manifest(tmp_path: Path) -> tuple[Path, Path]:
    source = tmp_path / "src"
    source.mkdir()
    samples = []
    pts = [[float(x), float(x)] for x in np.linspace(40, 210, 68)]
    for idx in range(6):
        name = f"img_{idx}.jpg"
        img = np.full((128, 128, 3), 90, dtype=np.uint8)
        assert cv2.imwrite(str(source / name), img)
        samples.append(
            {
                "sample_id": f"s{idx}",
                "image": name,
                "landmarks": pts,
                "source_schema": "2d_68",
            }
        )
    (source / "samples.json").write_text(
        json.dumps({"samples": samples}), encoding="utf-8"
    )
    out = tmp_path / "out"
    manifest = builder.build(
        builder._parser().parse_args(
            [
                "--dataset",
                "300vw",
                "--source-dir",
                str(source),
                "--output-dir",
                str(out),
            ]
        )
    )
    return manifest, out


def _overlay_fingerprint(report_path: Path) -> list[tuple]:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    return sorted((o["sample_id"], Path(o["overlay"]).name) for o in report["overlays"])


def test_overlay_audit_is_deterministic_across_worker_counts(tmp_path):
    manifest, out = _build_overlay_manifest(tmp_path)

    serial = builder._write_visual_audit(manifest, out / "a", limit=10, max_workers=1)
    parallel = builder._write_visual_audit(manifest, out / "b", limit=10, max_workers=4)

    fp_serial = _overlay_fingerprint(serial)
    fp_parallel = _overlay_fingerprint(parallel)
    assert fp_serial == fp_parallel
    assert len(fp_serial) == 6
    # Overlay images were actually written by the parallel run.
    report = json.loads(parallel.read_text(encoding="utf-8"))
    assert all(Path(o["overlay"]).is_file() for o in report["overlays"])
