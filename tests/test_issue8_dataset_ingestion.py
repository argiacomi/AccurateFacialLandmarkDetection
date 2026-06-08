from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from lib.landmarks.core.schema import projection_audit_for_schema
from lib.landmarks.datasets.video_frames import selected_frame_indices
from tools.landmarks import build_quality_dataset as builder


def _points(count: int) -> np.ndarray:
    return np.stack(
        [np.linspace(16, 220, count), np.linspace(24, 224, count)],
        axis=1,
    ).astype(np.float32)


def _write_image(path: Path, *, size: tuple[int, int] = (256, 256)) -> Path:
    image = np.full((size[1], size[0], 3), 127, dtype=np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    assert cv2.imwrite(str(path), image)
    return path


def _builder_args(*items: str):
    return builder._parser().parse_args(list(items))


def _load_manifest(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_issue8_dataset_choices_and_projection_statuses_are_registered():
    parser = builder._parser()
    choices = set(parser._option_string_actions["--dataset"].choices)

    assert {
        "helen",
        "lapa",
        "jd-landmark",
        "ffl2",
        "fll3",
        "cofw-original",
        "xm2vts",
        "frgc",
        "300vw",
        "wflw-v",
    }.issubset(choices)

    assert projection_audit_for_schema("2d_98")["status"] == "audited"
    for schema in ("2d_29", "2d_106", "2d_194"):
        audit = projection_audit_for_schema(schema)
        assert audit["status"] == "not_projectable"
        assert audit["target_schema"] == "2d_68"


@pytest.mark.parametrize(
    ("dataset", "schema", "count", "head"),
    [
        ("helen", "2d_194", 194, "landmarks_194"),
        ("lapa", "2d_106", 106, "landmarks_106"),
        ("ffl2", "2d_106", 106, "landmarks_106"),
    ],
)
def test_issue8_still_image_builders_preserve_native_schema(tmp_path, dataset, schema, count, head):
    source = tmp_path / "source"
    output = tmp_path / "out"
    _write_image(source / "sample.jpg")
    np.save(source / "sample.npy", _points(count))

    manifest_path = builder.build(
        _builder_args(
            "--dataset",
            dataset,
            "--source-dir",
            str(source),
            "--output-dir",
            str(output),
        )
    )

    manifest = _load_manifest(manifest_path)
    sample = manifest["samples"][0]
    saved_points = np.load(output / sample["landmarks"])

    assert saved_points.shape == (count, 2)
    assert sample["source_schema"] == schema
    assert sample["target_schema"] == schema
    assert sample["head_name"] == head
    assert sample["split"] in {"train", "test"}
    assert manifest["projection_status"] == {"not_projectable": 1}


def test_cofw_original_json_preserves_29_point_visibility_and_head(tmp_path):
    source = tmp_path / "source"
    output = tmp_path / "out"
    image_path = _write_image(source / "images" / "cofw.png")
    payload = {
        "samples": [
            {
                "sample_id": "cofw/original/0001",
                "dataset": "cofw-original",
                "image": str(image_path.relative_to(source)),
                "points": _points(29).tolist(),
                "source_schema": "2d_29",
                "visibility": ([1, 0] * 15)[:29],
                "metadata": {"subject_id": "cofw-subject"},
            }
        ]
    }
    (source / "samples.json").write_text(json.dumps(payload), encoding="utf-8")

    manifest_path = builder.build(
        _builder_args(
            "--dataset",
            "cofw-original",
            "--source-dir",
            str(source),
            "--output-dir",
            str(output),
        )
    )

    manifest = _load_manifest(manifest_path)
    sample = manifest["samples"][0]

    assert sample["source_schema"] == "2d_29"
    assert sample["head_name"] == "landmarks_29"
    assert sample["visibility"][:2] == [1, 0]
    assert sample["subject_id"] == "cofw-subject"
    assert manifest["heads"] == {"landmarks_29": 1}
    assert manifest["projection_status"] == {"not_projectable": 1}


def test_wflw_builder_keeps_native_98_points(tmp_path):
    source = tmp_path / "wflw"
    output = tmp_path / "out"
    image_rel = "0--Parade/sample.jpg"
    _write_image(source / "images" / image_rel)
    points = _points(98).reshape(-1).tolist()
    bbox = [10.0, 10.0, 230.0, 230.0]
    attrs = [0, 0, 0, 0, 0, 0]
    line = " ".join(str(value) for value in [*points, *bbox, *attrs, image_rel])
    ann = source / "list_98pt_rect_attr_train.txt"
    ann.write_text(line + "\n", encoding="utf-8")

    manifest_path = builder.build(
        _builder_args(
            "--dataset",
            "wflw",
            "--wflw-annotations",
            str(ann),
            "--image-root",
            str(source / "images"),
            "--output-dir",
            str(output),
        )
    )

    manifest = _load_manifest(manifest_path)
    sample = manifest["samples"][0]
    saved_points = np.load(output / sample["landmarks"])

    assert saved_points.shape == (98, 2)
    assert sample["target_schema"] == "2d_98"
    assert sample["head_name"] == "landmarks_98"
    assert manifest["projection_status"] == {"audited": 1}


def test_video_frame_extraction_and_300vw_manifest_are_video_split_safe(tmp_path):
    assert selected_frame_indices(10, stride=2, max_frames=3) == [0, 4, 8]

    source = tmp_path / "source"
    videos = source / "videos"
    videos.mkdir(parents=True)
    video_path = videos / "clip.avi"
    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        5.0,
        (64, 64),
    )
    assert writer.isOpened()
    for index in range(5):
        frame = np.full((64, 64, 3), 20 + index, dtype=np.uint8)
        writer.write(frame)
    writer.release()

    ann_dir = source / "annotations" / "clip"
    ann_dir.mkdir(parents=True)
    for frame_index in (0, 4):
        points = _points(68)
        (ann_dir / f"frame_{frame_index:06d}.pts").write_text(
            "version: 1\nn_points: 68\n{\n"
            + "\n".join(f"{x} {y}" for x, y in points)
            + "\n}\n",
            encoding="utf-8",
        )

    output = tmp_path / "out"
    manifest_path = builder.build(
        _builder_args(
            "--dataset",
            "300vw",
            "--source-dir",
            str(source),
            "--video-root",
            str(videos),
            "--output-dir",
            str(output),
            "--frame-stride",
            "2",
            "--max-frames-per-video",
            "2",
        )
    )

    manifest = _load_manifest(manifest_path)
    samples = manifest["samples"]

    assert len(samples) == 2
    assert {sample["video_id"] for sample in samples} == {"clip"}
    assert {sample["split_safe_id"] for sample in samples} == {"clip"}
    assert len({sample["split"] for sample in samples}) == 1
    assert {sample["frame_index"] for sample in samples} == {0, 4}
    assert manifest["projection_status"] == {"native": 2}
