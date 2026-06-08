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


def _write_counted_txt(path: Path, points: np.ndarray) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"{points.shape[0]}\n" + "\n".join(f"{x} {y}" for x, y in points) + "\n",
        encoding="utf-8",
    )
    return path


def _write_pts(path: Path, points: np.ndarray) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"version: 1\nn_points: {points.shape[0]}\n{{\n"
        + "\n".join(f"{x} {y}" for x, y in points)
        + "\n}\n",
        encoding="utf-8",
    )
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
    ("dataset", "schema", "count", "head", "parser_name", "landmarks_rel", "image_rel"),
    [
        ("helen", "2d_194", 194, "landmarks_194", "helen_194", "annotation/sample.txt", "images/sample.jpg"),
        ("lapa", "2d_106", 106, "landmarks_106", "lapa_106", "train/landmarks/sample.txt", "train/images/sample.jpg"),
        ("jd-landmark", "2d_106", 106, "landmarks_106", "jd_landmark_106", "labels/sample.pts", "images/sample.jpg"),
        ("ffl2", "2d_106", 106, "landmarks_106", "ffl2_106", "landmarks/sample.pts", "images/sample.jpg"),
        ("fll3", "2d_106", 106, "landmarks_106", "fll3_106", "landmarks/sample.pts", "images/sample.jpg"),
    ],
)
def test_issue8_still_image_builders_use_dataset_specific_parsers(
    tmp_path,
    dataset,
    schema,
    count,
    head,
    parser_name,
    landmarks_rel,
    image_rel,
):
    source = tmp_path / "source"
    output = tmp_path / "out"
    _write_image(source / image_rel)
    points = _points(count)
    landmark_path = source / landmarks_rel
    if landmark_path.suffix == ".pts":
        _write_pts(landmark_path, points)
    else:
        _write_counted_txt(landmark_path, points)

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
    assert sample["metadata"]["dataset_parser"] == parser_name
    assert sample["metadata"]["parser_type"] == "dataset_specific"
    assert manifest["projection_status"] == {"not_projectable": 1}


def test_lapa_parser_rejects_non_106_point_landmarks(tmp_path):
    source = tmp_path / "source"
    output = tmp_path / "out"
    _write_image(source / "train" / "images" / "sample.jpg")
    _write_counted_txt(source / "train" / "landmarks" / "sample.txt", _points(68))

    with pytest.raises(ValueError, match="no lapa samples built"):
        builder.build(
            _builder_args(
                "--dataset",
                "lapa",
                "--source-dir",
                str(source),
                "--output-dir",
                str(output),
            )
        )


def test_cofw_original_json_path_preserves_29_point_visibility_and_head(tmp_path):
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


def test_cofw_original_mat_parser_preserves_29_point_visibility_and_occlusion(tmp_path):
    scipy = pytest.importorskip("scipy.io")
    source = tmp_path / "source"
    source.mkdir()
    output = tmp_path / "out"
    points = _points(29).reshape(1, 58)
    images = np.full((1, 64, 64, 3), 127, dtype=np.uint8)
    occlusions = np.zeros((1, 29), dtype=np.uint8)
    occlusions[0, 3] = 1
    scipy.savemat(
        source / "COFW_train_color.mat",
        {
            "phisTr": points,
            "IsTr": images,
            "occlusionsTr": occlusions,
        },
    )

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
    assert sample["split"] == "train"
    assert sample["visibility"][3] is False
    assert sample["metadata"]["occlusion_mask"][3] is True
    assert sample["metadata"]["dataset_parser"] == "cofw_original_29"
    assert manifest["projection_status"] == {"not_projectable": 1}


@pytest.mark.parametrize("dataset", ["xm2vts", "frgc"])
def test_xm2vts_and_frgc_use_menpo_style_subject_session_builder(tmp_path, dataset):
    source = tmp_path / "source"
    output = tmp_path / "out"
    rel = Path("subject-01") / "session-02" / "capture-03"
    _write_image(source / rel.with_suffix(".jpg"))
    _write_pts(source / rel.with_suffix(".pts"), _points(68))

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

    sample = _load_manifest(manifest_path)["samples"][0]

    assert sample["source_schema"] == "2d_68"
    assert sample["subject_id"] == "subject-01"
    assert sample["session_id"] == "session-02"
    assert sample["capture_id"] == "capture-03"
    assert sample["metadata"]["dataset_parser"] == f"{dataset}_menpo_style"
    assert sample["metadata"]["parser_type"] == "dataset_specific"


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


def test_write_overlays_generates_visual_audit_for_native_schema(tmp_path):
    source = tmp_path / "source"
    output = tmp_path / "out"
    _write_image(source / "images" / "sample.jpg")
    _write_counted_txt(source / "annotation" / "sample.txt", _points(194))

    builder.build(
        _builder_args(
            "--dataset",
            "helen",
            "--source-dir",
            str(source),
            "--output-dir",
            str(output),
            "--write-overlays",
            "--audit-overlay-limit",
            "1",
        )
    )

    report = _load_manifest(output / "visual_audit" / "visual_audit.json")

    assert report["schema_counts"] == {"2d_194": 1}
    assert report["overlay_count"] == 1
    assert Path(report["overlays"][0]["overlay"]).is_file()


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
