from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from lib.core.schema import projection_audit_for_schema
from lib.datasets.video_frames import selected_frame_indices
from tools import build_quality_dataset as builder


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


def _menpo_list_row(image_rel: str, points: np.ndarray) -> str:
    bbox = [10.0, 20.0, 230.0, 240.0]
    coarse = points[:5].reshape(-1).tolist()
    values = [*bbox, *coarse, *points.reshape(-1).tolist()]
    return " ".join([image_rel, *(str(value) for value in values)])


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
        "fll2",
        "fll3",
        "cofw29",
        "xm2vts",
        "frgc",
        "300vw",
        "wflw-v",
        "wflwv",
    }.issubset(choices)

    assert projection_audit_for_schema("2d_98")["status"] == "audited"
    for schema in ("2d_29", "2d_106", "2d_194"):
        audit = projection_audit_for_schema(schema)
        assert audit["status"] == "not_projectable"
        assert audit["target_schema"] == "2d_68"


def test_helen_annotations_json_parser_uses_native_release_format(tmp_path):
    source = tmp_path / "source"
    cache = tmp_path / "300w" / "data" / "300w" / "300w"
    output = tmp_path / "out"
    points = _points(194)
    image_path = _write_image(cache / "helen" / "trainset" / "sample.jpg")
    source.mkdir(parents=True)
    (source / "annotations.json").write_text(
        json.dumps([[["sample.jpg", 256, 256], points.tolist()]]),
        encoding="utf-8",
    )

    manifest_path = builder.build(
        _builder_args(
            "--dataset",
            "helen",
            "--source-dir",
            str(source),
            "--image-root",
            str(cache),
            "--output-dir",
            str(output),
        )
    )

    manifest = _load_manifest(manifest_path)
    sample = manifest["samples"][0]
    saved_points = np.load(output / sample["landmarks"])

    assert saved_points.shape == (194, 2)
    assert sample["source_schema"] == "2d_194"
    assert sample["target_schema"] == "2d_194"
    assert sample["head_name"] == "landmarks_194"
    assert sample["metadata"]["dataset_parser"] == "helen_annotations_json"
    assert sample["metadata"]["parser_type"] == "dataset_specific"
    assert sample["metadata"]["image_width"] == 256
    assert sample["metadata"]["resolved_300w_image_path"] == str(image_path.resolve())
    assert manifest["projection_status"] == {"not_projectable": 1}


def test_helen_annotations_require_300w_cache(tmp_path):
    source = tmp_path / "source"
    output = tmp_path / "out"
    (source / "annotations.json").parent.mkdir(parents=True)
    (source / "annotations.json").write_text(
        json.dumps([[["sample.jpg", 256, 256], _points(194).tolist()]]),
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError, match="HELEN dense annotations require a 300W Helen image cache"
    ):
        builder.build(
            _builder_args(
                "--dataset",
                "helen",
                "--source-dir",
                str(source),
                "--output-dir",
                str(output),
            )
        )


def test_helen_annotations_reject_ambiguous_300w_image_matches(tmp_path):
    source = tmp_path / "source"
    cache = tmp_path / "300w" / "data" / "300w" / "300w"
    output = tmp_path / "out"
    _write_image(cache / "helen" / "trainset" / "sample.jpg")
    _write_image(cache / "helen" / "testset" / "sample.jpg")
    (source / "annotations.json").parent.mkdir(parents=True)
    (source / "annotations.json").write_text(
        json.dumps([[["sample.jpg", 256, 256], _points(194).tolist()]]),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="ambiguous"):
        builder.build(
            _builder_args(
                "--dataset",
                "helen",
                "--source-dir",
                str(source),
                "--image-root",
                str(cache),
                "--output-dir",
                str(output),
            )
        )


def test_lapa_release_parser_preserves_split_and_segmentation_label(tmp_path):
    source = tmp_path / "source"
    output = tmp_path / "out"
    points = _points(106)
    _write_image(source / "LaPa" / "train" / "images" / "sample.jpg")
    _write_image(source / "LaPa" / "train" / "labels" / "sample.png")
    _write_counted_txt(source / "LaPa" / "train" / "landmarks" / "sample.txt", points)

    manifest_path = builder.build(
        _builder_args(
            "--dataset",
            "lapa",
            "--source-dir",
            str(source),
            "--output-dir",
            str(output),
        )
    )

    sample = _load_manifest(manifest_path)["samples"][0]

    assert sample["split"] == "train"
    assert sample["source_schema"] == "2d_106"
    assert sample["head_name"] == "landmarks_106"
    assert sample["metadata"]["dataset_parser"] == "lapa_release_106"
    assert sample["metadata"]["source_split"] == "train"
    assert sample["metadata"]["semantic_label"].endswith("sample.png")


def test_jd_landmark_test_data_parser_pairs_jpg_txt_images_and_rects(tmp_path):
    source = tmp_path / "source"
    output = tmp_path / "out"
    points = _points(106)
    _write_image(source / "Test_data1" / "picture" / "sample.jpg")
    _write_counted_txt(source / "Test_data1" / "landmark" / "sample.jpg.txt", points)
    (source / "Test_data1" / "rect").mkdir(parents=True)
    (source / "Test_data1" / "rect" / "sample.jpg.rect").write_text(
        "10 20 210 220\n", encoding="utf-8"
    )

    manifest_path = builder.build(
        _builder_args(
            "--dataset",
            "jd-landmark",
            "--source-dir",
            str(source),
            "--output-dir",
            str(output),
        )
    )

    sample = _load_manifest(manifest_path)["samples"][0]

    assert sample["split"] == "test"
    assert sample["source_schema"] == "2d_106"
    assert sample["metadata"]["dataset_parser"] == "jd_landmark_release_106"
    assert sample["metadata"]["source_release"] == "test_data1"
    assert sample["metadata"]["bbox_xyxy"] == [10.0, 20.0, 210.0, 220.0]


def test_jd_landmark_maps_afw_annotation_names_to_300w_cache_and_applies_corrected_override(
    tmp_path,
):
    source = tmp_path / "source"
    cache = tmp_path / "300w" / "data" / "300w" / "300w"
    output = tmp_path / "out"
    image_path = _write_image(cache / "afw" / "134212_1.jpg")
    original = _points(106)
    corrected = original + 7.0
    name = "AFW_134212_1_0.jpg.txt"
    _write_counted_txt(source / "Test_data1" / "landmark" / name, original)
    _write_counted_txt(source / "Corrected_landmark" / name, corrected)
    bbox_dir = (
        source
        / "training_dataset_face_detection_bounding_box_v1"
        / "training_dataset_face_detection_bounding_box"
    )
    bbox_dir.mkdir(parents=True)
    (bbox_dir / "AFW_134212_1_0.jpg.rect").write_text(
        "10 20 210 220\n", encoding="utf-8"
    )

    manifest_path = builder.build(
        _builder_args(
            "--dataset",
            "jd-landmark",
            "--source-dir",
            str(source),
            "--image-root",
            str(cache),
            "--output-dir",
            str(output),
        )
    )

    manifest = _load_manifest(manifest_path)
    sample = manifest["samples"][0]
    saved_points = np.load(output / sample["landmarks"])

    assert len(manifest["samples"]) == 1
    assert np.allclose(saved_points, corrected)
    assert sample["image"] == str(image_path.resolve())
    assert sample["metadata"]["resolved_image_source"] == "300w_cache"
    assert sample["metadata"]["resolved_300w_image_path"] == str(image_path.resolve())
    assert sample["metadata"]["base_subset"] == "afw"
    assert sample["metadata"]["corrected_annotation"].endswith(
        "Corrected_landmark/AFW_134212_1_0.jpg.txt"
    )
    assert sample["metadata"]["bbox_xyxy"] == [10.0, 20.0, 210.0, 220.0]


def test_jd_landmark_corrected_annotations_require_300w_cache_images(tmp_path):
    source = tmp_path / "source"
    output = tmp_path / "out"
    _write_counted_txt(
        source / "Corrected_landmark" / "AFW_134212_1_0.jpg.txt", _points(106)
    )

    with pytest.raises(ValueError, match="JD-landmark requires a 300W image cache"):
        builder.build(
            _builder_args(
                "--dataset",
                "jd-landmark",
                "--source-dir",
                str(source),
                "--output-dir",
                str(output),
            )
        )


def test_jd_landmark_rejects_ambiguous_300w_cache_matches(tmp_path):
    source = tmp_path / "source"
    cache = tmp_path / "300w" / "data" / "300w" / "300w"
    output = tmp_path / "out"
    _write_counted_txt(
        source / "Corrected_landmark" / "LFPW_image_test_0237_0.jpg.txt", _points(106)
    )
    _write_image(cache / "lfpw" / "trainset" / "image_0237.png")
    _write_image(cache / "lfpw" / "testset" / "image_0237.png")

    with pytest.raises(ValueError, match="ambiguous"):
        builder.build(
            _builder_args(
                "--dataset",
                "jd-landmark",
                "--source-dir",
                str(source),
                "--image-root",
                str(cache),
                "--output-dir",
                str(output),
            )
        )


@pytest.mark.parametrize(
    ("dataset", "landmark_rel", "image_rel", "bbox_rel"),
    [
        (
            "fll2",
            "train/landmark/sample.txt",
            "train/picture/sample.jpg",
            "train/bbox/sample.txt",
        ),
        (
            "fll3",
            "FLL3_dataset/train/landmark/sample.txt",
            "FLL3_dataset/train/picture_mask/sample.jpg",
            "FLL3_dataset/train/bbox/sample.txt",
        ),
    ],
)
def test_ffl_release_parsers_pair_landmarks_pictures_and_bboxes(
    tmp_path,
    dataset,
    landmark_rel,
    image_rel,
    bbox_rel,
):
    source = tmp_path / "source"
    output = tmp_path / "out"
    _write_image(source / image_rel)
    _write_counted_txt(source / landmark_rel, _points(106))
    (source / bbox_rel).parent.mkdir(parents=True, exist_ok=True)
    (source / bbox_rel).write_text("10 20 210 220\n", encoding="utf-8")

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

    assert sample["split"] == "train"
    assert sample["source_schema"] == "2d_106"
    assert sample["head_name"] == "landmarks_106"
    assert sample["metadata"]["dataset_parser"] == f"{dataset}_release_106"
    assert sample["metadata"]["bbox_xyxy"] == [10.0, 20.0, 210.0, 220.0]


def test_lapa_parser_rejects_non_106_point_landmarks(tmp_path):
    source = tmp_path / "source"
    output = tmp_path / "out"
    _write_image(source / "LaPa" / "train" / "images" / "sample.jpg")
    _write_counted_txt(
        source / "LaPa" / "train" / "landmarks" / "sample.txt", _points(68)
    )

    with pytest.raises(ValueError, match="no LaPa native release samples built"):
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


def test_cofw68_original_json_path_preserves_29_point_visibility_and_head(tmp_path):
    source = tmp_path / "source"
    output = tmp_path / "out"
    image_path = _write_image(source / "images" / "cofw68.png")
    payload = {
        "samples": [
            {
                "sample_id": "cofw68/original/0001",
                "dataset": "cofw29",
                "image": str(image_path.relative_to(source)),
                "points": _points(29).tolist(),
                "source_schema": "2d_29",
                "visibility": ([1, 0] * 15)[:29],
                "metadata": {"subject_id": "cofw68-subject"},
            }
        ]
    }
    (source / "samples.json").write_text(json.dumps(payload), encoding="utf-8")

    manifest_path = builder.build(
        _builder_args(
            "--dataset",
            "cofw29",
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
    assert sample["subject_id"] == "cofw68-subject"
    assert manifest["heads"] == {"landmarks_29": 1}
    assert manifest["projection_status"] == {"not_projectable": 1}


def test_cofw68_original_mat_parser_preserves_29_point_visibility_and_occlusion(
    tmp_path,
):
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
            "cofw29",
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


def test_cofw68_original_hdf5_mat_parser_reads_native_caltech_color_release(tmp_path):
    h5py = pytest.importorskip("h5py")
    source = tmp_path / "source"
    source.mkdir()
    output = tmp_path / "out"
    points = _points(29)
    phis = np.zeros((87, 1), dtype=np.float32)
    phis[:29, 0] = points[:, 0]
    phis[29:58, 0] = points[:, 1]
    phis[58 + 3, 0] = 1.0
    image = np.full((3, 64, 64), 127, dtype=np.uint8)

    with h5py.File(source / "COFW_train_color.mat", "w") as handle:
        image_ds = handle.create_dataset("image_0000", data=image)
        refs = handle.create_dataset("IsTr", (1, 1), dtype=h5py.ref_dtype)
        refs[0, 0] = image_ds.ref
        handle.create_dataset("phisTr", data=phis)
        handle.create_dataset(
            "bboxesTr", data=np.asarray([[1], [2], [30], [40]], dtype=np.float32)
        )

    manifest_path = builder.build(
        _builder_args(
            "--dataset",
            "cofw29",
            "--source-dir",
            str(source),
            "--output-dir",
            str(output),
        )
    )

    sample = _load_manifest(manifest_path)["samples"][0]

    assert sample["source_schema"] == "2d_29"
    assert sample["visibility"][3] is False
    assert sample["metadata"]["bbox_xyxy"] == [1.0, 2.0, 30.0, 40.0]
    assert Path(sample["image"]).is_file()
    assert sample["metadata"]["dataset_parser"] == "cofw_original_29"


def test_cofw68_original_hdf5_image_is_reoriented_to_annotation_frame(tmp_path):
    h5py = pytest.importorskip("h5py")
    source = tmp_path / "source"
    source.mkdir()
    output = tmp_path / "out"

    # 29 points inside a 40(H) x 60(W) annotation frame.
    phis = np.zeros((87, 1), dtype=np.float32)
    phis[:29, 0] = np.linspace(2, 58, 29)  # x in [0, 60)
    phis[29:58, 0] = np.linspace(2, 38, 29)  # y in [0, 40)

    # HDF5/MATLAB layout is channel-first and H/W swapped vs the annotation frame,
    # so input[c, x, y] maps to output[y, x, c]: build a (C=3, W=60, H=40) plane.
    image = np.zeros((3, 60, 40), dtype=np.uint8)
    image[:, 5, 10] = 255  # white marker at annotation (x=5, y=10)

    with h5py.File(source / "COFW_train_color.mat", "w") as handle:
        image_ds = handle.create_dataset("image_0000", data=image)
        refs = handle.create_dataset("IsTr", (1, 1), dtype=h5py.ref_dtype)
        refs[0, 0] = image_ds.ref
        handle.create_dataset("phisTr", data=phis)

    manifest_path = builder.build(
        _builder_args(
            "--dataset",
            "cofw29",
            "--source-dir",
            str(source),
            "--output-dir",
            str(output),
        )
    )
    sample = _load_manifest(manifest_path)["samples"][0]
    saved = cv2.imread(sample["image"], cv2.IMREAD_COLOR)

    # Annotation frame is 40 rows x 60 cols; a transposed image would be 60x40.
    assert saved.shape[:2] == (40, 60)
    # The marker at annotation (x=5, y=10) must land at image[row=10, col=5].
    assert saved[10, 5].min() > 200
    assert saved[0, 0].max() < 50


def test_overlay_uses_small_uniform_visibility_colors(tmp_path):
    image_path = tmp_path / "img.png"
    assert cv2.imwrite(str(image_path), np.zeros((256, 256, 3), dtype=np.uint8))
    # Five well-separated points; index 3 is occluded, the rest visible.
    points = np.asarray(
        [[40, 40], [80, 40], [120, 40], [160, 40], [200, 40]], dtype=np.float32
    )
    landmarks_path = tmp_path / "lmk.npy"
    np.save(landmarks_path, points)
    visibility = [True, True, True, False, True]

    out = tmp_path / "overlay.png"
    builder._draw_manifest_overlay(
        image_path, landmarks_path, out, visibility=visibility
    )
    drawn = cv2.imread(str(out), cv2.IMREAD_COLOR)  # BGR

    def is_green(px):
        b, g, r = (int(v) for v in px)
        return g > 180 and r < 100 and b < 100

    def is_red(px):
        b, g, r = (int(v) for v in px)
        return r > 180 and g < 100 and b < 100

    # Visible points are green -- including index 0, which the old code colored
    # differently for every 5th point.
    assert is_green(drawn[40, 40])
    assert is_green(drawn[40, 80])
    assert is_green(drawn[40, 120])
    assert is_green(drawn[40, 200])
    # The occluded point is red.
    assert is_red(drawn[40, 160])
    # Points are small: a pixel a few px away from a center is background.
    assert drawn[55, 40].max() < 40


@pytest.mark.parametrize(
    ("dataset", "release_dir", "list_name", "image_rel", "expected_identity"),
    [
        (
            "xm2vts",
            "XM2VTS",
            "xm2vts_train.txt",
            "image/161_4_1.jpg",
            ("161", "4", "1"),
        ),
        (
            "frgc",
            "FRGC",
            "frgc_train.txt",
            "image/02463d453.jpg",
            ("02463", "d", "453"),
        ),
    ],
)
def test_xm2vts_and_frgc_use_native_menpo_train_list_builder(
    tmp_path,
    dataset,
    release_dir,
    list_name,
    image_rel,
    expected_identity,
):
    source = tmp_path / "source"
    output = tmp_path / "out"
    release = source / release_dir
    points = _points(68)
    _write_image(release / image_rel)
    (release / list_name).write_text(
        _menpo_list_row(image_rel, points) + "\n", encoding="utf-8"
    )

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
    subject_id, session_id, capture_id = expected_identity

    assert sample["source_schema"] == "2d_68"
    assert sample["split"] == "train"
    assert sample["subject_id"] == subject_id
    assert sample["session_id"] == session_id
    assert sample["capture_id"] == capture_id
    assert sample["metadata"]["dataset_parser"] == f"{dataset}_menpo_list_68"
    assert sample["metadata"]["parser_type"] == "dataset_specific"
    assert sample["metadata"]["bbox_xyxy"] == [10.0, 20.0, 230.0, 240.0]
    assert len(sample["metadata"]["five_point_landmarks"]) == 5


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


def test_samples_per_scenario_prunes_unreferenced_crop_artifacts(tmp_path):
    source = tmp_path / "wflw"
    output = tmp_path / "out"
    points = _points(98).reshape(-1).tolist()
    bbox = [10.0, 10.0, 230.0, 230.0]
    attrs = [0, 0, 0, 0, 0, 0]
    candidate_count = 6
    lines = []
    for index in range(candidate_count):
        image_rel = f"0--Parade/sample_{index}.jpg"
        _write_image(source / "images" / image_rel)
        lines.append(
            " ".join(str(value) for value in [*points, *bbox, *attrs, image_rel])
        )
    ann = source / "list_98pt_rect_attr_train.txt"
    ann.write_text("\n".join(lines) + "\n", encoding="utf-8")

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
            "--samples-per-scenario",
            "1",
        )
    )

    manifest = _load_manifest(manifest_path)
    kept = len(manifest["samples"])
    # The limit must actually drop candidates, otherwise the prune path is not
    # exercised. All candidates share one condition, so limit 1 keeps exactly 1.
    assert 0 < kept < candidate_count

    # On-disk crops and landmarks must match the filtered manifest, not the full
    # unfiltered candidate set: --samples-per-scenario only trimmed the manifest,
    # leaving orphaned crops/landmarks on disk before the prune step.
    image_files = [p for p in (output / "images" / "wflw").glob("*") if p.is_file()]
    landmark_files = [
        p for p in (output / "landmarks" / "wflw").glob("*") if p.is_file()
    ]
    assert len(image_files) == kept
    assert len(landmark_files) == kept

    for sample in manifest["samples"]:
        assert Path(sample["image"]).is_file()
        assert (output / sample["landmarks"]).is_file()


def test_write_overlays_generates_visual_audit_for_native_schema(tmp_path):
    source = tmp_path / "source"
    cache = tmp_path / "300w" / "data" / "300w" / "300w"
    output = tmp_path / "out"
    points = _points(194)
    _write_image(cache / "helen" / "trainset" / "sample.jpg")
    source.mkdir(parents=True)
    (source / "annotations.json").write_text(
        json.dumps([[["sample.jpg", 256, 256], points.tolist()]]),
        encoding="utf-8",
    )

    builder.build(
        _builder_args(
            "--dataset",
            "helen",
            "--source-dir",
            str(source),
            "--image-root",
            str(cache),
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


def test_split_marker_is_secondary_condition_not_primary_bucket():
    # A path with a "train" split directory must not make "trainset" the primary
    # hard-negative bucket; it is recorded only as a secondary condition.
    primary, conds = builder._condition_for_landmark_file(
        "fll2", Path("train/landmark/sample.txt"), "default"
    )
    assert primary != "trainset"
    assert conds[0] != "trainset"
    assert "trainset" in conds

    # A real visual token still wins the primary bucket, split stays secondary.
    primary2, conds2 = builder._condition_for_landmark_file(
        "wflw", Path("profile/train/img.txt"), "default"
    )
    assert primary2 == "profile"
    assert "trainset" in conds2


def test_dataset_condition_label_buckets_by_dataset():
    assert builder._dataset_condition_label("fll2") == "fll2"
    assert builder._dataset_condition_label("jd-landmark") == "jd_landmark"
    assert builder._dataset_condition_label("merl-rav") == "merl_rav"


def test_fll_native_build_uses_dataset_bucket_not_trainset(tmp_path):
    source = tmp_path / "source"
    output = tmp_path / "out"
    _write_image(source / "train" / "picture" / "sample.jpg")
    _write_counted_txt(source / "train" / "landmark" / "sample.txt", _points(106))
    (source / "train" / "bbox").mkdir(parents=True)
    (source / "train" / "bbox" / "sample.txt").write_text(
        "10 20 210 220\n", encoding="utf-8"
    )

    manifest_path = builder.build(
        _builder_args(
            "--dataset",
            "fll2",
            "--source-dir",
            str(source),
            "--output-dir",
            str(output),
        )
    )

    sample = _load_manifest(manifest_path)["samples"][0]
    # "default" is remapped to the dataset bucket, "trainset" kept secondary.
    assert sample["condition"] == "fll2"
    assert sample["condition"] != "trainset"
    assert "trainset" in sample["conditions"]
    assert sample["split"] == "train"


def _stage_lapa(root: Path, *, stem: str = "s") -> None:
    _write_image(root / "LaPa" / "train" / "images" / f"{stem}.jpg")
    _write_counted_txt(
        root / "LaPa" / "train" / "landmarks" / f"{stem}.txt", _points(106)
    )


def _stage_fll2(root: Path, *, stem: str = "s") -> None:
    _write_image(root / "train" / "picture" / f"{stem}.jpg")
    _write_counted_txt(root / "train" / "landmark" / f"{stem}.txt", _points(106))


def test_single_dataset_manifest_metadata_keeps_dataset(tmp_path):
    output = tmp_path / "out"
    src = tmp_path / "lapa"
    _stage_lapa(src)
    manifest_path = builder.build(
        _builder_args(
            "--dataset", "lapa", "--source-dir", str(src), "--output-dir", str(output)
        )
    )
    manifest = _load_manifest(manifest_path)
    assert manifest["metadata"]["dataset"] == "lapa"


def test_merged_manifest_metadata_marks_multi_dataset(tmp_path):
    output = tmp_path / "out"
    _stage_lapa(tmp_path / "lapa")
    builder.build(
        _builder_args(
            "--dataset",
            "lapa",
            "--source-dir",
            str(tmp_path / "lapa"),
            "--output-dir",
            str(output),
        )
    )

    _stage_fll2(tmp_path / "fll2")
    manifest_path = builder.build(
        _builder_args(
            "--dataset",
            "fll2",
            "--source-dir",
            str(tmp_path / "fll2"),
            "--output-dir",
            str(output),
            "--manifest-mode",
            "merge",
        )
    )

    manifest = _load_manifest(manifest_path)
    assert {s["dataset"] for s in manifest["samples"]} == {"lapa", "fll2"}
    # Top-level dataset must reflect the multi-dataset manifest, not the last one.
    assert manifest["metadata"]["dataset"] == "multi_dataset"
