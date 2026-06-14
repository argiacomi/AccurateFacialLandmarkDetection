import csv

import pytest

from lib.evaluation.split_safe import (
    build_slice_report,
    entry_in_eval_split,
    metrics_for_nmes,
    slice_labels,
    stable_random_hash_split,
    validate_no_train_test_leakage,
    write_eval_csv,
    write_eval_records_jsonl,
)


def test_leave_one_dataset_out_splits_heldout_dataset():
    wflw = {"sample_id": "a", "dataset": "wflw", "image": "a.jpg", "landmarks": "a.npy"}
    cofw68 = {
        "sample_id": "b",
        "dataset": "cofw68",
        "image": "b.jpg",
        "landmarks": "b.npy",
    }

    assert entry_in_eval_split(
        wflw,
        0,
        split="test",
        eval_mode="leave_one_dataset_out",
        heldout_datasets=("wflw",),
    )
    assert not entry_in_eval_split(
        cofw68,
        1,
        split="test",
        eval_mode="leave_one_dataset_out",
        heldout_datasets=("wflw",),
    )
    assert not entry_in_eval_split(
        wflw,
        0,
        split="train",
        eval_mode="leave_one_dataset_out",
        heldout_datasets=("wflw",),
    )
    assert entry_in_eval_split(
        cofw68,
        1,
        split="train",
        eval_mode="leave_one_dataset_out",
        heldout_datasets=("wflw",),
    )


def test_dataset_modes_require_heldout_dataset():
    with pytest.raises(ValueError, match="requires at least one --heldout-dataset"):
        entry_in_eval_split(
            {"sample_id": "a", "dataset": "wflw"},
            0,
            split="test",
            eval_mode="by_dataset",
            heldout_datasets=(),
        )
    with pytest.raises(ValueError, match="requires exactly one --heldout-dataset"):
        entry_in_eval_split(
            {"sample_id": "a", "dataset": "wflw"},
            0,
            split="test",
            eval_mode="leave_one_dataset_out",
            heldout_datasets=("wflw", "cofw68"),
        )
    assert entry_in_eval_split(
        {"sample_id": "a", "dataset": "wflw"},
        0,
        split="test",
        eval_mode="by_dataset",
        heldout_datasets=("wflw", "cofw68"),
    )


def test_split_policy_can_force_declared_or_hash_split():
    sample = None
    for index in range(1000):
        candidate = {"sample_id": f"sample-{index}", "dataset": "wflw", "split": "test"}
        if stable_random_hash_split(candidate, index) == "train":
            sample = candidate
            break
    assert sample is not None

    assert entry_in_eval_split(
        sample,
        0,
        split="test",
        eval_mode="random_hash",
        has_declared_splits=True,
        split_policy="declared",
    )
    assert not entry_in_eval_split(
        sample,
        0,
        split="test",
        eval_mode="random_hash",
        has_declared_splits=True,
        split_policy="random_hash",
    )


def test_leakage_check_fails_on_duplicate_image_or_landmark_sources():
    train = [
        {
            "sample_id": "train",
            "image": "/data/source/frame.jpg",
            "landmarks": "/data/source/points.npy",
        }
    ]
    test = [
        {
            "sample_id": "test",
            "image": "/data/source/frame.jpg",
            "landmarks": "/data/source/other_points.npy",
            "metadata": {"original_landmarks": "/data/source/points.npy"},
        }
    ]

    with pytest.raises(ValueError, match="train/test source leakage detected"):
        validate_no_train_test_leakage(train, test)


def test_leakage_check_catches_identity_sequence_and_archive_sources():
    train = [
        {
            "sample_id": "train",
            "image": "/data/train.jpg",
            "landmarks": "/data/train.npy",
            "metadata": {
                "subject_id": "person-1",
                "video_id": "video-1",
                "archive_path": "/archives/source.zip",
            },
        }
    ]
    test = [
        {
            "sample_id": "test",
            "image": "/data/test.jpg",
            "landmarks": "/data/test.npy",
            "metadata": {
                "person_id": "person-1",
                "clip_id": "video-1",
                "original_archive": "/archives/source.zip",
            },
        }
    ]

    with pytest.raises(ValueError) as err:
        validate_no_train_test_leakage(train, test)

    message = str(err.value)
    assert "duplicate_identity_count" in message
    assert "duplicate_sequence_count" in message
    assert "duplicate_archive_count" in message


def test_slice_report_includes_required_metrics_and_ci():
    report = build_slice_report(
        [
            {
                "nme": 0.02,
                "by_dataset": "wflw",
                "by_schema": "2d_98",
                "by_hard_negative_bucket": "profile",
                "by_pose_bucket": "profile_left",
                "by_occlusion": "no_occlusion",
                "by_profile_side": "left",
                "by_roll_bucket": "horizontal",
                "by_face_size": "medium",
                "by_production_source": "unknown",
            },
            {
                "nme": 0.04,
                "by_dataset": "wflw",
                "by_schema": "2d_98",
                "by_hard_negative_bucket": "occlusion",
                "by_pose_bucket": "frontal",
                "by_occlusion": "occlusion",
                "by_profile_side": "not_profile",
                "by_roll_bucket": "upright",
                "by_face_size": "large",
                "by_production_source": "unknown",
            },
        ]
    )

    overall = report["overall"]
    assert overall["sample_count"] == 2
    assert overall["nme"] == pytest.approx(0.03)
    assert overall["fr"] == pytest.approx(0.0)
    assert overall["auc"] is not None
    assert overall["nme_ci95"]["low"] is not None
    assert report["by_dataset"]["wflw"]["sample_count"] == 2
    assert report["by_roll_bucket"]["horizontal"]["sample_count"] == 1
    assert report["by_roll_bucket"]["upright"]["sample_count"] == 1


def test_slice_report_includes_visible_occluded_and_visibility_metrics():
    report = build_slice_report(
        [
            {
                "nme": 0.03,
                "nme_visible": 0.02,
                "nme_occluded": 0.05,
                "visible_landmark_count": 2,
                "occluded_landmark_count": 1,
                "visibility_label_skipped_count": 1,
                "visibility_targets": [1, 1, 0, -1],
                "visibility_scores": [0.95, 0.8, 0.1, 0.4],
                "by_dataset": "wflw",
                "by_schema": "2d_68",
                "by_hard_negative_bucket": "profile",
                "by_pose_bucket": "profile_left",
                "by_occlusion": "occlusion",
                "by_profile_side": "left",
                "by_face_size": "medium",
                "by_production_source": "unknown",
            },
            {
                "nme": 0.06,
                "nme_visible": 0.04,
                "nme_occluded": None,
                "visible_landmark_count": 1,
                "occluded_landmark_count": 0,
                "visibility_label_skipped_count": 2,
                "by_dataset": "cofw68",
                "by_schema": "2d_68",
                "by_hard_negative_bucket": "occlusion",
                "by_pose_bucket": "frontal",
                "by_occlusion": "occlusion",
                "by_profile_side": "not_profile",
                "by_face_size": "large",
                "by_production_source": "unknown",
            },
        ]
    )

    overall = report["overall"]
    assert overall["NME_all"] == pytest.approx(0.045)
    assert overall["NME_visible"] == pytest.approx(0.03)
    assert overall["NME_occluded"] == pytest.approx(0.05)
    assert overall["visible_landmark_count"] == 3
    assert overall["occluded_landmark_count"] == 1
    assert overall["visibility_label_skipped_count"] == 3
    assert overall["visibility_AP"] == pytest.approx(1.0)
    assert overall["visibility_F1@0.5"] == pytest.approx(1.0)
    assert overall["visibility_ROC_AUC"] == pytest.approx(1.0)
    assert report["by_dataset"]["wflw"]["NME_occluded"] == pytest.approx(0.05)


def test_eval_csv_includes_visibility_summary_fields(tmp_path):
    report = build_slice_report(
        [
            {
                "nme": 0.03,
                "nme_visible": 0.02,
                "nme_occluded": 0.05,
                "visible_landmark_count": 1,
                "occluded_landmark_count": 1,
                "visibility_label_skipped_count": 1,
                "visibility_targets": [1, 0, -1],
                "visibility_scores": [0.9, 0.2, 0.5],
                "by_dataset": "wflw",
            }
        ]
    )
    path = tmp_path / "summary.csv"

    write_eval_csv(path, {"model": report})

    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    overall = next(row for row in rows if row["slice"] == "overall")
    assert float(overall["NME_visible"]) == pytest.approx(0.02)
    assert float(overall["NME_occluded"]) == pytest.approx(0.05)
    assert overall["visibility_label_count"] == "2"
    assert overall["visibility_label_skipped_count"] == "1"
    assert overall["visibility_prediction_skipped_count"] == "0"


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("1", "occlusion"),
        ("true", "occlusion"),
        ("yes", "occlusion"),
        ("occluded", "occlusion"),
        ("0", "no_occlusion"),
        ("false", "no_occlusion"),
        ("no", "no_occlusion"),
        ("none", "no_occlusion"),
        ("clean", "no_occlusion"),
        ("clear", "no_occlusion"),
        ("possibly", "unknown"),
        (True, "occlusion"),
        (False, "no_occlusion"),
    ],
)
def test_occlusion_labels_are_normalized(value, expected):
    assert slice_labels({"occlusion": value})["by_occlusion"] == expected


def test_face_size_bucket_respects_explicit_bbox_format():
    assert (
        slice_labels({"bbox": [0, 0, 63, 63], "bbox_format": "xyxy"})["by_face_size"]
        == "small"
    )
    assert (
        slice_labels({"bbox": [20, 20, 90, 90], "bbox_format": "xywh"})["by_face_size"]
        == "medium"
    )
    assert slice_labels({"bbox": [0, 0, 63, 63]})["by_face_size"] == "unknown"
    assert (
        slice_labels({"bbox": {"x": 0, "y": 0, "w": 129, "h": 129}})["by_face_size"]
        == "large"
    )


@pytest.mark.parametrize(
    ("roll", "expected"),
    [
        (0.0, "upright"),
        (29.9, "upright"),
        (-30.0, "diagonal"),
        (69.9, "diagonal"),
        (70.0, "horizontal"),
        (-90.0, "horizontal"),
        (None, "unknown"),
    ],
)
def test_roll_bucket_uses_horizontal_face_thresholds(roll, expected):
    assert slice_labels({"pose_roll_deg": roll})["by_roll_bucket"] == expected


def test_eval_records_jsonl_can_be_written(tmp_path):
    path = tmp_path / "records.jsonl"
    write_eval_records_jsonl(
        path, [{"sample_id": "a", "nme": 0.1}, {"sample_id": "b", "nme": 0.2}]
    )

    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert '"sample_id": "a"' in lines[0]


def test_empty_metrics_are_reportable():
    metrics = metrics_for_nmes([])

    assert metrics["sample_count"] == 0
    assert metrics["nme"] is None
    assert metrics["nme_ci95"] == {"low": None, "high": None}
