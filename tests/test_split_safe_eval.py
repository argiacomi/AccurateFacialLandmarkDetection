import pytest

from lib.landmarks.evaluation.split_safe import (
    build_slice_report,
    entry_in_eval_split,
    metrics_for_nmes,
    validate_no_train_test_leakage,
)


def test_leave_one_dataset_out_splits_heldout_dataset():
    wflw = {"sample_id": "a", "dataset": "wflw", "image": "a.jpg", "landmarks": "a.npy"}
    cofw = {"sample_id": "b", "dataset": "cofw", "image": "b.jpg", "landmarks": "b.npy"}

    assert entry_in_eval_split(
        wflw,
        0,
        split="test",
        eval_mode="leave_one_dataset_out",
        heldout_datasets=("wflw",),
    )
    assert not entry_in_eval_split(
        cofw,
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
        cofw,
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


def test_empty_metrics_are_reportable():
    metrics = metrics_for_nmes([])

    assert metrics["sample_count"] == 0
    assert metrics["nme"] is None
    assert metrics["nme_ci95"] == {"low": None, "high": None}
