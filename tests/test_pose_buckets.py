from __future__ import annotations

import numpy as np
import pytest

from lib.core import pose
from tools import build_quality_dataset as builder


# ---------------------------------------------------------------------------
# Bucketing
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("yaw", "expected"),
    [
        (0.0, "frontal"),
        (14.9, "frontal"),
        (15.0, "right_slight"),
        (-15.0, "left_slight"),
        (29.9, "right_slight"),
        (30.0, "right_profile"),
        (-45.0, "left_profile"),
        (59.9, "right_profile"),
        (60.0, "right_extreme"),
        (-90.0, "left_extreme"),
        (None, "unknown"),
        (float("nan"), "unknown"),
    ],
)
def test_yaw_bucket(yaw, expected):
    assert pose.yaw_bucket(yaw) == expected


@pytest.mark.parametrize(
    ("pitch", "expected"),
    [
        (0.0, "neutral"),
        (14.0, "neutral"),
        (15.0, "up"),
        (-15.0, "down"),
        (29.9, "up"),
        (30.0, "up_extreme"),
        (-40.0, "down_extreme"),
        (None, "unknown"),
    ],
)
def test_pitch_bucket(pitch, expected):
    assert pose.pitch_bucket(pitch) == expected


@pytest.mark.parametrize(
    ("yaw", "tier", "side"),
    [
        (0.0, "frontal", "frontal"),
        (10.0, "frontal", "frontal"),
        (20.0, "slight", "right"),
        (-45.0, "profile", "left"),
        (70.0, "extreme", "right"),
        (None, "unknown", "unknown"),
    ],
)
def test_yaw_tier_and_side(yaw, tier, side):
    assert pose.yaw_tier(yaw) == tier
    assert pose.yaw_side(yaw) == side


# ---------------------------------------------------------------------------
# Geometry estimator
# ---------------------------------------------------------------------------
def _base68() -> np.ndarray:
    pts = np.zeros((68, 2), dtype=np.float64)
    pts[0] = [-1.0, 1.0]  # left jaw
    pts[16] = [1.0, 1.0]  # right jaw
    pts[8] = [0.0, 1.6]  # chin
    for i in range(36, 42):
        pts[i] = [-0.5 + 0.04 * (i - 36), 0.0]
    for i in range(42, 48):
        pts[i] = [0.3 + 0.04 * (i - 42), 0.0]
    pts[30] = [0.0, 0.4]  # nose tip centered
    for i in range(48, 68):
        pts[i] = [-0.2 + 0.02 * (i - 48), 0.9]
    return pts


def test_estimate_pose_sign_convention():
    yaw_front, _, _ = pose.estimate_pose_from_68(_base68())
    assert pose.yaw_bucket(yaw_front) == "frontal"

    right = _base68()
    right[30] = [0.6, 0.4]  # nose toward right jaw
    yaw_right, _, _ = pose.estimate_pose_from_68(right)
    assert yaw_right > 0  # positive yaw == subject turns image-right

    left = _base68()
    left[30] = [-0.6, 0.4]
    yaw_left, _, _ = pose.estimate_pose_from_68(left)
    assert yaw_left < 0


def test_estimate_pose_handles_degenerate_input():
    assert pose.estimate_pose_from_68(np.zeros((10, 2))) is None
    bad = _base68()
    bad[0] = [np.nan, np.nan]
    assert pose.estimate_pose_from_68(bad) is None


# ---------------------------------------------------------------------------
# Source resolution (_pose_metadata)
# ---------------------------------------------------------------------------
def test_pose_metadata_prefers_annotation_angles():
    fields = builder._pose_metadata(
        "aflw2000-3d",
        _base68(),
        "2d_68",
        {"pose_yaw_deg": 40.0, "pose_pitch_deg": -20.0, "pose_roll_deg": 5.0},
    )
    assert fields["pose_source"] == "annotation"
    assert fields["pose_yaw_deg"] == 40.0
    assert fields["pose_abs_yaw_deg"] == 40.0
    assert fields["pose_side"] == "right"
    assert fields["pose_bucket"] == "right_profile"
    assert fields["pitch_bucket"] == "down"
    assert fields["pose_roll_deg"] == 5.0


def test_pose_metadata_multipie_label_without_side_is_unknown():
    # _base68() is near-frontal, so geometry cannot resolve a profile's side:
    # emit a side-agnostic magnitude bucket, not a guessed left/right.
    fields = builder._pose_metadata(
        "multipie",
        _base68(),
        "2d_68",
        {
            "image_id": "session01/profile/img.png",
            "annotation_file": "profile_train.txt",
        },
    )
    assert fields["pose_source"] == "dataset_label"
    assert fields["pose_side"] == "unknown"
    assert fields["pose_bucket"] == "extreme"
    assert fields["pose_abs_yaw_deg"] == 65.0
    assert "pose_yaw_deg" not in fields  # never guess a direction
    assert fields["pitch_bucket"] == "unknown"  # label has no pitch evidence


def test_pose_metadata_multipie_label_side_from_strong_geometry():
    turned = _base68()
    turned[30] = [0.6, 0.4]  # nose strongly toward the right jaw
    fields = builder._pose_metadata(
        "multipie",
        turned,
        "2d_68",
        {"image_id": "x/profile/img.png", "annotation_file": "profile.txt"},
    )
    assert fields["pose_source"] == "dataset_label"
    assert fields["pose_side"] == "right"
    assert fields["pose_yaw_deg"] == 65.0
    assert fields["pose_bucket"] == "right_extreme"


def test_pose_metadata_falls_back_to_geometry():
    fields = builder._pose_metadata("300w", _base68(), "2d_68", {})
    assert fields["pose_source"] == "landmark_geometry"
    assert "pose_bucket" in fields and "pitch_bucket" in fields
    assert fields["pose_side"] in {"frontal", "left", "right"}
    assert "pose_roll_deg" in fields


def test_pose_metadata_omitted_for_sparse_unprojectable_schema():
    # 2d_29 has no audited 68 projection and no label -> no pose fields.
    fields = builder._pose_metadata("cofw29", np.zeros((29, 2)), "2d_29", {})
    assert fields == {}


# ---------------------------------------------------------------------------
# End-to-end: WFLW (98-pt) build attaches geometry pose to each sample
# ---------------------------------------------------------------------------
def _write_image(path, size=(256, 256)):
    import cv2

    path.parent.mkdir(parents=True, exist_ok=True)
    assert cv2.imwrite(str(path), np.full((size[1], size[0], 3), 127, dtype=np.uint8))


def test_wflw_build_attaches_landmark_geometry_pose(tmp_path):
    source = tmp_path / "wflw"
    output = tmp_path / "out"
    image_rel = "0--Parade/sample.jpg"
    _write_image(source / "images" / image_rel)
    pts = np.stack(
        [np.linspace(20, 230, 98), np.linspace(30, 210, 98)], axis=1
    ).reshape(-1)
    bbox = [10.0, 10.0, 230.0, 230.0]
    attrs = [0, 0, 0, 0, 0, 0]
    line = " ".join(str(v) for v in [*pts.tolist(), *bbox, *attrs, image_rel])
    ann = source / "list_98pt_rect_attr_train.txt"
    ann.write_text(line + "\n", encoding="utf-8")

    manifest_path = builder.build(
        builder._parser().parse_args(
            [
                "--dataset",
                "wflw",
                "--wflw-annotations",
                str(ann),
                "--image-root",
                str(source / "images"),
                "--output-dir",
                str(output),
            ]
        )
    )
    sample = __import__("json").loads(manifest_path.read_text())["samples"][0]
    meta = sample["metadata"]
    assert meta["pose_source"] == "landmark_geometry"
    assert meta["pose_bucket"] in {
        "frontal",
        "left_slight",
        "right_slight",
        "left_profile",
        "right_profile",
        "left_extreme",
        "right_extreme",
    }
    assert meta["pitch_bucket"].startswith(("neutral", "up", "down"))
    assert meta["pose_side"] in {"frontal", "left", "right"}
    assert "pose_yaw_deg" in meta and "pose_roll_deg" in meta
