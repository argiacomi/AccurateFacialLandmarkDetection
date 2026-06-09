#!/usr/bin/env python3
"""Head-pose estimation and bucketing for landmark manifests.

Sign conventions (image space, x right / y down):

* ``yaw``   > 0  -> the subject turns toward image right.
* ``pitch`` > 0  -> the face tilts up.
* ``roll``  > 0  -> the head rolls clockwise in the image.

Annotation pose (e.g. AFLW2000-3D ``Pose_Para``) is passed through with the
source dataset's own convention; the geometry estimator below matches the same
signs so buckets stay consistent across sources.
"""

from __future__ import annotations

import typing as T

import numpy as np

# Yaw bucket edges in degrees: |yaw| < 15 frontal, < 30 slight, < 60 profile,
# else extreme. Pitch: |pitch| < 15 neutral, < 30 up/down, else *_extreme.
YAW_BUCKET_THRESHOLDS = (15.0, 30.0, 60.0)
PITCH_BUCKET_THRESHOLDS = (15.0, 30.0)


def _finite(value: T.Any) -> bool:
    try:
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


def yaw_bucket(yaw_deg: float | None) -> str:
    """Bucket a signed yaw angle into frontal / *_slight / *_profile / *_extreme.

    Returns ``"unknown"`` for missing/non-finite input so callers never turn an
    absence of evidence into a real ``frontal`` bucket.
    """
    if not _finite(yaw_deg):
        return "unknown"
    yaw = float(yaw_deg)
    magnitude = abs(yaw)
    slight, profile, extreme = YAW_BUCKET_THRESHOLDS
    if magnitude < slight:
        return "frontal"
    side = "right" if yaw > 0 else "left"
    if magnitude < profile:
        return f"{side}_slight"
    if magnitude < extreme:
        return f"{side}_profile"
    return f"{side}_extreme"


def yaw_tier(yaw_deg: float | None) -> str:
    """Side-agnostic yaw magnitude tier: frontal / slight / profile / extreme.

    Used when a yaw magnitude is known but its left/right side is not (e.g. a
    profile capture label with no geometry to disambiguate).
    """
    if not _finite(yaw_deg):
        return "unknown"
    magnitude = abs(float(yaw_deg))
    slight, profile, extreme = YAW_BUCKET_THRESHOLDS
    if magnitude < slight:
        return "frontal"
    if magnitude < profile:
        return "slight"
    if magnitude < extreme:
        return "profile"
    return "extreme"


def yaw_side(yaw_deg: float | None) -> str:
    """Resolve the turn direction: frontal / left / right / unknown."""
    if not _finite(yaw_deg):
        return "unknown"
    if abs(float(yaw_deg)) < YAW_BUCKET_THRESHOLDS[0]:
        return "frontal"
    return "right" if float(yaw_deg) > 0 else "left"


def pitch_bucket(pitch_deg: float | None) -> str:
    """Bucket a pitch angle into neutral / up|down / up|down_extreme.

    Returns ``"unknown"`` for missing/non-finite input rather than ``neutral``.
    """
    if not _finite(pitch_deg):
        return "unknown"
    pitch = float(pitch_deg)
    magnitude = abs(pitch)
    moderate, extreme = PITCH_BUCKET_THRESHOLDS
    if magnitude < moderate:
        return "neutral"
    direction = "up" if pitch > 0 else "down"
    if magnitude < extreme:
        return direction
    return f"{direction}_extreme"


def estimate_pose_from_68(
    points68: T.Sequence[T.Sequence[float]] | np.ndarray,
) -> tuple[float, float, float] | None:
    """Approximate (yaw, pitch, roll) degrees from a 68-point face.

    A cheap 2D heuristic for datasets without annotated pose: yaw from the
    horizontal offset of the nose tip between the jaw extremes, roll from the
    inter-eye line, and a coarse pitch proxy from the nose's vertical position
    between the eye line and the mouth. Scale-invariant, so it works on
    normalized or pixel coordinates. Returns ``None`` for degenerate input.
    """
    pts = np.asarray(points68, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[0] < 68 or pts.shape[1] < 2:
        return None
    xy = pts[:, :2]
    if not np.all(np.isfinite(xy)):
        return None

    jaw_left = xy[0]
    jaw_right = xy[16]
    nose_tip = xy[30]
    left_eye = xy[36:42].mean(axis=0)
    right_eye = xy[42:48].mean(axis=0)
    eye_center = (left_eye + right_eye) / 2.0
    mouth_center = xy[48:68].mean(axis=0)

    # Yaw: nose tip horizontal position between the jaw extremes. As the face
    # turns right the right jaw foreshortens, so d_right shrinks -> asym > 0.
    d_left = nose_tip[0] - jaw_left[0]
    d_right = jaw_right[0] - nose_tip[0]
    denom = d_left + d_right
    if abs(denom) < 1e-6:
        yaw = 0.0
    else:
        yaw = float(np.clip((d_left - d_right) / denom, -1.0, 1.0) * 90.0)

    # Roll: inter-eye line angle, folded into [-90, 90].
    roll = float(
        np.degrees(np.arctan2(right_eye[1] - left_eye[1], right_eye[0] - left_eye[0]))
    )
    if roll > 90.0:
        roll -= 180.0
    elif roll < -90.0:
        roll += 180.0

    # Pitch proxy: nose tip vertical position between eye line and mouth. ~0.55
    # is neutral; looking up moves the nose up (smaller y) -> ratio drops.
    span = mouth_center[1] - eye_center[1]
    if abs(span) < 1e-6:
        pitch = 0.0
    else:
        ratio = (nose_tip[1] - eye_center[1]) / span
        pitch = float(np.clip((0.55 - ratio) * 120.0, -45.0, 45.0))

    return yaw, pitch, roll
