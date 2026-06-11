"""Training-loader geometry simulation helpers.

These helpers intentionally mirror ``LandmarkDataset._load_image_and_landmarks``
and ``LandmarkDataset.MakeLMKInsideImage``. Builders, staging tools, and
manifest validation use them to catch image/landmark coordinate-frame mismatches
before training DataLoader workers crash.
"""

from __future__ import annotations

import math
import typing as T
from pathlib import Path

import cv2
import numpy as np


LOADER_IMAGE_SIZE = 256
LOADER_PADDING_MARGIN = 5.0
LOADER_MAX_PADDED_SIDE = 2048
LOADER_MAX_PADDED_PIXELS = 2048 * 2048


def image_hw(path: str | Path) -> tuple[int, int]:
    """Return ``(height, width)`` for an image path using the loader decoder."""

    image_path = Path(path)
    img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(str(image_path))
    h, w = img.shape[:2]
    return int(h), int(w)


def _resolve_path(base_dir: Path, value: T.Any) -> Path:
    raw = str(value or "")
    p = Path(raw)
    return p if p.is_absolute() else (base_dir / p).resolve()


def prepared_image_is_usable(path: str | Path) -> bool:
    """Return True only when the training loader would use ``prepared_image``."""

    try:
        return image_hw(path) == (LOADER_IMAGE_SIZE, LOADER_IMAGE_SIZE)
    except (FileNotFoundError, OSError):
        return False


def resolve_loader_source_hw(
    sample: T.Mapping[str, T.Any],
    *,
    base_dir: str | Path = ".",
) -> tuple[tuple[int, int] | None, str, str | None]:
    """Resolve the image size the training loader will use for geometry.

    The real loader only takes the prepared fast path when ``prepared_image`` is
    readable and exactly 256x256 and ``prepared_image_orig_hw`` is present.
    Otherwise it falls back to the native image path. This helper mirrors that
    choice so validators do not validate against stale prepared metadata.
    """

    base = Path(base_dir)

    prepared = sample.get("prepared_image")
    prepared_orig_hw = sample.get("prepared_image_orig_hw")
    if prepared and prepared_orig_hw:
        prepared_path = _resolve_path(base, prepared)
        if prepared_image_is_usable(prepared_path):
            try:
                hw = (int(prepared_orig_hw[0]), int(prepared_orig_hw[1]))
            except Exception as err:  # noqa: BLE001
                return (
                    None,
                    "prepared_image_orig_hw",
                    f"invalid_prepared_image_orig_hw:{err}",
                )
            if hw[0] <= 0 or hw[1] <= 0:
                return (
                    None,
                    "prepared_image_orig_hw",
                    f"invalid_prepared_image_orig_hw:{hw}",
                )
            return hw, "prepared_image", None

    image_value = sample.get("image") or sample.get("image_path") or sample.get("path")
    if not image_value:
        return None, "image", "missing_image"

    image_path = _resolve_path(base, image_value)
    try:
        return image_hw(image_path), "image", None
    except Exception as err:  # noqa: BLE001
        return None, "image", f"unreadable_image:{err}"


def _as_xy(points: T.Any) -> np.ndarray:
    arr = np.asarray(points, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] < 2:
        raise ValueError(f"landmarks must be shaped (N, >=2), got {arr.shape}")
    return arr[:, :2].copy()


def _loader_scaled_points(
    points: T.Any,
    source_image_hw: tuple[int, int],
) -> tuple[tuple[int, int], np.ndarray]:
    """Simulate the loader's coordinate scaling into the 256x256 training image."""

    h, w = int(source_image_hw[0]), int(source_image_hw[1])
    if h <= 0 or w <= 0:
        raise ValueError(f"invalid source image size {(h, w)}")

    lmk = _as_xy(points)

    if lmk.size and float(np.nanmax(lmk)) <= 1.5:
        # Mirrors LandmarkDataset exactly. Normalized labels are assumed to be
        # normalized to the 256x256 training frame, not source image dimensions.
        lmk *= 255.0

    if h != LOADER_IMAGE_SIZE or w != LOADER_IMAGE_SIZE:
        lmk[:, 0] *= float(LOADER_IMAGE_SIZE) / float(w)
        lmk[:, 1] *= float(LOADER_IMAGE_SIZE) / float(h)
        h = w = LOADER_IMAGE_SIZE

    return (h, w), lmk.astype(np.float32)


def loader_padding_for_points(
    points: T.Any,
    image_hw_256: tuple[int, int] = (LOADER_IMAGE_SIZE, LOADER_IMAGE_SIZE),
    *,
    landmark_mask: T.Any = None,
    margin: float = LOADER_PADDING_MARGIN,
) -> dict[str, T.Any]:
    """Return ``MakeLMKInsideImage`` padding diagnostics for loader-scaled points."""

    lmk = _as_xy(points)

    if landmark_mask is None:
        valid = np.ones((lmk.shape[0],), dtype=bool)
    else:
        valid = np.asarray(landmark_mask, dtype=np.float32) > 0.5
        if valid.shape[0] != lmk.shape[0] or not valid.any():
            valid = np.ones((lmk.shape[0],), dtype=bool)

    finite = np.isfinite(lmk).all(axis=1)
    valid = valid & finite
    if not valid.any():
        return {
            "ok": False,
            "reason": "no_finite_valid_landmarks",
            "padding": None,
            "padded_shape": None,
            "lt": None,
            "rb": None,
            "image_shape": [int(image_hw_256[0]), int(image_hw_256[1])],
            "landmarks_outside_image": False,
        }

    valid_lmk = lmk[valid]
    lt = np.min(valid_lmk, axis=0)
    rb = np.max(valid_lmk, axis=0)

    h, w = int(image_hw_256[0]), int(image_hw_256[1])
    padding = 0.0
    if lt[0] < margin:
        padding = margin - float(lt[0])
    if lt[1] < margin:
        padding = max(margin - float(lt[1]), padding)
    if rb[0] > w - margin:
        padding = max(padding, float(rb[0]) - w + margin)
    if rb[1] > h - margin:
        padding = max(padding, float(rb[1]) - h + margin)

    if not np.isfinite(padding):
        return {
            "ok": False,
            "reason": "non_finite_landmark_padding",
            "padding": None,
            "padded_shape": None,
            "lt": lt.astype(float).tolist(),
            "rb": rb.astype(float).tolist(),
            "image_shape": [h, w],
            "landmarks_outside_image": True,
        }

    padded_h = h + 2 * int(math.ceil(float(padding)))
    padded_w = w + 2 * int(math.ceil(float(padding)))
    unreasonable = (
        padded_h > LOADER_MAX_PADDED_SIDE
        or padded_w > LOADER_MAX_PADDED_SIDE
        or padded_h * padded_w > LOADER_MAX_PADDED_PIXELS
    )

    return {
        "ok": not unreasonable,
        "reason": "unreasonable_loader_padding" if unreasonable else "",
        "padding": float(padding),
        "padded_shape": [int(padded_h), int(padded_w)],
        "lt": lt.astype(float).tolist(),
        "rb": rb.astype(float).tolist(),
        "image_shape": [h, w],
        "landmarks_outside_image": bool(padding > 0.0),
    }


def simulate_loader_geometry(
    points: T.Any,
    source_image_hw: tuple[int, int],
    *,
    landmark_mask: T.Any = None,
) -> dict[str, T.Any]:
    """Simulate loader scaling + ``MakeLMKInsideImage`` padding checks.

    ``source_image_hw`` is the native image size the loader would use for
    coordinate scaling. For prepared crops, pass ``prepared_image_orig_hw``.
    """

    try:
        loader_hw, scaled = _loader_scaled_points(points, source_image_hw)
        diag = loader_padding_for_points(
            scaled,
            loader_hw,
            landmark_mask=landmark_mask,
        )
        diag["source_image_hw"] = [int(source_image_hw[0]), int(source_image_hw[1])]
        return diag
    except Exception as err:  # noqa: BLE001
        return {
            "ok": False,
            "reason": f"geometry_simulation_error:{err}",
            "padding": None,
            "padded_shape": None,
            "lt": None,
            "rb": None,
            "image_shape": None,
            "source_image_hw": [int(source_image_hw[0]), int(source_image_hw[1])]
            if source_image_hw
            else None,
            "landmarks_outside_image": False,
        }
