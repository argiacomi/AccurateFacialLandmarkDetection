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
# Stricter review threshold than the loader's crash guard. The guard only
# rejects padding that would explode MakeLMKInsideImage's temporary image;
# wrong-coordinate-frame annotations routinely land *under* it (e.g. landmarks
# 150px outside a 240px image pad to ~590px, well below 2048). Padding above
# this fraction of the 256 crop marks a sample as suspicious: quarantined for
# review rather than hard-failed, since chins/profile noses legitimately
# overflow a little.
SUSPICIOUS_LOADER_PADDING = 48.0


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
            "suspicious": False,
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
            "suspicious": False,
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
    suspicious = not unreasonable and float(padding) > SUSPICIOUS_LOADER_PADDING

    reason = ""
    if unreasonable:
        reason = "unreasonable_loader_padding"
    elif suspicious:
        reason = "suspicious_loader_padding"

    return {
        "ok": not unreasonable,
        "suspicious": bool(suspicious),
        "reason": reason,
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
            "suspicious": False,
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


def _normalize_mask_label(value: T.Any) -> str:
    label = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    while "__" in label:
        label = label.replace("__", "_")
    return label.strip("_")


def as_bool_landmark_mask(value: T.Any, landmark_count: int = 68) -> np.ndarray | None:
    """Coerce a manifest mask value the way the training loader does."""

    if value is None:
        return None
    if isinstance(value, dict):
        # Accept dicts keyed by landmark index.
        arr = [
            value.get(str(i), value.get(i, True)) for i in range(int(landmark_count))
        ]
    else:
        arr = value
    if isinstance(arr, np.ndarray):
        arr = arr.tolist()
    if not isinstance(arr, (list, tuple)) or len(arr) != int(landmark_count):
        return None

    out = []
    for item in arr:
        if isinstance(item, str):
            label = _normalize_mask_label(item)
            out.append(
                label
                not in {
                    "",
                    "0",
                    "false",
                    "none",
                    "invalid",
                    "missing",
                    "self_occluded",
                    "selfoccluded",
                }
            )
        else:
            out.append(bool(item))
    return np.asarray(out, dtype=np.float32)


def landmark_mask_from_entry(
    entry: T.Mapping[str, T.Any],
    metadata: T.Mapping[str, T.Any],
    landmark_count: int = 68,
) -> np.ndarray:
    """Mirror ``LandmarkDataset``'s landmark-mask resolution for one sample.

    Priority matters. For MERL-RAV, coordinate-valid includes visible plus
    externally occluded estimated points, and excludes only true no-coordinate
    self-occlusion. Geometry simulation must use the same mask the loader
    passes to ``MakeLMKInsideImage``, otherwise zeroed masked-out points look
    like out-of-frame landmarks.
    """

    for key in (
        "landmark_mask",
        "landmark_coordinate_valid_mask",
        "landmark_source_valid_mask",
        "landmark_in_image_mask",
        "coordinate_valid_mask",
        "source_valid_mask",
        "valid_mask",
    ):
        mask = as_bool_landmark_mask(entry.get(key), landmark_count)
        if mask is not None:
            return mask
        mask = as_bool_landmark_mask(metadata.get(key), landmark_count)
        if mask is not None:
            return mask

    # Lower priority: visibility often means score-visible only, which would drop
    # externally occluded but coordinate-valid MERL-RAV points.
    for key in (
        "visibility",
        "landmark_score_visibility_mask",
        "score_visibility_mask",
    ):
        mask = as_bool_landmark_mask(entry.get(key), landmark_count)
        if mask is not None:
            return mask
        mask = as_bool_landmark_mask(metadata.get(key), landmark_count)
        if mask is not None:
            return mask

    return np.ones((int(landmark_count),), dtype=np.float32)


def landmark_mask_for_sample(sample: T.Mapping[str, T.Any]) -> np.ndarray | None:
    """Loader-parity landmark mask for a manifest sample, or None for all-valid."""

    metadata = (
        sample.get("metadata") if isinstance(sample.get("metadata"), dict) else {}
    )
    count = sample.get("landmark_count") or (metadata or {}).get("landmark_count")
    try:
        count = int(count)
    except (TypeError, ValueError):
        landmarks = sample.get("landmarks")
        count = 0
        if landmarks:
            try:
                count = int(
                    np.load(str(landmarks), mmap_mode="r", allow_pickle=False).shape[0]
                )
            except Exception:  # noqa: BLE001
                return None
    if count <= 0:
        return None
    return landmark_mask_from_entry(sample, metadata or {}, count)


def points_look_normalized(points: T.Any) -> bool:
    """True when the loader would treat ``points`` as [0, 1]-normalized.

    The loader scales such points by 255 and then assumes they live in the
    256x256 training frame. Points normalized to a non-256 source image are
    silently misplaced (in-bounds but wrong), so builders/validators flag them.
    """

    lmk = _as_xy(points)
    return bool(lmk.size) and float(np.nanmax(lmk)) <= 1.5


def write_geometry_overlay(
    out_path: str | Path,
    image_path: str | Path | None,
    points: T.Any,
    source_image_hw: tuple[int, int],
    *,
    landmark_mask: T.Any = None,
    diag: T.Mapping[str, T.Any] | None = None,
    max_canvas_side: int = 1024,
) -> Path | None:
    """Write a review PNG showing loader-scaled landmarks over the 256 crop.

    The canvas is the loader's padded frame (capped at ``max_canvas_side``) with
    the 256x256 training crop pasted at its padding offset, the crop border
    drawn in white, in-frame landmarks in green, and out-of-frame landmarks in
    red. Returns the written path, or None when the image cannot be decoded
    (the overlay is best-effort review tooling and must never fail a build).
    """

    try:
        _, scaled = _loader_scaled_points(points, source_image_hw)
        if diag is None:
            diag = loader_padding_for_points(scaled, landmark_mask=landmark_mask)

        if image_path is not None:
            decoded = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        else:
            decoded = None
        if decoded is None:
            crop = np.full((LOADER_IMAGE_SIZE, LOADER_IMAGE_SIZE, 3), 64, np.uint8)
        elif decoded.shape[:2] != (LOADER_IMAGE_SIZE, LOADER_IMAGE_SIZE):
            crop = cv2.resize(
                decoded,
                (LOADER_IMAGE_SIZE, LOADER_IMAGE_SIZE),
                interpolation=cv2.INTER_LINEAR,
            )
        else:
            crop = decoded

        pad = int(math.ceil(float(diag.get("padding") or 0.0)))
        pad = min(pad, max(0, (max_canvas_side - LOADER_IMAGE_SIZE) // 2))
        side = LOADER_IMAGE_SIZE + 2 * pad
        canvas = np.zeros((side, side, 3), dtype=np.uint8)
        canvas[pad : pad + LOADER_IMAGE_SIZE, pad : pad + LOADER_IMAGE_SIZE] = crop
        cv2.rectangle(
            canvas,
            (pad, pad),
            (pad + LOADER_IMAGE_SIZE - 1, pad + LOADER_IMAGE_SIZE - 1),
            (255, 255, 255),
            1,
        )

        finite = np.isfinite(scaled).all(axis=1)
        for idx in range(scaled.shape[0]):
            if not finite[idx]:
                continue
            x, y = float(scaled[idx, 0]), float(scaled[idx, 1])
            inside = 0.0 <= x <= LOADER_IMAGE_SIZE and 0.0 <= y <= LOADER_IMAGE_SIZE
            cx = int(round(x)) + pad
            cy = int(round(y)) + pad
            if not (0 <= cx < side and 0 <= cy < side):
                # Beyond the capped canvas: clamp to the edge so the reviewer
                # still sees the direction of the overflow.
                cx = min(max(cx, 0), side - 1)
                cy = min(max(cy, 0), side - 1)
            color = (0, 200, 0) if inside else (0, 0, 255)
            cv2.circle(canvas, (cx, cy), 2, color, -1)

        label = f"pad={diag.get('padding')!s:.6} reason={diag.get('reason') or 'ok'}"
        cv2.putText(
            canvas,
            label,
            (4, 14),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(out), canvas):
            return None
        return out
    except Exception:  # noqa: BLE001
        return None
