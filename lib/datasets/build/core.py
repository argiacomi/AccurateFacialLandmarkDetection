"""Shared infrastructure for landmark dataset builders.

Constants, landmark parsing, image indexing/cropping, sample + manifest
emission, pose metadata, and the generic directory/JSON builders shared by
every dataset-specific module under :mod:`lib.datasets.build`.
"""

# ruff: noqa: E402, F401
# (F401: imports below are re-exported to sibling
# modules via `from lib.datasets.build.core import *`.)
from __future__ import annotations

import argparse
import contextlib
import hashlib
import logging
import os
import re
import sys
import threading
import typing as T
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lib.core import pose
from lib.core.schema import (
    canonicalize_schema,
    flip_map_for_schema,
    head_name_for_schema,
    normalize_landmark_array,
    normalize_landmarks,
    point_count_for_schema,
    projection_audit_for_schema,
    to_canonical_68,
)
from lib.datasets.loader_geometry import (
    image_hw as loader_image_hw,
    loader_padding_for_points as _shared_loader_padding_for_points,
    simulate_loader_geometry,
)
from lib.datasets.parallel import parallel_map
from lib.datasets.progress import track as _dataset_progress_track


def track(
    iterable=None,
    *,
    desc: str,
    total=None,
    unit: str = "it",
    unit_scale: bool = False,
    leave: bool = False,
    disable=None,
):
    """Force visible progress for dataset build loops.

    Builder functions know their sample counts. The prepare orchestrator does
    not, so build progress belongs here rather than in prepare_landmark_dataset.
    """

    if str(desc).startswith("Build "):
        leave = True
        disable = False
    return _dataset_progress_track(
        iterable,
        desc=desc,
        total=total,
        unit=unit,
        unit_scale=unit_scale,
        leave=leave,
        disable=disable,
    )


from lib.datasets.sources import extract_archive_to_temp
from lib.datasets.video_frames import extract_video_frames, video_files
from lib.io_utils import read_json, relative_or_absolute, safe_id, write_json
from lib.logging_utils import (
    Verbosity,
    configure_console_logging,
    fmt_count,
    fmt_mapping,
    log_error,
    log_event,
    verbosity_from_name,
)
from lib.manifest.contract import (
    TRAINING_MANIFEST_CONTRACT,
    TRAINING_MANIFEST_VERSION,
    manifest_summary,
    split_safe_id_for_sample,
)

logger = logging.getLogger(__name__)

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff")
LANDMARK_EXTS = (".npy", ".pts", ".mat", ".txt")
SUPPORTED_DATASETS = (
    "wflw",
    "cofw68",
    "cofw29",
    "helen",
    "lapa",
    "jd-landmark",
    "fll2",
    "fll3",
    "xm2vts",
    "frgc",
    "300vw",
    "wflw-v",
    "wflwv",
    "merl-rav",
    "aflw2000-3d",
    "300w",
    "menpo2d",
    "multipie",
    "directory",
)
WFLW_ATTRIBUTE_NAMES = (
    "pose",
    "expression",
    "illumination",
    "makeup",
    "occlusion",
    "blur",
)
DEFAULT_NORMALIZER_SOURCE = "interocular_outer_eye_corners_36_45"


def _label(value: T.Any) -> str:
    label = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    while "__" in label:
        label = label.replace("__", "_")
    return label.strip("_") or "default"


def _dataset(value: str) -> str:
    key = str(value or "").strip().lower().replace("_", "-")
    aliases = {
        "aflw2000": "aflw2000-3d",
        "aflw2000-3d": "aflw2000-3d",
        "merlrav": "merl-rav",
        "merl-rav": "merl-rav",
        "menpo": "menpo2d",
        "menpo2d": "menpo2d",
        "menpo-2d": "menpo2d",
        "multi-pie": "multipie",
        "multipie": "multipie",
        "w300": "300w",
        "300w": "300w",
        "300-w": "300w",
        "wflw": "wflw",
        "cofw68": "cofw68",
        "cofw29": "cofw29",
        "cofw29-29": "cofw29",
        "cofw29-color": "cofw29",
        "helen": "helen",
        "lapa": "lapa",
        "jd": "jd-landmark",
        "jdlandmark": "jd-landmark",
        "jd-landmark": "jd-landmark",
        "jd-landmarks": "jd-landmark",
        "fll2": "fll2",
        "fll3": "fll3",
        "xm2vts": "xm2vts",
        "frgc": "frgc",
        "300vw": "300vw",
        "300-vw": "300vw",
        "wflw-v": "wflw-v",
        "wflwv": "wflw-v",
        "wflwvideo": "wflw-v",
        "wflw-video": "wflw-v",
        "directory": "directory",
    }
    return aliases.get(key, key)


def _dataset_condition_label(dataset: str) -> str:
    """Hard-negative bucket name for a dataset that lacks richer condition labels.

    Datasets without per-sample visual conditions (e.g. HELEN, LaPa, FLL2/FLL3,
    XM2VTS, FRGC, plain WFLW/cofw29 samples) would otherwise all collapse into one
    oversized generic ``default`` bucket. Falling back to the dataset id keeps the
    buckets balanceable and meaningful.
    """
    return _label(_dataset(dataset))


def _parse_csv(value: str | None) -> tuple[str, ...] | None:
    if value is None:
        return None
    parsed = tuple(_label(item) for item in value.split(",") if item.strip())
    return parsed or None


def _resolve_path(value: T.Any, *, base_dir: Path) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else (base_dir / path).resolve()


def _schema_from_declared_or_count(source_schema: str | None, count: int) -> str:
    if source_schema:
        raw = str(source_schema).strip().lower().replace("-", "_")
        if raw in {"3d_68", "68_3d", "lm_3d_68"}:
            return "2d_68"
        schema = canonicalize_schema(source_schema)
        if point_count_for_schema(schema) != int(count):
            raise ValueError(
                f"declared source_schema {source_schema!r} expects "
                f"{point_count_for_schema(schema)} points, got {count}"
            )
        return schema
    return canonicalize_schema(f"2d_{int(count)}")


def _canonical_points(
    raw: T.Any, *, source_schema: str | None = None
) -> tuple[np.ndarray, str]:
    """Return native trainable 2D points and the canonical source schema label."""
    arr = np.asarray(raw, dtype=np.float32)

    while arr.ndim > 2 and 1 in arr.shape:
        arr = np.squeeze(arr)

    if arr.ndim == 1:
        for count, dims in (
            (29, 2),
            (39, 2),
            (68, 3),
            (68, 2),
            (98, 2),
            (106, 2),
            (194, 2),
        ):
            if arr.size == count * dims:
                arr = arr.reshape(count, dims)
                break
        else:
            raise ValueError(f"flat landmark array has unsupported size {arr.size}")

    if arr.ndim != 2:
        raise ValueError(f"landmarks must be 2D, got shape {arr.shape}")

    if arr.shape[0] in (2, 3) and arr.shape[1] in (29, 39, 68, 98, 106, 194):
        arr = arr.T

    if not np.all(np.isfinite(arr)):
        raise ValueError("landmarks contain NaN or infinite values")

    if arr.shape[1] < 2:
        raise ValueError(
            f"landmarks must contain x/y coordinates, got shape {arr.shape}"
        )

    if arr.shape[0] not in (29, 39, 68, 98, 106, 194):
        raise ValueError(
            f"unsupported landmark shape {arr.shape}; expected 29, 39, 68, 98, 106, or 194 points"
        )

    schema = _schema_from_declared_or_count(source_schema, int(arr.shape[0]))
    points = normalize_landmark_array(arr[:, :2], schema=schema)
    return points, schema


def _parse_pts(path: Path) -> np.ndarray:
    rows: list[list[float]] = []
    in_block = False
    saw_brace = False
    for raw_line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line == "{":
            in_block = True
            saw_brace = True
            continue
        if line == "}":
            break
        if saw_brace and not in_block:
            continue
        if ":" in line and not re.match(r"^[+-]?\d", line):
            continue
        parts = line.replace(",", " ").split()
        if len(parts) < 2:
            continue
        try:
            rows.append([float(parts[0]), float(parts[1])])
        except ValueError:
            continue
    if not rows:
        raise ValueError(f"no point rows found in {path}")
    return np.asarray(rows, dtype=np.float32)


def _parse_numeric_text(path: Path) -> np.ndarray:
    text = path.read_text(encoding="utf-8", errors="ignore")
    values = [
        float(item)
        for item in re.findall(r"[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?", text)
    ]
    for count, dims in (
        (29, 2),
        (39, 2),
        (68, 3),
        (68, 2),
        (98, 2),
        (106, 2),
        (194, 2),
    ):
        total = count * dims
        for offset in (0, 1):
            if len(values) - offset == total and (
                offset == 0 or int(values[0]) == count
            ):
                return np.asarray(values[offset:], dtype=np.float32).reshape(
                    count, dims
                )
    rows: list[list[float]] = []
    for line in text.splitlines():
        parts = line.replace(",", " ").split()
        if len(parts) < 2:
            continue
        try:
            if len(parts) >= 3 and re.fullmatch(r"[+-]?\d+", parts[0]):
                row_index = int(parts[0])
                if row_index in {len(rows), len(rows) + 1}:
                    rows.append([float(parts[1]), float(parts[2])])
                    continue
            rows.append([float(parts[0]), float(parts[1])])
        except ValueError:
            continue
    if rows:
        return np.asarray(rows, dtype=np.float32)
    raise ValueError(f"could not parse numeric landmarks from {path}")


def _parse_mat(path: Path) -> np.ndarray:
    try:
        import scipy.io as sio
    except ImportError as err:
        raise RuntimeError("scipy is required to read .mat landmark files") from err

    payload = sio.loadmat(path)
    preferred = (
        "pt2d",
        "pts_2d",
        "points_2d",
        "landmarks",
        "landmark",
        "pts",
        "points",
        "shape",
        "lms",
        "lm",
        "keypoints",
    )

    def candidates() -> T.Iterator[tuple[str, T.Any]]:
        for key in preferred:
            if key in payload:
                yield key, payload[key]
        for key, value in payload.items():
            if not key.startswith("__") and key not in preferred:
                yield key, value

    errors: list[str] = []
    for key, value in candidates():
        try:
            arr = np.asarray(value, dtype=np.float32)
        except Exception:
            continue
        if arr.size < 29 * 2:
            continue
        try:
            _canonical_points(arr)
            return arr
        except Exception as err:  # noqa: BLE001
            errors.append(f"{key}: {err}")
            continue
    raise ValueError(f"no supported landmark array found in {path}; tried {errors[:5]}")


def _npy_shape_looks_like_single_landmark(path: Path) -> bool:
    """Cheaply reject bbox/index/cache .npy files before full parsing."""

    try:
        arr = np.load(path, mmap_mode="r", allow_pickle=False)
    except Exception:  # noqa: BLE001
        # Let the normal parser produce the real error for unusual files.
        return True

    shape = tuple(int(item) for item in getattr(arr, "shape", ()))
    counts = {29, 39, 68, 98, 106, 194}
    flat_sizes = {count * dims for count in counts for dims in (2, 3)}

    if len(shape) == 1:
        return shape[0] in flat_sizes
    if len(shape) == 2:
        if shape[0] in counts and shape[1] >= 2:
            return True
        if shape[0] in {2, 3} and shape[1] in counts:
            return True
        if shape[1] in flat_sizes:
            return True

    # Sequence arrays such as (N, 68, 2) need a dataset-specific reader, not the
    # generic one-file-per-sample path.
    return False


def _merl_rav_landmark_candidate(path: Path) -> bool:
    lowered = path.as_posix().lower()
    if path.suffix.lower() == ".npy":
        if any(
            token in lowered
            for token in (
                "bbox",
                "bboxes",
                "box",
                "boxes",
                "rect",
                "face_detection",
                "bounding",
            )
        ):
            return False
        return _npy_shape_looks_like_single_landmark(path)
    return True


def _load_landmark_file(path: Path) -> tuple[np.ndarray, str]:
    suffix = path.suffix.lower()
    if suffix == ".npy":
        return _canonical_points(np.load(path), source_schema=None)
    if suffix == ".pts":
        raw = _parse_pts(path)
        return _canonical_points(raw, source_schema=f"2d_{raw.shape[0]}")
    if suffix == ".mat":
        raw = _parse_mat(path)
        return _canonical_points(raw, source_schema=None)
    if suffix == ".txt":
        raw = _parse_numeric_text(path)
        return _canonical_points(raw, source_schema=f"2d_{raw.shape[0]}")
    raise ValueError(f"unsupported landmark file: {path}")


def _load_points(
    value: T.Any, *, base_dir: Path, source_schema: str | None = None
) -> tuple[np.ndarray, str]:
    if isinstance(value, (list, tuple, np.ndarray)):
        return _canonical_points(value, source_schema=source_schema)
    path = _resolve_path(value, base_dir=base_dir)
    if path.suffix.lower() in LANDMARK_EXTS:
        points, detected_schema = _load_landmark_file(path)
        return points, source_schema or detected_schema
    if path.suffix.lower() == ".json":
        return _canonical_points(read_json(path), source_schema=source_schema)
    raise ValueError(f"unsupported landmark input: {value!r}")


def _normalizer(points68: np.ndarray, sample_id: str) -> float:
    value = float("nan")
    if points68.shape[0] > 45:
        value = float(np.linalg.norm(points68[36] - points68[45]))
    if np.isfinite(value) and value > 0.0:
        return value

    span = np.ptp(points68[:, :2], axis=0)
    fallback = float(max(span[0], span[1]))
    if np.isfinite(fallback) and fallback > 0.0:
        if points68.shape[0] <= 45:
            return fallback
        logger.warning(
            "invalid interocular normalizer for %s: %s; using landmark span fallback %s",
            sample_id,
            value,
            fallback,
        )
        return fallback

    raise ValueError(
        f"invalid normalizer for {sample_id}: interocular={value}, span={fallback}"
    )


def _bbox_from_points_xyxy(points: np.ndarray) -> list[float]:
    valid = np.asarray(points, dtype=np.float32)
    valid = valid[np.isfinite(valid).all(axis=1)]
    if valid.size == 0:
        raise ValueError("cannot derive crop bbox from empty/non-finite landmarks")
    left, top = np.min(valid, axis=0)
    right, bottom = np.max(valid, axis=0)
    return [float(left), float(top), float(right), float(bottom)]


def _bbox_to_square_with_padding(
    bbox: T.Sequence[float], *, image_hw: tuple[int, int], pad_ratio: float
) -> tuple[float, float, float, float]:
    if len(bbox) != 4:
        raise ValueError(f"bbox must have 4 values, got {bbox!r}")
    x1, y1, x2, y2 = [float(v) for v in bbox]

    # If a source accidentally provides x,y,w,h, recover it.
    if x2 <= x1 and x2 > 0:
        x2 = x1 + x2
    if y2 <= y1 and y2 > 0:
        y2 = y1 + y2

    if not all(np.isfinite([x1, y1, x2, y2])) or x2 <= x1 or y2 <= y1:
        raise ValueError(f"invalid crop bbox {bbox!r}")

    width = x2 - x1
    height = y2 - y1
    side = max(width, height) * (1.0 + 2.0 * float(pad_ratio))
    cx = (x1 + x2) * 0.5
    cy = (y1 + y2) * 0.5

    left = cx - side * 0.5
    top = cy - side * 0.5
    right = cx + side * 0.5
    bottom = cy + side * 0.5
    return left, top, right, bottom


def _crop_image_and_remap_points(
    image_path: Path,
    points68: np.ndarray,
    bbox_xyxy: T.Sequence[float],
    *,
    pad_ratio: float = 0.25,
    output_size: int = 256,
) -> tuple[np.ndarray, np.ndarray, list[float]]:
    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError(f"could not read image for crop: {image_path}")

    image_rgb = image_bgr[:, :, [2, 1, 0]]
    height, width = image_rgb.shape[:2]
    left, top, right, bottom = _bbox_to_square_with_padding(
        bbox_xyxy,
        image_hw=(height, width),
        pad_ratio=pad_ratio,
    )
    side = max(right - left, bottom - top)
    if not np.isfinite(side) or side <= 1:
        raise ValueError(f"invalid crop side for {image_path}: {side}")

    virtual_w = max(1.0, right - left)
    virtual_h = max(1.0, bottom - top)
    max_virtual_side = 4096.0
    max_virtual_pixels = 4096.0 * 4096.0
    if (
        virtual_w > max_virtual_side
        or virtual_h > max_virtual_side
        or virtual_w * virtual_h > max_virtual_pixels
    ):
        raise ValueError(
            "unreasonable crop bbox for "
            f"{image_path}: bbox={(left, top, right, bottom)} "
            f"virtual_shape=({virtual_h:.1f}, {virtual_w:.1f})"
        )

    ix1 = int(np.floor(max(0.0, left)))
    iy1 = int(np.floor(max(0.0, top)))
    ix2 = int(np.ceil(min(float(width), right)))
    iy2 = int(np.ceil(min(float(height), bottom)))

    if ix2 <= ix1 or iy2 <= iy1:
        raise ValueError(f"empty crop for {image_path}: {(left, top, right, bottom)}")

    crop = image_rgb[iy1:iy2, ix1:ix2]
    pad_left = int(round(max(0.0, -left)))
    pad_top = int(round(max(0.0, -top)))
    pad_right = int(round(max(0.0, right - width)))
    pad_bottom = int(round(max(0.0, bottom - height)))

    if any(v > 0 for v in (pad_left, pad_top, pad_right, pad_bottom)):
        crop = cv2.copyMakeBorder(
            crop, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT
        )

    # Ensure exact virtual crop size before resize.
    virtual_w = max(1.0, right - left)
    virtual_h = max(1.0, bottom - top)
    scale_x = float(output_size) / virtual_w
    scale_y = float(output_size) / virtual_h

    crop_resized = cv2.resize(
        crop, (output_size, output_size), interpolation=cv2.INTER_LINEAR
    )

    remapped = np.asarray(points68, dtype=np.float32).copy()
    remapped[:, 0] = (remapped[:, 0] - float(left)) * scale_x
    remapped[:, 1] = (remapped[:, 1] - float(top)) * scale_y

    return (
        crop_resized,
        remapped.astype(np.float32),
        [float(left), float(top), float(right), float(bottom)],
    )


def _artifact_stem(dataset: str, sample_id: str) -> str:
    """Stable filename stem for prepared image/landmark artifacts."""

    dataset_key = _dataset(dataset)
    raw = str(sample_id or "sample").replace("\\", "/").strip("/")
    prefixes = (
        f"{dataset}/",
        f"{dataset_key}/",
        f"{dataset.replace('_', '-')}/",
        f"{dataset.replace('-', '_')}/",
    )
    for prefix in dict.fromkeys(prefixes):
        if raw.lower().startswith(prefix.lower()):
            raw = raw[len(prefix) :]
            break

    archive_suffixes = (".tar.gz", ".zip", ".tar", ".tgz", ".rar", ".7z")
    noisy_labels = {
        "labels",
        "landmarks",
        "annotations",
        "annot",
        "images",
        "frames",
        "videos",
        "bboxes",
        "bbox",
        "extracted",
        "wflw_v_release",
        "wflw_v_release_v1",
        "wflw_v_release_v2",
        "wflw_release",
        "wflw_v",
        "300vw_dataset_2015_12_14",
        "merl_rav_dataset_master",
    }

    parts = [part for part in raw.split("/") if part and part not in {".", ".."}]
    cleaned: list[str] = []
    skip_next_duplicate_archive_root: str | None = None

    for part in parts:
        lowered = part.lower()
        label = _label(part)

        archive_base = None
        for suffix in archive_suffixes:
            if lowered.endswith(suffix):
                archive_base = part[: -len(suffix)]
                break
        if archive_base is not None:
            skip_next_duplicate_archive_root = _label(archive_base)
            continue

        if skip_next_duplicate_archive_root is not None:
            if label == skip_next_duplicate_archive_root:
                skip_next_duplicate_archive_root = None
                continue
            skip_next_duplicate_archive_root = None

        if label in noisy_labels:
            continue

        cleaned.append(part)

    if not cleaned:
        cleaned = parts or [raw or "sample"]

    return safe_id("_".join(cleaned)).replace("#", "_").replace("/", "_")


def _write_crop_image(
    output_dir: Path, dataset: str, sample_id: str, crop_rgb: np.ndarray
) -> Path:
    safe = _artifact_stem(dataset, sample_id)
    out = output_dir / "images" / _dataset(dataset) / f"{safe}.jpg"
    out.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(
        str(out), crop_rgb[:, :, [2, 1, 0]], [int(cv2.IMWRITE_JPEG_QUALITY), 95]
    )
    if not ok:
        raise OSError(f"failed to write crop image: {out}")
    return out


def _loader_padding_for_points(
    points: np.ndarray,
    image_hw_256: tuple[int, int] = (256, 256),
    *,
    landmark_mask: T.Any = None,
) -> dict[str, T.Any]:
    """Return training-loader padding diagnostics for already loader-scaled points."""

    return _shared_loader_padding_for_points(
        points,
        image_hw_256,
        landmark_mask=landmark_mask,
    )


def _simulate_loader_geometry(
    image_path: Path,
    points: np.ndarray,
    *,
    landmark_mask: T.Any = None,
) -> dict[str, T.Any]:
    """Simulate the training loader's geometry for ``points`` on ``image_path``.

    Proves the landmarks are usable in the resolved image's coordinate frame:
    the loader scales them into the 256x256 crop and pads via
    ``MakeLMKInsideImage``, which raises when the required padding is
    pathological. Returns diagnostics with ``ok=False`` instead of raising for
    unreadable images.
    """

    try:
        hw = loader_image_hw(image_path)
    except (FileNotFoundError, OSError) as err:
        return {"ok": False, "reason": f"unreadable_image:{err}", "image_hw": None}
    return simulate_loader_geometry(points, hw, landmark_mask=landmark_mask)


def _crop_sample_image(
    *,
    output_dir: Path,
    dataset: str,
    sample_id: str,
    image_path: Path,
    points68: np.ndarray,
    bbox_xyxy: T.Sequence[float] | None,
    bbox_source: str,
    pad_ratio: float = 0.25,
) -> tuple[Path, np.ndarray, dict[str, T.Any]]:
    if bbox_xyxy is None or len(bbox_xyxy) != 4:
        bbox_xyxy = _bbox_from_points_xyxy(points68)

    crop_rgb, crop_points68, crop_bbox = _crop_image_and_remap_points(
        image_path,
        points68,
        bbox_xyxy,
        pad_ratio=pad_ratio,
        output_size=256,
    )
    crop_path = _write_crop_image(output_dir, dataset, sample_id, crop_rgb)
    return (
        crop_path,
        crop_points68,
        {
            "original_image": str(image_path.resolve()),
            "crop_bbox_xyxy": crop_bbox,
            "crop_padding_ratio": float(pad_ratio),
            "crop_bbox_source": bbox_source,
            "crop_output_size": 256,
        },
    )


def _build_image_index(root: Path) -> dict[str, list[Path]]:
    index: dict[str, list[Path]] = {}
    for ext in IMAGE_EXTS:
        for path in root.rglob(f"*{ext}"):
            index.setdefault(path.stem.lower(), []).append(path)
    return index


def _build_combined_image_index(roots: T.Iterable[Path]) -> dict[str, list[Path]]:
    index: dict[str, list[Path]] = {}
    seen: set[str] = set()
    for root in roots:
        if not root.is_dir():
            continue
        for key, paths in _build_image_index(root).items():
            bucket = index.setdefault(key, [])
            for path in paths:
                lexical_key = path.absolute().as_posix()
                if lexical_key not in seen:
                    seen.add(lexical_key)
                    bucket.append(path)
    return index


def _matching_image(
    landmarks: Path,
    *,
    root: Path | None = None,
    image_index: dict[str, list[Path]] | None = None,
) -> Path | None:
    for ext in IMAGE_EXTS:
        candidate = landmarks.with_suffix(ext)
        if candidate.is_file():
            return candidate
    for ext in IMAGE_EXTS:
        candidate = landmarks.parent / "images" / f"{landmarks.stem}{ext}"
        if candidate.is_file():
            return candidate
    if image_index is not None:
        matches = image_index.get(landmarks.stem.lower(), [])
        if matches:
            return sorted(matches, key=lambda item: len(item.parts))[0]
    if root is not None:
        # Fallback for less common builders/callers that have not precomputed an
        # image index. Hot paths should still pass image_index explicitly.
        fallback_index = _build_combined_image_index((root,))
        matches = fallback_index.get(landmarks.stem.lower(), [])
        if matches:
            return sorted(matches, key=lambda item: len(item.parts))[0]
    return None


def _conditions(entry: T.Mapping[str, T.Any], fallback: str) -> tuple[str, ...]:
    labels: list[str] = []
    for raw in (
        entry.get("conditions"),
        entry.get("condition"),
        entry.get("scenario"),
        fallback,
    ):
        if isinstance(raw, dict):
            items = [key for key, present in raw.items() if present]
        elif isinstance(raw, str):
            items = [raw]
        elif isinstance(raw, (list, tuple, set)):
            items = list(raw)
        else:
            items = []
        for item in items:
            label = _label(item)
            if label not in labels:
                labels.append(label)
    return tuple(labels or (_label(fallback),))


IDENTITY_METADATA_FIELDS = (
    "subject_id",
    "person_id",
    "identity_id",
    "session_id",
    "capture_id",
    "video_id",
    "clip_id",
    "sequence_id",
    "frame_id",
    "frame_index",
    "archive_id",
    "image_id",
    "quality",
    "attributes",
)


def _entry_metadata(
    entry: T.Mapping[str, T.Any], *, dataset: str, source_file: Path | None = None
) -> dict[str, T.Any]:
    metadata = (
        dict(entry.get("metadata", {}))
        if isinstance(entry.get("metadata"), dict)
        else {}
    )
    for key in IDENTITY_METADATA_FIELDS:
        if entry.get(key) not in (None, ""):
            metadata.setdefault(key, entry[key])
    metadata.setdefault("dataset", _dataset(dataset))
    if source_file is not None:
        metadata.setdefault("source_file", str(source_file.resolve()))
    return metadata


def _path_identity_metadata(
    path: Path, *, root: Path, dataset: str
) -> dict[str, T.Any]:
    rel = path.relative_to(root).with_suffix("")
    parts = list(rel.parts)
    metadata: dict[str, T.Any] = {
        "source_landmarks": str(path.resolve()),
        "image_id": rel.as_posix(),
    }
    if dataset in {"xm2vts", "frgc", "multipie", "menpo2d"} and parts:
        metadata.setdefault("subject_id", parts[0])
    if dataset in {"xm2vts", "frgc"} and len(parts) > 1:
        metadata.setdefault("session_id", parts[1])
    if dataset in {"xm2vts", "frgc"} and len(parts) > 2:
        metadata.setdefault("capture_id", parts[-1])
    return metadata


def _split_from_entry_or_identity(
    entry: T.Mapping[str, T.Any],
    metadata: T.Mapping[str, T.Any],
    *,
    dataset: str,
    sample_id: str,
) -> str:
    explicit = _label(entry.get("split") or metadata.get("split") or "")
    if explicit in {"train", "test"}:
        return explicit
    if explicit in {"val", "valid", "validation", "dev"}:
        return "test"
    split_identity = (
        metadata.get("split_safe_id")
        or metadata.get("video_id")
        or metadata.get("clip_id")
        or metadata.get("sequence_id")
        or metadata.get("session_id")
        or metadata.get("subject_id")
        or sample_id
    )
    return _deterministic_split(dataset, str(split_identity))


def _save_landmarks(
    output_dir: Path,
    dataset: str,
    sample_id: str,
    points68: np.ndarray,
) -> Path:
    safe = _artifact_stem(dataset, sample_id)
    path = output_dir / "landmarks" / _dataset(dataset) / f"{safe}.npy"
    path.parent.mkdir(parents=True, exist_ok=True)

    arr = np.asarray(points68, dtype=np.float32)
    if path.is_file():
        try:
            existing = np.load(path, mmap_mode="r", allow_pickle=False)
            if existing.shape == arr.shape and np.array_equal(
                np.asarray(existing), arr
            ):
                return path
        except Exception:  # noqa: BLE001
            pass

    np.save(path, arr)
    return path


def _is_finite_number(value: T.Any) -> bool:
    try:
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


def _pose_points_68(points: T.Any, source_schema: str) -> np.ndarray | None:
    """Return 68-point coordinates for pose geometry, or None if unavailable."""
    try:
        audit = projection_audit_for_schema(source_schema)
    except Exception:  # noqa: BLE001
        return None
    if audit.get("status") not in ("audited", "native"):
        return None
    try:
        arr = np.asarray(points, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[0] < 2:
            return None
        if arr.shape[0] == 68:
            return arr[:, :2]
        return to_canonical_68(arr, source_schema=source_schema)
    except Exception:  # noqa: BLE001
        return None


def _multipie_label_yaw(metadata: T.Mapping[str, T.Any]) -> float | None:
    """Coarse yaw magnitude for MultiPIE profile/semifrontal capture labels."""
    text = (
        f"{metadata.get('annotation_file', '')} {metadata.get('image_id', '')}".lower()
    )
    if "profile" in text:
        return 65.0
    if "semifrontal" in text or "semi_frontal" in text:
        return 20.0
    return None


def _pose_fields(
    *,
    source: str,
    yaw_signed: T.Any = None,
    yaw_abs: T.Any = None,
    side: str | None = None,
    pitch: T.Any = None,
    roll: T.Any = None,
) -> dict[str, T.Any]:
    """Assemble pose metadata for one sample.

    When the yaw sign (side) is known, emit signed ``pose_yaw_deg`` and a
    ``left_``/``right_`` bucket. When only a magnitude is known (a profile label
    with no side evidence), emit ``pose_abs_yaw_deg`` with a side-agnostic bucket
    and ``pose_side="unknown"`` instead of guessing a direction. Missing pitch is
    recorded as ``"unknown"`` rather than ``"neutral"``.
    """
    fields: dict[str, T.Any] = {"pose_source": source}

    if _is_finite_number(yaw_signed):
        yaw = float(yaw_signed)
        fields["pose_yaw_deg"] = round(yaw, 2)
        fields["pose_abs_yaw_deg"] = round(abs(yaw), 2)
        fields["pose_bucket"] = pose.yaw_bucket(yaw)
        fields["pose_side"] = pose.yaw_side(yaw)
    elif _is_finite_number(yaw_abs):
        magnitude = abs(float(yaw_abs))
        fields["pose_abs_yaw_deg"] = round(magnitude, 2)
        fields["pose_bucket"] = pose.yaw_tier(magnitude)
        fields["pose_side"] = side or "unknown"
    else:
        fields["pose_bucket"] = "unknown"
        fields["pose_side"] = "unknown"

    if _is_finite_number(pitch):
        fields["pose_pitch_deg"] = round(float(pitch), 2)
        fields["pitch_bucket"] = pose.pitch_bucket(float(pitch))
    else:
        fields["pitch_bucket"] = "unknown"

    if _is_finite_number(roll):
        fields["pose_roll_deg"] = round(float(roll), 2)
    return fields


def _geometry_yaw_side(pts68: np.ndarray | None) -> str:
    """Resolve left/right from 68-point geometry, only when confidently off-center.

    A near-frontal geometry estimate cannot disambiguate the side of a profile
    capture, so anything below the slight-yaw threshold returns ``"unknown"``.
    """
    if pts68 is None:
        return "unknown"
    est = pose.estimate_pose_from_68(pts68)
    if est is None or abs(est[0]) < pose.YAW_BUCKET_THRESHOLDS[0]:
        return "unknown"
    return "right" if est[0] > 0 else "left"


def _dense_flip_pair_yaw(points: T.Any, source_schema: str) -> float | None:
    """Estimate signed yaw from dense schemas with audited flip-pair geometry.

    This is intentionally pose-only. It does not mark the schema as projectable to
    68 landmarks. For HELEN/2d_194, the dense face outline and verified flip map
    are enough to derive a coarse yaw bucket from the apparent centerline shift
    inside the visible face silhouette.
    """
    try:
        schema = canonicalize_schema(source_schema)
    except Exception:  # noqa: BLE001
        return None
    if schema != "2d_194":
        return None

    try:
        arr = np.asarray(points, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[0] < 2 or arr.shape[1] < 2:
            return None
        xy = arr[:, :2]
        if not np.all(np.isfinite(xy)):
            return None
        flip = flip_map_for_schema(schema)
    except Exception:  # noqa: BLE001
        return None

    pairs = np.asarray(
        [(idx, int(peer)) for idx, peer in enumerate(flip) if idx < int(peer)],
        dtype=np.int64,
    )
    if pairs.size == 0:
        return None

    # Pair midpoints approximate the face's semantic centerline. Use a median to
    # avoid over-weighting any one region of the 194-point markup.
    pair_mid_x = (xy[pairs[:, 0], 0] + xy[pairs[:, 1], 0]) / 2.0
    center_x = float(np.median(pair_mid_x))

    # Percentile silhouette bounds are more stable than raw min/max if an outer
    # contour point is noisy.
    left_x, right_x = np.percentile(xy[:, 0], [2.0, 98.0])
    face_width = float(right_x - left_x)
    if not np.isfinite(face_width) or face_width <= 1e-6:
        return None

    d_left = center_x - float(left_x)
    d_right = float(right_x) - center_x
    denom = d_left + d_right
    if abs(denom) <= 1e-6:
        return 0.0

    yaw = float(np.clip((d_left - d_right) / denom, -1.0, 1.0) * 90.0)
    if abs(yaw) < 1.0:
        return 0.0
    return yaw


def _dense_pose_fields(points: T.Any, source_schema: str) -> dict[str, T.Any]:
    """Return pose metadata for dense non-68 schemas with pose-only heuristics."""
    yaw = _dense_flip_pair_yaw(points, source_schema)
    if yaw is None:
        return {}

    schema = canonicalize_schema(source_schema)
    fields = _pose_fields(source=f"landmark_geometry_{schema}", yaw_signed=yaw)
    fields["pose_geometry_schema"] = schema
    fields["pose_geometry_audit"] = "flip_pair_centerline_heuristic"
    return fields


def _pose_condition_tags(meta: T.Mapping[str, T.Any]) -> tuple[str, ...]:
    """Derive secondary balancing tags from pose metadata.

    These tags are appended to ``conditions`` only. They never replace the
    existing primary condition.
    """
    tags: list[str] = []

    pose_bucket = _label(meta.get("pose_bucket"))
    if pose_bucket not in ("default", "unknown"):
        tags.append(f"pose_{pose_bucket}")

    pitch_bucket = _label(meta.get("pitch_bucket"))
    if pitch_bucket not in ("default", "unknown"):
        tags.append(f"pitch_{pitch_bucket}")

    pose_side = _label(meta.get("pose_side"))
    if pose_side not in ("default", "unknown"):
        tags.append(f"pose_side_{pose_side}")

    return tuple(dict.fromkeys(tags))


def _pose_metadata(
    dataset: str,
    points: T.Any,
    source_schema: str,
    metadata: T.Mapping[str, T.Any],
) -> dict[str, T.Any]:
    """Resolve head-pose angles + buckets for one sample.

    Source priority: dataset-provided annotation angles (e.g. AFLW2000-3D
    ``Pose_Para``) > MultiPIE profile/semifrontal capture labels > approximate
    landmark geometry (any schema with an audited 68-point projection). Sparse
    schemas without a 68 projection or a usable label get no pose fields.
    """
    if _is_finite_number(metadata.get("pose_yaw_deg")):
        return _pose_fields(
            source="annotation",
            yaw_signed=metadata.get("pose_yaw_deg"),
            pitch=metadata.get("pose_pitch_deg"),
            roll=metadata.get("pose_roll_deg"),
        )

    pts68 = _pose_points_68(points, source_schema)

    if _dataset(dataset) == "multipie":
        magnitude = _multipie_label_yaw(metadata)
        if magnitude is not None:
            side = _geometry_yaw_side(pts68)
            if side in ("left", "right"):
                signed = magnitude if side == "right" else -magnitude
                return _pose_fields(
                    source="dataset_label", yaw_signed=signed, side=side
                )
            return _pose_fields(
                source="dataset_label", yaw_abs=magnitude, side="unknown"
            )

    if pts68 is not None:
        est = pose.estimate_pose_from_68(pts68)
        if est is not None:
            return _pose_fields(
                source="landmark_geometry",
                yaw_signed=est[0],
                pitch=est[1],
                roll=est[2],
            )

    dense_pose = _dense_pose_fields(points, source_schema)
    if dense_pose:
        return dense_pose

    return {}


def _sample(
    *,
    output_dir: Path,
    dataset: str,
    sample_id: str,
    image: Path,
    points68: np.ndarray,
    condition: str,
    conditions: tuple[str, ...],
    source_schema: str,
    source_id: str | None = None,
    metadata: dict[str, T.Any] | None = None,
    visibility: T.Any = None,
    normalizer: T.Any = None,
) -> dict[str, T.Any]:
    sample_id = safe_id(sample_id)
    source_schema = canonicalize_schema(source_schema)
    target_schema = source_schema
    head_name = head_name_for_schema(target_schema)
    landmarks = _save_landmarks(output_dir, dataset, sample_id, points68)
    meta = dict(metadata or {})

    normalizer_value = float("nan")
    if normalizer is not None:
        try:
            normalizer_value = float(normalizer)
        except (TypeError, ValueError):
            normalizer_value = float("nan")

    if not np.isfinite(normalizer_value) or normalizer_value <= 0.0:
        normalizer_value = _normalizer(points68, sample_id)
        meta.setdefault("normalizer_source", DEFAULT_NORMALIZER_SOURCE)
    else:
        meta.setdefault("normalizer_source", "explicit_manifest_normalizer")

    source_block = {"dataset": dataset, "source_id": source_id or sample_id}
    meta.setdefault("source_schema", source_schema)
    meta.setdefault("target_schema", target_schema)
    meta.setdefault("landmark_count", int(points68.shape[0]))
    meta.setdefault("head_name", head_name)
    if not isinstance(meta.get("mapping_audit"), dict):
        meta["mapping_audit"] = {
            "status": "native",
            "source_schema": source_schema,
            "target_schema": target_schema,
            "projection_to_68": projection_audit_for_schema(source_schema),
        }
    meta["mapping_audit"].setdefault(
        "projection_to_68", projection_audit_for_schema(source_schema)
    )

    for pose_key, pose_value in _pose_metadata(
        dataset, points68, source_schema, meta
    ).items():
        meta[pose_key] = pose_value

    primary_condition = _label(condition)
    sample_conditions = tuple(dict.fromkeys(_label(item) for item in conditions))
    # "default" is a non-bucket. Replace it with a dataset-specific hard-negative
    # bucket so datasets without richer condition labels do not all collapse into
    # one oversized generic bucket. Split markers stay only as secondary entries.
    if primary_condition == "default":
        dataset_bucket = _dataset_condition_label(dataset)
        sample_conditions = tuple(
            dataset_bucket if item == "default" else item for item in sample_conditions
        )
        if dataset_bucket not in sample_conditions:
            sample_conditions = (dataset_bucket, *sample_conditions)
        sample_conditions = tuple(dict.fromkeys(sample_conditions))
        primary_condition = dataset_bucket

    pose_condition_tags = _pose_condition_tags(meta)
    if pose_condition_tags:
        sample_conditions = tuple(
            dict.fromkeys((*sample_conditions, *pose_condition_tags))
        )

    out: dict[str, T.Any] = {
        "sample_id": sample_id,
        "dataset": dataset,
        "condition": primary_condition,
        "conditions": sample_conditions,
        "image": str(image.resolve()),
        "landmarks": relative_or_absolute(landmarks, output_dir),
        "source_schema": source_schema,
        "target_schema": target_schema,
        "landmark_count": int(points68.shape[0]),
        "head_name": head_name,
        "normalizer": normalizer_value,
        "source": source_block,
        "metadata": meta,
        "mapping_audit": dict(meta["mapping_audit"]),
    }
    for identity_key in (
        "subject_id",
        "person_id",
        "identity_id",
        "session_id",
        "capture_id",
        "video_id",
        "clip_id",
        "sequence_id",
        "frame_id",
        "frame_index",
        "archive_id",
        "image_id",
    ):
        if meta.get(identity_key) not in (None, ""):
            out[identity_key] = meta[identity_key]
    out["split_safe_id"] = split_safe_id_for_sample(out)
    out["metadata"].setdefault("split_safe_id", out["split_safe_id"])

    if visibility is not None:
        out["visibility"] = visibility
        out["metadata"].setdefault("visibility", visibility)
    return out


def _deterministic_split(dataset: str, sample_id: str, *, test_percent: int = 5) -> str:
    split_key = f"{_dataset(dataset)}|{sample_id}"
    split_hash = int(hashlib.sha256(split_key.encode()).hexdigest()[:8], 16)
    return "test" if (split_hash % 100) < int(test_percent) else "train"


def _with_split(sample: dict[str, T.Any], split: str) -> dict[str, T.Any]:
    sample["split"] = split
    metadata = sample.setdefault("metadata", {})
    if isinstance(metadata, dict):
        metadata["split"] = split
    return sample


def _limit_reached_for_build(
    samples: list[dict[str, T.Any]],
    scenarios: tuple[str, ...] | None,
    limit: int | None,
) -> bool:
    """Return True when enough samples have been built for a smoke/capped run.

    With explicit --scenarios, stop once every requested scenario has at least
    limit matching samples. Without --scenarios, stop once the primary emitted
    condition has reached limit. This makes --samples-per-scenario useful for
    fast smoke runs instead of only trimming the manifest after all files were
    parsed.
    """

    if not limit or limit <= 0 or not samples:
        return False

    if scenarios:
        requested = {_label(item) for item in scenarios}
        counts: Counter[str] = Counter()
        for sample in samples:
            for condition in sample.get("conditions", ()) or ():
                label = _label(condition)
                if label in requested:
                    counts[label] += 1
        return all(counts.get(label, 0) >= limit for label in requested)

    counts = Counter(_label(sample.get("condition") or "default") for sample in samples)
    return bool(counts) and all(count >= limit for count in counts.values())


def _filter(
    samples: list[dict[str, T.Any]],
    scenarios: tuple[str, ...] | None,
    limit: int | None,
) -> list[dict[str, T.Any]]:
    if scenarios:
        allowed = set(scenarios)
        samples = [
            sample
            for sample in samples
            if allowed.intersection(sample.get("conditions", ()))
        ]
    if not limit:
        return samples
    counts: Counter[str] = Counter()
    out = []
    for sample in samples:
        condition = str(sample.get("condition") or "default")
        if counts.get(condition, 0) >= limit:
            continue
        counts[condition] += 1
        out.append(sample)
    return out


def _prune_unreferenced_artifacts(
    output_dir: Path, dataset: str, samples: list[dict[str, T.Any]]
) -> int:
    """Delete prepared image/landmark files not referenced by the manifest.

    Crop-writing builders (WFLW, cofw68, cofw29, 300W, ...) emit one cropped
    image and one landmark file per *candidate* sample, but the
    ``--samples-per-scenario`` limit and ``allow_overlap=False`` dedup only trim
    the manifest afterwards. Without this step the filesystem keeps crops and
    landmarks for samples that were dropped from the manifest. Scoped to this
    dataset's own ``images/<dataset>`` and ``landmarks/<dataset>`` subtrees and
    removes only files the manifest does not point at, so the prepared artifacts
    always match the written manifest. ``source_images/`` is left untouched: it
    is a deliberate intermediate decode cache reused across runs.
    """
    ds = _dataset(dataset)
    referenced: set[Path] = set()
    for sample in samples:
        for key in ("image", "landmarks"):
            value = sample.get(key)
            if value:
                referenced.add((output_dir / Path(str(value))).resolve())

    removed = 0
    for subdir in ("images", "landmarks"):
        artifact_dir = output_dir / subdir / ds
        if not artifact_dir.is_dir():
            continue
        for path in artifact_dir.rglob("*"):
            if path.is_file() and path.resolve() not in referenced:
                path.unlink()
                removed += 1
    if removed:
        logger.info(
            "Pruned %s unreferenced %s artifact file(s) not in the manifest",
            removed,
            ds,
        )
    return removed


def _write_manifest(
    output_dir: Path,
    dataset: str,
    scenario: str,
    samples: list[dict[str, T.Any]],
    *,
    mode: str,
    allow_overlap: bool,
    scenarios: tuple[str, ...] | None,
    skipped: list[dict[str, str]] | None = None,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.json"
    merged: list[dict[str, T.Any]] = []
    if mode == "merge" and manifest_path.is_file():
        payload = read_json(manifest_path)
        merged = [
            dict(item) for item in payload.get("samples", []) if isinstance(item, dict)
        ]
    seen = {str(item.get("image")) for item in merged}
    for sample in samples:
        image = str(sample.get("image"))
        if not allow_overlap and image in seen:
            continue
        seen.add(image)
        merged.append(sample)

    summary = manifest_summary(merged)
    # A merged manifest can span multiple datasets; reflect that at the top level
    # instead of leaking the last-processed dataset. Per-sample "dataset" fields
    # remain authoritative.
    distinct_datasets = {
        str(sample.get("dataset")) for sample in merged if sample.get("dataset")
    }
    manifest_dataset = (
        next(iter(distinct_datasets))
        if len(distinct_datasets) == 1
        else ("multi_dataset" if distinct_datasets else dataset)
    )
    payload = {
        "version": TRAINING_MANIFEST_VERSION,
        "manifest_contract": TRAINING_MANIFEST_CONTRACT,
        "landmark_schema": "multi_schema",
        "metadata": {
            "builder": "AccurateFacialLandmarkDetection.tools.landmarks.build_quality_dataset",
            "dataset": manifest_dataset,
            "scenario": _label(scenario),
            "scenarios": list(scenarios or []),
            "sample_count": len(merged),
            "skipped_count": len(skipped or []),
        },
        **summary,
        "samples": merged,
    }
    write_json(manifest_path, payload)

    write_json(
        output_dir / "dataset_audit.json",
        {
            "manifest": str(manifest_path),
            "manifest_contract": TRAINING_MANIFEST_CONTRACT,
            "version": TRAINING_MANIFEST_VERSION,
            "sample_count": len(merged),
            "skipped_count": len(skipped or []),
            "skipped_examples": (skipped or [])[:50],
            **summary,
        },
    )

    # Crops/landmarks are written per candidate during the build, before the
    # manifest is filtered. Remove this dataset's artifacts that the final
    # manifest does not reference so disk usage tracks the manifest, not the
    # unfiltered candidate set.
    _prune_unreferenced_artifacts(output_dir, dataset, merged)
    return manifest_path


def _draw_manifest_overlay(
    image_path: Path,
    landmarks_path: Path,
    output_path: Path,
    *,
    visibility: T.Sequence[T.Any] | None = None,
) -> None:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"could not read overlay image: {image_path}")
    points = np.load(landmarks_path).astype(np.float32)[:, :2]
    height, width = image.shape[:2]
    if points.size and float(np.nanmax(points)) <= 1.5:
        points = points * np.asarray([width - 1, height - 1], dtype=np.float32)
    radius = max(1, int(round(max(width, height) / 512.0)))
    visible_color = (0, 255, 0)  # green (BGR)
    occluded_color = (0, 0, 255)  # red (BGR)
    vis = list(visibility) if visibility is not None else None
    for index, (x, y) in enumerate(points):
        occluded = vis is not None and index < len(vis) and not bool(vis[index])
        color = occluded_color if occluded else visible_color
        cv2.circle(
            image,
            (int(round(x)), int(round(y))),
            radius,
            color,
            -1,
            lineType=cv2.LINE_AA,
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(output_path), image)
    if not ok:
        raise OSError(f"failed to write overlay image: {output_path}")


@dataclass(frozen=True, slots=True)
class _OverlayTask:
    """Inputs for rendering one audit overlay in a worker."""

    sample_id: str
    dataset: str
    schema: str
    image_path: Path
    landmarks_path: Path
    overlay_path: Path
    visibility: tuple[T.Any, ...] | None


def _draw_overlay_task(task: _OverlayTask) -> tuple[_OverlayTask, str | None]:
    """Render one overlay; return (task, error)."""
    try:
        _draw_manifest_overlay(
            task.image_path,
            task.landmarks_path,
            task.overlay_path,
            visibility=task.visibility,
        )
        return task, None
    except Exception as err:  # noqa: BLE001
        return task, str(err)


def _write_visual_audit(
    manifest_path: Path, output_dir: Path, *, limit: int = 50, max_workers: int = 1
) -> Path:
    payload = read_json(manifest_path)
    entries = payload.get("samples", [])
    if not isinstance(entries, list):
        raise ValueError(
            f"manifest {manifest_path} must contain samples list for visual audit"
        )
    base_dir = manifest_path.parent
    audit_dir = output_dir / "visual_audit"
    overlays: list[dict[str, T.Any]] = []
    skipped: list[dict[str, str]] = []
    schema_counts: Counter[str] = Counter()

    # Select up to `limit` overlay tasks per dataset deterministically, then render
    # them in parallel (image decode/encode releases the GIL). Output is organized
    # by dataset/schema and results stay input-ordered.
    tasks: list[_OverlayTask] = []
    per_dataset_selected: Counter[str] = Counter()
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            continue
        schema = str(
            entry.get("target_schema") or entry.get("source_schema") or "unknown"
        )
        schema_counts[schema] += 1
        dataset_name = str(entry.get("dataset") or "unknown")
        if per_dataset_selected.get(dataset_name, 0) >= int(limit):
            continue
        image_value = entry.get("image")
        landmarks_value = entry.get("landmarks") or entry.get("ground_truth")
        sample_id = str(entry.get("sample_id") or index)
        if not image_value or not landmarks_value:
            skipped.append(
                {"sample_id": sample_id, "reason": "missing image or landmarks"}
            )
            continue
        visibility = entry.get("visibility")
        if visibility is None and isinstance(entry.get("metadata"), dict):
            visibility = entry["metadata"].get("visibility")
        overlay_name = safe_id(sample_id).replace("/", "_").replace("#", "_")
        dataset_dir = safe_id(dataset_name).replace("/", "_").replace("#", "_")
        tasks.append(
            _OverlayTask(
                sample_id=sample_id,
                dataset=dataset_name,
                schema=schema,
                image_path=_resolve_path(image_value, base_dir=base_dir),
                landmarks_path=_resolve_path(landmarks_value, base_dir=base_dir),
                overlay_path=audit_dir
                / "overlays"
                / dataset_dir
                / schema
                / f"{overlay_name}.jpg",
                visibility=tuple(visibility) if visibility is not None else None,
            )
        )
        per_dataset_selected[dataset_name] += 1

    for task, error in parallel_map(
        _draw_overlay_task, tasks, workers=max_workers, desc="Overlays", unit="overlay"
    ):
        if error is not None:
            skipped.append({"sample_id": task.sample_id, "reason": error})
            continue
        overlays.append(
            {
                "sample_id": task.sample_id,
                "dataset": task.dataset,
                "schema": task.schema,
                "image": str(task.image_path),
                "landmarks": str(task.landmarks_path),
                "overlay": str(task.overlay_path),
            }
        )

    report = {
        "manifest": str(manifest_path),
        "schema_counts": dict(sorted(schema_counts.items())),
        "overlay_count": len(overlays),
        "overlays": overlays,
        "skipped_count": len(skipped),
        "skipped_examples": skipped[:50],
    }
    report_path = audit_dir / "visual_audit.json"
    write_json(report_path, report)
    return report_path


def _json_source(root: Path) -> Path | None:
    candidates = [root] if root.is_file() else sorted(root.rglob("*.json"))
    for path in candidates:
        if not path.is_file() or path.name == "dataset_audit.json":
            continue
        try:
            payload = read_json(path)
        except (OSError, ValueError):
            continue
        if isinstance(payload, list) or (
            isinstance(payload, dict)
            and any(key in payload for key in ("samples", "entries"))
        ):
            return path
    return None


def _build_json(
    path: Path,
    output_dir: Path,
    *,
    dataset: str,
    scenario: str,
    scenarios: tuple[str, ...] | None,
    limit: int | None,
    mode: str,
    allow_overlap: bool,
    image_root: str | None,
) -> Path:
    payload = read_json(path)
    entries = (
        payload.get("samples", payload.get("entries", payload))
        if isinstance(payload, dict)
        else payload
    )
    if not isinstance(entries, list):
        raise ValueError(
            f"JSON source must contain list, entries, or samples list: {path}"
        )
    image_base = Path(image_root) if image_root else path.parent
    samples: list[dict[str, T.Any]] = []
    skipped: list[dict[str, str]] = []
    for idx, entry in track(
        enumerate(entries), desc=f"Build {dataset}", total=len(entries), unit="sample"
    ):
        if not isinstance(entry, dict):
            continue
        image_value = entry.get("image") or entry.get("image_path") or entry.get("path")
        landmark_value = (
            entry.get("landmarks")
            or entry.get("points")
            or entry.get("ground_truth")
            or entry.get("pts")
        )
        if image_value is None or landmark_value is None:
            skipped.append(
                {"sample_id": str(idx), "reason": "missing image or landmarks"}
            )
            continue
        sample_id = str(
            entry.get("sample_id") or entry.get("id") or entry.get("name") or idx
        )
        entry_dataset = _dataset(str(entry.get("dataset") or dataset))
        metadata = _entry_metadata(entry, dataset=entry_dataset, source_file=path)
        source_schema = (
            str(entry.get("source_schema") or metadata.get("source_schema") or "")
            or None
        )
        try:
            points68, detected_schema = _load_points(
                landmark_value, base_dir=path.parent, source_schema=source_schema
            )
            image_path = _resolve_path(image_value, base_dir=image_base)
            if not image_path.is_file():
                raise FileNotFoundError(f"image not found: {image_path}")
        except Exception as err:  # noqa: BLE001
            skipped.append({"sample_id": sample_id, "reason": str(err)})
            continue
        conds = _conditions(entry, scenario)
        split = _split_from_entry_or_identity(
            entry, metadata, dataset=entry_dataset, sample_id=sample_id
        )
        conds = tuple(dict.fromkeys((*conds, f"{split}set")))
        samples.append(
            _with_split(
                _sample(
                    output_dir=output_dir,
                    dataset=entry_dataset,
                    sample_id=sample_id,
                    image=image_path,
                    points68=points68,
                    condition=str(entry.get("condition") or conds[0]),
                    conditions=conds,
                    source_schema=source_schema or detected_schema,
                    source_id=str(entry.get("source_id") or sample_id),
                    metadata=metadata,
                    visibility=entry.get("visibility", metadata.get("visibility")),
                    normalizer=entry.get("normalizer", metadata.get("normalizer")),
                ),
                split,
            )
        )
    if not samples:
        raise ValueError(f"no JSON samples built from {path}; skipped={skipped[:5]}")
    return _write_manifest(
        output_dir,
        dataset,
        scenario,
        _filter(samples, scenarios, limit),
        mode=mode,
        allow_overlap=allow_overlap,
        scenarios=scenarios,
        skipped=skipped,
    )


def _raise_if_interrupted() -> None:
    """Give long dataset-build loops a cheap, explicit Ctrl-C checkpoint."""
    # A no-op Python bytecode checkpoint; pending KeyboardInterrupt is raised here.
    return None


def _condition_for_landmark_file(
    dataset: str, path: Path, scenario: str
) -> tuple[str, tuple[str, ...]]:
    parts = {_label(part) for part in path.parts}
    visual_labels: list[str] = []
    if dataset == "cofw68":
        visual_labels.append("occlusion")
    if dataset in {"300w", "w300"}:
        visual_labels.append("anchor")
    for token in (
        "profile",
        "pose",
        "occlusion",
        "occluded",
        "frontal",
        "normal",
        "clean",
        "challenging",
    ):
        if token in parts or any(token in part for part in parts):
            visual_labels.append(token)

    # Split markers are recorded as secondary conditions only: "trainset" is a
    # split, not a visual condition, and must never become the primary
    # hard-negative bucket.
    split_labels: list[str] = []
    for token in ("train", "training"):
        if token in parts or any(token == part for part in parts):
            split_labels.append("trainset")
    for token in ("test", "testing", "validation", "val"):
        if token in parts or any(token == part for part in parts):
            split_labels.append("testset")

    # Primary bucket is a real visual condition when present; otherwise fall back
    # to the scenario label ("default" by default), which _sample() maps to a
    # dataset-specific bucket.
    primary = visual_labels[0] if visual_labels else _label(scenario)
    labels = list(
        dict.fromkeys(_label(item) for item in (primary, *visual_labels, *split_labels))
    )
    return labels[0], tuple(labels)


def _aflw2000_pose_metadata(path: Path) -> dict[str, T.Any]:
    """Read AFLW2000-3D pose metadata from .mat files."""

    try:
        import scipy.io as sio
    except ImportError:
        return {}

    try:
        payload = sio.loadmat(path)
    except Exception:  # noqa: BLE001
        return {}

    raw_pose = None
    for key in ("Pose_Para", "pose_para", "Pose", "pose"):
        if key in payload:
            raw_pose = payload[key]
            break
    if raw_pose is None:
        return {}

    pose = np.asarray(raw_pose, dtype=np.float32).reshape(-1)
    if pose.size < 3:
        return {}

    angles = pose[:3].astype(np.float32)
    if float(np.nanmax(np.abs(angles))) <= float(2.0 * np.pi + 1e-3):
        angles = np.degrees(angles)

    pitch, yaw, roll = [float(value) for value in angles[:3]]
    if not all(np.isfinite([pitch, yaw, roll])):
        return {}

    return {
        "pose_pitch_deg": pitch,
        "pose_yaw_deg": yaw,
        "pose_roll_deg": roll,
        "pose_abs_yaw_deg": abs(yaw),
        "pose_source": "aflw2000_3d_pose_para",
    }


def _aflw2000_pose_conditions(
    scenario: str,
    metadata: T.Mapping[str, T.Any],
) -> tuple[str, tuple[str, ...]]:
    try:
        yaw = float(metadata["pose_yaw_deg"])
    except (KeyError, TypeError, ValueError):
        label = _label(scenario)
        return label, (label,)

    abs_yaw = abs(yaw)
    if abs_yaw >= 45.0:
        direction = "large_yaw_right" if yaw > 0 else "large_yaw_left"
        conditions = ("profile", "large_yaw", direction)
    elif abs_yaw >= 25.0:
        direction = "yaw_right" if yaw > 0 else "yaw_left"
        conditions = ("pose", direction)
    else:
        conditions = ("frontal",)

    label = _label(scenario)
    if label != "default":
        conditions = (*conditions, label)

    return conditions[0], tuple(dict.fromkeys(_label(item) for item in conditions))


def _build_directory(
    root: Path,
    output_dir: Path,
    *,
    dataset: str,
    scenario: str,
    scenarios: tuple[str, ...] | None,
    limit: int | None,
    mode: str,
    allow_overlap: bool,
    image_root: str | None,
    workers: int | None = None,
) -> Path:
    json_path = _json_source(root)
    if json_path is not None:
        return _build_json(
            json_path,
            output_dir,
            dataset=dataset,
            scenario=scenario,
            scenarios=scenarios,
            limit=limit,
            mode=mode,
            allow_overlap=allow_overlap,
            image_root=image_root,
        )

    image_base = Path(image_root) if image_root else root
    landmark_paths = _landmark_paths(root)
    if dataset == "merl-rav":
        before_count = len(landmark_paths)
        landmark_paths = [
            path for path in landmark_paths if _merl_rav_landmark_candidate(path)
        ]
        skipped_non_landmark_count = before_count - len(landmark_paths)
        if skipped_non_landmark_count:
            logger.info(
                "MERL-RAV skipped %s non-landmark candidate files before parsing",
                skipped_non_landmark_count,
            )

    # The recursive image index is only consulted when a landmark file has no
    # co-located (same-stem) or ``images/`` sibling image. 300W keeps the .pts
    # next to its image, so the index is never needed there -- build it lazily,
    # once, behind a lock so the parallel path below stays thread-safe.
    _index_cache: dict[str, dict[str, list[Path]]] = {}
    _index_lock = threading.Lock()

    def _image_index() -> dict[str, list[Path]]:
        with _index_lock:
            index = _index_cache.get("value")
            if index is None:
                index = _build_image_index(image_base)
                _index_cache["value"] = index
            return index

    def _process(
        landmark_path: Path,
    ) -> tuple[str, dict[str, T.Any] | None]:
        """Parse, match, crop and assemble one sample (thread-safe).

        Returns ``("sample", entry)``, ``("skip", reason)`` or ``("drop", None)``.
        Reads only shared read-only state and writes per-sample artifact files
        whose paths are unique to ``sample_id``, so it is safe to run across a
        thread pool.
        """
        if (
            landmark_path.suffix.lower() == ".txt"
            and "98pt" in landmark_path.name.lower()
        ):
            return ("drop", None)
        try:
            aflw2000_pose_metadata: dict[str, T.Any] = {}
            points68, source_schema = _load_landmark_file(landmark_path)
            if dataset == "aflw2000-3d" and landmark_path.suffix.lower() == ".mat":
                aflw2000_pose_metadata = _aflw2000_pose_metadata(landmark_path)
        except Exception as err:  # noqa: BLE001
            return ("skip", {"sample_id": landmark_path.as_posix(), "reason": str(err)})
        # Cheap co-located / ``images/`` lookup first; only fall back to the
        # (lazily built) recursive index when that misses.
        image = _matching_image(landmark_path)
        if image is None:
            image = _matching_image(landmark_path, image_index=_image_index())
        if image is None:
            return (
                "skip",
                {
                    "sample_id": landmark_path.as_posix(),
                    "reason": "matching image not found",
                },
            )
        sample_id = landmark_path.relative_to(root).with_suffix("").as_posix()
        condition, conds = _condition_for_landmark_file(
            dataset, landmark_path.relative_to(root), scenario
        )
        sample_image = image
        sample_points68 = points68
        sample_metadata = _path_identity_metadata(
            landmark_path, root=root, dataset=dataset
        )
        if dataset == "aflw2000-3d" and aflw2000_pose_metadata:
            sample_metadata.update(aflw2000_pose_metadata)
        entry_for_split: dict[str, T.Any] = {}
        if "trainset" in conds:
            entry_for_split["split"] = "train"
        elif "testset" in conds:
            entry_for_split["split"] = "test"
        split = _split_from_entry_or_identity(
            entry_for_split,
            sample_metadata,
            dataset=dataset,
            sample_id=sample_id,
        )
        if f"{split}set" not in conds:
            conds = tuple(dict.fromkeys((*conds, f"{split}set")))
        if dataset in {"300w", "w300"}:
            sample_image, sample_points68, crop_metadata = _crop_sample_image(
                output_dir=output_dir,
                dataset="300w",
                sample_id=sample_id,
                image_path=image,
                points68=points68,
                bbox_xyxy=_bbox_from_points_xyxy(points68),
                bbox_source="landmark_bbox",
                pad_ratio=0.25,
            )
            sample_metadata.update(crop_metadata)

        if dataset == "aflw2000-3d" and aflw2000_pose_metadata:
            cond, pose_conds = _aflw2000_pose_conditions(
                scenario, aflw2000_pose_metadata
            )
            return (
                "sample",
                _with_split(
                    _sample(
                        output_dir=output_dir,
                        dataset=dataset,
                        sample_id=sample_id,
                        image=sample_image,
                        points68=sample_points68,
                        condition=cond,
                        conditions=tuple(dict.fromkeys((*pose_conds, *conds))),
                        source_schema=source_schema,
                        source_id=sample_id,
                        metadata=sample_metadata,
                    ),
                    split,
                ),
            )
        return (
            "sample",
            _with_split(
                _sample(
                    output_dir=output_dir,
                    dataset=dataset,
                    sample_id=sample_id,
                    image=sample_image,
                    points68=sample_points68,
                    condition=condition,
                    conditions=conds,
                    source_schema=source_schema,
                    source_id=sample_id,
                    metadata=sample_metadata,
                ),
                split,
            ),
        )

    samples: list[dict[str, T.Any]] = []
    skipped: list[dict[str, str]] = []

    # Decode -> crop -> encode is the wall-clock bottleneck (especially 300W,
    # which re-encodes every sample). OpenCV releases the GIL during imread/
    # resize/imwrite, so a thread pool gives a near-linear speedup with
    # input-ordered, byte-identical output. Only parallelize a full build:
    # capped runs keep the serial early-break so they stop after ``limit``
    # samples instead of decoding every candidate first.
    want_workers = 1 if workers is None else workers
    if not limit and want_workers != 1 and len(landmark_paths) > 1:
        for kind, payload in parallel_map(
            _process,
            landmark_paths,
            workers=workers,
            desc=f"Build {dataset}",
            unit="file",
            # The serial path forces "Build ..." bars visible+persistent via the
            # track() wrapper; keep parity on the parallel path.
            leave=True,
            disable=False,
        ):
            if kind == "sample":
                samples.append(T.cast("dict[str, T.Any]", payload))
            elif kind == "skip":
                skipped.append(T.cast("dict[str, str]", payload))
    else:
        for landmark_path in track(
            landmark_paths,
            desc=f"Build {dataset}",
            total=len(landmark_paths),
            unit="file",
        ):
            kind, payload = _process(landmark_path)
            if kind == "skip":
                skipped.append(T.cast("dict[str, str]", payload))
                continue
            if kind == "drop":
                continue
            samples.append(T.cast("dict[str, T.Any]", payload))
            if _limit_reached_for_build(samples, scenarios, limit):
                break
    if not samples:
        raise ValueError(
            f"no usable schema-aware landmark samples found under {root}; skipped={skipped[:5]}"
        )
    return _write_manifest(
        output_dir,
        dataset,
        scenario,
        _filter(samples, scenarios, limit),
        mode=mode,
        allow_overlap=allow_overlap,
        scenarios=scenarios,
        skipped=skipped,
    )


def _landmark_paths(root: Path) -> list[Path]:
    return [
        path
        for suffix in LANDMARK_EXTS
        for path in sorted(root.rglob(f"*{suffix}"))
        if path.name != "manifest.json" and not path.name.startswith(".")
    ]


def _source_image_roots(root: Path, dataset: str) -> tuple[Path, ...]:
    labels = {
        "helen": ("images", "annotation", "annotations", "labels"),
        "lapa": ("images", "landmarks", "labels", "LaPa"),
        "jd-landmark": ("images", "landmarks", "labels"),
        "fll2": ("images", "landmarks", "labels"),
        "fll3": ("images", "landmarks", "labels"),
        "cofw29": ("images", "annotations", "landmarks"),
        "xm2vts": ("images", "annotations", "landmarks"),
        "frgc": ("images", "annotations", "landmarks"),
    }.get(dataset, ("images",))
    roots = [root]
    for label in labels:
        roots.extend(path for path in root.rglob(label) if path.is_dir())
    out: list[Path] = []
    seen: set[Path] = set()
    for item in roots:
        resolved = item.resolve()
        if resolved not in seen:
            seen.add(resolved)
            out.append(item)
    return tuple(out)


def _schema_parser_metadata(
    dataset: str, parser_name: str, landmark_path: Path, root: Path
) -> dict[str, T.Any]:
    metadata = _path_identity_metadata(landmark_path, root=root, dataset=dataset)
    metadata.update(
        {
            "dataset_parser": parser_name,
            "parser_type": "dataset_specific",
        }
    )
    if dataset == "lapa":
        rel_parts = {_label(part) for part in landmark_path.relative_to(root).parts}
        for split_label in ("train", "val", "test"):
            if split_label in rel_parts:
                metadata.setdefault("source_split", split_label)
                break
    return metadata


def _image_for_dataset_landmarks(
    landmark_path: Path,
    *,
    dataset: str,
    root: Path,
    image_root: str | None,
    roots: T.Sequence[Path] | None = None,
    image_indexes: T.Mapping[Path, dict[str, list[Path]]] | None = None,
) -> Path | None:
    search_roots = (
        tuple(roots)
        if roots is not None
        else ((Path(image_root),) if image_root else _source_image_roots(root, dataset))
    )
    for candidate_root in search_roots:
        image_index = (
            image_indexes.get(candidate_root) if image_indexes is not None else None
        )
        if image_index is None:
            image_index = _build_image_index(candidate_root)
        image = _matching_image(
            landmark_path,
            root=candidate_root,
            image_index=image_index,
        )
        if image is not None:
            return image
    return None


def _find_image_by_stem(directory: Path, stem: str) -> Path | None:
    for ext in IMAGE_EXTS:
        candidate = directory / f"{stem}{ext}"
        if candidate.is_file():
            return candidate
    return None


def _find_named_image(
    roots: T.Iterable[Path],
    image_name: str,
    *,
    image_index: dict[str, list[Path]] | None = None,
) -> Path | None:
    raw = Path(str(image_name))
    if raw.is_absolute() and raw.is_file():
        return raw

    for root in roots:
        candidate = root / raw
        if candidate.is_file():
            return candidate
        if raw.suffix.lower() in IMAGE_EXTS:
            candidate = root / raw.name
            if candidate.is_file():
                return candidate
        else:
            image = _find_image_by_stem(root, raw.name)
            if image is not None:
                return image

    if image_index is not None:
        key = raw.stem.lower() if raw.suffix.lower() in IMAGE_EXTS else raw.name.lower()
        matches = image_index.get(key, [])
        if matches:
            return sorted(matches, key=lambda item: len(item.parts))[0]
    return None


def _image_name_from_landmark_name(path: Path) -> str:
    name = path.name
    if name.lower().endswith(".txt"):
        name = name[:-4]
    return name


def _read_bbox_file(path: Path | None) -> list[float] | None:
    if path is None or not path.is_file():
        return None
    values = [
        float(item)
        for item in re.findall(
            r"[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?",
            path.read_text(encoding="utf-8", errors="ignore"),
        )
    ]
    if len(values) < 4:
        return None
    return [float(value) for value in values[:4]]


def _bbox_file_for_landmark(
    bbox_dir: Path | None,
    landmark_path: Path,
    *,
    image_name: str | None = None,
) -> Path | None:
    if bbox_dir is None or not bbox_dir.is_dir():
        return None
    stem = landmark_path.stem
    names = [f"{stem}.txt", f"{stem}.rect"]
    if image_name:
        names.extend((f"{image_name}.txt", f"{image_name}.rect"))
    for name in dict.fromkeys(names):
        candidate = bbox_dir / name
        if candidate.is_file():
            return candidate
    return None


def _manifest_split_for_source_split(source_split: str) -> str:
    return "train" if _label(source_split) == "train" else "test"


def _native_conditions_for_split(
    scenario: str, split: str
) -> tuple[str, tuple[str, ...]]:
    conds = tuple(dict.fromkeys((_label(scenario), f"{split}set")))
    return conds[0], conds


def _path_under_roots(path: Path, roots: T.Sequence[Path]) -> bool:
    resolved = path.resolve()
    for root in roots:
        with contextlib.suppress(ValueError):
            resolved.relative_to(root.resolve())
            return True
    return False


def _images_matching_name(
    roots: T.Iterable[Path],
    image_name: str,
    *,
    image_index: dict[str, list[Path]] | None = None,
) -> list[Path]:
    raw = Path(str(image_name))
    root_list = tuple(roots)
    out: list[Path] = []
    seen: set[Path] = set()

    def add(candidate: Path) -> None:
        if candidate.is_file() and candidate.suffix.lower() in IMAGE_EXTS:
            resolved = candidate.resolve()
            if resolved not in seen:
                seen.add(resolved)
                out.append(candidate)

    if raw.is_absolute():
        add(raw)
        return out

    search_names = [raw.name]
    if raw.suffix.lower() not in IMAGE_EXTS:
        search_names = [f"{raw.name}{ext}" for ext in IMAGE_EXTS]
    if image_index is not None:
        index_key = (
            raw.stem.lower() if raw.suffix.lower() in IMAGE_EXTS else raw.name.lower()
        )
        for match in image_index.get(index_key, []):
            if match.name in search_names and _path_under_roots(match, root_list):
                add(match)
        if out:
            return out

    for root in root_list:
        for name in search_names:
            add(root / raw.parent / name if raw.parent != Path(".") else root / name)
            for match in sorted(root.rglob(name), key=lambda item: len(item.parts)):
                add(match)
    return out


def _resolve_unique_image(
    roots: T.Iterable[Path],
    image_name: str,
    *,
    context: str,
    image_index: dict[str, list[Path]] | None = None,
) -> Path:
    matches = _images_matching_name(roots, image_name, image_index=image_index)
    if not matches:
        raise FileNotFoundError(f"{context} image not found: {image_name}")
    if len(matches) > 1:
        rendered = ", ".join(str(path) for path in matches[:5])
        raise ValueError(
            f"{context} image match is ambiguous for {image_name}: {rendered}"
        )
    return matches[0]


def _build_expected_schema_dataset(
    root: Path,
    output_dir: Path,
    *,
    dataset: str,
    expected_schema: str,
    parser_name: str,
    scenario: str,
    scenarios: tuple[str, ...] | None,
    limit: int | None,
    mode: str,
    allow_overlap: bool,
    image_root: str | None,
) -> Path:
    json_path = _json_source(root)
    if json_path is not None:
        return _build_expected_schema_json(
            json_path,
            output_dir,
            dataset=dataset,
            expected_schema=expected_schema,
            parser_name=parser_name,
            scenario=scenario,
            scenarios=scenarios,
            limit=limit,
            mode=mode,
            allow_overlap=allow_overlap,
            image_root=image_root,
        )

    samples: list[dict[str, T.Any]] = []
    skipped: list[dict[str, str]] = []
    source_image_roots = (
        (Path(image_root),) if image_root else _source_image_roots(root, dataset)
    )
    source_image_indexes = {
        candidate_root: _build_image_index(candidate_root)
        for candidate_root in source_image_roots
    }
    landmark_files = _landmark_paths(root)
    for landmark_path in track(
        landmark_files,
        desc=f"Build {dataset}",
        total=len(landmark_files),
        unit="file",
    ):
        _raise_if_interrupted()
        try:
            points, detected_schema = _load_landmark_file(landmark_path)
            if detected_schema != expected_schema:
                raise ValueError(
                    f"{parser_name} expected {expected_schema}, got {detected_schema}"
                )
            image = _image_for_dataset_landmarks(
                landmark_path,
                dataset=dataset,
                root=root,
                image_root=image_root,
                roots=source_image_roots,
                image_indexes=source_image_indexes,
            )
            if image is None:
                raise FileNotFoundError("matching image not found")
        except KeyboardInterrupt:
            raise
        except Exception as err:  # noqa: BLE001
            skipped.append({"sample_id": landmark_path.as_posix(), "reason": str(err)})
            continue

        sample_id = landmark_path.relative_to(root).with_suffix("").as_posix()
        condition, conds = _condition_for_landmark_file(
            dataset, landmark_path.relative_to(root), scenario
        )
        metadata = _schema_parser_metadata(dataset, parser_name, landmark_path, root)
        entry_for_split: dict[str, T.Any] = {}
        if "trainset" in conds or metadata.get("source_split") == "train":
            entry_for_split["split"] = "train"
        elif "testset" in conds or metadata.get("source_split") in {"val", "test"}:
            entry_for_split["split"] = "test"
        split = _split_from_entry_or_identity(
            entry_for_split,
            metadata,
            dataset=dataset,
            sample_id=sample_id,
        )
        conds = tuple(dict.fromkeys((*conds, f"{split}set")))
        samples.append(
            _with_split(
                _sample(
                    output_dir=output_dir,
                    dataset=dataset,
                    sample_id=sample_id,
                    image=image,
                    points68=points,
                    condition=condition,
                    conditions=conds,
                    source_schema=expected_schema,
                    source_id=sample_id,
                    metadata=metadata,
                ),
                split,
            )
        )

    if not samples:
        raise ValueError(
            f"no {dataset} samples built with {parser_name}; skipped={skipped[:10]}"
        )

    return _write_manifest(
        output_dir,
        dataset,
        scenario,
        _filter(samples, scenarios, limit),
        mode=mode,
        allow_overlap=allow_overlap,
        scenarios=scenarios,
        skipped=skipped,
    )


def _build_expected_schema_json(
    path: Path,
    output_dir: Path,
    *,
    dataset: str,
    expected_schema: str,
    parser_name: str,
    scenario: str,
    scenarios: tuple[str, ...] | None,
    limit: int | None,
    mode: str,
    allow_overlap: bool,
    image_root: str | None,
) -> Path:
    payload = read_json(path)
    entries = (
        payload.get("samples", payload.get("entries", payload))
        if isinstance(payload, dict)
        else payload
    )
    if not isinstance(entries, list):
        raise ValueError(
            f"{parser_name} JSON source must contain list, entries, or samples list: {path}"
        )
    image_base = Path(image_root) if image_root else path.parent
    samples: list[dict[str, T.Any]] = []
    skipped: list[dict[str, str]] = []
    for idx, entry in track(
        enumerate(entries), desc=f"Build {dataset}", total=len(entries), unit="sample"
    ):
        if not isinstance(entry, dict):
            continue
        image_value = entry.get("image") or entry.get("image_path") or entry.get("path")
        landmark_value = (
            entry.get("landmarks")
            or entry.get("points")
            or entry.get("ground_truth")
            or entry.get("pts")
        )
        sample_id = str(
            entry.get("sample_id") or entry.get("id") or entry.get("name") or idx
        )
        if image_value is None or landmark_value is None:
            skipped.append(
                {"sample_id": sample_id, "reason": "missing image or landmarks"}
            )
            continue
        metadata = _entry_metadata(entry, dataset=dataset, source_file=path)
        metadata["dataset_parser"] = parser_name
        metadata["parser_type"] = "dataset_specific"
        source_schema = str(
            entry.get("source_schema")
            or metadata.get("source_schema")
            or expected_schema
        )
        try:
            points, detected_schema = _load_points(
                landmark_value,
                base_dir=path.parent,
                source_schema=source_schema,
            )
            if detected_schema != expected_schema:
                raise ValueError(
                    f"{parser_name} expected {expected_schema}, got {detected_schema}"
                )
            image_path = _resolve_path(image_value, base_dir=image_base)
            if not image_path.is_file():
                raise FileNotFoundError(f"image not found: {image_path}")
        except Exception as err:  # noqa: BLE001
            skipped.append({"sample_id": sample_id, "reason": str(err)})
            continue

        conds = _conditions(entry, scenario)
        split = _split_from_entry_or_identity(
            entry, metadata, dataset=dataset, sample_id=sample_id
        )
        conds = tuple(dict.fromkeys((*conds, f"{split}set")))
        samples.append(
            _with_split(
                _sample(
                    output_dir=output_dir,
                    dataset=dataset,
                    sample_id=sample_id,
                    image=image_path,
                    points68=points,
                    condition=str(entry.get("condition") or conds[0]),
                    conditions=conds,
                    source_schema=expected_schema,
                    source_id=str(entry.get("source_id") or sample_id),
                    metadata=metadata,
                    visibility=entry.get("visibility", metadata.get("visibility")),
                    normalizer=entry.get("normalizer", metadata.get("normalizer")),
                ),
                split,
            )
        )

    if not samples:
        raise ValueError(
            f"no {dataset} JSON samples built with {parser_name}; skipped={skipped[:10]}"
        )
    return _write_manifest(
        output_dir,
        dataset,
        scenario,
        _filter(samples, scenarios, limit),
        mode=mode,
        allow_overlap=allow_overlap,
        scenarios=scenarios,
        skipped=skipped,
    )


def _normalizer_from_bbox(bbox: list[float]) -> float:
    value = max(float(bbox[2] - bbox[0]), float(bbox[3] - bbox[1]))
    return value if np.isfinite(value) and value > 0.0 else 1.0


# Re-export every module-level name (including the single-underscore build
# helpers) so `from lib.datasets.build.<mod> import *` resolves bare-name
# calls in sibling modules exactly as they did in the original flat module.
__all__ = [_n for _n in dict(globals()) if not _n.startswith("__")]
