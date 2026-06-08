#!/usr/bin/env python3
"""Build CD-ViT/faceswap-compatible landmark manifests.

This local builder covers the faceswap landmark dataset names while emitting a
CD-ViT-friendly contract: every manifest entry points to a materialized
trainable landmark ``.npy`` file.

Supported raw inputs by dataset:

* WFLW: official 98-point annotation text plus images, or generic sources.
* COFW: faceswap-style 68-point JSON export, or generic 68/98 landmark files.
* 300W: iBUG ``.pts`` files plus same-stem images, JSON, ``.npy``, or ``.mat``.
* AFLW2000-3D: same-stem ``.mat`` files with 68 2D/3D landmarks plus images.
* HELEN, LaPa, JD-landmark, FFL2, FLL3, COFW original, XM2VTS, FRGC:
  native release layouts, with generic JSON/``.npy``/``.pts``/``.mat`` staging
  retained as a fallback.
* MERL-RAV, Menpo2D, MultiPIE: JSON, ``.npy``, ``.pts``, ``.mat`` sources.
* 300VW and WFLW-V: video/frame JSON, pre-extracted frame directories, or
  video extraction plus same-frame annotations.

Registered non-68 schemas are preserved for schema-aware multi-head training.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import logging
import re
import sys
import typing as T
from pathlib import Path

import numpy as np
import cv2

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lib.landmarks.core.schema import (
    canonicalize_schema,
    head_name_for_schema,
    normalize_landmark_array,
    normalize_landmarks,
    point_count_for_schema,
    projection_audit_for_schema,
)
from lib.landmarks.datasets.video_frames import extract_video_frames, video_files
from lib.landmarks.manifest.contract import (
    TRAINING_MANIFEST_CONTRACT,
    TRAINING_MANIFEST_VERSION,
    manifest_summary,
    split_safe_id_for_sample,
)
from lib.landmarks.datasets.sources import extract_archive_to_temp

logger = logging.getLogger(__name__)

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff")
LANDMARK_EXTS = (".npy", ".pts", ".mat", ".txt")
SUPPORTED_DATASETS = (
    "wflw",
    "cofw",
    "cofw-original",
    "helen",
    "lapa",
    "jd-landmark",
    "ffl2",
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
WFLW_ATTRIBUTE_NAMES = ("pose", "expression", "illumination", "makeup", "occlusion", "blur")
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
        "cofw": "cofw",
        "cofw-original": "cofw-original",
        "cofw-original-29": "cofw-original",
        "cofw29": "cofw-original",
        "cofw-original-color": "cofw-original",
        "helen": "helen",
        "lapa": "lapa",
        "jd": "jd-landmark",
        "jdlandmark": "jd-landmark",
        "jd-landmark": "jd-landmark",
        "jd-landmarks": "jd-landmark",
        "ffl2": "ffl2",
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


def _parse_csv(value: str | None) -> tuple[str, ...] | None:
    if value is None:
        return None
    parsed = tuple(_label(item) for item in value.split(",") if item.strip())
    return parsed or None


def _safe_id(value: T.Any) -> str:
    text = str(value or "sample").strip().replace("\\", "/").strip("/") or "sample"
    return "".join(ch if ch.isalnum() or ch in "._-/#" else "_" for ch in text)


def _relative_or_absolute(path: Path, base: Path) -> str:
    try:
        return path.resolve().relative_to(base.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def _jsonable(value: T.Any) -> T.Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    return value


def _read_json(path: Path) -> T.Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: T.Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


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


def _canonical_points(raw: T.Any, *, source_schema: str | None = None) -> tuple[np.ndarray, str]:
    """Return native trainable 2D points and the canonical source schema label."""
    arr = np.asarray(raw, dtype=np.float32)

    while arr.ndim > 2 and 1 in arr.shape:
        arr = np.squeeze(arr)

    if arr.ndim == 1:
        for count, dims in ((29, 2), (39, 2), (68, 3), (68, 2), (98, 2), (106, 2), (194, 2)):
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
        raise ValueError(f"landmarks must contain x/y coordinates, got shape {arr.shape}")

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
    values = [float(item) for item in re.findall(r"[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?", text)]
    for count, dims in ((29, 2), (39, 2), (68, 3), (68, 2), (98, 2), (106, 2), (194, 2)):
        total = count * dims
        for offset in (0, 1):
            if len(values) - offset == total and (offset == 0 or int(values[0]) == count):
                return np.asarray(values[offset:], dtype=np.float32).reshape(count, dims)
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


def _load_points(value: T.Any, *, base_dir: Path, source_schema: str | None = None) -> tuple[np.ndarray, str]:
    if isinstance(value, (list, tuple, np.ndarray)):
        return _canonical_points(value, source_schema=source_schema)
    path = _resolve_path(value, base_dir=base_dir)
    if path.suffix.lower() in LANDMARK_EXTS:
        points, detected_schema = _load_landmark_file(path)
        return points, source_schema or detected_schema
    if path.suffix.lower() == ".json":
        return _canonical_points(_read_json(path), source_schema=source_schema)
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

    raise ValueError(f"invalid normalizer for {sample_id}: interocular={value}, span={fallback}")


def _bbox_from_points_xyxy(points: np.ndarray) -> list[float]:
    valid = np.asarray(points, dtype=np.float32)
    valid = valid[np.isfinite(valid).all(axis=1)]
    if valid.size == 0:
        raise ValueError("cannot derive crop bbox from empty/non-finite landmarks")
    left, top = np.min(valid, axis=0)
    right, bottom = np.max(valid, axis=0)
    return [float(left), float(top), float(right), float(bottom)]


def _bbox_to_square_with_padding(bbox: T.Sequence[float], *, image_hw: tuple[int, int], pad_ratio: float) -> tuple[float, float, float, float]:
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
        crop = cv2.copyMakeBorder(crop, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT)

    # Ensure exact virtual crop size before resize.
    virtual_w = max(1.0, right - left)
    virtual_h = max(1.0, bottom - top)
    scale_x = float(output_size) / virtual_w
    scale_y = float(output_size) / virtual_h

    crop_resized = cv2.resize(crop, (output_size, output_size), interpolation=cv2.INTER_LINEAR)

    remapped = np.asarray(points68, dtype=np.float32).copy()
    remapped[:, 0] = (remapped[:, 0] - float(left)) * scale_x
    remapped[:, 1] = (remapped[:, 1] - float(top)) * scale_y

    return crop_resized, remapped.astype(np.float32), [float(left), float(top), float(right), float(bottom)]


def _write_crop_image(output_dir: Path, dataset: str, sample_id: str, crop_rgb: np.ndarray) -> Path:
    safe = _safe_id(sample_id).replace("#", "_").replace("/", "_")
    out = output_dir / "images" / dataset / f"{safe}.jpg"
    out.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(out), crop_rgb[:, :, [2, 1, 0]], [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    if not ok:
        raise OSError(f"failed to write crop image: {out}")
    return out


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
    return crop_path, crop_points68, {
        "original_image": str(image_path.resolve()),
        "crop_bbox_xyxy": crop_bbox,
        "crop_padding_ratio": float(pad_ratio),
        "crop_bbox_source": bbox_source,
        "crop_output_size": 256,
    }


def _build_image_index(root: Path) -> dict[str, list[Path]]:
    index: dict[str, list[Path]] = {}
    for ext in IMAGE_EXTS:
        for path in root.rglob(f"*{ext}"):
            index.setdefault(path.stem.lower(), []).append(path)
    return index


def _matching_image(landmarks: Path, *, root: Path | None = None, image_index: dict[str, list[Path]] | None = None) -> Path | None:
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
        for ext in IMAGE_EXTS:
            matches = sorted(root.rglob(f"{landmarks.stem}{ext}"), key=lambda item: len(item.parts))
            if matches:
                return matches[0]
    return None


def _conditions(entry: T.Mapping[str, T.Any], fallback: str) -> tuple[str, ...]:
    labels: list[str] = []
    for raw in (entry.get("conditions"), entry.get("condition"), entry.get("scenario"), fallback):
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


def _entry_metadata(entry: T.Mapping[str, T.Any], *, dataset: str, source_file: Path | None = None) -> dict[str, T.Any]:
    metadata = dict(entry.get("metadata", {})) if isinstance(entry.get("metadata"), dict) else {}
    for key in IDENTITY_METADATA_FIELDS:
        if entry.get(key) not in (None, ""):
            metadata.setdefault(key, entry[key])
    metadata.setdefault("dataset", _dataset(dataset))
    if source_file is not None:
        metadata.setdefault("source_file", str(source_file.resolve()))
    return metadata


def _path_identity_metadata(path: Path, *, root: Path, dataset: str) -> dict[str, T.Any]:
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


def _save_landmarks(output_dir: Path, sample_id: str, points68: np.ndarray) -> Path:
    safe = _safe_id(sample_id).replace("#", "_")
    path = output_dir / "landmarks" / f"{safe}.npy"
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, points68.astype(np.float32))
    return path


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
    sample_id = _safe_id(sample_id)
    source_schema = canonicalize_schema(source_schema)
    target_schema = source_schema
    head_name = head_name_for_schema(target_schema)
    landmarks = _save_landmarks(output_dir, sample_id, points68)
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
    meta["mapping_audit"].setdefault("projection_to_68", projection_audit_for_schema(source_schema))

    out: dict[str, T.Any] = {
        "sample_id": sample_id,
        "dataset": dataset,
        "condition": _label(condition),
        "conditions": tuple(_label(item) for item in conditions),
        "image": str(image.resolve()),
        "landmarks": _relative_or_absolute(landmarks, output_dir),
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


def _filter(samples: list[dict[str, T.Any]], scenarios: tuple[str, ...] | None, limit: int | None) -> list[dict[str, T.Any]]:
    if scenarios:
        allowed = set(scenarios)
        samples = [sample for sample in samples if allowed.intersection(sample.get("conditions", ()))]
    if not limit:
        return samples
    counts: dict[str, int] = {}
    out = []
    for sample in samples:
        condition = str(sample.get("condition") or "default")
        if counts.get(condition, 0) >= limit:
            continue
        counts[condition] = counts.get(condition, 0) + 1
        out.append(sample)
    return out


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
        payload = _read_json(manifest_path)
        merged = [dict(item) for item in payload.get("samples", []) if isinstance(item, dict)]
    seen = {str(item.get("image")) for item in merged}
    for sample in samples:
        image = str(sample.get("image"))
        if not allow_overlap and image in seen:
            continue
        seen.add(image)
        merged.append(sample)

    summary = manifest_summary(merged)
    payload = {
        "version": TRAINING_MANIFEST_VERSION,
        "manifest_contract": TRAINING_MANIFEST_CONTRACT,
        "landmark_schema": "multi_schema",
        "metadata": {
            "builder": "AccurateFacialLandmarkDetection.tools.landmarks.build_quality_dataset",
            "dataset": dataset,
            "scenario": _label(scenario),
            "scenarios": list(scenarios or []),
            "sample_count": len(merged),
            "skipped_count": len(skipped or []),
        },
        **summary,
        "samples": merged,
    }
    _write_json(manifest_path, payload)

    _write_json(
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
    return manifest_path


def _draw_manifest_overlay(image_path: Path, landmarks_path: Path, output_path: Path) -> None:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"could not read overlay image: {image_path}")
    points = np.load(landmarks_path).astype(np.float32)[:, :2]
    height, width = image.shape[:2]
    if points.size and float(np.nanmax(points)) <= 1.5:
        points = points * np.asarray([width - 1, height - 1], dtype=np.float32)
    radius = max(1, int(round(max(width, height) / 256.0 * 2.0)))
    for index, (x, y) in enumerate(points):
        color = (0, 255, 255) if index % 5 else (0, 0, 255)
        cv2.circle(image, (int(round(x)), int(round(y))), radius, color, -1)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(output_path), image)
    if not ok:
        raise OSError(f"failed to write overlay image: {output_path}")


def _write_visual_audit(manifest_path: Path, output_dir: Path, *, limit: int = 50) -> Path:
    payload = _read_json(manifest_path)
    entries = payload.get("samples", [])
    if not isinstance(entries, list):
        raise ValueError(f"manifest {manifest_path} must contain samples list for visual audit")
    base_dir = manifest_path.parent
    audit_dir = output_dir / "visual_audit"
    overlays: list[dict[str, T.Any]] = []
    skipped: list[dict[str, str]] = []
    schema_counts: dict[str, int] = {}
    emitted = 0
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            continue
        schema = str(entry.get("target_schema") or entry.get("source_schema") or "unknown")
        schema_counts[schema] = schema_counts.get(schema, 0) + 1
        if emitted >= int(limit):
            continue
        image_value = entry.get("image")
        landmarks_value = entry.get("landmarks") or entry.get("ground_truth")
        sample_id = str(entry.get("sample_id") or index)
        if not image_value or not landmarks_value:
            skipped.append({"sample_id": sample_id, "reason": "missing image or landmarks"})
            continue
        image_path = _resolve_path(image_value, base_dir=base_dir)
        landmarks_path = _resolve_path(landmarks_value, base_dir=base_dir)
        overlay_name = _safe_id(sample_id).replace("/", "_").replace("#", "_")
        overlay_path = audit_dir / "overlays" / schema / f"{overlay_name}.jpg"
        try:
            _draw_manifest_overlay(image_path, landmarks_path, overlay_path)
        except Exception as err:  # noqa: BLE001
            skipped.append({"sample_id": sample_id, "reason": str(err)})
            continue
        overlays.append(
            {
                "sample_id": sample_id,
                "schema": schema,
                "image": str(image_path),
                "landmarks": str(landmarks_path),
                "overlay": str(overlay_path),
            }
        )
        emitted += 1

    report = {
        "manifest": str(manifest_path),
        "schema_counts": dict(sorted(schema_counts.items())),
        "overlay_count": len(overlays),
        "overlays": overlays,
        "skipped_count": len(skipped),
        "skipped_examples": skipped[:50],
    }
    report_path = audit_dir / "visual_audit.json"
    _write_json(report_path, report)
    return report_path


def _json_source(root: Path) -> Path | None:
    candidates = [root] if root.is_file() else sorted(root.rglob("*.json"))
    for path in candidates:
        if not path.is_file() or path.name == "dataset_audit.json":
            continue
        try:
            payload = _read_json(path)
        except Exception:
            continue
        if isinstance(payload, list) or (
            isinstance(payload, dict) and any(key in payload for key in ("samples", "entries"))
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
    payload = _read_json(path)
    entries = payload.get("samples", payload.get("entries", payload)) if isinstance(payload, dict) else payload
    if not isinstance(entries, list):
        raise ValueError(f"JSON source must contain list, entries, or samples list: {path}")
    image_base = Path(image_root) if image_root else path.parent
    samples: list[dict[str, T.Any]] = []
    skipped: list[dict[str, str]] = []
    for idx, entry in enumerate(entries):
        if not isinstance(entry, dict):
            continue
        image_value = entry.get("image") or entry.get("image_path") or entry.get("path")
        landmark_value = entry.get("landmarks") or entry.get("points") or entry.get("ground_truth") or entry.get("pts")
        if image_value is None or landmark_value is None:
            skipped.append({"sample_id": str(idx), "reason": "missing image or landmarks"})
            continue
        sample_id = str(entry.get("sample_id") or entry.get("id") or entry.get("name") or idx)
        entry_dataset = _dataset(str(entry.get("dataset") or dataset))
        metadata = _entry_metadata(entry, dataset=entry_dataset, source_file=path)
        source_schema = str(entry.get("source_schema") or metadata.get("source_schema") or "") or None
        try:
            points68, detected_schema = _load_points(landmark_value, base_dir=path.parent, source_schema=source_schema)
            image_path = _resolve_path(image_value, base_dir=image_base)
            if not image_path.is_file():
                raise FileNotFoundError(f"image not found: {image_path}")
        except Exception as err:  # noqa: BLE001
            skipped.append({"sample_id": sample_id, "reason": str(err)})
            continue
        conds = _conditions(entry, scenario)
        split = _split_from_entry_or_identity(entry, metadata, dataset=entry_dataset, sample_id=sample_id)
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


def _condition_for_landmark_file(dataset: str, path: Path, scenario: str) -> tuple[str, tuple[str, ...]]:
    parts = {_label(part) for part in path.parts}
    labels: list[str] = []
    if dataset == "cofw":
        labels.append("occlusion")
    if dataset in {"300w", "w300"}:
        labels.append("anchor")
    for token in ("profile", "pose", "occlusion", "occluded", "frontal", "normal", "clean", "challenging"):
        if token in parts or any(token in part for part in parts):
            labels.append(token)
    for token in ("train", "training"):
        if token in parts or any(token == part for part in parts):
            labels.append("trainset")
    for token in ("test", "testing", "validation", "val"):
        if token in parts or any(token == part for part in parts):
            labels.append("testset")
    if not labels:
        labels.append(_label(scenario))
    labels = list(dict.fromkeys(_label(item) for item in labels))
    return labels[0], tuple(labels)


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
    image_index = _build_image_index(image_base)
    samples: list[dict[str, T.Any]] = []
    skipped: list[dict[str, str]] = []
    landmark_paths = [
        path
        for suffix in LANDMARK_EXTS
        for path in sorted(root.rglob(f"*{suffix}"))
        if path.name != "manifest.json" and not path.name.startswith(".")
    ]
    for landmark_path in landmark_paths:
        if landmark_path.suffix.lower() == ".txt" and "98pt" in landmark_path.name.lower():
            continue
        try:
            points68, source_schema = _load_landmark_file(landmark_path)
        except Exception as err:  # noqa: BLE001
            skipped.append({"sample_id": landmark_path.as_posix(), "reason": str(err)})
            continue
        image = _matching_image(landmark_path, root=image_base, image_index=image_index)
        if image is None:
            skipped.append({"sample_id": landmark_path.as_posix(), "reason": "matching image not found"})
            continue
        sample_id = landmark_path.relative_to(root).with_suffix("").as_posix()
        condition, conds = _condition_for_landmark_file(dataset, landmark_path.relative_to(root), scenario)
        sample_image = image
        sample_points68 = points68
        sample_metadata = _path_identity_metadata(landmark_path, root=root, dataset=dataset)
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

        samples.append(
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
            )
        )
    if not samples:
        raise ValueError(f"no usable schema-aware landmark samples found under {root}; skipped={skipped[:5]}")
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
        "ffl2": ("images", "landmarks", "labels"),
        "fll3": ("images", "landmarks", "labels"),
        "cofw-original": ("images", "annotations", "landmarks"),
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


def _schema_parser_metadata(dataset: str, parser_name: str, landmark_path: Path, root: Path) -> dict[str, T.Any]:
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
) -> Path | None:
    roots = (Path(image_root),) if image_root else _source_image_roots(root, dataset)
    for candidate_root in roots:
        image_index = _build_image_index(candidate_root)
        image = _matching_image(landmark_path, root=candidate_root, image_index=image_index)
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
    values = [float(item) for item in re.findall(r"[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?", path.read_text(encoding="utf-8", errors="ignore"))]
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


def _native_conditions_for_split(scenario: str, split: str) -> tuple[str, tuple[str, ...]]:
    conds = tuple(dict.fromkeys((_label(scenario), f"{split}set")))
    return conds[0], conds


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
    for landmark_path in _landmark_paths(root):
        try:
            points, detected_schema = _load_landmark_file(landmark_path)
            if detected_schema != expected_schema:
                raise ValueError(f"{parser_name} expected {expected_schema}, got {detected_schema}")
            image = _image_for_dataset_landmarks(
                landmark_path,
                dataset=dataset,
                root=root,
                image_root=image_root,
            )
            if image is None:
                raise FileNotFoundError("matching image not found")
        except Exception as err:  # noqa: BLE001
            skipped.append({"sample_id": landmark_path.as_posix(), "reason": str(err)})
            continue

        sample_id = landmark_path.relative_to(root).with_suffix("").as_posix()
        condition, conds = _condition_for_landmark_file(dataset, landmark_path.relative_to(root), scenario)
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
        raise ValueError(f"no {dataset} samples built with {parser_name}; skipped={skipped[:10]}")

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
    payload = _read_json(path)
    entries = payload.get("samples", payload.get("entries", payload)) if isinstance(payload, dict) else payload
    if not isinstance(entries, list):
        raise ValueError(f"{parser_name} JSON source must contain list, entries, or samples list: {path}")
    image_base = Path(image_root) if image_root else path.parent
    samples: list[dict[str, T.Any]] = []
    skipped: list[dict[str, str]] = []
    for idx, entry in enumerate(entries):
        if not isinstance(entry, dict):
            continue
        image_value = entry.get("image") or entry.get("image_path") or entry.get("path")
        landmark_value = entry.get("landmarks") or entry.get("points") or entry.get("ground_truth") or entry.get("pts")
        sample_id = str(entry.get("sample_id") or entry.get("id") or entry.get("name") or idx)
        if image_value is None or landmark_value is None:
            skipped.append({"sample_id": sample_id, "reason": "missing image or landmarks"})
            continue
        metadata = _entry_metadata(entry, dataset=dataset, source_file=path)
        metadata["dataset_parser"] = parser_name
        metadata["parser_type"] = "dataset_specific"
        source_schema = str(entry.get("source_schema") or metadata.get("source_schema") or expected_schema)
        try:
            points, detected_schema = _load_points(
                landmark_value,
                base_dir=path.parent,
                source_schema=source_schema,
            )
            if detected_schema != expected_schema:
                raise ValueError(f"{parser_name} expected {expected_schema}, got {detected_schema}")
            image_path = _resolve_path(image_value, base_dir=image_base)
            if not image_path.is_file():
                raise FileNotFoundError(f"image not found: {image_path}")
        except Exception as err:  # noqa: BLE001
            skipped.append({"sample_id": sample_id, "reason": str(err)})
            continue

        conds = _conditions(entry, scenario)
        split = _split_from_entry_or_identity(entry, metadata, dataset=dataset, sample_id=sample_id)
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
        raise ValueError(f"no {dataset} JSON samples built with {parser_name}; skipped={skipped[:10]}")
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


def _helen_annotations_path(root: Path) -> Path | None:
    if root.is_file() and root.suffix.lower() == ".json":
        return root
    exact = sorted(root.rglob("annotations.json"), key=lambda item: len(item.parts))
    return exact[0] if exact else None


def _build_helen(
    root: Path,
    output_dir: Path,
    *,
    scenario: str,
    scenarios: tuple[str, ...] | None,
    limit: int | None,
    mode: str,
    allow_overlap: bool,
    image_root: str | None,
) -> Path:
    annotations = _helen_annotations_path(root)
    if annotations is None:
        return _build_expected_schema_dataset(
            root,
            output_dir,
            dataset="helen",
            expected_schema="2d_194",
            parser_name="helen_194",
            scenario=scenario,
            scenarios=scenarios,
            limit=limit,
            mode=mode,
            allow_overlap=allow_overlap,
            image_root=image_root,
        )

    payload = _read_json(annotations)
    if not isinstance(payload, list):
        raise ValueError(f"HELEN annotations.json must contain a list: {annotations}")

    image_base = Path(image_root) if image_root else annotations.parent
    image_index = _build_image_index(image_base) if image_base.is_dir() else {}
    samples: list[dict[str, T.Any]] = []
    skipped: list[dict[str, str]] = []
    for index, entry in enumerate(payload):
        sample_id = f"annotations/{index:05d}"
        try:
            if isinstance(entry, dict):
                image_name = str(entry.get("image") or entry.get("image_path") or entry.get("filename") or "")
                raw_points = entry.get("landmarks") or entry.get("points")
                width = entry.get("width") or entry.get("image_width")
                height = entry.get("height") or entry.get("image_height")
            elif isinstance(entry, (list, tuple)) and len(entry) >= 2:
                image_info = entry[0]
                raw_points = entry[1]
                if not isinstance(image_info, (list, tuple)) or not image_info:
                    raise ValueError("missing HELEN image info")
                image_name = str(image_info[0])
                width = image_info[1] if len(image_info) > 1 else None
                height = image_info[2] if len(image_info) > 2 else None
            else:
                raise ValueError("unsupported HELEN annotation row")
            if not image_name or raw_points is None:
                raise ValueError("missing image name or landmarks")
            points, detected_schema = _canonical_points(raw_points, source_schema="2d_194")
            if detected_schema != "2d_194":
                raise ValueError(f"HELEN expected 2d_194, got {detected_schema}")
            image = _find_named_image((image_base, annotations.parent, root), image_name, image_index=image_index)
            if image is None:
                raise FileNotFoundError(f"HELEN image not found: {image_name}")
        except Exception as err:  # noqa: BLE001
            skipped.append({"sample_id": sample_id, "reason": str(err)})
            continue

        sample_id = f"helen/{Path(image_name).stem}"
        split = _deterministic_split("helen", sample_id)
        condition, conds = _native_conditions_for_split(scenario, split)
        metadata = {
            "dataset": "helen",
            "dataset_parser": "helen_annotations_json",
            "parser_type": "dataset_specific",
            "annotation_file": str(annotations.resolve()),
            "source_image_name": image_name,
            "source_schema": "2d_194",
        }
        if width is not None:
            metadata["image_width"] = int(width)
        if height is not None:
            metadata["image_height"] = int(height)
        samples.append(
            _with_split(
                _sample(
                    output_dir=output_dir,
                    dataset="helen",
                    sample_id=sample_id,
                    image=image,
                    points68=points,
                    condition=condition,
                    conditions=conds,
                    source_schema="2d_194",
                    source_id=sample_id,
                    metadata=metadata,
                ),
                split,
            )
        )

    if not samples:
        raise ValueError(f"no HELEN annotation samples built; skipped={skipped[:10]}")
    return _write_manifest(
        output_dir,
        "helen",
        scenario,
        _filter(samples, scenarios, limit),
        mode=mode,
        allow_overlap=allow_overlap,
        scenarios=scenarios,
        skipped=skipped,
    )


def _lapa_release_roots(root: Path) -> list[Path]:
    candidates = [root, root / "LaPa"]
    candidates.extend(sorted(path for path in root.rglob("LaPa") if path.is_dir()))
    out: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        if not candidate.is_dir():
            continue
        resolved = candidate.resolve()
        if resolved in seen:
            continue
        if any((candidate / split / "landmarks").is_dir() for split in ("train", "val", "test")):
            seen.add(resolved)
            out.append(candidate)
    return out


def _build_lapa(
    root: Path,
    output_dir: Path,
    *,
    scenario: str,
    scenarios: tuple[str, ...] | None,
    limit: int | None,
    mode: str,
    allow_overlap: bool,
    image_root: str | None,
) -> Path:
    release_roots = _lapa_release_roots(root)
    if not release_roots:
        return _build_expected_schema_dataset(
            root,
            output_dir,
            dataset="lapa",
            expected_schema="2d_106",
            parser_name="lapa_106",
            scenario=scenario,
            scenarios=scenarios,
            limit=limit,
            mode=mode,
            allow_overlap=allow_overlap,
            image_root=image_root,
        )

    samples: list[dict[str, T.Any]] = []
    skipped: list[dict[str, str]] = []
    for release_root in release_roots:
        for source_split in ("train", "val", "test"):
            split_dir = release_root / source_split
            landmark_dir = split_dir / "landmarks"
            image_dir = split_dir / "images"
            label_dir = split_dir / "labels"
            if not landmark_dir.is_dir():
                continue
            for landmark_path in sorted(landmark_dir.glob("*.txt")):
                try:
                    points, detected_schema = _load_landmark_file(landmark_path)
                    if detected_schema != "2d_106":
                        raise ValueError(f"LaPa expected 2d_106, got {detected_schema}")
                    roots = [image_dir]
                    if image_root:
                        roots.insert(0, Path(image_root))
                    image = _find_named_image(roots, landmark_path.stem)
                    if image is None:
                        raise FileNotFoundError(f"LaPa image not found for {landmark_path.name}")
                except Exception as err:  # noqa: BLE001
                    skipped.append({"sample_id": landmark_path.as_posix(), "reason": str(err)})
                    continue

                split = _manifest_split_for_source_split(source_split)
                condition, conds = _native_conditions_for_split(scenario, split)
                label_path = label_dir / f"{landmark_path.stem}.png"
                sample_id = f"{source_split}/{landmark_path.stem}"
                metadata = _path_identity_metadata(landmark_path, root=root, dataset="lapa")
                metadata.update(
                    {
                        "dataset_parser": "lapa_release_106",
                        "parser_type": "dataset_specific",
                        "source_split": source_split,
                        "source_schema": "2d_106",
                        "source_image": str(image.resolve()),
                    }
                )
                if label_path.is_file():
                    metadata["semantic_label"] = str(label_path.resolve())
                samples.append(
                    _with_split(
                        _sample(
                            output_dir=output_dir,
                            dataset="lapa",
                            sample_id=sample_id,
                            image=image,
                            points68=points,
                            condition=condition,
                            conditions=conds,
                            source_schema="2d_106",
                            source_id=sample_id,
                            metadata=metadata,
                        ),
                        split,
                    )
                )

    if not samples:
        raise ValueError(f"no LaPa native release samples built; skipped={skipped[:10]}")
    return _write_manifest(
        output_dir,
        "lapa",
        scenario,
        _filter(samples, scenarios, limit),
        mode=mode,
        allow_overlap=allow_overlap,
        scenarios=scenarios,
        skipped=skipped,
    )


def _jd_landmark_sources(root: Path) -> list[tuple[Path, Path | None, Path | None, str, str]]:
    out: list[tuple[Path, Path | None, Path | None, str, str]] = []
    test_roots = [root / "Test_data1"]
    if root.name == "Test_data1":
        test_roots.insert(0, root)
    for test_root in test_roots:
        landmark_dir = test_root / "landmark"
        if landmark_dir.is_dir():
            out.append((landmark_dir, test_root / "picture", test_root / "rect", "test", "test_data1"))

    corrected_roots = [root / "Corrected_landmark"]
    if root.name == "Corrected_landmark":
        corrected_roots.insert(0, root)
    for corrected_root in corrected_roots:
        if corrected_root.is_dir():
            out.append((corrected_root, None, None, "corrected", "corrected_landmark"))
    return out


def _split_hint_from_jd_name(name: str) -> str | None:
    lowered = name.lower()
    if "image_train" in lowered or "_train_" in lowered:
        return "train"
    if "image_test" in lowered or "_test_" in lowered:
        return "test"
    return None


def _build_jd_landmark(
    root: Path,
    output_dir: Path,
    *,
    scenario: str,
    scenarios: tuple[str, ...] | None,
    limit: int | None,
    mode: str,
    allow_overlap: bool,
    image_root: str | None,
) -> Path:
    sources = _jd_landmark_sources(root)
    if not sources:
        return _build_expected_schema_dataset(
            root,
            output_dir,
            dataset="jd-landmark",
            expected_schema="2d_106",
            parser_name="jd_landmark_106",
            scenario=scenario,
            scenarios=scenarios,
            limit=limit,
            mode=mode,
            allow_overlap=allow_overlap,
            image_root=image_root,
        )

    index_root = Path(image_root) if image_root else root
    image_index = _build_image_index(index_root) if index_root.is_dir() else {}
    samples: list[dict[str, T.Any]] = []
    skipped: list[dict[str, str]] = []
    for landmark_dir, image_dir, bbox_dir, source_split, source_name in sources:
        for landmark_path in sorted(landmark_dir.glob("*.txt")):
            image_name = _image_name_from_landmark_name(landmark_path)
            try:
                points, detected_schema = _load_landmark_file(landmark_path)
                if detected_schema != "2d_106":
                    raise ValueError(f"JD-landmark expected 2d_106, got {detected_schema}")
                roots = []
                if image_root:
                    roots.append(Path(image_root))
                if image_dir is not None:
                    roots.append(image_dir)
                roots.append(root)
                image = _find_named_image(roots, image_name, image_index=image_index)
                if image is None:
                    raise FileNotFoundError(f"JD-landmark image not found: {image_name}")
            except Exception as err:  # noqa: BLE001
                skipped.append({"sample_id": landmark_path.as_posix(), "reason": str(err)})
                continue

            bbox_path = _bbox_file_for_landmark(bbox_dir, landmark_path, image_name=image_name)
            bbox = _read_bbox_file(bbox_path)
            sample_id = f"{source_name}/{Path(image_name).stem}"
            metadata = _path_identity_metadata(landmark_path, root=root, dataset="jd-landmark")
            metadata.update(
                {
                    "dataset_parser": "jd_landmark_release_106",
                    "parser_type": "dataset_specific",
                    "source_release": source_name,
                    "source_split": source_split,
                    "source_schema": "2d_106",
                    "source_image_name": image_name,
                    "source_image": str(image.resolve()),
                }
            )
            if bbox_path is not None and bbox is not None:
                metadata["source_bbox"] = str(bbox_path.resolve())
                metadata["bbox_xyxy"] = bbox

            split_hint = "test" if source_split == "test" else _split_hint_from_jd_name(image_name)
            split = _split_from_entry_or_identity(
                {"split": split_hint} if split_hint else {},
                metadata,
                dataset="jd-landmark",
                sample_id=sample_id,
            )
            condition, conds = _native_conditions_for_split(scenario, split)
            samples.append(
                _with_split(
                    _sample(
                        output_dir=output_dir,
                        dataset="jd-landmark",
                        sample_id=sample_id,
                        image=image,
                        points68=points,
                        condition=condition,
                        conditions=conds,
                        source_schema="2d_106",
                        source_id=sample_id,
                        metadata=metadata,
                    ),
                    split,
                )
            )

    if not samples:
        raise ValueError(f"no JD-landmark native release samples built; skipped={skipped[:10]}")
    return _write_manifest(
        output_dir,
        "jd-landmark",
        scenario,
        _filter(samples, scenarios, limit),
        mode=mode,
        allow_overlap=allow_overlap,
        scenarios=scenarios,
        skipped=skipped,
    )


def _ffl_split_dirs(root: Path, dataset: str) -> list[tuple[Path, str]]:
    if dataset == "ffl2":
        candidates = [(root / "train", "train"), (root, "train")]
    else:
        base_candidates = [root / "FLL3_dataset", root]
        candidates = []
        for base in base_candidates:
            candidates.extend((base / split, split) for split in ("train", "val", "test"))
    out: list[tuple[Path, str]] = []
    seen: set[Path] = set()
    for split_dir, split in candidates:
        if not (split_dir / "landmark").is_dir():
            continue
        resolved = split_dir.resolve()
        if resolved not in seen:
            seen.add(resolved)
            out.append((split_dir, split))
    return out


def _build_ffl_family(
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
) -> Path:
    split_dirs = _ffl_split_dirs(root, dataset)
    if not split_dirs:
        return _build_expected_schema_dataset(
            root,
            output_dir,
            dataset=dataset,
            expected_schema="2d_106",
            parser_name=f"{dataset}_106",
            scenario=scenario,
            scenarios=scenarios,
            limit=limit,
            mode=mode,
            allow_overlap=allow_overlap,
            image_root=image_root,
        )

    samples: list[dict[str, T.Any]] = []
    skipped: list[dict[str, str]] = []
    for split_dir, source_split in split_dirs:
        landmark_dir = split_dir / "landmark"
        image_dir = split_dir / ("picture_mask" if (split_dir / "picture_mask").is_dir() else "picture")
        bbox_dir = split_dir / "bbox"
        for landmark_path in sorted(landmark_dir.glob("*.txt")):
            try:
                points, detected_schema = _load_landmark_file(landmark_path)
                if detected_schema != "2d_106":
                    raise ValueError(f"{dataset} expected 2d_106, got {detected_schema}")
                roots = [image_dir]
                if image_root:
                    roots.insert(0, Path(image_root))
                image = _find_named_image(roots, landmark_path.stem)
                if image is None:
                    raise FileNotFoundError(f"{dataset} image not found for {landmark_path.name}")
            except Exception as err:  # noqa: BLE001
                skipped.append({"sample_id": landmark_path.as_posix(), "reason": str(err)})
                continue

            bbox_path = _bbox_file_for_landmark(bbox_dir, landmark_path)
            bbox = _read_bbox_file(bbox_path)
            split = _manifest_split_for_source_split(source_split)
            condition, conds = _native_conditions_for_split(scenario, split)
            sample_id = f"{source_split}/{landmark_path.stem}"
            metadata = _path_identity_metadata(landmark_path, root=root, dataset=dataset)
            metadata.update(
                {
                    "dataset_parser": f"{dataset}_release_106",
                    "parser_type": "dataset_specific",
                    "source_split": source_split,
                    "source_schema": "2d_106",
                    "source_image": str(image.resolve()),
                }
            )
            if bbox_path is not None and bbox is not None:
                metadata["source_bbox"] = str(bbox_path.resolve())
                metadata["bbox_xyxy"] = bbox
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
                        source_schema="2d_106",
                        source_id=sample_id,
                        metadata=metadata,
                    ),
                    split,
                )
            )

    if not samples:
        raise ValueError(f"no {dataset} native release samples built; skipped={skipped[:10]}")
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


def _menpo_list_files(root: Path, dataset: str) -> list[Path]:
    names = {f"{dataset}_train.txt", f"{dataset}_test.txt", f"{dataset}_val.txt"}
    return sorted(path for path in root.rglob("*.txt") if path.name.lower() in names)


def _list_split_from_path(path: Path) -> str:
    lowered = path.stem.lower()
    if "train" in lowered:
        return "train"
    if "val" in lowered or "test" in lowered:
        return "test"
    return "train"


def _menpo_identity_from_image(dataset: str, image_name: str) -> dict[str, str]:
    stem = Path(image_name).stem
    metadata = {"image_id": stem}
    if dataset == "xm2vts":
        parts = stem.split("_")
        if parts:
            metadata["subject_id"] = parts[0]
        if len(parts) > 1:
            metadata["session_id"] = parts[1]
        if len(parts) > 2:
            metadata["capture_id"] = parts[2]
        return metadata
    match = re.match(r"^(?P<subject>\d+)(?P<session>[A-Za-z])(?P<capture>\d+)$", stem)
    if match:
        metadata["subject_id"] = match.group("subject")
        metadata["session_id"] = match.group("session")
        metadata["capture_id"] = match.group("capture")
    else:
        metadata["subject_id"] = stem
    return metadata


def _parse_menpo_list_line(line: str) -> tuple[str, list[float] | None, list[list[float]] | None, np.ndarray]:
    parts = line.split()
    if len(parts) < 2:
        raise ValueError("empty Menpo-style list row")
    image_name = parts[0]
    values = [float(item) for item in parts[1:]]
    bbox: list[float] | None = None
    coarse: list[list[float]] | None = None
    landmark_values = values
    if len(values) == 150:
        bbox = [float(item) for item in values[:4]]
        coarse = np.asarray(values[4:14], dtype=np.float32).reshape(5, 2).astype(float).tolist()
        landmark_values = values[14:]
    points, detected_schema = _canonical_points(np.asarray(landmark_values, dtype=np.float32), source_schema="2d_68")
    if detected_schema != "2d_68":
        raise ValueError(f"Menpo-style list expected 2d_68, got {detected_schema}")
    return image_name, bbox, coarse, points


def _build_subject_session_dataset(
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
) -> Path:
    list_files = _menpo_list_files(root, dataset)
    if list_files:
        samples: list[dict[str, T.Any]] = []
        skipped: list[dict[str, str]] = []
        for list_path in list_files:
            source_split = _list_split_from_path(list_path)
            split = _manifest_split_for_source_split(source_split)
            for line_number, line in enumerate(list_path.read_text(encoding="utf-8", errors="ignore").splitlines(), start=1):
                if not line.strip() or line.lstrip().startswith("#"):
                    continue
                try:
                    image_name, bbox, coarse, points = _parse_menpo_list_line(line)
                    roots = [list_path.parent, root]
                    if image_root:
                        roots.insert(0, Path(image_root))
                    image = _find_named_image(roots, image_name)
                    if image is None:
                        raise FileNotFoundError(f"{dataset} image not found: {image_name}")
                except Exception as err:  # noqa: BLE001
                    skipped.append({"sample_id": f"{list_path}:{line_number}", "reason": str(err)})
                    continue

                sample_id = f"{list_path.stem}/{Path(image_name).stem}"
                condition, conds = _native_conditions_for_split(scenario, split)
                metadata: dict[str, T.Any] = {
                    "dataset": dataset,
                    "dataset_parser": f"{dataset}_menpo_list_68",
                    "parser_type": "dataset_specific",
                    "source_annotation": str(list_path.resolve()),
                    "source_line": line_number,
                    "source_split": source_split,
                    "source_schema": "2d_68",
                    "source_image_name": image_name,
                    "source_image": str(image.resolve()),
                    **_menpo_identity_from_image(dataset, image_name),
                }
                if bbox is not None:
                    metadata["bbox_xyxy"] = bbox
                if coarse is not None:
                    metadata["five_point_landmarks"] = coarse
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
                            source_schema="2d_68",
                            source_id=sample_id,
                            metadata=metadata,
                        ),
                        split,
                    )
                )

        if not samples:
            raise ValueError(f"no {dataset} Menpo-style list samples built; skipped={skipped[:10]}")
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

    json_path = _json_source(root)
    if json_path is not None:
        manifest = _build_json(
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
        payload = _read_json(manifest)
        for sample in payload.get("samples", []):
            if not isinstance(sample, dict):
                continue
            metadata = sample.setdefault("metadata", {})
            if isinstance(metadata, dict):
                metadata.setdefault("dataset_parser", f"{dataset}_menpo_style")
                metadata.setdefault("parser_type", "dataset_specific")
        _write_json(manifest, payload)
        return manifest

    image_base = Path(image_root) if image_root else root
    image_index = _build_image_index(image_base)
    samples: list[dict[str, T.Any]] = []
    skipped: list[dict[str, str]] = []
    for landmark_path in _landmark_paths(root):
        try:
            points, source_schema = _load_landmark_file(landmark_path)
            image = _matching_image(landmark_path, root=image_base, image_index=image_index)
            if image is None:
                raise FileNotFoundError("matching image not found")
        except Exception as err:  # noqa: BLE001
            skipped.append({"sample_id": landmark_path.as_posix(), "reason": str(err)})
            continue
        rel = landmark_path.relative_to(root)
        sample_id = rel.with_suffix("").as_posix()
        condition, conds = _condition_for_landmark_file(dataset, rel, scenario)
        metadata = _path_identity_metadata(landmark_path, root=root, dataset=dataset)
        metadata.update(
            {
                "dataset_parser": f"{dataset}_menpo_style",
                "parser_type": "dataset_specific",
                "source_schema": source_schema,
            }
        )
        split = _split_from_entry_or_identity({}, metadata, dataset=dataset, sample_id=sample_id)
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
                    source_schema=source_schema,
                    source_id=sample_id,
                    metadata=metadata,
                ),
                split,
            )
        )

    if not samples:
        raise ValueError(f"no {dataset} Menpo-style samples built; skipped={skipped[:10]}")
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


def _cofw_original_mat_files(root: Path) -> list[tuple[Path, str]]:
    out: list[tuple[Path, str]] = []
    for path in sorted(root.rglob("*.mat"), key=lambda item: len(item.parts)):
        name = path.name.lower()
        if "cofw" not in name:
            continue
        if "train" in name:
            out.append((path, "train"))
        elif "test" in name:
            out.append((path, "test"))
    return out


def _mat_first_key(payload: T.Mapping[str, T.Any], names: tuple[str, ...]) -> T.Any:
    lowered = {key.lower(): key for key in payload if not key.startswith("__")}
    for name in names:
        if name.lower() in lowered:
            return payload[lowered[name.lower()]]
    for key in payload:
        key_l = key.lower()
        if key.startswith("__"):
            continue
        if any(name.lower() in key_l for name in names):
            return payload[key]
    return None


def _cofw_original_points_array(value: T.Any) -> list[np.ndarray]:
    if value is None:
        return []
    arr = np.asarray(value)
    if arr.dtype == object:
        out = []
        for item in arr.reshape(-1):
            try:
                points, _ = _canonical_points(item, source_schema="2d_29")
            except Exception:
                continue
            out.append(points)
        return out
    arr = np.asarray(value, dtype=np.float32)
    if arr.ndim == 3 and arr.shape[-2:] == (29, 2):
        return [arr[index].astype(np.float32) for index in range(arr.shape[0])]
    if arr.ndim == 3 and arr.shape[:2] == (29, 2):
        return [arr[:, :, index].astype(np.float32) for index in range(arr.shape[2])]
    if arr.ndim == 2:
        if arr.shape[0] == 87:
            return [
                np.stack((arr[:29, index], arr[29:58, index]), axis=1).astype(np.float32)
                for index in range(arr.shape[1])
            ]
        if arr.shape[1] == 87:
            return [
                np.stack((arr[index, :29], arr[index, 29:58]), axis=1).astype(np.float32)
                for index in range(arr.shape[0])
            ]
        if arr.shape[1] == 58:
            return [row.reshape(29, 2).astype(np.float32) for row in arr]
        if arr.shape[0] == 58:
            return [arr[:, index].reshape(29, 2).astype(np.float32) for index in range(arr.shape[1])]
        if arr.shape == (29, 2):
            return [arr.astype(np.float32)]
    return []


def _cofw_original_image_array(value: T.Any) -> list[np.ndarray]:
    if value is None:
        return []
    arr = np.asarray(value)
    if arr.dtype == object:
        return [np.asarray(item) for item in arr.reshape(-1)]
    if arr.ndim == 4:
        if arr.shape[-1] in (1, 3, 4):
            return [arr[index] for index in range(arr.shape[0])]
        if arr.shape[0] in (1, 3, 4):
            return [np.moveaxis(arr[:, :, :, index], 0, -1) for index in range(arr.shape[-1])]
        return [arr[:, :, :, index] for index in range(arr.shape[-1])]
    if arr.ndim in (2, 3):
        return [arr]
    return []


def _cofw_original_visibility(value: T.Any, count: int) -> list[list[bool]]:
    if value is None:
        return [[True] * 29 for _ in range(count)]
    arr = np.asarray(value)
    if arr.dtype == object:
        rows = [np.asarray(item).reshape(-1) for item in arr.reshape(-1)]
    elif arr.ndim == 2 and arr.shape[1] == 29:
        rows = [arr[index] for index in range(arr.shape[0])]
    elif arr.ndim == 2 and arr.shape[0] == 29:
        rows = [arr[:, index] for index in range(arr.shape[1])]
    elif arr.size == 29:
        rows = [arr.reshape(-1)]
    else:
        rows = []
    out: list[list[bool]] = []
    for row in rows[:count]:
        # COFW stores occlusion flags in common releases: 1 means occluded.
        out.append([not bool(item) for item in np.asarray(row).reshape(-1)[:29]])
    while len(out) < count:
        out.append([True] * 29)
    return out


def _write_cofw_original_image(output_dir: Path, sample_id: str, image: np.ndarray) -> Path:
    path = output_dir / "images" / "cofw-original" / f"{_safe_id(sample_id).replace('/', '_')}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(image)
    if arr.ndim == 3 and arr.shape[0] in (1, 3, 4) and arr.shape[-1] not in (1, 3, 4):
        arr = np.moveaxis(arr, 0, -1)
    if arr.ndim == 2:
        write_arr = arr
    else:
        if arr.shape[-1] == 1:
            arr = arr[:, :, 0]
            write_arr = arr
        else:
            write_arr = arr[:, :, [2, 1, 0]] if arr.shape[-1] >= 3 else arr
    write_arr = np.clip(write_arr, 0, 255).astype(np.uint8)
    ok = cv2.imwrite(str(path), write_arr)
    if not ok:
        raise OSError(f"failed to write COFW original image: {path}")
    return path


def _is_hdf5_mat(path: Path) -> bool:
    try:
        with path.open("rb") as handle:
            header = handle.read(128)
        return header.startswith(b"\x89HDF\r\n\x1a\n") or b"MATLAB 7.3 MAT-file" in header
    except OSError:
        return False


def _cofw_original_hdf5_arrays(
    path: Path,
    declared_split: str,
) -> tuple[list[np.ndarray], list[np.ndarray], list[list[bool]], list[list[float] | None]]:
    try:
        import h5py
    except ImportError as err:
        raise RuntimeError("h5py is required to read COFW original MATLAB v7.3 files") from err

    with h5py.File(path, "r") as handle:
        trainish = declared_split == "train"
        phis_key = "phisTr" if trainish and "phisTr" in handle else "phisT" if "phisT" in handle else "phisTr"
        images_key = "IsTr" if trainish and "IsTr" in handle else "IsT" if "IsT" in handle else "IsTr"
        bboxes_key = "bboxesTr" if trainish and "bboxesTr" in handle else "bboxesT" if "bboxesT" in handle else None
        phis = np.asarray(handle[phis_key], dtype=np.float32)
        if phis.ndim != 2:
            raise ValueError(f"COFW original phis must be 2D, got {phis.shape}")
        if phis.shape[0] == 87:
            columns = [phis[:, index] for index in range(phis.shape[1])]
        elif phis.shape[1] == 87:
            columns = [phis[index, :] for index in range(phis.shape[0])]
        else:
            raise ValueError(f"COFW original phis must have 87 rows/columns, got {phis.shape}")

        points_rows = [
            np.stack((column[:29], column[29:58]), axis=1).astype(np.float32)
            for column in columns
        ]
        visibility_rows = [
            [not bool(item) for item in np.asarray(column[58:87]).reshape(-1)[:29]]
            for column in columns
        ]

        images: list[np.ndarray] = []
        image_refs = handle[images_key]
        for index in range(len(points_rows)):
            ref = image_refs[0, index] if image_refs.ndim == 2 and image_refs.shape[0] == 1 else image_refs[index]
            images.append(np.asarray(handle[ref]))

        bbox_rows: list[list[float] | None] = [None] * len(points_rows)
        if bboxes_key and bboxes_key in handle:
            bboxes = np.asarray(handle[bboxes_key], dtype=np.float32)
            if bboxes.ndim == 2 and bboxes.shape[0] == 4:
                bbox_rows = [[float(value) for value in bboxes[:, index]] for index in range(bboxes.shape[1])]
            elif bboxes.ndim == 2 and bboxes.shape[1] == 4:
                bbox_rows = [[float(value) for value in bboxes[index, :]] for index in range(bboxes.shape[0])]
            bbox_rows = (bbox_rows + [None] * len(points_rows))[:len(points_rows)]

    return points_rows, images, visibility_rows, bbox_rows


def _build_cofw_original(
    root: Path,
    output_dir: Path,
    *,
    scenario: str,
    scenarios: tuple[str, ...] | None,
    limit: int | None,
    mode: str,
    allow_overlap: bool,
    image_root: str | None,
) -> Path:
    mat_files = _cofw_original_mat_files(root)
    if not mat_files:
        return _build_expected_schema_dataset(
            root,
            output_dir,
            dataset="cofw-original",
            expected_schema="2d_29",
            parser_name="cofw_original_29",
            scenario=scenario,
            scenarios=scenarios,
            limit=limit,
            mode=mode,
            allow_overlap=allow_overlap,
            image_root=image_root,
        )

    samples: list[dict[str, T.Any]] = []
    skipped: list[dict[str, str]] = []
    sio: T.Any | None = None
    for mat_path, declared_split in mat_files:
        try:
            if _is_hdf5_mat(mat_path):
                points_rows, images, visibility_rows, bbox_rows = _cofw_original_hdf5_arrays(mat_path, declared_split)
            else:
                if sio is None:
                    try:
                        import scipy.io as sio_module
                    except ImportError as err:
                        raise RuntimeError("scipy is required to read COFW original .mat files") from err
                    sio = sio_module
                payload = sio.loadmat(mat_path)
                points_rows = _cofw_original_points_array(
                    _mat_first_key(payload, ("phisTr", "phisT", "phis", "points", "landmarks"))
                )
                images = _cofw_original_image_array(
                    _mat_first_key(payload, ("IsTr", "IsT", "images", "image"))
                )
                visibility_rows = _cofw_original_visibility(
                    _mat_first_key(payload, ("occlusionsTr", "occlusionsT", "occlusion", "occ", "occluded")),
                    len(points_rows),
                )
                bbox_rows = [None] * len(points_rows)
        except Exception as err:  # noqa: BLE001
            skipped.append({"sample_id": mat_path.as_posix(), "reason": str(err)})
            continue
        for index, points in enumerate(points_rows):
            sample_id = f"cofw_original/{declared_split}/{mat_path.stem}_{index:04d}"
            try:
                points29 = normalize_landmark_array(points, schema="2d_29")
                if index < len(images):
                    image_path = _write_cofw_original_image(output_dir, sample_id, images[index])
                else:
                    image = _matching_image(mat_path, root=Path(image_root) if image_root else root)
                    if image is None:
                        raise FileNotFoundError("COFW original image not found in MAT or image root")
                    image_path = image
            except Exception as err:  # noqa: BLE001
                skipped.append({"sample_id": sample_id, "reason": str(err)})
                continue
            visibility = visibility_rows[index] if index < len(visibility_rows) else [True] * 29
            metadata = {
                "dataset": "cofw-original",
                "dataset_parser": "cofw_original_29",
                "parser_type": "dataset_specific",
                "annotation_file": str(mat_path.resolve()),
                "source_schema": "2d_29",
                "split": declared_split,
                "cofw_original_index": index,
                "occlusion_mask": [not bool(item) for item in visibility],
                "landmark_score_visibility_mask": visibility,
            }
            if index < len(bbox_rows) and bbox_rows[index] is not None:
                metadata["bbox_xyxy"] = bbox_rows[index]
            condition = "occlusion" if any(not bool(item) for item in visibility) else scenario
            samples.append(
                _with_split(
                    _sample(
                        output_dir=output_dir,
                        dataset="cofw-original",
                        sample_id=sample_id,
                        image=image_path,
                        points68=points29,
                        condition=condition,
                        conditions=tuple(dict.fromkeys((_label(condition), f"{declared_split}set"))),
                        source_schema="2d_29",
                        source_id=sample_id,
                        metadata=metadata,
                        visibility=visibility,
                    ),
                    declared_split,
                )
            )

    if not samples:
        raise ValueError(f"no COFW original 29-point samples built; skipped={skipped[:10]}")
    return _write_manifest(
        output_dir,
        "cofw-original",
        scenario,
        _filter(samples, scenarios, limit),
        mode=mode,
        allow_overlap=allow_overlap,
        scenarios=scenarios,
        skipped=skipped,
    )



def _cofw68_annotation_paths(root: Path) -> list[Path]:
    return sorted(
        path for path in root.rglob("*_points.mat")
        if path.is_file() and "test_annotations" in path.as_posix()
    )


def _cofw_test_color_mat(root: Path) -> Path:
    matches = sorted(root.rglob("COFW_test_color.mat"), key=lambda item: len(item.parts))
    if not matches:
        raise FileNotFoundError(f"COFW_test_color.mat not found below {root}")
    return matches[0]


def _cofw_test_bboxes(root: Path) -> np.ndarray | None:
    matches = sorted(root.rglob("cofw68_test_bboxes.mat"), key=lambda item: len(item.parts))
    if not matches:
        return None
    try:
        import scipy.io as sio
        payload = sio.loadmat(matches[0])
        boxes = np.asarray(payload.get("bboxes"), dtype=np.float32)
        return boxes if boxes.ndim == 2 and boxes.shape[1] == 4 else None
    except Exception:
        return None


def _cofw_annotation_index(path: Path) -> int:
    text = path.stem.replace("_points", "")
    return int(text) - 1


def _cofw_points_and_occ(path: Path) -> tuple[np.ndarray, list[bool], dict[str, T.Any]]:
    import scipy.io as sio

    payload = sio.loadmat(path)
    if "Points" not in payload:
        raise ValueError(f"COFW68 annotation missing Points: {path}")
    points68, schema = _canonical_points(payload["Points"], source_schema="2d_68")

    occ_raw = payload.get("Occ")
    occ_mask: list[bool] = []
    visibility: list[bool] = []
    if occ_raw is not None:
        occ_arr = np.asarray(occ_raw).reshape(-1)
        occ_mask = [bool(x) for x in occ_arr[:68]]
        visibility = [not bool(x) for x in occ_arr[:68]]
    if len(visibility) != 68:
        visibility = [True] * 68

    metadata = {
        "source_schema": schema,
        "occlusion_mask": occ_mask,
        "landmark_score_visibility_mask": visibility,
    }
    return points68, visibility, metadata


def _cofw_hdf5_image_by_index(mat_path: Path, index: int) -> np.ndarray:
    import h5py

    with h5py.File(mat_path, "r") as h5:
        refs = h5["IsT"][()]
        ref = refs.reshape(-1)[index]
        arr = np.asarray(h5[ref])

    # COFW HDF5 images are usually channel-first: C,H,W.
    if arr.ndim == 3 and arr.shape[0] in (1, 3, 4):
        arr = np.moveaxis(arr, 0, -1)

    # MATLAB/HDF5 stores COFW image planes transposed relative to the
    # annotation coordinate frame. Points/bboxes align after swapping H/W.
    if arr.ndim == 3:
        arr = np.transpose(arr, (1, 0, 2))
    elif arr.ndim == 2:
        arr = arr.T

    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr[:, :, 0]
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(arr)


def _write_cofw_image(output_dir: Path, index: int, image: np.ndarray) -> Path:
    from PIL import Image

    path = output_dir / "images" / f"cofw_test_{index + 1:04d}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.is_file():
        Image.fromarray(image).save(path)
    return path


def _build_cofw(
    root: Path,
    output_dir: Path,
    *,
    scenario: str,
    scenarios: tuple[str, ...] | None,
    limit: int | None,
    mode: str,
    allow_overlap: bool,
) -> Path:
    color_mat = _cofw_test_color_mat(root)
    annotations = _cofw68_annotation_paths(root)
    boxes = _cofw_test_bboxes(root)

    samples: list[dict[str, T.Any]] = []
    skipped: list[dict[str, str]] = []

    for ann in annotations:
        try:
            idx = _cofw_annotation_index(ann)
            sample_id = f"cofw_test_{idx + 1:04d}"
            split = _deterministic_split("cofw", sample_id)
            points68, visibility, metadata = _cofw_points_and_occ(ann)
            image_arr = _cofw_hdf5_image_by_index(color_mat, idx)
            image_path = _write_cofw_image(output_dir, idx, image_arr)

            raw_bbox = None
            if boxes is not None and 0 <= idx < len(boxes):
                raw_bbox = [float(x) for x in boxes[idx].tolist()]
                x, y, width, height = raw_bbox
                metadata["face_bbox_raw"] = raw_bbox
                metadata["face_bbox_raw_format"] = "xywh"
                metadata["face_bbox_raw_source"] = "cofw68_test_bboxes"
                metadata["face_bbox"] = [x, y, x + width, y + height]
                metadata["face_bbox_format"] = "ltrb"
                metadata["face_bbox_source"] = "cofw68_test_bboxes"

            metadata.update(
                {
                    "annotation_file": str(ann.resolve()),
                    "cofw_index": idx + 1,
                    "split": split,
                    "image_source_mat": str(color_mat.resolve()),
                    "source_schema": "2d_68",
                }
            )

            entry_for_crop = {"visibility": visibility}
            visible_mask, visible_mask_source = _cofw_visibility_mask_and_source(entry_for_crop, metadata)
            bbox_ltrb, bbox_source = _cofw_choose_crop_bbox(
                entry_for_crop,
                metadata,
                image_path,
                points68,
                visible_mask,
            )
            crop_image_path, crop_points68, crop_metadata = _crop_sample_image(
                output_dir=output_dir,
                dataset="cofw",
                sample_id=sample_id,
                image_path=image_path,
                points68=points68,
                bbox_xyxy=bbox_ltrb,
                bbox_source=bbox_source,
                pad_ratio=0.25,
            )
        except Exception as err:  # noqa: BLE001
            skipped.append({"sample_id": ann.as_posix(), "reason": str(err)})
            continue

        metadata.update(crop_metadata)
        metadata["face_bbox"] = [float(v) for v in bbox_ltrb]
        metadata["face_bbox_format"] = "ltrb"
        metadata["face_bbox_source"] = bbox_source
        metadata["crop_visibility_mask_source"] = visible_mask_source
        metadata["crop_visible_landmark_count"] = int(np.asarray(visible_mask, dtype=bool).sum())
        metadata["visibility"] = visibility

        samples.append(
            _with_split(
                _sample(
                    output_dir=output_dir,
                    dataset="cofw",
                    sample_id=sample_id,
                    image=crop_image_path,
                    points68=crop_points68,
                    condition="occlusion",
                    conditions=("occlusion", f"{split}set"),
                    source_schema="2d_68",
                    source_id=sample_id,
                    metadata=metadata,
                    visibility=visibility,
                ),
                split,
            )
        )

    if not samples:
        raise ValueError(f"no COFW68 test samples built; skipped={skipped[:5]}")

    return _write_manifest(
        output_dir,
        "cofw",
        scenario,
        _filter(samples, scenarios, limit),
        mode=mode,
        allow_overlap=allow_overlap,
        scenarios=scenarios,
        skipped=skipped,
    )



def _find_multipie_root(root: Path) -> Path:
    candidates = sorted(
        path.parent for path in root.rglob("MultiPIE_*_train.txt")
        if path.is_file()
    )
    if candidates:
        return candidates[0]
    if (root / "image").is_dir():
        return root
    raise FileNotFoundError(f"MultiPIE root not found below {root}")


def _multipie_annotation_files(root: Path) -> list[Path]:
    multipie_root = _find_multipie_root(root)
    files = sorted(multipie_root.glob("MultiPIE_*_train.txt"))
    if not files:
        raise FileNotFoundError(f"MultiPIE train txt files not found in {multipie_root}")
    return files


def _multipie_conditions(annotation_file: Path, image_rel: str, scenario: str) -> tuple[str, tuple[str, ...]]:
    text = f"{annotation_file.name} {image_rel}".lower()
    labels: list[str] = []
    if "profile" in text:
        labels.append("profile")
    if "semifrontal" in text or "semi_frontal" in text:
        labels.append("semifrontal")
    if "train" in annotation_file.name.lower():
        labels.append("trainset")
    if not labels:
        labels.append(_label(scenario))
    labels = list(dict.fromkeys(_label(item) for item in labels))
    return labels[0], tuple(labels)


def _multipie_parse_line(line: str, *, line_no: int, path: Path) -> tuple[str, np.ndarray, list[float], str]:
    parts = line.strip().split()
    if len(parts) < 2:
        raise ValueError("empty or malformed line")

    image_rel = parts[0].replace("\\", "/")
    try:
        values = [float(item) for item in parts[1:]]
    except ValueError as err:
        raise ValueError(f"non-numeric landmark value on line {line_no}") from err

    header_values = 14  # 4 bbox + 5 detector/reference points * 2
    dense_count = len(values) - header_values

    if dense_count == 78:
        bbox = [float(item) for item in values[:4]]
        raw = values[header_values:]
        points = np.asarray(raw, dtype=np.float32).reshape(39, 2)
        return image_rel, points, bbox, "multipie_profile_39"

    if dense_count != 136:
        raise ValueError(
            f"line {line_no} in {path} has {len(values)} numeric values; "
            "expected 150 for 68-point rows or 92 for 39-point profile rows"
        )

    bbox = [float(item) for item in values[:4]]
    raw = values[header_values:]
    points = np.asarray(raw, dtype=np.float32).reshape(68, 2)
    points = normalize_landmarks(points, source_schema="2d_68")
    return image_rel, points, bbox, "2d_68"


def _bbox_from_points(points68: np.ndarray) -> list[float]:
    left, top = np.min(points68, axis=0)
    right, bottom = np.max(points68, axis=0)
    return [float(left), float(top), float(right), float(bottom)]


def _normalizer_from_bbox(bbox: list[float]) -> float:
    value = max(float(bbox[2] - bbox[0]), float(bbox[3] - bbox[1]))
    return value if np.isfinite(value) and value > 0.0 else 1.0


def _build_multipie(
    root: Path,
    output_dir: Path,
    *,
    scenario: str,
    scenarios: tuple[str, ...] | None,
    limit: int | None,
    mode: str,
    allow_overlap: bool,
) -> Path:
    multipie_root = _find_multipie_root(root)
    annotation_files = _multipie_annotation_files(root)

    samples: list[dict[str, T.Any]] = []
    skipped: list[dict[str, str]] = []

    for annotation_file in annotation_files:
        for line_no, line in enumerate(annotation_file.read_text(encoding="utf-8", errors="ignore").splitlines(), 1):
            if not line.strip():
                continue
            try:
                image_rel, points68, bbox, source_schema = _multipie_parse_line(
                    line,
                    line_no=line_no,
                    path=annotation_file,
                )
                image_path = (multipie_root / image_rel).resolve()
                if not image_path.is_file():
                    raise FileNotFoundError(f"image not found: {image_path}")

                condition, conds = _multipie_conditions(annotation_file, image_rel, scenario)
                bbox = bbox or _bbox_from_points(points68)
                sample_id = Path(image_rel).with_suffix("").as_posix()
                normalizer = _normalizer(points68, sample_id)
                split = "train" if "trainset" in conds else _deterministic_split("multipie", sample_id)

                metadata = {
                    "annotation_file": str(annotation_file.resolve()),
                    "annotation_line": line_no,
                    "image_id": image_rel,
                    "face_bbox": bbox,
                    "face_bbox_source": "multipie_landmark_bounds",
                    "normalizer_source": DEFAULT_NORMALIZER_SOURCE,
                    "source_schema": source_schema,
                    "split": split,
                }

                sample_kwargs = dict(
                    output_dir=output_dir,
                    dataset="multipie",
                    sample_id=sample_id,
                    image=image_path,
                    points68=points68,
                    condition=condition,
                    conditions=conds,
                    source_schema=source_schema,
                    source_id=sample_id,
                    metadata=metadata,
                )
                try:
                    sample = _sample(**sample_kwargs, normalizer=normalizer)
                except TypeError:
                    sample = _sample(**sample_kwargs)

                samples.append(_with_split(sample, split))
            except Exception as err:  # noqa: BLE001
                skipped.append(
                    {
                        "sample_id": f"{annotation_file.as_posix()}:{line_no}",
                        "reason": str(err),
                    }
                )
                continue

    if not samples:
        raise ValueError(f"no MultiPIE samples built; skipped={skipped[:5]}")

    return _write_manifest(
        output_dir,
        "multipie",
        scenario,
        _filter(samples, scenarios, limit),
        mode=mode,
        allow_overlap=allow_overlap,
        scenarios=scenarios,
        skipped=skipped,
    )


def _parse_wflw_line(line: str, line_no: int) -> tuple[np.ndarray, list[float], dict[str, int], str]:
    parts = line.split()
    if len(parts) < 197:
        raise ValueError(f"WFLW line {line_no} has too few fields")
    points = np.asarray([float(value) for value in parts[:196]], dtype=np.float32).reshape(98, 2)
    bbox: list[float] = []
    if len(parts) >= 201:
        bbox = [float(value) for value in parts[196:200]]
    attrs = dict.fromkeys(WFLW_ATTRIBUTE_NAMES, 0)
    if len(parts) >= 207:
        values = [int(float(value)) for value in parts[200:206]]
        attrs = dict(zip(WFLW_ATTRIBUTE_NAMES, values, strict=True))
        image_rel = " ".join(parts[206:])
    else:
        image_rel = parts[-1]
    return points, bbox, attrs, image_rel


def _find_wflw_annotations(root: Path) -> Path | None:
    for pattern in (
        "list_98pt_rect_attr_train_test.txt",
        "list_98pt_rect_attr_train.txt",
        "list_98pt_rect_attr_test.txt",
        "*98pt*rect*attr*.txt",
    ):
        matches = sorted(root.rglob(pattern), key=lambda item: len(item.parts))
        if matches:
            return matches[0]
    return None


def _find_wflw_images(root: Path) -> Path:
    for name in ("WFLW_images", "images", "Images", "WFLW"):
        matches = [path for path in root.rglob(name) if path.is_dir()]
        if matches:
            return sorted(matches, key=lambda item: len(item.parts))[0]
    return root


def _build_wflw(
    root: Path | None,
    output_dir: Path,
    *,
    annotation_file: str | None,
    image_root: str | None,
    scenario: str,
    scenarios: tuple[str, ...] | None,
    limit: int | None,
    mode: str,
    allow_overlap: bool,
) -> Path:
    if annotation_file:
        annotations = Path(annotation_file)
        root_for_images = annotations.parent
    else:
        root_for_images = root or Path(".")
        annotations = _find_wflw_annotations(root_for_images)
    if annotations is None or not annotations.is_file():
        if root is None:
            raise FileNotFoundError("WFLW annotation file not found; pass --wflw-annotations or --source-dir")
        logger.info("WFLW annotations not found; falling back to generic directory parsing")
        return _build_directory(
            root,
            output_dir,
            dataset="wflw",
            scenario=scenario,
            scenarios=scenarios,
            limit=limit,
            mode=mode,
            allow_overlap=allow_overlap,
            image_root=image_root,
        )

    image_base = Path(image_root) if image_root else _find_wflw_images(root_for_images)
    rows = []
    counts: dict[str, int] = {}
    for line_no, line in enumerate(annotations.read_text(encoding="utf-8", errors="ignore").splitlines(), 1):
        if not line.strip():
            continue
        row = _parse_wflw_line(line, line_no)
        rows.append(row)
        counts[row[3]] = counts.get(row[3], 0) + 1

    seen: dict[str, int] = {}
    samples: list[dict[str, T.Any]] = []
    skipped: list[dict[str, str]] = []
    for points98, bbox, attrs, image_rel in rows:
        seen[image_rel] = seen.get(image_rel, 0) + 1
        base_id = Path(image_rel).with_suffix("").as_posix()
        sample_id = base_id if counts[image_rel] <= 1 else f"{base_id}#face-{seen[image_rel]:02d}"
        conds = tuple(name for name in WFLW_ATTRIBUTE_NAMES if attrs.get(name)) or (_label(scenario),)
        image_path = (image_base / image_rel).resolve()
        if not image_path.is_file():
            skipped.append({"sample_id": sample_id, "reason": f"image not found: {image_path}"})
            continue
        points98 = normalize_landmark_array(points98, schema="2d_98")
        crop_image_path, crop_points68, crop_metadata = _crop_sample_image(
            output_dir=output_dir,
            dataset="wflw",
            sample_id=sample_id,
            image_path=image_path,
            points68=points98,
            bbox_xyxy=bbox,
            bbox_source="wflw_rect_attr_bbox",
            pad_ratio=0.25,
        )
        split = _deterministic_split("wflw", sample_id)
        metadata = {"bbox": bbox, "attributes": attrs, "image_id": image_rel, "split": split}
        metadata.update(crop_metadata)
        samples.append(
            _with_split(
                _sample(
                    output_dir=output_dir,
                    dataset="wflw",
                    sample_id=sample_id,
                    image=crop_image_path,
                    points68=crop_points68,
                    condition=conds[0],
                    conditions=tuple(dict.fromkeys((*(_label(item) for item in conds), f"{split}set"))),
                    source_schema="2d_98",
                    source_id=sample_id,
                    metadata=metadata,
                ),
                split,
            )
        )
    if not samples:
        raise ValueError(f"no WFLW samples built; skipped={skipped[:5]}")
    return _write_manifest(
        output_dir,
        "wflw",
        scenario,
        _filter(samples, scenarios, limit),
        mode=mode,
        allow_overlap=allow_overlap,
        scenarios=scenarios,
        skipped=skipped,
    )


def _candidate_frame_stems(frame_index: int) -> tuple[str, ...]:
    one_based = int(frame_index) + 1
    return tuple(
        dict.fromkeys(
            (
                f"frame_{frame_index:06d}",
                f"{frame_index:06d}",
                f"{frame_index:05d}",
                f"{frame_index:04d}",
                str(frame_index),
                f"frame_{one_based:06d}",
                f"{one_based:06d}",
                f"{one_based:05d}",
                f"{one_based:04d}",
                str(one_based),
            )
        )
    )


def _find_frame_landmark_file(root: Path, video_id: str, frame_index: int) -> Path | None:
    safe_video_id = str(video_id).replace("\\", "/").strip("/")
    search_roots = [
        root / "annotations" / safe_video_id,
        root / "landmarks" / safe_video_id,
        root / "labels" / safe_video_id,
        root / safe_video_id,
    ]
    for search_root in search_roots:
        if not search_root.is_dir():
            continue
        for stem in _candidate_frame_stems(frame_index):
            for suffix in LANDMARK_EXTS:
                candidate = search_root / f"{stem}{suffix}"
                if candidate.is_file():
                    return candidate

    for stem in _candidate_frame_stems(frame_index):
        for suffix in LANDMARK_EXTS:
            matches = sorted(root.rglob(f"{safe_video_id}*{stem}{suffix}"), key=lambda item: len(item.parts))
            if matches:
                return matches[0]
    return None


def _build_video_dataset(
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
    video_root: str | None,
    frame_output_dir: str | None,
    frame_stride: int,
    max_frames_per_video: int | None,
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

    videos_root = Path(video_root) if video_root else root
    videos = video_files(videos_root)
    if not videos:
        return _build_directory(
            root,
            output_dir,
            dataset=dataset,
            scenario=scenario,
            scenarios=scenarios,
            limit=limit,
            mode=mode,
            allow_overlap=allow_overlap,
            image_root=image_root,
        )

    frame_root = Path(frame_output_dir) if frame_output_dir else output_dir / "frames" / dataset
    samples: list[dict[str, T.Any]] = []
    skipped: list[dict[str, str]] = []
    for video_path in videos:
        video_id = video_path.resolve().relative_to(videos_root.resolve()).with_suffix("").as_posix()
        split = _deterministic_split(dataset, video_id)
        try:
            frame_records = extract_video_frames(
                video_path,
                frame_root,
                stride=frame_stride,
                max_frames=max_frames_per_video,
                video_id=video_id,
            )
        except Exception as err:  # noqa: BLE001
            skipped.append({"sample_id": video_id, "reason": str(err)})
            continue

        for record in frame_records:
            frame_index = int(record["frame_index"])
            sample_id = f"{dataset}/{video_id}/frame_{frame_index:06d}"
            landmark_path = _find_frame_landmark_file(root, video_id, frame_index)
            if landmark_path is None:
                skipped.append({"sample_id": sample_id, "reason": "matching frame landmarks not found"})
                continue
            try:
                points, source_schema = _load_landmark_file(landmark_path)
            except Exception as err:  # noqa: BLE001
                skipped.append({"sample_id": sample_id, "reason": str(err)})
                continue

            metadata = {
                "dataset": dataset,
                "video_id": video_id,
                "frame_index": frame_index,
                "frame_id": record["frame_id"],
                "split": split,
                "split_safe_id": video_id,
                "source_video": str(video_path.resolve()),
                "source_landmarks": str(landmark_path.resolve()),
            }
            conditions = ("video_frame", f"{split}set")
            samples.append(
                _with_split(
                    _sample(
                        output_dir=output_dir,
                        dataset=dataset,
                        sample_id=sample_id,
                        image=Path(record["image"]),
                        points68=points,
                        condition="video_frame",
                        conditions=conditions,
                        source_schema=source_schema,
                        source_id=sample_id,
                        metadata=metadata,
                    ),
                    split,
                )
            )

    if not samples:
        raise ValueError(f"no {dataset} video-frame samples built; skipped={skipped[:10]}")

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


@contextlib.contextmanager
def _source_context(source_dir: str | None, source_zip: str | None) -> T.Iterator[Path | None]:
    if source_dir and source_zip:
        raise ValueError("pass only one of --source-dir or --source-zip")
    if source_dir:
        path = Path(source_dir)
        if not path.is_dir():
            raise FileNotFoundError(f"source directory not found: {path}")
        yield path
    elif source_zip:
        with extract_archive_to_temp(source_zip) as root:
            yield root
    else:
        yield None


def build(args: argparse.Namespace) -> Path:
    dataset = _dataset(args.dataset)
    if dataset not in SUPPORTED_DATASETS:
        raise ValueError(f"unsupported dataset {args.dataset!r}")
    output_dir = Path(args.output_dir)
    scenarios = _parse_csv(args.scenarios)
    limit = None if not args.samples_per_scenario else int(args.samples_per_scenario)
    if args.cofw_json and dataset != "cofw":
        raise ValueError("--cofw-json is only valid with --dataset cofw")

    with _source_context(args.source_dir, args.source_zip) as root:
        if dataset == "wflw":
            manifest_path = _build_wflw(
                root,
                output_dir,
                annotation_file=args.wflw_annotations,
                image_root=args.image_root,
                scenario=args.scenario,
                scenarios=scenarios,
                limit=limit,
                mode=args.manifest_mode,
                allow_overlap=args.allow_overlap,
            )
        elif dataset == "cofw":
            if args.cofw_json:
                manifest_path = _build_cofw_json_cropped(
                    Path(args.cofw_json),
                    output_dir,
                    scenario=args.scenario,
                    scenarios=scenarios,
                    limit=limit,
                    mode=args.manifest_mode,
                    allow_overlap=args.allow_overlap,
                    image_root=args.image_root,
                )
            else:
                if root is None:
                    raise ValueError("--source-dir or --source-zip is required for COFW")
                manifest_path = _build_cofw(
                    root,
                    output_dir,
                    scenario=args.scenario,
                    scenarios=scenarios,
                    limit=limit,
                    mode=args.manifest_mode,
                    allow_overlap=args.allow_overlap,
                )
        elif dataset == "multipie":
            if root is None:
                raise ValueError("--source-dir or --source-zip is required for MultiPIE")
            manifest_path = _build_multipie(
                root,
                output_dir,
                scenario=args.scenario,
                scenarios=scenarios,
                limit=limit,
                mode=args.manifest_mode,
                allow_overlap=args.allow_overlap,
            )
        elif dataset == "cofw-original":
            if root is None:
                raise ValueError("--source-dir or --source-zip is required for COFW original")
            manifest_path = _build_cofw_original(
                root,
                output_dir,
                scenario=args.scenario,
                scenarios=scenarios,
                limit=limit,
                mode=args.manifest_mode,
                allow_overlap=args.allow_overlap,
                image_root=args.image_root,
            )
        elif dataset == "helen":
            if root is None:
                raise ValueError("--source-dir or --source-zip is required for HELEN")
            manifest_path = _build_helen(
                root,
                output_dir,
                scenario=args.scenario,
                scenarios=scenarios,
                limit=limit,
                mode=args.manifest_mode,
                allow_overlap=args.allow_overlap,
                image_root=args.image_root,
            )
        elif dataset == "lapa":
            if root is None:
                raise ValueError("--source-dir or --source-zip is required for LaPa")
            manifest_path = _build_lapa(
                root,
                output_dir,
                scenario=args.scenario,
                scenarios=scenarios,
                limit=limit,
                mode=args.manifest_mode,
                allow_overlap=args.allow_overlap,
                image_root=args.image_root,
            )
        elif dataset == "jd-landmark":
            if root is None:
                raise ValueError("--source-dir or --source-zip is required for JD-landmark")
            manifest_path = _build_jd_landmark(
                root,
                output_dir,
                scenario=args.scenario,
                scenarios=scenarios,
                limit=limit,
                mode=args.manifest_mode,
                allow_overlap=args.allow_overlap,
                image_root=args.image_root,
            )
        elif dataset in {"ffl2", "fll3"}:
            if root is None:
                raise ValueError(f"--source-dir or --source-zip is required for {dataset}")
            manifest_path = _build_ffl_family(
                root,
                output_dir,
                dataset=dataset,
                scenario=args.scenario,
                scenarios=scenarios,
                limit=limit,
                mode=args.manifest_mode,
                allow_overlap=args.allow_overlap,
                image_root=args.image_root,
            )
        elif dataset in {"xm2vts", "frgc"}:
            if root is None:
                raise ValueError(f"--source-dir or --source-zip is required for {dataset}")
            manifest_path = _build_subject_session_dataset(
                root,
                output_dir,
                dataset=dataset,
                scenario=args.scenario,
                scenarios=scenarios,
                limit=limit,
                mode=args.manifest_mode,
                allow_overlap=args.allow_overlap,
                image_root=args.image_root,
            )
        elif dataset in {"300vw", "wflw-v"}:
            if root is None and not args.video_root:
                raise ValueError("--source-dir, --source-zip, or --video-root is required for video datasets")
            manifest_path = _build_video_dataset(
                root or Path(args.video_root),
                output_dir,
                dataset=dataset,
                scenario=args.scenario,
                scenarios=scenarios,
                limit=limit,
                mode=args.manifest_mode,
                allow_overlap=args.allow_overlap,
                image_root=args.image_root,
                video_root=args.video_root,
                frame_output_dir=args.frame_output_dir,
                frame_stride=args.frame_stride,
                max_frames_per_video=args.max_frames_per_video,
            )
        else:
            if root is None:
                raise ValueError("--source-dir, --source-zip, --wflw-annotations, or --cofw-json is required")
            manifest_path = _build_directory(
                root,
                output_dir,
                dataset=dataset,
                scenario=args.scenario,
                scenarios=scenarios,
                limit=limit,
                mode=args.manifest_mode,
                allow_overlap=args.allow_overlap,
                image_root=args.image_root,
            )

    if args.write_overlays:
        _write_visual_audit(manifest_path, output_dir, limit=args.audit_overlay_limit)
    return manifest_path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=SUPPORTED_DATASETS)
    parser.add_argument("--source-dir", default=None)
    parser.add_argument("--source-zip", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--scenario", default="default")
    parser.add_argument("--scenarios", default=None)
    parser.add_argument("--samples-per-scenario", type=int, default=None)
    parser.add_argument("--manifest-mode", choices=("replace", "merge"), default="replace")
    parser.add_argument("--allow-overlap", action="store_true")
    parser.add_argument("--image-root", default=None)
    parser.add_argument("--video-root", default=None)
    parser.add_argument("--frame-output-dir", default=None)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--max-frames-per-video", type=int, default=None)
    parser.add_argument("--recursive", action="store_true", help="Accepted for compatibility; scans are recursive.")
    parser.add_argument("--wflw-annotations", default=None)
    parser.add_argument("--cofw-json", default=None)
    parser.add_argument("--write-overlays", action="store_true", help="Write visual landmark overlay audit images for built samples.")
    parser.add_argument("--audit-overlay-limit", type=int, default=50)
    parser.add_argument("--no-39pt-profile", action="store_true", help="Accepted for compatibility; non-68 samples are skipped.")
    parser.add_argument("--include-39pt-profile", action="store_true", help="Accepted for compatibility; non-68 samples are skipped.")
    parser.add_argument("--cache-dir", default=None, help="Accepted for compatibility; explicit sources are preferred.")
    parser.add_argument("--download-url", default=None, help="Accepted for compatibility; explicit sources are preferred.")
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--log-level", default="INFO", choices=("DEBUG", "INFO", "WARNING", "ERROR"))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(levelname)s:%(name)s:%(message)s")
    try:
        manifest = build(args)
    except Exception as err:  # noqa: BLE001
        logger.error("manifest build failed: %s", err)
        return 1
    logger.info("Wrote landmark manifest: %s", manifest)
    print(f"Wrote landmark manifest: {manifest}")
    return 0


# ---------------------------------------------------------------------------
# COFW bbox helpers.
#
# Some local COFW materializations mark benchmark boxes as ltrb even when the
# values are effectively xywh. Choose a bbox by checking whether it contains the
# visible/valid landmarks. Fall back to visible-landmark bbox when the benchmark
# bbox is inconsistent.
# ---------------------------------------------------------------------------
def _cofw_visibility_mask_and_source(entry, metadata):
    raw = entry.get("visibility", metadata.get("visibility"))
    if isinstance(raw, (list, tuple)) and len(raw) == 68:
        return np.asarray([bool(v) for v in raw], dtype=bool), "visibility"

    raw = metadata.get("landmark_score_visibility_mask")
    if isinstance(raw, (list, tuple)) and len(raw) == 68:
        return np.asarray([bool(v) for v in raw], dtype=bool), "landmark_score_visibility_mask"

    # COFW Occ is occluded=True. If present, invert it.
    occ = metadata.get("occlusion", entry.get("occlusion"))
    if not isinstance(occ, (list, tuple)):
        occ = metadata.get("occlusion_mask")
    if isinstance(occ, (list, tuple)) and len(occ) == 68:
        return np.asarray([not bool(v) for v in occ], dtype=bool), "occlusion_mask"

    return np.ones((68,), dtype=bool), "all_landmarks_fallback"


def _cofw_visibility_mask_for_crop(entry, metadata):
    return _cofw_visibility_mask_and_source(entry, metadata)[0]


def _cofw_bbox_candidates(entry, metadata):
    candidates = []

    def add(label, bbox, fmt):
        if bbox is None:
            return
        try:
            vals = [float(v) for v in list(bbox)[:4]]
        except Exception:
            return
        if len(vals) != 4 or not all(np.isfinite(vals)):
            return
        x, y, a, b = vals
        if fmt == "xywh":
            if a > 0 and b > 0:
                candidates.append((label + "_xywh", [x, y, x + a, y + b]))
        elif fmt == "ltrb":
            if a > x and b > y:
                candidates.append((label + "_ltrb", [x, y, a, b]))
        else:
            # Include both interpretations; the scorer will choose.
            if a > x and b > y:
                candidates.append((label + "_as_ltrb", [x, y, a, b]))
            if a > 0 and b > 0:
                candidates.append((label + "_as_xywh", [x, y, x + a, y + b]))

    source = str(
        entry.get("face_bbox_source")
        or entry.get("bbox_source")
        or metadata.get("face_bbox_source")
        or metadata.get("bbox_source")
        or ""
    ).lower()
    raw_fmt = str(
        entry.get("face_bbox_raw_format")
        or entry.get("bbox_raw_format")
        or metadata.get("face_bbox_raw_format")
        or metadata.get("bbox_raw_format")
        or ""
    ).lower()
    fmt = str(
        entry.get("face_bbox_format")
        or entry.get("bbox_format")
        or metadata.get("face_bbox_format")
        or metadata.get("bbox_format")
        or ""
    ).lower()

    # Prefer raw benchmark bbox if available.
    raw_bbox = entry.get("face_bbox_raw") or entry.get("bbox_raw") or metadata.get("face_bbox_raw") or metadata.get("bbox_raw")
    if raw_bbox is not None:
        add("face_bbox_raw", raw_bbox, raw_fmt or "xywh")

    bbox = entry.get("face_bbox") or entry.get("bbox") or metadata.get("face_bbox") or metadata.get("bbox")
    if bbox is not None:
        if "cofw" in source and raw_bbox is None:
            # The local builder has shown stale/misleading "ltrb" metadata for
            # COFW. For COFW benchmark boxes, consider xywh first.
            add("face_bbox", bbox, "xywh")
            add("face_bbox", bbox, "ltrb")
        else:
            add("face_bbox", bbox, fmt or None)

    # Deduplicate.
    out = []
    seen = set()
    for label, box in candidates:
        key = tuple(round(float(v), 4) for v in box)
        if key not in seen:
            seen.add(key)
            out.append((label, box))
    return out


def _cofw_score_bbox_candidate(bbox_ltrb, points68, visible_mask, image_hw):
    try:
        left, top, right, bottom = _bbox_to_square_with_padding(
            bbox_ltrb,
            image_hw=image_hw,
            pad_ratio=0.25,
        )
    except Exception:
        return -1, float("inf")

    pts = np.asarray(points68, dtype=np.float32)
    mask = np.asarray(visible_mask, dtype=bool)
    if not mask.any():
        mask = np.ones((68,), dtype=bool)

    valid = pts[mask]
    inside = (
        (valid[:, 0] >= left - 2)
        & (valid[:, 0] <= right + 2)
        & (valid[:, 1] >= top - 2)
        & (valid[:, 1] <= bottom + 2)
    )
    count_inside = int(inside.sum())
    area = float(max(right - left, 1.0) * max(bottom - top, 1.0))
    return count_inside, area


def _cofw_choose_crop_bbox(entry, metadata, image_path, points68, visible_mask):
    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError(f"could not read COFW image: {image_path}")
    image_hw = image_bgr.shape[:2]

    visible_mask = np.asarray(visible_mask, dtype=bool)
    if not visible_mask.any():
        visible_mask = np.ones((68,), dtype=bool)
    required = int(visible_mask.sum())
    best = None

    for label, bbox in _cofw_bbox_candidates(entry, metadata):
        score, area = _cofw_score_bbox_candidate(bbox, points68, visible_mask, image_hw)
        if best is None or score > best[0] or (score == best[0] and area < best[1]):
            best = (score, area, label, bbox)

    if best is not None and best[0] >= max(1, int(0.95 * required)):
        return best[3], f"cofw_bbox_v2:{best[2]}"

    # Benchmark bbox is inconsistent with visible landmarks. Use visible
    # landmarks to derive the crop. This is safer for CD-ViT than training on
    # exploded coordinates.
    pts = np.asarray(points68, dtype=np.float32)
    return _bbox_from_points_xyxy(pts[visible_mask]), "cofw_bbox_v2:visible_landmark_bbox_fallback"


def _cofw_bbox4(value: T.Any) -> list[float] | None:
    if value is None:
        return None
    try:
        values = [float(v) for v in list(value)[:4]]
    except Exception:
        return None
    if len(values) != 4 or not all(np.isfinite(values)):
        return None
    return values


def _cofw_entry_is_materialized_crop(entry: T.Mapping[str, T.Any], metadata: T.Mapping[str, T.Any]) -> bool:
    crop_bbox = entry.get("crop_bbox_xyxy") or metadata.get("crop_bbox_xyxy")
    crop_output_size = entry.get("crop_output_size") or metadata.get("crop_output_size")
    original_image = entry.get("original_image") or metadata.get("original_image")
    try:
        output_size = int(crop_output_size)
    except (TypeError, ValueError):
        output_size = None
    return crop_bbox is not None or output_size == 256 or original_image is not None


def _build_cofw_json_cropped(
    path: Path,
    output_dir: Path,
    *,
    scenario: str,
    scenarios: tuple[str, ...] | None,
    limit: int | None,
    mode: str,
    allow_overlap: bool,
    image_root: str | None,
) -> Path:
    payload = _read_json(path)
    entries = payload.get("samples", payload.get("entries", payload)) if isinstance(payload, dict) else payload
    if not isinstance(entries, list):
        raise ValueError(f"COFW JSON source must contain list, entries, or samples list: {path}")

    image_base = Path(image_root) if image_root else path.parent
    samples: list[dict[str, T.Any]] = []
    skipped: list[dict[str, str]] = []

    for idx, entry in enumerate(entries):
        if not isinstance(entry, dict):
            continue

        metadata = dict(entry.get("metadata", {})) if isinstance(entry.get("metadata"), dict) else {}
        image_value = entry.get("image") or entry.get("image_path") or entry.get("path")
        landmark_value = entry.get("landmarks") or entry.get("points") or entry.get("ground_truth") or entry.get("pts")
        sample_id = str(entry.get("sample_id") or entry.get("id") or entry.get("name") or f"cofw/{idx:04d}")

        if _cofw_entry_is_materialized_crop(entry, metadata):
            raise ValueError(
                "--cofw-json points to an already-cropped manifest entry "
                f"{sample_id!r}; use raw COFW JSON/source instead"
            )

        if image_value is None or landmark_value is None:
            skipped.append({"sample_id": sample_id, "reason": "missing image or landmarks"})
            continue

        try:
            image_path = _resolve_path(image_value, base_dir=image_base)
            source_schema = str(entry.get("source_schema") or metadata.get("source_schema") or "") or None
            points68, detected_schema = _load_points(
                landmark_value,
                base_dir=path.parent,
                source_schema=source_schema,
            )
            visibility = entry.get("visibility", metadata.get("visibility"))
            visible_mask, visible_mask_source = _cofw_visibility_mask_and_source(entry, metadata)
            bbox_ltrb, bbox_source = _cofw_choose_crop_bbox(entry, metadata, image_path, points68, visible_mask)
            crop_image_path, crop_points68, crop_metadata = _crop_sample_image(
                output_dir=output_dir,
                dataset="cofw",
                sample_id=sample_id,
                image_path=image_path,
                points68=points68,
                bbox_xyxy=bbox_ltrb,
                bbox_source=bbox_source,
                pad_ratio=0.25,
            )
        except Exception as err:  # noqa: BLE001
            skipped.append({"sample_id": sample_id, "reason": str(err)})
            continue

        is_occluded = True
        if isinstance(visibility, (list, tuple)) and visibility:
            is_occluded = any(not bool(v) for v in visibility)

        explicit_split = _label(entry.get("split") or metadata.get("split") or "")
        split = explicit_split if explicit_split in {"train", "test"} else _deterministic_split("cofw", sample_id)

        conds = _conditions(entry, "occlusion" if is_occluded else scenario)
        if is_occluded and "occlusion" not in conds:
            conds = tuple(dict.fromkeys((*conds, "occlusion")))
        split_condition = f"{split}set"
        if split_condition not in conds:
            conds = tuple(dict.fromkeys((*conds, split_condition)))

        merged_metadata = dict(metadata)
        input_bbox = _cofw_bbox4(entry.get("face_bbox") or entry.get("bbox") or metadata.get("face_bbox") or metadata.get("bbox"))
        if input_bbox is not None:
            merged_metadata.setdefault("face_bbox_input", input_bbox)
            input_format = str(
                entry.get("face_bbox_format")
                or entry.get("bbox_format")
                or metadata.get("face_bbox_format")
                or metadata.get("bbox_format")
                or ""
            ).strip()
            if input_format:
                merged_metadata.setdefault("face_bbox_input_format", input_format)
            input_source = str(
                entry.get("face_bbox_source")
                or entry.get("bbox_source")
                or metadata.get("face_bbox_source")
                or metadata.get("bbox_source")
                or "cofw_json"
            )
            merged_metadata.setdefault("face_bbox_input_source", input_source)

        raw_bbox = _cofw_bbox4(entry.get("face_bbox_raw") or entry.get("bbox_raw") or metadata.get("face_bbox_raw") or metadata.get("bbox_raw"))
        if raw_bbox is not None:
            merged_metadata.setdefault("face_bbox_raw", raw_bbox)
            raw_format = str(
                entry.get("face_bbox_raw_format")
                or entry.get("bbox_raw_format")
                or metadata.get("face_bbox_raw_format")
                or metadata.get("bbox_raw_format")
                or "xywh"
            )
            merged_metadata.setdefault("face_bbox_raw_format", raw_format)
            raw_source = str(
                entry.get("face_bbox_raw_source")
                or entry.get("bbox_raw_source")
                or metadata.get("face_bbox_raw_source")
                or metadata.get("bbox_raw_source")
                or "cofw_json"
            )
            merged_metadata.setdefault("face_bbox_raw_source", raw_source)

        merged_metadata.update(crop_metadata)
        merged_metadata["face_bbox"] = [float(v) for v in bbox_ltrb]
        merged_metadata["face_bbox_format"] = "ltrb"
        merged_metadata["face_bbox_source"] = bbox_source
        merged_metadata["crop_visibility_mask_source"] = visible_mask_source
        merged_metadata["crop_visible_landmark_count"] = int(np.asarray(visible_mask, dtype=bool).sum())
        merged_metadata.setdefault("source_schema", detected_schema)
        if visibility is not None:
            merged_metadata["visibility"] = visibility

        samples.append(
            _with_split(
                _sample(
                    output_dir=output_dir,
                    dataset="cofw",
                    sample_id=sample_id,
                    image=crop_image_path,
                    points68=crop_points68,
                    condition="occlusion" if is_occluded else str(entry.get("condition") or conds[0]),
                    conditions=tuple(_label(item) for item in conds),
                    source_schema=source_schema or detected_schema,
                    source_id=str(entry.get("source_id") or sample_id),
                    metadata=merged_metadata,
                    visibility=visibility,
                ),
                split,
            )
        )

    if not samples:
        raise ValueError(f"no cropped COFW JSON samples built; skipped={skipped[:10]}")

    return _write_manifest(
        output_dir,
        "cofw",
        scenario,
        _filter(samples, scenarios, limit),
        mode=mode,
        allow_overlap=allow_overlap,
        scenarios=scenarios,
        skipped=skipped,
    )


if __name__ == "__main__":
    raise SystemExit(main())
