#!/usr/bin/env python3
"""Build CD-ViT/faceswap-compatible landmark manifests.

This local builder covers the faceswap landmark dataset names while emitting a
CD-ViT-friendly contract: every manifest entry points to a materialized
canonical ``(68, 2)`` ``.npy`` file.

Supported raw inputs by dataset:

* WFLW: official 98-point annotation text plus images, or generic sources.
* COFW: faceswap-style 68-point JSON export, or generic 68/98 landmark files.
* 300W: iBUG ``.pts`` files plus same-stem images, JSON, ``.npy``, or ``.mat``.
* AFLW2000-3D: same-stem ``.mat`` files with 68 2D/3D landmarks plus images.
* MERL-RAV, Menpo2D, MultiPIE: JSON, ``.npy``, ``.pts``, ``.mat`` sources.

Non-68/non-98 samples are skipped because ``DatasetFS68Manifest`` trains a
68-point model.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
import re
import sys
import typing as T
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lib.landmarks.core.schema import normalize_landmarks
from lib.landmarks.datasets.sources import extract_archive_to_temp

logger = logging.getLogger(__name__)

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff")
LANDMARK_EXTS = (".npy", ".pts", ".mat", ".txt")
SUPPORTED_DATASETS = (
    "wflw",
    "cofw",
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


def _canonical_points(raw: T.Any, *, source_schema: str | None = None) -> tuple[np.ndarray, str]:
    """Return canonical 68x2 points and the source schema label."""
    arr = np.asarray(raw, dtype=np.float32)

    while arr.ndim > 2 and 1 in arr.shape:
        arr = np.squeeze(arr)

    if arr.ndim == 1:
        if arr.size == 68 * 3:
            arr = arr.reshape(68, 3)
        elif arr.size == 98 * 2:
            arr = arr.reshape(98, 2)
        elif arr.size == 68 * 2:
            arr = arr.reshape(68, 2)
        else:
            raise ValueError(f"flat landmark array has unsupported size {arr.size}")

    if arr.ndim != 2:
        raise ValueError(f"landmarks must be 2D, got shape {arr.shape}")

    if arr.shape[0] in (2, 3) and arr.shape[1] in (68, 98):
        arr = arr.T

    if not np.all(np.isfinite(arr)):
        raise ValueError("landmarks contain NaN or infinite values")

    if arr.shape == (68, 3):
        return np.ascontiguousarray(arr[:, :2], dtype=np.float32), "3d_68"
    if arr.shape[0] == 68 and arr.shape[1] >= 2:
        return normalize_landmarks(arr[:, :2], source_schema="2d_68"), source_schema or "2d_68"
    if arr.shape[0] == 98 and arr.shape[1] >= 2:
        return normalize_landmarks(arr[:, :2], source_schema="2d_98"), source_schema or "2d_98"

    raise ValueError(f"unsupported landmark shape {arr.shape}; expected 68 or 98 points")


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
    for count, dims in ((68, 2), (68, 3), (98, 2)):
        total = count * dims
        if len(values) == total:
            return np.asarray(values, dtype=np.float32).reshape(count, dims)
    rows: list[list[float]] = []
    for line in text.splitlines():
        parts = line.replace(",", " ").split()
        if len(parts) < 2:
            continue
        try:
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
        if arr.size < 68 * 2:
            continue
        try:
            _canonical_points(arr)
            return arr
        except Exception as err:  # noqa: BLE001
            errors.append(f"{key}: {err}")
            continue
    raise ValueError(f"no 68/98-point landmark array found in {path}; tried {errors[:5]}")


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
    value = float(np.linalg.norm(points68[36] - points68[45]))
    if np.isfinite(value) and value > 0.0:
        return value

    span = np.ptp(points68[:, :2], axis=0)
    fallback = float(max(span[0], span[1]))
    if np.isfinite(fallback) and fallback > 0.0:
        logger.warning(
            "invalid interocular normalizer for %s: %s; using landmark span fallback %s",
            sample_id,
            value,
            fallback,
        )
        return fallback

    raise ValueError(f"invalid normalizer for {sample_id}: interocular={value}, span={fallback}")


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

    meta.setdefault("source_schema", source_schema)
    out: dict[str, T.Any] = {
        "sample_id": sample_id,
        "dataset": dataset,
        "condition": _label(condition),
        "conditions": tuple(_label(item) for item in conditions),
        "image": str(image.resolve()),
        "landmarks": _relative_or_absolute(landmarks, output_dir),
        "source_schema": "2d_68",
        "normalizer": normalizer_value,
        "source": {"dataset": dataset, "source_id": source_id or sample_id},
        "metadata": meta,
    }
    if visibility is not None:
        out["visibility"] = visibility
        out["metadata"].setdefault("visibility", visibility)
    return out


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

    payload = {
        "version": 1,
        "landmark_schema": "2d_68",
        "metadata": {
            "builder": "AccurateFacialLandmarkDetection.tools.landmarks.build_quality_dataset",
            "dataset": dataset,
            "scenario": _label(scenario),
            "scenarios": list(scenarios or []),
            "sample_count": len(merged),
            "skipped_count": len(skipped or []),
        },
        "samples": merged,
    }
    _write_json(manifest_path, payload)

    condition_counts: dict[str, int] = {}
    dataset_counts: dict[str, int] = {}
    for sample in merged:
        condition = str(sample.get("condition", "default"))
        condition_counts[condition] = condition_counts.get(condition, 0) + 1
        sample_dataset = str(sample.get("dataset", dataset))
        dataset_counts[sample_dataset] = dataset_counts.get(sample_dataset, 0) + 1
    _write_json(
        output_dir / "dataset_audit.json",
        {
            "manifest": str(manifest_path),
            "sample_count": len(merged),
            "skipped_count": len(skipped or []),
            "skipped_examples": (skipped or [])[:50],
            "datasets": dataset_counts,
            "conditions": condition_counts,
            "landmark_schema": "2d_68",
        },
    )
    return manifest_path


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
        metadata = dict(entry.get("metadata", {})) if isinstance(entry.get("metadata"), dict) else {}
        source_schema = str(entry.get("source_schema") or metadata.get("source_schema") or "") or None
        try:
            points68, detected_schema = _load_points(landmark_value, base_dir=path.parent, source_schema=source_schema)
        except Exception as err:  # noqa: BLE001
            skipped.append({"sample_id": sample_id, "reason": str(err)})
            continue
        conds = _conditions(entry, scenario)
        samples.append(
            _sample(
                output_dir=output_dir,
                dataset=_dataset(str(entry.get("dataset") or dataset)),
                sample_id=sample_id,
                image=_resolve_path(image_value, base_dir=image_base),
                points68=points68,
                condition=str(entry.get("condition") or conds[0]),
                conditions=conds,
                source_schema=source_schema or detected_schema,
                source_id=str(entry.get("source_id") or sample_id),
                metadata=metadata,
                visibility=entry.get("visibility", metadata.get("visibility")),
                normalizer=entry.get("normalizer", metadata.get("normalizer")),
            )
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
        samples.append(
            _sample(
                output_dir=output_dir,
                dataset=dataset,
                sample_id=sample_id,
                image=image,
                points68=points68,
                condition=condition,
                conditions=conds,
                source_schema=source_schema,
                source_id=sample_id,
                metadata={"source_landmarks": str(landmark_path.resolve())},
            )
        )
    if not samples:
        raise ValueError(f"no usable 68/98-point landmark samples found under {root}; skipped={skipped[:5]}")
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
            points68, visibility, metadata = _cofw_points_and_occ(ann)
            image_arr = _cofw_hdf5_image_by_index(color_mat, idx)
            image_path = _write_cofw_image(output_dir, idx, image_arr)
        except Exception as err:  # noqa: BLE001
            skipped.append({"sample_id": ann.as_posix(), "reason": str(err)})
            continue

        if boxes is not None and 0 <= idx < len(boxes):
            metadata["face_bbox"] = [float(x) for x in boxes[idx].tolist()]
            metadata["face_bbox_source"] = "cofw68_test_bboxes"

        metadata.update(
            {
                "annotation_file": str(ann.resolve()),
                "cofw_index": idx + 1,
                "split": "test",
                "image_source_mat": str(color_mat.resolve()),
                "source_schema": "2d_68",
            }
        )

        samples.append(
            _sample(
                output_dir=output_dir,
                dataset="cofw",
                sample_id=f"cofw_test_{idx + 1:04d}",
                image=image_path,
                points68=points68,
                condition="occlusion",
                conditions=("occlusion", "testset"),
                source_schema="2d_68",
                source_id=f"cofw_test_{idx + 1:04d}",
                metadata=metadata,
                visibility=visibility,
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


def _multipie_parse_line(line: str, *, line_no: int, path: Path) -> tuple[str, np.ndarray, list[float]]:
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
        raise ValueError("39-point profile sample skipped for 68-point CD-ViT manifest")

    if dense_count != 136:
        raise ValueError(
            f"line {line_no} in {path} has {len(values)} numeric values; "
            "expected 150 for 68-point rows or 92 for 39-point profile rows"
        )

    bbox = [float(item) for item in values[:4]]
    raw = values[header_values:]
    points = np.asarray(raw, dtype=np.float32).reshape(68, 2)
    points = normalize_landmarks(points, source_schema="2d_68")
    return image_rel, points, bbox


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
                image_rel, points68, bbox = _multipie_parse_line(
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

                metadata = {
                    "annotation_file": str(annotation_file.resolve()),
                    "annotation_line": line_no,
                    "image_id": image_rel,
                    "face_bbox": bbox,
                    "face_bbox_source": "multipie_landmark_bounds",
                    "normalizer_source": DEFAULT_NORMALIZER_SOURCE,
                    "source_schema": "2d_68",
                }

                sample_kwargs = dict(
                    output_dir=output_dir,
                    dataset="multipie",
                    sample_id=sample_id,
                    image=image_path,
                    points68=points68,
                    condition=condition,
                    conditions=conds,
                    source_schema="2d_68",
                    source_id=sample_id,
                    metadata=metadata,
                )
                try:
                    sample = _sample(**sample_kwargs, normalizer=normalizer)
                except TypeError:
                    sample = _sample(**sample_kwargs)

                samples.append(sample)
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
        points68 = normalize_landmarks(points98, source_schema="2d_98")
        samples.append(
            _sample(
                output_dir=output_dir,
                dataset="wflw",
                sample_id=sample_id,
                image=image_path,
                points68=points68,
                condition=conds[0],
                conditions=tuple(_label(item) for item in conds),
                source_schema="2d_98",
                source_id=sample_id,
                metadata={"bbox": bbox, "attributes": attrs, "image_id": image_rel},
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

    with _source_context(args.source_dir, args.source_zip) as root:
        if dataset == "wflw":
            return _build_wflw(
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
        if args.cofw_json:
            return _build_json(
                Path(args.cofw_json),
                output_dir,
                dataset=dataset,
                scenario=args.scenario,
                scenarios=scenarios,
                limit=limit,
                mode=args.manifest_mode,
                allow_overlap=args.allow_overlap,
                image_root=args.image_root,
            )
        if dataset == "cofw":
            if root is None:
                raise ValueError("--source-dir or --source-zip is required for COFW")
            return _build_cofw(
                root,
                output_dir,
                scenario=args.scenario,
                scenarios=scenarios,
                limit=limit,
                mode=args.manifest_mode,
                allow_overlap=args.allow_overlap,
            )
        if dataset == "multipie":
            if root is None:
                raise ValueError("--source-dir or --source-zip is required for MultiPIE")
            return _build_multipie(
                root,
                output_dir,
                scenario=args.scenario,
                scenarios=scenarios,
                limit=limit,
                mode=args.manifest_mode,
                allow_overlap=args.allow_overlap,
            )
        if root is None:
            raise ValueError("--source-dir, --source-zip, --wflw-annotations, or --cofw-json is required")
        return _build_directory(
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
    parser.add_argument("--recursive", action="store_true", help="Accepted for compatibility; scans are recursive.")
    parser.add_argument("--wflw-annotations", default=None)
    parser.add_argument("--cofw-json", default=None)
    parser.add_argument("--write-overlays", action="store_true", help="Accepted for compatibility; overlays are not generated.")
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


if __name__ == "__main__":
    raise SystemExit(main())
