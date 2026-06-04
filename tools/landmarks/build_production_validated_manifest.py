#!/usr/bin/env python3
"""Build a CD-ViT production_validated manifest from a Faceswap production directory.

The production directory is expected to contain source images and exactly one
Faceswap ``.fsa`` alignments file. The ``.fsa`` file is Faceswap's compressed
pickle alignment format, so only run this helper on trusted local files.

Example:
    python tools/landmarks/build_production_validated_manifest.py \
      --prod-dir /path/to/production_dir \
      --output-dir data/landmarks/production_validated
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import re
import sys
import typing as T
import zlib
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lib.landmarks.core.schema import normalize_landmarks

logger = logging.getLogger(__name__)

DEFAULT_DATASET = "production_validated"
DEFAULT_SOURCE = "faceswap_fsa_production_dir"
DEFAULT_LABEL_QUALITY = "human_validated"
IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff")
PRODUCTION_RUNTIME_BUCKET_KEYS = (
    "runtime_bucket",
    "bucket",
    "landmark_ensemble_runtime_bucket",
    "landmark_ensemble_bucket",
)


def _jsonable(value: T.Any) -> T.Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _write_json(path: Path, payload: T.Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _safe_sample_id(frame_name: str, face_index: int) -> str:
    stem = Path(frame_name).stem
    safe_stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", stem).strip("_")
    return f"{safe_stem or 'frame'}_face{face_index}"


def _find_fsa(prod_dir: Path) -> Path:
    direct = sorted(path for path in prod_dir.iterdir() if path.is_file() and path.suffix.lower() == ".fsa")
    recursive = sorted(path for path in prod_dir.rglob("*.fsa") if path.is_file())
    candidates = direct or recursive
    if not candidates:
        raise FileNotFoundError(f"no .fsa file found under {prod_dir}")
    if len(candidates) > 1:
        names = ", ".join(str(path) for path in candidates[:10])
        raise ValueError(f"expected exactly one .fsa file under {prod_dir}; found {len(candidates)}: {names}")
    return candidates[0]


def _load_fsa(path: Path) -> dict[str, T.Any]:
    """Load a Faceswap compressed-pickle alignment file.

    This mirrors Faceswap's compressed serializer: zlib-compressed pickle with
    top-level ``__meta__`` and ``__data__`` keys.
    """
    try:
        payload = pickle.loads(zlib.decompress(path.read_bytes()))
    except Exception as err:  # noqa: BLE001
        raise ValueError(f"failed to read Faceswap .fsa alignments at {path}: {err}") from err
    if not isinstance(payload, dict):
        raise ValueError(f"Faceswap .fsa payload must be a dict: {path}")
    data = payload.get("__data__", payload)
    if not isinstance(data, dict):
        raise ValueError(f"Faceswap .fsa payload has no dict __data__: {path}")
    return data


def _build_image_index(prod_dir: Path, fsa_path: Path) -> dict[str, Path]:
    index: dict[str, Path] = {}
    for ext in IMAGE_EXTS:
        for path in sorted(prod_dir.rglob(f"*{ext}")):
            if not path.is_file() or path == fsa_path:
                continue
            keys = {
                path.name.lower(),
                path.stem.lower(),
                path.relative_to(prod_dir).as_posix().lower(),
            }
            for key in keys:
                index.setdefault(key, path.resolve())
    return index


def _resolve_image(prod_dir: Path, frame_name: str, image_index: T.Mapping[str, Path]) -> Path | None:
    frame_path = Path(str(frame_name))
    candidates = [
        prod_dir / frame_path,
        prod_dir / frame_path.name,
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    for key in (str(frame_name).lower(), frame_path.name.lower(), frame_path.stem.lower()):
        match = image_index.get(key)
        if match is not None and match.is_file():
            return match.resolve()
    return None


def _faces_from_entry(entry: T.Any) -> list[dict[str, T.Any]]:
    if isinstance(entry, dict):
        faces = entry.get("faces", entry.get("face", []))
    else:
        faces = getattr(entry, "faces", [])
    if isinstance(faces, dict):
        faces = list(faces.values())
    if not isinstance(faces, (list, tuple)):
        return []
    out: list[dict[str, T.Any]] = []
    for face in faces:
        if isinstance(face, dict):
            out.append(dict(face))
        else:
            out.append(dict(getattr(face, "__dict__", {})))
    return out


def _face_value(face: T.Mapping[str, T.Any], key: str, default: T.Any = None) -> T.Any:
    return face.get(key, default)


def _first_present(mapping: T.Mapping[str, T.Any], keys: T.Sequence[str]) -> T.Any:
    for key in keys:
        if key in mapping and mapping[key] is not None:
            return mapping[key]
    return None


def _landmarks(face: T.Mapping[str, T.Any]) -> np.ndarray:
    raw = _first_present(face, ("landmarks_xy", "landmarks", "landmarksXY", "landmark"))
    if raw is None:
        raise ValueError("face has no landmarks_xy")
    points = normalize_landmarks(np.asarray(raw, dtype=np.float32), source_schema="2d_68")
    if points.shape != (68, 2):
        raise ValueError(f"expected 68x2 landmarks, got {points.shape}")
    if not np.all(np.isfinite(points)):
        raise ValueError("landmarks contain NaN or infinite values")
    return np.ascontiguousarray(points, dtype=np.float32)


def _bbox(face: T.Mapping[str, T.Any], points: np.ndarray) -> list[float]:
    try:
        x = float(_face_value(face, "x"))
        y = float(_face_value(face, "y"))
        w = float(_face_value(face, "w"))
        h = float(_face_value(face, "h"))
        if np.isfinite([x, y, w, h]).all() and w > 0 and h > 0:
            return [x, y, x + w, y + h]
    except Exception:
        pass
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    return [float(mins[0]), float(mins[1]), float(maxs[0]), float(maxs[1])]


def _normalizer(face: T.Mapping[str, T.Any], points: np.ndarray) -> float:
    try:
        w = float(_face_value(face, "w"))
        h = float(_face_value(face, "h"))
        value = float(np.hypot(w, h))
        if np.isfinite(value) and value > 0:
            return value
    except Exception:
        pass
    value = float(np.linalg.norm(points[36] - points[45]))
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"invalid normalizer: {value}")
    return value


def _metadata(face: T.Mapping[str, T.Any], *, frame_name: str, fsa_path: Path, face_index: int) -> dict[str, T.Any]:
    metadata = dict(face.get("metadata", {})) if isinstance(face.get("metadata"), dict) else {}
    metadata.setdefault("review_status", "accepted")
    metadata.setdefault("label_quality", DEFAULT_LABEL_QUALITY)
    metadata.setdefault("source", DEFAULT_SOURCE)
    metadata.setdefault("frame", frame_name)
    metadata.setdefault("face_index", face_index)
    metadata.setdefault("alignments_file", str(fsa_path.resolve()))
    return _jsonable(metadata)


def _runtime_bucket(metadata: T.Mapping[str, T.Any]) -> str | None:
    for key in PRODUCTION_RUNTIME_BUCKET_KEYS:
        value = metadata.get(key)
        if value:
            return str(value)
    landmark_ensemble = metadata.get("landmark_ensemble")
    if isinstance(landmark_ensemble, dict):
        for key in ("runtime_bucket", "bucket"):
            value = landmark_ensemble.get(key)
            if value:
                return str(value)
        resolver = landmark_ensemble.get("resolver")
        if isinstance(resolver, dict):
            for key in ("runtime_bucket", "bucket"):
                value = resolver.get(key)
                if value:
                    return str(value)
    return None


def _write_landmarks(output_dir: Path, sample_id: str, points: np.ndarray) -> str:
    landmarks_dir = output_dir / "landmarks"
    landmarks_dir.mkdir(parents=True, exist_ok=True)
    path = landmarks_dir / f"{sample_id}.npy"
    np.save(path, points.astype(np.float32))
    return path.relative_to(output_dir).as_posix()


def build_manifest(prod_dir: Path, output_dir: Path, *, dataset_name: str = DEFAULT_DATASET) -> dict[str, T.Any]:
    prod_dir = prod_dir.resolve()
    if not prod_dir.is_dir():
        raise FileNotFoundError(f"production directory not found: {prod_dir}")
    fsa_path = _find_fsa(prod_dir)
    alignments = _load_fsa(fsa_path)
    image_index = _build_image_index(prod_dir, fsa_path)

    output_dir.mkdir(parents=True, exist_ok=True)
    samples: list[dict[str, T.Any]] = []
    used_ids: dict[str, int] = {}
    skipped_missing_image = 0
    skipped_invalid_face = 0

    for frame_name, entry in sorted(alignments.items()):
        if str(frame_name).startswith("__"):
            continue
        image_path = _resolve_image(prod_dir, str(frame_name), image_index)
        faces = _faces_from_entry(entry)
        if image_path is None:
            skipped_missing_image += len(faces) or 1
            continue
        for face_index, face in enumerate(faces):
            sample_id = _safe_sample_id(str(frame_name), face_index)
            used_ids[sample_id] = used_ids.get(sample_id, 0) + 1
            if used_ids[sample_id] > 1:
                sample_id = f"{sample_id}_{used_ids[sample_id]}"
            try:
                points = _landmarks(face)
                metadata = _metadata(face, frame_name=str(frame_name), fsa_path=fsa_path, face_index=face_index)
                condition = _runtime_bucket(metadata) or "unknown"
                landmark_path = _write_landmarks(output_dir, sample_id, points)
                sample = {
                    "sample_id": sample_id,
                    "dataset": dataset_name,
                    "condition": condition,
                    "conditions": [condition],
                    "source_schema": "2d_68",
                    "image": str(image_path),
                    "landmarks": landmark_path,
                    "face_bbox": _bbox(face, points),
                    "normalizer": _normalizer(face, points),
                    "source": {"dataset": dataset_name, "source_id": sample_id},
                    "metadata": metadata,
                }
            except Exception as err:  # noqa: BLE001
                skipped_invalid_face += 1
                logger.debug("Skipping invalid production face %s[%d]: %s", frame_name, face_index, err)
                continue
            samples.append(sample)

    payload = {
        "dataset": dataset_name,
        "metadata": {
            "source": DEFAULT_SOURCE,
            "prod_dir": str(prod_dir),
            "alignments": str(fsa_path.resolve()),
            "label_quality": DEFAULT_LABEL_QUALITY,
            "review_status": "accepted",
            "sample_count": len(samples),
            "skipped_missing_image": skipped_missing_image,
            "skipped_invalid_face": skipped_invalid_face,
        },
        "samples": sorted(samples, key=lambda item: str(item["sample_id"])),
    }
    _write_json(output_dir / "manifest.json", payload)
    return payload["metadata"]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prod-dir", "--production-dir", dest="prod_dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset-name", default=DEFAULT_DATASET)
    parser.add_argument("--log-level", default="INFO", choices=("DEBUG", "INFO", "WARNING", "ERROR"))
    return parser


def main(argv: T.Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(levelname)s:%(name)s:%(message)s")
    try:
        metadata = build_manifest(args.prod_dir, args.output_dir, dataset_name=args.dataset_name)
    except Exception as err:  # noqa: BLE001
        logger.error("production manifest build failed: %s", err)
        return 1
    logger.info("Wrote %d production_validated samples to %s", metadata["sample_count"], args.output_dir / "manifest.json")
    print(f"Wrote production_validated manifest: {args.output_dir / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
