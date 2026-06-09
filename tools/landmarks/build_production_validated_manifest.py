#!/usr/bin/env python3
"""Build a CD-ViT production_validated manifest from a Faceswap production source.

The production source can be either a directory or a ``.zip`` archive containing
source images and exactly one Faceswap ``.fsa`` alignments file. The ``.fsa``
file is Faceswap's compressed pickle alignment format, so only run this helper
on trusted local files.

Example:
    python tools/landmarks/build_production_validated_manifest.py \
      --prod-dir /path/to/production_dir_or_zip \
      --output-dir data/landmarks/production_validated
"""

from __future__ import annotations

import argparse
import logging
import pickle
import re
import shutil
import sys
import typing as T
import zipfile
import zlib
from collections import Counter
from pathlib import Path, PurePosixPath

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lib.landmarks.core.schema import normalize_landmarks
from lib.landmarks.io_utils import jsonable, write_json

logger = logging.getLogger(__name__)

DEFAULT_DATASET = "production_validated"
DEFAULT_SOURCE = "faceswap_fsa_production_source"
DEFAULT_LABEL_QUALITY = "human_validated"
IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff")
PRODUCTION_RUNTIME_BUCKET_KEYS = (
    "runtime_bucket",
    "bucket",
    "landmark_ensemble_runtime_bucket",
    "landmark_ensemble_bucket",
)
EXTRACTED_SOURCE_DIRNAME = "extracted_source"


def _safe_sample_id(frame_name: str, face_index: int) -> str:
    stem = Path(frame_name).stem
    safe_stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", stem).strip("_")
    return f"{safe_stem or 'frame'}_face{face_index}"


def _zip_member_target(root: Path, member_name: str) -> Path:
    member_path = PurePosixPath(member_name)
    if member_path.is_absolute():
        raise ValueError(f"zip archive contains absolute path: {member_name!r}")
    if any(part in {"..", ""} for part in member_path.parts):
        raise ValueError(f"zip archive contains unsafe path: {member_name!r}")
    target = (root / Path(*member_path.parts)).resolve()
    root_resolved = root.resolve()
    if target != root_resolved and root_resolved not in target.parents:
        raise ValueError(f"zip archive member escapes extraction root: {member_name!r}")
    return target


def _extract_zip_source(zip_path: Path, output_dir: Path) -> Path:
    if not zipfile.is_zipfile(zip_path):
        raise ValueError(f"production source is not a valid .zip archive: {zip_path}")
    extract_dir = output_dir / EXTRACTED_SOURCE_DIRNAME
    if extract_dir.exists():
        shutil.rmtree(extract_dir)
    extract_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as archive:
        for member in archive.infolist():
            target = _zip_member_target(extract_dir, member.filename)
            if member.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(member, "r") as src, target.open("wb") as dst:
                shutil.copyfileobj(src, dst)
    return extract_dir.resolve()


def _resolve_prod_source(
    prod_source: Path, output_dir: Path
) -> tuple[Path, Path | None]:
    prod_source = prod_source.resolve()
    if prod_source.is_dir():
        return prod_source, None
    if prod_source.is_file() and prod_source.suffix.lower() == ".zip":
        return _extract_zip_source(prod_source, output_dir), prod_source
    raise FileNotFoundError(
        f"production source must be a directory or .zip file: {prod_source}"
    )


def _is_ignored_macos_sidecar(path: Path) -> bool:
    parts = set(path.parts)
    return (
        "__MACOSX" in parts
        or path.name.startswith("._")
        or any(part.startswith("._") for part in path.parts)
    )


def _is_real_fsa(path: Path) -> bool:
    return (
        path.is_file()
        and path.suffix.lower() == ".fsa"
        and not path.name.startswith(".")
        and not _is_ignored_macos_sidecar(path)
    )


def _find_fsa(prod_dir: Path) -> Path:
    direct = sorted(path for path in prod_dir.iterdir() if _is_real_fsa(path))
    recursive = sorted(path for path in prod_dir.rglob("*.fsa") if _is_real_fsa(path))
    candidates = direct or recursive
    if not candidates:
        raise FileNotFoundError(f"no .fsa file found under {prod_dir}")
    if len(candidates) > 1:
        names = ", ".join(str(path) for path in candidates[:10])
        raise ValueError(
            f"expected exactly one .fsa file under {prod_dir}; found {len(candidates)}: {names}"
        )
    return candidates[0]


def _load_fsa(path: Path) -> dict[str, T.Any]:
    """Load a Faceswap compressed-pickle alignment file.

    This mirrors Faceswap's compressed serializer: zlib-compressed pickle with
    top-level ``__meta__`` and ``__data__`` keys.
    """
    try:
        payload = pickle.loads(zlib.decompress(path.read_bytes()))
    except Exception as err:  # noqa: BLE001
        raise ValueError(
            f"failed to read Faceswap .fsa alignments at {path}: {err}"
        ) from err
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


def _resolve_image(
    prod_dir: Path, frame_name: str, image_index: T.Mapping[str, Path]
) -> Path | None:
    frame_path = Path(str(frame_name))
    candidates = [
        prod_dir / frame_path,
        prod_dir / frame_path.name,
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    for key in (
        str(frame_name).lower(),
        frame_path.name.lower(),
        frame_path.stem.lower(),
    ):
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

    raw_points = np.asarray(raw, dtype=np.float32)
    if raw_points.ndim == 1:
        if raw_points.size % 2 != 0:
            raise ValueError(
                f"flat landmark array has odd value count: {raw_points.size}"
            )
        raw_points = raw_points.reshape((-1, 2))
    if raw_points.ndim != 2 or raw_points.shape[1] < 2:
        raise ValueError(f"expected Nx2 landmarks, got {raw_points.shape}")

    source_schema = "2d_98" if raw_points.shape[0] == 98 else "2d_68"
    points = normalize_landmarks(raw_points[:, :2], source_schema=source_schema)

    if points.shape != (68, 2):
        raise ValueError(
            f"expected canonical 68x2 landmarks from {source_schema}, got {points.shape}"
        )
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
    # Match the quality dataset/WFLW path: normalize source landmarks to
    # canonical 68 first, then use canonical eye-corner interocular distance.
    del face
    value = float(np.linalg.norm(points[36] - points[45]))
    if not np.isfinite(value) or value <= 0:
        raise ValueError(
            f"invalid interocular normalizer after canonical 68 conversion: {value}"
        )
    return value


def _metadata(
    face: T.Mapping[str, T.Any], *, frame_name: str, fsa_path: Path, face_index: int
) -> dict[str, T.Any]:
    metadata = (
        dict(face.get("metadata", {})) if isinstance(face.get("metadata"), dict) else {}
    )
    metadata.setdefault("review_status", "accepted")
    metadata.setdefault("label_quality", DEFAULT_LABEL_QUALITY)
    metadata.setdefault("source", DEFAULT_SOURCE)
    metadata.setdefault("frame", frame_name)
    metadata.setdefault("face_index", face_index)
    metadata.setdefault("alignments_file", str(fsa_path.resolve()))
    return jsonable(metadata)


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


def build_manifest(
    prod_dir: Path, output_dir: Path, *, dataset_name: str = DEFAULT_DATASET
) -> dict[str, T.Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    prod_source = prod_dir.resolve()
    prod_dir, zip_source = _resolve_prod_source(prod_source, output_dir)
    fsa_path = _find_fsa(prod_dir)
    alignments = _load_fsa(fsa_path)
    image_index = _build_image_index(prod_dir, fsa_path)

    samples: list[dict[str, T.Any]] = []
    used_ids: Counter[str] = Counter()
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
            used_ids[sample_id] += 1
            if used_ids[sample_id] > 1:
                sample_id = f"{sample_id}_{used_ids[sample_id]}"
            try:
                points = _landmarks(face)
                metadata = _metadata(
                    face,
                    frame_name=str(frame_name),
                    fsa_path=fsa_path,
                    face_index=face_index,
                )
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
                logger.debug(
                    "Skipping invalid production face %s[%d]: %s",
                    frame_name,
                    face_index,
                    err,
                )
                continue
            samples.append(sample)

    metadata = {
        "source": DEFAULT_SOURCE,
        "prod_source": str(prod_source),
        "prod_dir": str(prod_dir),
        "alignments": str(fsa_path.resolve()),
        "label_quality": DEFAULT_LABEL_QUALITY,
        "review_status": "accepted",
        "sample_count": len(samples),
        "skipped_missing_image": skipped_missing_image,
        "skipped_invalid_face": skipped_invalid_face,
    }
    if zip_source is not None:
        metadata["zip_source"] = str(zip_source)
        metadata["extracted_source"] = str(prod_dir)
    payload = {
        "dataset": dataset_name,
        "metadata": metadata,
        "samples": sorted(samples, key=lambda item: str(item["sample_id"])),
    }
    write_json(output_dir / "manifest.json", payload)
    return metadata


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prod-dir",
        "--production-dir",
        dest="prod_dir",
        type=Path,
        required=True,
        help="Production source directory or .zip archive containing images and exactly one .fsa file.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset-name", default=DEFAULT_DATASET)
    parser.add_argument(
        "--log-level", default="INFO", choices=("DEBUG", "INFO", "WARNING", "ERROR")
    )
    return parser


def main(argv: T.Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(levelname)s:%(name)s:%(message)s",
    )
    try:
        metadata = build_manifest(
            args.prod_dir, args.output_dir, dataset_name=args.dataset_name
        )
    except Exception as err:  # noqa: BLE001
        logger.error("production manifest build failed: %s", err)
        return 1
    logger.info(
        "Wrote %d production_validated samples to %s",
        metadata["sample_count"],
        args.output_dir / "manifest.json",
    )
    print(f"Wrote production_validated manifest: {args.output_dir / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
