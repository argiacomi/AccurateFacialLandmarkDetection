#!/usr/bin/env python3
"""Prepare MERL-RAV annotations against native AFLW images.

MERL-RAV provides signed 68-point landmark annotations over AFLW images. This
helper matches MERL-RAV ``.pts`` files to native AFLW ``flickr/`` images by the
``imageNNNNN`` token, converts signed MERL-RAV coordinates into finite 68-point
``.npy`` files, and writes a JSON manifest that ``build_quality_dataset.py`` can
consume.

Signed MERL-RAV semantics:

* positive ``x y``: visible landmark
* negative ``-x -y``: externally occluded landmark estimated at ``abs(x), abs(y)``
* ``-1 -1``: self-occluded landmark without a valid coordinate; emitted as ``0,0``
  with visibility masks preserved in metadata

Typical usage after downloading/extracting sources:

    python tools/landmarks/prepare_merl_rav_aflw.py \
      --merl-rav-root data/landmarks/merl-rav/extracted/MERL-RAV_dataset-master.zip \
      --aflw-root data/landmarks/aflw/extracted/AFLW.zip \
      --output-dir data/landmarks/merl-rav/organized

Then build the CD-ViT dataset manifest:

    python tools/landmarks/build_quality_dataset.py \
      --dataset merl-rav \
      --source-dir data/landmarks/merl-rav/organized \
      --output-dir runs/landmarks/build_merl_rav
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import shutil
import sys
import typing as T
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lib.landmarks.io_utils import relative_or_absolute, safe_id, write_json

logger = logging.getLogger(__name__)

AFLW_IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
AFLW_SOURCE_STEM_RE = re.compile(r"(image\d+)", re.IGNORECASE)


@dataclass
class PrepStats:
    labels: int = 0
    matched: int = 0
    written: int = 0
    skipped_split: int = 0
    skipped_bad_label: int = 0
    skipped_no_image_token: int = 0
    skipped_no_image: int = 0
    skipped_bad_image: int = 0
    skipped_no_visible_landmarks: int = 0
    copied_images: int = 0
    symlinked_images: int = 0
    duplicate_aflw_stems: int = 0
    aflw_image_count: int = 0
    visibility_counts: dict[str, int] = field(default_factory=lambda: {
        "visible": 0,
        "externally_occluded": 0,
        "self_occluded": 0,
    })

    def to_json(self) -> dict[str, T.Any]:
        return {
            "labels": self.labels,
            "matched": self.matched,
            "written": self.written,
            "skipped_split": self.skipped_split,
            "skipped_bad_label": self.skipped_bad_label,
            "skipped_no_image_token": self.skipped_no_image_token,
            "skipped_no_image": self.skipped_no_image,
            "skipped_bad_image": self.skipped_bad_image,
            "skipped_no_visible_landmarks": self.skipped_no_visible_landmarks,
            "copied_images": self.copied_images,
            "symlinked_images": self.symlinked_images,
            "duplicate_aflw_stems": self.duplicate_aflw_stems,
            "aflw_image_count": self.aflw_image_count,
            "visibility_counts": dict(self.visibility_counts),
        }


def _labels_from_path(path: Path) -> tuple[str, ...]:
    parts = tuple(part.lower().replace("-", "_") for part in path.parts)
    labels: list[str] = []
    for pose in ("frontal", "left", "lefthalf", "right", "righthalf"):
        if pose in parts:
            labels.append(pose)
            break
    for split in ("testset", "trainset"):
        if split in parts:
            labels.append(split)
            break
    return tuple(labels) or ("default",)


def _split_from_annotation_path(path: Path) -> str | None:
    parts = {part.lower().replace("-", "_") for part in path.parts}
    if "trainset" in parts or "train" in parts:
        return "train"
    if "testset" in parts or "test" in parts:
        return "test"
    return None


def _find_label_roots(root: Path) -> list[Path]:
    candidates = [path for path in root.rglob("merl_rav_labels") if path.is_dir()]
    if candidates:
        return sorted(candidates, key=lambda item: len(item.parts))
    if any(path.name == "labels" and path.is_dir() for path in root.rglob("*")):
        return [root]
    return [root]


def _label_files(root: Path) -> list[Path]:
    files: list[Path] = []
    for label_root in _find_label_roots(root):
        files.extend(path for path in label_root.rglob("*.pts") if path.is_file())
    return sorted(set(files))


def _parse_pts_signed(path: Path) -> np.ndarray:
    rows: list[tuple[float, float]] = []
    inside = False
    for line_number, line in enumerate(path.read_text(encoding="utf-8", errors="ignore").splitlines(), start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith("{"):
            inside = True
            continue
        if stripped.startswith("}"):
            break
        if not inside and any(stripped.lower().startswith(prefix) for prefix in ("version", "n_points")):
            continue
        parts = stripped.split()
        if len(parts) < 2:
            continue
        try:
            rows.append((float(parts[0]), float(parts[1])))
        except ValueError as err:
            raise ValueError(f"invalid MERL-RAV .pts row {line_number} in {path}: {stripped}") from err
    if len(rows) != 68:
        raise ValueError(f"MERL-RAV .pts file must contain 68 points, got {len(rows)}: {path}")
    return np.asarray(rows, dtype=np.float32)


def _visibility_and_points(signed_xy: np.ndarray) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray]:
    visibility: list[str] = []
    points = np.zeros(signed_xy.shape, dtype=np.float32)
    coordinate_valid = np.zeros((signed_xy.shape[0],), dtype=bool)
    score_visible = np.zeros((signed_xy.shape[0],), dtype=bool)

    for idx, (x_value, y_value) in enumerate(signed_xy):
        if x_value == -1 and y_value == -1:
            visibility.append("self_occluded")
            points[idx] = (0.0, 0.0)
            continue
        if x_value < 0 or y_value < 0:
            visibility.append("externally_occluded")
            points[idx] = (abs(float(x_value)), abs(float(y_value)))
            coordinate_valid[idx] = True
            continue
        visibility.append("visible")
        points[idx] = (float(x_value), float(y_value))
        coordinate_valid[idx] = True
        score_visible[idx] = True

    return visibility, points, coordinate_valid, score_visible


def _source_stem(stem: str) -> str | None:
    match = AFLW_SOURCE_STEM_RE.search(stem)
    return match.group(1).lower() if match else None


def _find_aflw_root(root: Path) -> Path:
    if (root / "flickr").is_dir():
        return root
    nested = root / "aflw"
    if (nested / "flickr").is_dir():
        return nested
    candidates = sorted(path.parent for path in root.rglob("flickr") if path.is_dir())
    if candidates:
        return candidates[0]
    raise FileNotFoundError(f"AFLW root not found below {root}. Expected a directory containing flickr/.")


def _build_aflw_image_index(aflw_root: Path, stats: PrepStats) -> dict[str, Path]:
    flickr_root = aflw_root / "flickr"
    if not flickr_root.is_dir():
        raise FileNotFoundError(f"AFLW flickr directory not found: {flickr_root}")

    index: dict[str, Path] = {}
    for image in sorted(flickr_root.rglob("*")):
        if not image.is_file() or image.suffix.lower() not in AFLW_IMAGE_EXTS:
            continue
        stem = _source_stem(image.stem)
        if stem is None:
            continue
        if stem in index:
            stats.duplicate_aflw_stems += 1
            continue
        index[stem] = image
    stats.aflw_image_count = len(index)
    if not index:
        raise FileNotFoundError(f"No AFLW images found below {flickr_root}")
    return index


def _read_image_size(path: Path) -> tuple[int, int] | None:
    try:
        from PIL import Image
    except ImportError:
        return None
    with Image.open(path) as image:
        width, height = image.size
    return height, width


def _in_image_mask(points_xy: np.ndarray, coordinate_valid: np.ndarray, image_hw: tuple[int, int] | None) -> np.ndarray:
    if image_hw is None:
        return coordinate_valid.copy()
    height, width = image_hw
    return coordinate_valid & (
        np.isfinite(points_xy).all(axis=1)
        & (points_xy[:, 0] >= 0)
        & (points_xy[:, 1] >= 0)
        & (points_xy[:, 0] < float(width))
        & (points_xy[:, 1] < float(height))
    )


def _bbox_from_valid(points: np.ndarray, mask: np.ndarray, image_hw: tuple[int, int] | None) -> list[float]:
    if mask.any():
        valid = points[mask]
        left, top = np.min(valid, axis=0)
        right, bottom = np.max(valid, axis=0)
        return [float(left), float(top), float(right), float(bottom)]
    if image_hw is not None:
        height, width = image_hw
        return [0.0, 0.0, float(width), float(height)]
    return [0.0, 0.0, 1.0, 1.0]


def _materialize_image(image_path: Path, output_dir: Path, aflw_root: Path, *, mode: str, stats: PrepStats) -> str:
    if mode == "absolute":
        return str(image_path.resolve())

    relative = image_path.relative_to(aflw_root).as_posix()
    target = output_dir / "images" / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        return relative_or_absolute(target, output_dir)

    if mode == "copy":
        shutil.copy2(image_path, target)
        stats.copied_images += 1
        return relative_or_absolute(target, output_dir)

    if mode == "symlink":
        os.symlink(image_path.resolve(), target)
        stats.symlinked_images += 1
        return relative_or_absolute(target, output_dir)

    raise ValueError(f"unsupported --image-mode {mode!r}")


def _normalizer(points: np.ndarray, mask: np.ndarray, image_hw: tuple[int, int] | None) -> float:
    if mask.any():
        bbox = _bbox_from_valid(points, mask, image_hw)
        return max(float(bbox[2] - bbox[0]), float(bbox[3] - bbox[1]), 1.0)
    if image_hw is not None:
        return float(max(image_hw))
    return 1.0


def _write_landmarks(path: Path, points: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, points.astype(np.float32))


def prepare(args: argparse.Namespace) -> tuple[Path, dict[str, T.Any]]:
    merl_root = Path(args.merl_rav_root)
    aflw_root = _find_aflw_root(Path(args.aflw_root))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stats = PrepStats()
    image_index = _build_aflw_image_index(aflw_root, stats)
    requested_splits = {item.strip().lower() for item in args.splits.split(",") if item.strip()}
    manifest_samples: list[dict[str, T.Any]] = []
    skipped_examples: list[dict[str, str]] = []

    label_files = _label_files(merl_root)
    label_roots = _find_label_roots(merl_root)
    label_root = label_roots[0] if len(label_roots) == 1 else merl_root

    for annotation in label_files:
        stats.labels += 1
        relative_annotation = annotation.relative_to(label_root) if annotation.is_relative_to(label_root) else annotation.relative_to(merl_root)
        split = _split_from_annotation_path(relative_annotation)
        if split is not None and requested_splits and split not in requested_splits:
            stats.skipped_split += 1
            continue

        stem = _source_stem(annotation.stem)
        if stem is None:
            stats.skipped_no_image_token += 1
            skipped_examples.append({"annotation": str(relative_annotation), "reason": "missing imageNNNNN token"})
            continue
        image_path = image_index.get(stem)
        if image_path is None:
            stats.skipped_no_image += 1
            skipped_examples.append({"annotation": str(relative_annotation), "reason": f"AFLW image not found for {stem}"})
            continue

        try:
            signed = _parse_pts_signed(annotation)
            visibility_labels, points, coordinate_valid, score_visible = _visibility_and_points(signed)
        except Exception as err:  # noqa: BLE001
            stats.skipped_bad_label += 1
            skipped_examples.append({"annotation": str(relative_annotation), "reason": str(err)})
            continue

        image_hw = None
        if not args.skip_image_validation:
            try:
                image_hw = _read_image_size(image_path)
            except Exception as err:  # noqa: BLE001
                stats.skipped_bad_image += 1
                skipped_examples.append({"annotation": str(relative_annotation), "reason": f"bad image: {err}"})
                continue

        in_image_mask = _in_image_mask(points, coordinate_valid, image_hw)
        score_visibility = in_image_mask & score_visible
        if not any(score_visibility):
            stats.skipped_no_visible_landmarks += 1
            skipped_examples.append({"annotation": str(relative_annotation), "reason": "no score-visible landmarks"})
            continue

        condition_labels = _labels_from_path(relative_annotation)
        if any(value != "visible" for value in visibility_labels):
            condition_labels = tuple(dict.fromkeys((*condition_labels, "occlusion")))

        sample_base_id = relative_annotation.with_suffix("").as_posix().replace("/", "_")
        sample_id = safe_id(f"{sample_base_id}__{image_path.stem}")
        landmark_rel = Path("landmarks") / f"{sample_id}.npy"
        _write_landmarks(output_dir / landmark_rel, points)
        image_value = _materialize_image(image_path, output_dir, aflw_root, mode=args.image_mode, stats=stats)

        for item in visibility_labels:
            stats.visibility_counts[item] = stats.visibility_counts.get(item, 0) + 1

        bbox = _bbox_from_valid(points, in_image_mask, image_hw)
        normalizer = _normalizer(points, in_image_mask, image_hw)
        top_level_visibility = [bool(value) for value in score_visibility.tolist()]
        coordinate_valid_mask = [bool(value) for value in in_image_mask.tolist()]
        source_valid_mask = [bool(value) for value in coordinate_valid.tolist()]

        metadata: dict[str, T.Any] = {
            "annotation_file": relative_annotation.as_posix(),
            "image_id": relative_or_absolute(image_path, aflw_root),
            "aflw_image_source": "aflw_native",
            "visibility": visibility_labels,
            "self_occluded_count": sum(1 for value in visibility_labels if value == "self_occluded"),
            "externally_occluded_count": sum(1 for value in visibility_labels if value == "externally_occluded"),
            "landmark_source_valid_mask": source_valid_mask,
            "landmark_in_image_mask": coordinate_valid_mask,
            "landmark_coordinate_valid_mask": coordinate_valid_mask,
            "landmark_score_visibility_mask": top_level_visibility,
            "landmark_source_valid_count": int(coordinate_valid.sum()),
            "landmark_in_image_count": int(in_image_mask.sum()),
            "landmark_score_visible_count": int(score_visibility.sum()),
            "face_bbox": bbox,
            "face_bbox_source": "merl_rav_native_aflw_image_landmarks",
            "normalizer_source": "merl_rav_coordinate_valid_landmark_bbox_max_side",
        }
        if image_hw is not None:
            metadata["image_height"] = int(image_hw[0])
            metadata["image_width"] = int(image_hw[1])

        manifest_samples.append(
            {
                "sample_id": sample_id,
                "dataset": "merl-rav",
                "condition": condition_labels[0],
                "conditions": condition_labels,
                "image": image_value,
                "landmarks": landmark_rel.as_posix(),
                "source_schema": "2d_68",
                "normalizer": float(normalizer),
                "visibility": top_level_visibility,
                "source": {"dataset": "merl-rav-aflw-native", "source_id": sample_id},
                "metadata": metadata,
            }
        )
        stats.matched += 1
        stats.written += 1

    manifest_path = output_dir / "manifest.json"
    write_json(
        manifest_path,
        {
            "version": 1,
            "landmark_schema": "2d_68",
            "metadata": {
                "builder": "AccurateFacialLandmarkDetection.tools.landmarks.prepare_merl_rav_aflw",
                "merl_rav_root": str(merl_root.resolve()),
                "aflw_root": str(aflw_root.resolve()),
                "image_mode": args.image_mode,
                "splits": sorted(requested_splits),
                "sample_count": len(manifest_samples),
            },
            "samples": manifest_samples,
        },
    )

    audit = stats.to_json()
    audit["skipped_examples"] = skipped_examples[:100]
    audit_path = output_dir / "prepare_audit.json"
    write_json(audit_path, audit)
    return manifest_path, {"audit": audit_path, **audit}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--merl-rav-root", required=True, help="Extracted MERL-RAV labels root or MERL-RAV_dataset-master directory.")
    parser.add_argument("--aflw-root", required=True, help="Extracted AFLW root or parent containing aflw/flickr.")
    parser.add_argument("--output-dir", required=True, help="Output organized source directory.")
    parser.add_argument("--splits", default="train,test", help="Comma-separated split filter: train,test or empty for all.")
    parser.add_argument("--image-mode", choices=("absolute", "symlink", "copy"), default="absolute", help="How manifest image paths should be materialized.")
    parser.add_argument("--skip-image-validation", action="store_true", help="Do not open images with Pillow for size/bounds checks.")
    parser.add_argument("--log-level", default="INFO", choices=("DEBUG", "INFO", "WARNING", "ERROR"))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(levelname)s:%(name)s:%(message)s")
    manifest, audit = prepare(args)
    print(f"Wrote MERL-RAV/AFLW organized manifest: {manifest}")
    print(f"Wrote audit: {audit['audit']}")
    print(json.dumps({key: value for key, value in audit.items() if key != "audit"}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
