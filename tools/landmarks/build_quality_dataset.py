#!/usr/bin/env python3
"""Build CD-ViT/faceswap-compatible landmark manifests.

This is a compact local port of the faceswap manifest builder. It supports:

* directory trees of image + matching .npy landmark pairs
* JSON exports/manifests with a samples list
* WFLW 98-point annotation text files

Every emitted landmark file is materialized as canonical 68-point .npy so it can
be consumed directly by DatasetFS68Manifest and TrainHeatmapStageFP16.py.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
import shutil
import sys
import tempfile
import typing as T
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lib.landmarks.core.schema import normalize_landmarks
from lib.landmarks.datasets.sources import extract_archive_to_temp

logger = logging.getLogger(__name__)
IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".webp")
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


def _load_points(value: T.Any, *, base_dir: Path, source_schema: str | None = None) -> np.ndarray:
    if isinstance(value, (list, tuple, np.ndarray)):
        raw = np.asarray(value, dtype=np.float32)
    else:
        path = _resolve_path(value, base_dir=base_dir)
        if path.suffix.lower() == ".npy":
            raw = np.load(path).astype(np.float32)
        elif path.suffix.lower() == ".json":
            raw = np.asarray(_read_json(path), dtype=np.float32)
        else:
            raise ValueError(f"unsupported landmark input: {value!r}")
    return normalize_landmarks(raw, source_schema=source_schema)


def _normalizer(points68: np.ndarray, sample_id: str) -> float:
    value = float(np.linalg.norm(points68[36] - points68[45]))
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"invalid interocular normalizer for {sample_id}: {value}")
    return value


def _matching_image(landmarks: Path) -> Path | None:
    for ext in IMAGE_EXTS:
        candidate = landmarks.with_suffix(ext)
        if candidate.is_file():
            return candidate
    for ext in IMAGE_EXTS:
        candidate = landmarks.parent / "images" / f"{landmarks.stem}{ext}"
        if candidate.is_file():
            return candidate
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
) -> dict[str, T.Any]:
    sample_id = _safe_id(sample_id)
    landmarks = _save_landmarks(output_dir, sample_id, points68)
    normalizer = _normalizer(points68, sample_id)
    meta = dict(metadata or {})
    meta.setdefault("normalizer_source", DEFAULT_NORMALIZER_SOURCE)
    meta.setdefault("source_schema", source_schema)
    out: dict[str, T.Any] = {
        "sample_id": sample_id,
        "dataset": dataset,
        "condition": _label(condition),
        "conditions": tuple(_label(item) for item in conditions),
        "image": str(image.resolve()),
        "landmarks": _relative_or_absolute(landmarks, output_dir),
        "source_schema": "2d_68",
        "normalizer": normalizer,
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
        samples = [sample for sample in samples if allowed.intersection(sample.get("conditions", ())) ]
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


def _write_manifest(output_dir: Path, dataset: str, scenario: str, samples: list[dict[str, T.Any]], *, mode: str, allow_overlap: bool, scenarios: tuple[str, ...] | None) -> Path:
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
        },
        "samples": merged,
    }
    _write_json(manifest_path, payload)
    condition_counts: dict[str, int] = {}
    dataset_counts: dict[str, int] = {}
    for sample in merged:
        condition_counts[str(sample.get("condition", "default"))] = condition_counts.get(str(sample.get("condition", "default")), 0) + 1
        dataset_counts[str(sample.get("dataset", dataset))] = dataset_counts.get(str(sample.get("dataset", dataset)), 0) + 1
    _write_json(output_dir / "dataset_audit.json", {
        "manifest": str(manifest_path),
        "sample_count": len(merged),
        "datasets": dataset_counts,
        "conditions": condition_counts,
        "landmark_schema": "2d_68",
    })
    return manifest_path


def _json_source(root: Path) -> Path | None:
    candidates = [root] if root.is_file() else sorted(root.rglob("*.json"))
    for path in candidates:
        if not path.is_file():
            continue
        try:
            payload = _read_json(path)
        except Exception:
            continue
        if isinstance(payload, list) or (isinstance(payload, dict) and "samples" in payload):
            return path
    return None


def _build_json(path: Path, output_dir: Path, *, dataset: str, scenario: str, scenarios: tuple[str, ...] | None, limit: int | None, mode: str, allow_overlap: bool, image_root: str | None) -> Path:
    payload = _read_json(path)
    entries = payload.get("samples", payload) if isinstance(payload, dict) else payload
    if not isinstance(entries, list):
        raise ValueError(f"JSON source must contain list or samples list: {path}")
    image_base = Path(image_root) if image_root else path.parent
    samples = []
    for idx, entry in enumerate(entries):
        if not isinstance(entry, dict):
            continue
        image_value = entry.get("image") or entry.get("image_path") or entry.get("path")
        landmark_value = entry.get("landmarks") or entry.get("points") or entry.get("ground_truth")
        if image_value is None or landmark_value is None:
            continue
        sample_id = str(entry.get("sample_id") or entry.get("id") or entry.get("name") or idx)
        metadata = dict(entry.get("metadata", {})) if isinstance(entry.get("metadata"), dict) else {}
        source_schema = str(entry.get("source_schema") or metadata.get("source_schema") or "") or None
        points68 = _load_points(landmark_value, base_dir=path.parent, source_schema=source_schema)
        conds = _conditions(entry, scenario)
        samples.append(_sample(
            output_dir=output_dir,
            dataset=_dataset(str(entry.get("dataset") or dataset)),
            sample_id=sample_id,
            image=_resolve_path(image_value, base_dir=image_base),
            points68=points68,
            condition=str(entry.get("condition") or conds[0]),
            conditions=conds,
            source_schema=source_schema or "inferred",
            source_id=str(entry.get("source_id") or sample_id),
            metadata=metadata,
            visibility=entry.get("visibility", metadata.get("visibility")),
        ))
    return _write_manifest(output_dir, dataset, scenario, _filter(samples, scenarios, limit), mode=mode, allow_overlap=allow_overlap, scenarios=scenarios)


def _build_directory(root: Path, output_dir: Path, *, dataset: str, scenario: str, scenarios: tuple[str, ...] | None, limit: int | None, mode: str, allow_overlap: bool, image_root: str | None) -> Path:
    json_path = _json_source(root)
    if json_path is not None:
        return _build_json(json_path, output_dir, dataset=dataset, scenario=scenario, scenarios=scenarios, limit=limit, mode=mode, allow_overlap=allow_overlap, image_root=image_root)
    samples = []
    cond = _label(scenario)
    for landmarks in sorted(root.rglob("*.npy")):
        image = _matching_image(landmarks)
        if image is None:
            continue
        sample_id = landmarks.relative_to(root).with_suffix("").as_posix()
        points68 = _load_points(landmarks, base_dir=root)
        samples.append(_sample(
            output_dir=output_dir,
            dataset=dataset,
            sample_id=sample_id,
            image=image,
            points68=points68,
            condition=cond,
            conditions=(cond,),
            source_schema="inferred",
            source_id=sample_id,
            metadata={"source_landmarks": str(landmarks.resolve())},
        ))
    if not samples:
        raise ValueError(f"no image/.npy landmark pairs found under {root}")
    return _write_manifest(output_dir, dataset, scenario, _filter(samples, scenarios, limit), mode=mode, allow_overlap=allow_overlap, scenarios=scenarios)


def _parse_wflw_line(line: str, line_no: int) -> tuple[np.ndarray, list[float], dict[str, int], str]:
    parts = line.split()
    if len(parts) < 207:
        raise ValueError(f"WFLW line {line_no} expected at least 207 fields, got {len(parts)}")
    points = np.asarray([float(value) for value in parts[:196]], dtype=np.float32).reshape(98, 2)
    bbox = [float(value) for value in parts[196:200]]
    attrs = {name: int(float(parts[200 + idx])) for idx, name in enumerate(WFLW_ATTRIBUTE_NAMES)}
    return points, bbox, attrs, parts[206]


def _find_wflw_annotations(root: Path) -> Path | None:
    for pattern in ("list_98pt_rect_attr_train_test.txt", "list_98pt_rect_attr_train.txt", "list_98pt_rect_attr_test.txt", "*98pt*rect*attr*.txt"):
        matches = sorted(root.rglob(pattern))
        if matches:
            return matches[0]
    return None


def _find_wflw_images(root: Path) -> Path:
    for name in ("WFLW_images", "images", "WFLW"):
        matches = [path for path in root.rglob(name) if path.is_dir()]
        if matches:
            return matches[0]
    return root


def _build_wflw(root: Path | None, output_dir: Path, *, annotation_file: str | None, image_root: str | None, scenario: str, scenarios: tuple[str, ...] | None, limit: int | None, mode: str, allow_overlap: bool) -> Path:
    annotations = Path(annotation_file) if annotation_file else (_find_wflw_annotations(root or Path(".")))
    if annotations is None or not annotations.is_file():
        raise FileNotFoundError("WFLW annotation file not found")
    image_base = Path(image_root) if image_root else _find_wflw_images(root or annotations.parent)
    rows = []
    counts: dict[str, int] = {}
    for line_no, line in enumerate(annotations.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = _parse_wflw_line(line, line_no)
        rows.append(row)
        counts[row[3]] = counts.get(row[3], 0) + 1
    seen: dict[str, int] = {}
    samples = []
    for points98, bbox, attrs, image_rel in rows:
        seen[image_rel] = seen.get(image_rel, 0) + 1
        base_id = Path(image_rel).with_suffix("").as_posix()
        sample_id = base_id if counts[image_rel] <= 1 else f"{base_id}#face-{seen[image_rel]:02d}"
        conds = tuple(name for name in WFLW_ATTRIBUTE_NAMES if attrs.get(name)) or (_label(scenario),)
        points68 = normalize_landmarks(points98, source_schema="2d_98")
        samples.append(_sample(
            output_dir=output_dir,
            dataset="wflw",
            sample_id=sample_id,
            image=(image_base / image_rel).resolve(),
            points68=points68,
            condition=conds[0],
            conditions=tuple(_label(item) for item in conds),
            source_schema="2d_98",
            source_id=sample_id,
            metadata={"bbox": bbox, "attributes": attrs, "image_id": image_rel},
        ))
    return _write_manifest(output_dir, "wflw", scenario, _filter(samples, scenarios, limit), mode=mode, allow_overlap=allow_overlap, scenarios=scenarios)


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
            return _build_wflw(root, output_dir, annotation_file=args.wflw_annotations, image_root=args.image_root, scenario=args.scenario, scenarios=scenarios, limit=limit, mode=args.manifest_mode, allow_overlap=args.allow_overlap)
        if args.cofw_json:
            return _build_json(Path(args.cofw_json), output_dir, dataset=dataset, scenario=args.scenario, scenarios=scenarios, limit=limit, mode=args.manifest_mode, allow_overlap=args.allow_overlap, image_root=args.image_root)
        if root is None:
            raise ValueError("--source-dir, --source-zip, --wflw-annotations, or --cofw-json is required")
        return _build_directory(root, output_dir, dataset=dataset, scenario=args.scenario, scenarios=scenarios, limit=limit, mode=args.manifest_mode, allow_overlap=args.allow_overlap, image_root=args.image_root)


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
    parser.add_argument("--no-39pt-profile", action="store_true", help="Accepted for compatibility; non-68 samples are not emitted.")
    parser.add_argument("--include-39pt-profile", action="store_true", help="Accepted for compatibility; non-68 samples are not emitted.")
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
