#!/usr/bin/env python3
"""Path-aware multi-dataset landmark preparation orchestrator.

This command stitches together the existing downloader and manifest builder so a
user can go from "download" to "training-ready manifest" without manually
plumbing source/cache paths between tools. For each requested dataset it will:

* download (or reuse cached) source archives into a default data root when needed,
* accept local production data via ``--datasets prod --prod-dir ...``,
* resolve the extracted source directory from the downloader registry,
* stage multi-archive datasets (e.g. JD-landmark) into a single source root,
* build the CD-ViT manifest (extracting video frames when required),
* write audit overlays when requested,
* validate the resulting manifest,
* merge every requested dataset into one combined manifest, and
* print the training command for the combined manifest.

Example::

    python tools/prepare_landmark_dataset.py \
      --datasets wflw-v \
      --write-overlays

    python tools/prepare_landmark_dataset.py \
      --datasets prod \
      --prod-dir /path/to/production_dir_or_zip
"""

# ruff: noqa: E402
from __future__ import annotations

import argparse
import contextlib
import copy
import os
import sys
import time
import typing as T
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lib.core.schema import (
    head_name_for_schema,
    point_count_for_schema,
    projection_audit_for_schema,
)
from lib.datasets.hard_negative_mining import annotate_sample_bucket_in_place
from lib.datasets.parallel import resolve_worker_count
from lib.datasets.progress import concurrent_progress
from lib.io_utils import read_json, write_json
from lib.logging_utils import (
    Verbosity,
    configure_console_logging,
    fmt_count,
    log_error,
    log_event,
    log_table,
    verbosity_from_name,
)
from lib.manifest.validator import validate_training_manifest
from tools import build_quality_dataset as builder
from tools import build_production_validated_manifest as production_builder
from tools import download_landmark_datasets as downloader
from tools.stage_prepared_crops import stage_crops

VIDEO_DATASETS = frozenset({"300vw", "wflw-v"})
PRODUCTION_DATASET = production_builder.DEFAULT_DATASET
PRODUCTION_DATASET_ALIASES = frozenset(
    {"prod", "production", "production-validated", "production_validated"}
)
PREPARE_BUILDABLE_DATASETS = frozenset(
    (*builder.SUPPORTED_DATASETS, PRODUCTION_DATASET)
)
# Datasets that are annotation layers over the existing 300W image cache.
DATASETS_NEEDING_300W_IMAGES = frozenset({"helen"})
# Datasets that are annotation layers over the native AFLW image cache.
DATASETS_NEEDING_AFLW_IMAGES = frozenset({"merl-rav"})


def _prepare_log_level_name(value: str | None) -> str:
    key = str(value or "info").lower()
    if key == "normal":
        return "info"
    if key in {"warning", "error", "critical"}:
        return "quiet"
    if key in {"quiet", "info", "verbose", "debug"}:
        return key
    return "info"


def _short_list(values: T.Sequence[str], *, limit: int = 8) -> str:
    shown = list(values[:limit])
    suffix = f" +{len(values) - limit} more" if len(values) > limit else ""
    return ", ".join(shown) + suffix


def _normalize_prepare_datasets(values: T.Iterable[str]) -> list[str]:
    """Normalize downloader ids plus the local production dataset aliases."""

    datasets: list[str] = []
    for dataset in downloader.normalize_datasets(values):
        key = str(dataset).strip().lower().replace("_", "-")
        canonical = (
            PRODUCTION_DATASET if key in PRODUCTION_DATASET_ALIASES else dataset
        )
        if canonical not in datasets:
            datasets.append(canonical)
    return datasets


# ---------------------------------------------------------------------------
# Source resolution / staging
# ---------------------------------------------------------------------------
def _find_dir_named(root: Path, name: str) -> Path | None:
    direct = root / name
    if direct.is_dir():
        return direct
    if not root.is_dir():
        return None
    for candidate in sorted(root.rglob(name)):
        # macOS zip extraction mirrors the tree under __MACOSX with AppleDouble
        # junk files; matching there would stage resource forks, not data.
        if candidate.is_dir() and "__MACOSX" not in candidate.parts:
            return candidate
    return None


def _is_300w_image_cache_root(path: Path) -> bool:
    return path.is_dir() and any(
        (path / subset).is_dir() for subset in ("afw", "helen", "lfpw", "ibug")
    )


def _normalize_300w_image_cache_candidate(path: Path) -> Path | None:
    if _is_300w_image_cache_root(path):
        return path

    # Accept a direct subset directory such as .../300w/helen by returning its parent.
    if path.name.lower() in {
        "afw",
        "helen",
        "lfpw",
        "ibug",
    } and _is_300w_image_cache_root(path.parent):
        return path.parent

    for nested in (
        path / "data" / "300w" / "300w",
        path / "300w",
        path / "extracted" / "data" / "300w" / "300w",
        path / "extracted" / "300w",
    ):
        if _is_300w_image_cache_root(nested):
            return nested

    return None


def _resolve_300w_image_cache(
    registry: dict[str, T.Any] | None,
    data_root: Path,
) -> Path | None:
    """Find the actual 300W image root containing afw/helen/lfpw/ibug.

    The downloader registry may point at an extraction wrapper directory such as
    data/datasets/300w/extracted/300w.tar.gz. The HELEN builder needs the nested
    image cache root, not just the extraction wrapper.
    """

    candidates: list[Path] = []
    resolved = downloader.resolve_source_dir(registry or {}, "300w", data_root)
    if resolved is not None:
        candidates.extend(
            (
                resolved,
                resolved / "data" / "300w" / "300w",
                resolved / "300w",
            )
        )

    candidates.extend(
        (
            data_root / "300w" / "extracted" / "data" / "300w" / "300w",
            data_root / "300w" / "extracted" / "300w",
            data_root / "300w" / "extracted",
            data_root / "300w",
            ROOT
            / "data"
            / "datasets"
            / "300w"
            / "extracted"
            / "data"
            / "300w"
            / "300w",
            ROOT / "data" / "datasets" / "300w" / "extracted" / "300w",
        )
    )

    seen: set[Path] = set()
    for candidate in candidates:
        try:
            key = candidate.resolve()
        except OSError:
            key = candidate
        if key in seen:
            continue
        seen.add(key)
        normalized = _normalize_300w_image_cache_candidate(candidate)
        if normalized is not None:
            return normalized

    # Last resort: search below the 300W data root for a directory containing HELEN.
    search_roots = []
    if resolved is not None:
        search_roots.append(resolved)
    search_roots.append(data_root / "300w")

    searched: set[Path] = set()
    for search_root in search_roots:
        if not search_root.is_dir():
            continue
        try:
            key = search_root.resolve()
        except OSError:
            key = search_root
        if key in searched:
            continue
        searched.add(key)

        for helen_dir in sorted(search_root.rglob("helen")):
            if not helen_dir.is_dir():
                continue
            normalized = _normalize_300w_image_cache_candidate(helen_dir)
            if normalized is not None:
                return normalized
            normalized = _normalize_300w_image_cache_candidate(helen_dir.parent)
            if normalized is not None:
                return normalized

    return None


def _symlink(link: Path, target: Path) -> None:
    if link.is_symlink() or link.exists():
        if link.is_symlink() and Path(link).resolve() == target.resolve():
            return
        link.unlink()
    link.symlink_to(target.resolve())


_JD_TRAINING_SUBSETS = ("AFW", "HELEN", "IBUG", "LFPW")


def _jd_find_training_data(extracted: Path) -> Path | None:
    """Find the Training_data root holding AFW/HELEN/IBUG/LFPW subset folders."""
    candidates = [extracted / "Training_data", extracted / "Training_data.zip"]
    candidates.extend(
        path.parent
        for subset in _JD_TRAINING_SUBSETS
        for path in sorted(extracted.rglob(subset))
    )
    for candidate in candidates:
        if "__MACOSX" in candidate.parts:
            continue
        if any(
            (candidate / subset / "landmark").is_dir()
            for subset in _JD_TRAINING_SUBSETS
        ):
            return candidate
    return None


def _jd_find_test_data1(extracted: Path) -> Path | None:
    """Find the Test_data1 root, skipping Training_data subset landmark dirs."""
    if (extracted / "Test_data1" / "landmark").is_dir():
        return extracted / "Test_data1"
    for landmark_dir in sorted(extracted.rglob("landmark")):
        if not landmark_dir.is_dir():
            continue
        candidate = landmark_dir.parent
        if "__MACOSX" in candidate.parts:
            continue
        if candidate.name in _JD_TRAINING_SUBSETS:
            continue
        return candidate
    return None


def _stage_jd_landmark(
    data_root: Path, registry: dict[str, T.Any] | None
) -> Path | None:
    """Stage Training_data, Test_data1, Corrected_landmark, and bbox dirs.

    The JD-landmark builder expects ``<root>/Training_data`` (AFW/HELEN/IBUG/LFPW
    with bundled landmark/picture pairs) and ``<root>/Test_data1`` plus
    ``<root>/Corrected_landmark`` and a discoverable training bbox directory.
    The downloader extracts each archive into its own folder, so we link the
    discovered artifacts into a single staging directory the builder can consume.
    """
    extracted = downloader.resolve_source_dir(registry or {}, "jd-landmark", data_root)
    if extracted is None:
        return None
    staged = Path(data_root) / "jd-landmark" / "staged"
    staged.mkdir(parents=True, exist_ok=True)

    training_data = _jd_find_training_data(extracted)
    if training_data is not None:
        _symlink(staged / "Training_data", training_data)
    test_data1 = _jd_find_test_data1(extracted)
    if test_data1 is not None:
        _symlink(staged / "Test_data1", test_data1)
    corrected = _find_dir_named(extracted, "Corrected_landmark")
    if corrected is not None:
        _symlink(staged / "Corrected_landmark", corrected)
    bbox = _find_dir_named(extracted, "training_dataset_face_detection_bounding_box")
    if bbox is not None:
        _symlink(staged / "training_dataset_face_detection_bounding_box", bbox)
    return staged


def _resolve_inputs(
    dataset: str,
    registry: dict[str, T.Any] | None,
    data_root: Path,
    image_root_override: str | None,
    prod_dir: Path | None = None,
) -> tuple[Path | None, str | None]:
    """Return the (source_dir, image_root) the builder should use for a dataset."""
    if dataset == PRODUCTION_DATASET:
        if prod_dir is None:
            raise ValueError("dataset 'prod' requires --prod-dir")
        return Path(prod_dir).expanduser(), None

    image_root = image_root_override
    if dataset == "jd-landmark":
        source = _stage_jd_landmark(data_root, registry)
    else:
        source = downloader.resolve_source_dir(registry or {}, dataset, data_root)
    if image_root is None and dataset in DATASETS_NEEDING_300W_IMAGES:
        cache_300w = _resolve_300w_image_cache(registry, data_root)
        if cache_300w is not None:
            image_root = str(cache_300w)
    if image_root is None and dataset in DATASETS_NEEDING_AFLW_IMAGES:
        aflw_cache = downloader.resolve_source_dir(registry or {}, "aflw", data_root)
        if aflw_cache is not None:
            image_root = str(aflw_cache)
    return source, image_root


# ---------------------------------------------------------------------------
# Build / validate
# ---------------------------------------------------------------------------
def _complete_production_sample_contract(sample: dict[str, T.Any]) -> None:
    """Add the schema-aware training fields omitted by the standalone builder."""

    source_schema = str(sample.get("source_schema") or "2d_68")
    target_schema = str(sample.get("target_schema") or source_schema)
    landmark_count = point_count_for_schema(target_schema)
    head_name = head_name_for_schema(target_schema)
    metadata = sample.setdefault("metadata", {})
    if not isinstance(metadata, dict):
        metadata = {}
        sample["metadata"] = metadata

    split = str(sample.get("split") or metadata.get("split") or "train")
    split_safe_id = str(
        sample.get("split_safe_id")
        or metadata.get("split_safe_id")
        or sample.get("image")
        or sample.get("sample_id")
    )
    mapping_audit = {
        "status": "native",
        "source_schema": source_schema,
        "target_schema": target_schema,
        "projection_to_68": projection_audit_for_schema(source_schema),
    }

    sample.update(
        {
            "source_schema": source_schema,
            "target_schema": target_schema,
            "landmark_count": landmark_count,
            "head_name": head_name,
            "split": split,
            "split_safe_id": split_safe_id,
            "mapping_audit": mapping_audit,
        }
    )
    metadata.setdefault("dataset", PRODUCTION_DATASET)
    metadata.setdefault("source_schema", source_schema)
    metadata.setdefault("target_schema", target_schema)
    metadata.setdefault("landmark_count", landmark_count)
    metadata.setdefault("head_name", head_name)
    metadata.setdefault("split", split)
    metadata.setdefault("split_safe_id", split_safe_id)
    metadata.setdefault("mapping_audit", mapping_audit)


def _build_production_dataset(
    source: Path,
    output_dir: Path,
    *,
    mode: str,
    args: argparse.Namespace,
) -> Path:
    """Build production data in isolation, then merge it into the main manifest."""

    dataset_output_dir = output_dir / "_production_validated"
    metadata = production_builder.build_manifest(
        source,
        dataset_output_dir,
        dataset_name=PRODUCTION_DATASET,
    )
    payload = read_json(dataset_output_dir / "manifest.json")
    payload = _rebase_dataset_payload(
        payload,
        dataset_output_dir=dataset_output_dir,
        output_root=output_dir,
    )
    samples = [
        sample for sample in payload.get("samples", []) if isinstance(sample, dict)
    ]
    for sample in samples:
        _complete_production_sample_contract(sample)

    skipped: list[dict[str, str]] = []
    for key, reason in (
        ("skipped_missing_image", "missing production source image"),
        ("skipped_invalid_face", "invalid production face"),
    ):
        count = int(metadata.get(key, 0))
        skipped.extend(
            [{"sample_id": PRODUCTION_DATASET, "reason": reason}] * max(0, count)
        )

    return builder._write_manifest(
        output_dir,
        PRODUCTION_DATASET,
        "default",
        samples,
        mode=mode,
        allow_overlap=args.allow_overlap,
        scenarios=None,
        skipped=skipped,
    )


def _build_dataset(
    dataset: str,
    source: Path | None,
    image_root: str | None,
    output_dir: Path,
    *,
    mode: str,
    args: argparse.Namespace,
) -> Path:
    if dataset == PRODUCTION_DATASET:
        if source is None:
            raise ValueError("dataset 'prod' requires --prod-dir")
        return _build_production_dataset(source, output_dir, mode=mode, args=args)

    # Guard before argparse: an unsupported id would otherwise hit the builder's
    # ``--dataset`` choices, printing a usage dump and raising SystemExit (which
    # bypasses the per-dataset error containment). Raise a clean, catchable error.
    if dataset not in builder.SUPPORTED_DATASETS:
        raise ValueError(
            f"dataset {dataset!r} is not buildable by build_quality_dataset"
        )
    arglist: list[str] = [
        "--dataset",
        dataset,
        "--output-dir",
        str(output_dir),
        "--manifest-mode",
        mode,
    ]
    if source is not None:
        arglist += ["--source-dir", str(source)]
    if image_root is not None:
        arglist += ["--image-root", str(image_root)]
    if args.allow_overlap:
        arglist += ["--allow-overlap"]
    arglist += ["--workers", str(args.workers)]
    arglist += ["--log-format", str(getattr(args, "log_format", "human"))]
    arglist += [
        "--log-level",
        _prepare_log_level_name(getattr(args, "log_level", "info")),
    ]
    if dataset in VIDEO_DATASETS:
        arglist += ["--frame-stride", str(args.frame_stride)]
        if args.max_frames_per_video is not None:
            arglist += ["--max-frames-per-video", str(args.max_frames_per_video)]
    if args.write_overlays:
        arglist += [
            "--write-overlays",
            "--audit-overlay-limit",
            str(args.audit_overlay_limit),
        ]
    if args.samples_per_scenario is not None:
        arglist += ["--samples-per-scenario", str(args.samples_per_scenario)]
    build_args = builder._parser().parse_args(arglist)
    return builder.build(build_args)


def _validate(
    manifest: Path,
    *,
    require_images: bool,
    manifest_payload: T.Mapping[str, T.Any] | None = None,
    allow_suspicious_geometry: bool = False,
    allow_normalized_non_256: bool = False,
) -> dict[str, T.Any]:
    return validate_training_manifest(
        manifest,
        manifest_payload=manifest_payload,
        require_images=require_images,
        raise_on_error=False,
        allow_suspicious_geometry=allow_suspicious_geometry,
        allow_normalized_non_256=allow_normalized_non_256,
    )


def _dataset_summary(payload: T.Mapping[str, T.Any]) -> dict[str, dict[str, T.Any]]:
    per_dataset: dict[str, dict[str, T.Any]] = {}
    for sample in payload.get("samples", []):
        if not isinstance(sample, dict):
            continue
        name = str(
            sample.get("dataset")
            or sample.get("source", {}).get("dataset")
            or "unknown"
        )
        entry = per_dataset.setdefault(name, {"samples": 0, "schemas": Counter()})
        entry["samples"] += 1
        schema = str(
            sample.get("target_schema") or sample.get("source_schema") or "unknown"
        )
        entry["schemas"][schema] += 1
    return per_dataset


def _is_pathlike(text: str) -> bool:
    """True when a source value looks like a file path rather than a bare id."""

    return "/" in text or "\\" in text or Path(text).is_absolute()


def _sample_image_group_key(sample: T.Mapping[str, T.Any]) -> str | None:
    """Return the source-image grouping key used to prevent same-file split leaks.

    Mirrors ``lib.evaluation.split_safe._source_values``: a real file path is a
    *global* identity with no dataset prefix, because different datasets can
    legitimately reference the same native file (e.g. the 300W image cache shared
    by helen/jd-landmark, or MERL-RAV labels over native AFLW frames). The leakage
    validator keys those shared paths globally, so the cleanup must group them
    globally too -- otherwise it cannot collapse the very same-file split that
    validation then flags. Only bare ids (e.g. ``image_id="212"``) stay namespaced
    to the dataset, to avoid false cross-dataset merges.
    """

    metadata = sample.get("metadata")
    metadata = metadata if isinstance(metadata, dict) else {}
    source = sample.get("source")
    source = source if isinstance(source, dict) else {}
    dataset = str(sample.get("dataset") or source.get("dataset") or "")

    # Prefer a resolved native-image path: it identifies the source file across
    # datasets. ``original_image`` is the native source recorded by crop builders;
    # ``image`` is the native file itself for native-image datasets.
    for key in ("original_image", "source_image", "image", "image_path", "path"):
        for container in (metadata, source, sample):
            text = str(container.get(key) or "").strip()
            if text and _is_pathlike(text):
                return str(Path(text).expanduser())

    # No path available: fall back to an image id. Path-like ids stay global; bare
    # ids are dataset-local, matching the validator's namespacing rule.
    for key in ("image_id", "merl_image_id", "frame_name"):
        for container in (sample, metadata, source):
            text = str(container.get(key) or "").strip()
            if text:
                return (
                    str(Path(text).expanduser())
                    if _is_pathlike(text)
                    else f"{dataset}|{text}"
                )

    return None


def _canonical_split_for_image_group(samples: list[dict[str, T.Any]]) -> str:
    """Choose one split for all samples from the same image/file.

    Test wins so an explicitly held-out source image never gets moved into train.
    Otherwise preserve the first concrete split in the group.
    """

    concrete: list[str] = []
    for sample in samples:
        split = str(sample.get("split") or "").strip().lower()
        if split and split != "unspecified":
            concrete.append(split)

    if "test" in concrete:
        return "test"
    if "train" in concrete:
        return "train"
    return "train"


def _clean_same_image_split_leakage(
    manifest_path: Path,
    payload: T.Mapping[str, T.Any] | None,
) -> T.Mapping[str, T.Any] | None:
    """Normalize same-image groups to one split before manifest validation.

    This keeps multiple landmark sets for the same source image, but prevents
    them from being split across train/test and tripping leakage validation.
    """

    if payload is None or not isinstance(payload, dict):
        return payload

    samples = payload.get("samples")
    if not isinstance(samples, list):
        return payload

    groups: dict[str, list[dict[str, T.Any]]] = {}
    for sample in samples:
        if not isinstance(sample, dict):
            continue
        key = _sample_image_group_key(sample)
        if key is not None:
            groups.setdefault(key, []).append(sample)

    changed = 0
    affected_groups = 0
    for group_samples in groups.values():
        splits = {
            str(sample.get("split") or "").strip().lower()
            for sample in group_samples
            if str(sample.get("split") or "").strip().lower() not in ("", "unspecified")
        }
        if len(splits) <= 1:
            continue

        affected_groups += 1
        canonical = _canonical_split_for_image_group(group_samples)
        for sample in group_samples:
            old_split = str(sample.get("split") or "").strip().lower()
            if old_split != canonical:
                sample["split"] = canonical
                metadata = sample.setdefault("metadata", {})
                if isinstance(metadata, dict):
                    metadata["split"] = canonical
                    metadata["same_image_split_cleanup"] = True
                changed += 1

    if changed:
        metadata = payload.setdefault("metadata", {})
        if isinstance(metadata, dict):
            metadata["same_image_split_cleanup"] = {
                "groups": affected_groups,
                "samples": changed,
            }
        write_json(manifest_path, payload)
        log_event(
            "prepare",
            (
                "cleaned same-image split leakage | "
                f"groups {fmt_count(affected_groups)} | samples {fmt_count(changed)}"
            ),
            level=Verbosity.INFO,
            same_image_groups=affected_groups,
            samples=changed,
        )

    return payload


def _annotate_hard_negative_buckets(
    manifest_path: Path,
    payload: T.Mapping[str, T.Any] | None,
) -> T.Mapping[str, T.Any] | None:
    """Write metadata.hard_negative_bucket on every un-annotated sample.

    The training sampler reads ``metadata.hard_negative_bucket`` first and only
    falls back to the overloaded ``condition`` field otherwise -- where a
    dataset/source label (lapa, fll2, 300vw, ...) would pollute the bucket
    dimension. Baking an authoritative bucket here (classifier, else dataset
    default, else anchor) keeps domain-balanced sampling balancing real hard
    buckets. Non-destructive: ``condition``/``conditions`` are left untouched so
    evaluation slicing is unaffected, and already-annotated samples are
    preserved (a manifest from build_hard_negative_manifest.py is not changed).
    """

    if payload is None or not isinstance(payload, dict):
        return payload
    samples = payload.get("samples")
    if not isinstance(samples, list):
        return payload

    annotated = 0
    by_bucket: Counter[str] = Counter()
    by_source: Counter[str] = Counter()
    for sample in samples:
        if not isinstance(sample, dict):
            continue
        metadata = sample.get("metadata")
        if isinstance(metadata, dict) and metadata.get("hard_negative_bucket"):
            continue
        bucket, source = annotate_sample_bucket_in_place(sample)
        annotated += 1
        by_bucket[bucket] += 1
        by_source[source] += 1

    if annotated:
        payload_meta = payload.setdefault("metadata", {})
        if isinstance(payload_meta, dict):
            payload_meta["hard_negative_bucket_annotation"] = {
                "samples": annotated,
                "by_bucket": dict(by_bucket),
                "by_source": dict(by_source),
            }
        write_json(manifest_path, payload)
        bucket_summary = ", ".join(
            f"{name} {count}" for name, count in sorted(by_bucket.items())
        )
        log_event(
            "prepare",
            (
                f"annotated hard-negative buckets | samples {fmt_count(annotated)}"
                f" | {bucket_summary}"
            ),
            level=Verbosity.INFO,
            samples=annotated,
            by_bucket=dict(by_bucket),
            by_source=dict(by_source),
        )

    return payload


def _short_build_path(value: Path | str | None, *, max_chars: int = 72) -> str:
    if value is None:
        return "-"
    text = str(value)
    if len(text) <= max_chars:
        return text
    parts = Path(text).parts
    if len(parts) >= 3:
        shortened = ".../" + "/".join(parts[-3:])
        if len(shortened) <= max_chars:
            return shortened
    return "..." + text[-max_chars + 3 :]


def _manifest_dataset_build_counts(
    manifest_path: Path, dataset: str
) -> tuple[int, int, int]:
    """Return per-dataset sample count, total manifest samples, skipped count."""

    try:
        payload = read_json(manifest_path)
    except Exception:  # noqa: BLE001
        return 0, 0, 0

    samples = payload.get("samples") if isinstance(payload, dict) else None
    if not isinstance(samples, list):
        return 0, 0, 0

    dataset_count = 0
    for sample in samples:
        if not isinstance(sample, dict):
            continue
        sample_dataset = str(
            sample.get("dataset") or sample.get("source", {}).get("dataset") or ""
        )
        if sample_dataset == dataset:
            dataset_count += 1

    metadata = payload.get("metadata") if isinstance(payload, dict) else {}
    skipped = 0
    if isinstance(metadata, dict):
        try:
            skipped = int(metadata.get("skipped_count", 0))
        except (TypeError, ValueError):
            skipped = 0

    return dataset_count, len(samples), skipped


# ---------------------------------------------------------------------------
# Outer (per-dataset) parallelism
#
# Several datasets can be built at once, but never against the same output root:
# each dataset is built in isolation into ``_datasets/NN-<dataset>/`` with
# manifest_mode="replace", then their manifests are merged serially into the
# combined ``manifest.json``. Validation and crop staging still run once on the
# final combined manifest, exactly as in the serial path.
# ---------------------------------------------------------------------------


def _resolve_parallel_budget(
    dataset_workers: int | None,
    inner_workers: int | None,
    dataset_count: int,
) -> tuple[int, int]:
    """Split the CPU budget between outer datasets and inner workers.

    ``--dataset-workers`` has priority: it is resolved first (clamped to the
    dataset count and the machine's CPU count), then the inner ``--workers``
    count is capped so that ``outer * inner`` never exceeds the CPU count. This
    keeps the total thread fan-out bounded when several dataset builds each
    spawn their own video-extraction / overlay workers. ``dataset_workers == 1``
    yields ``outer == 1`` so the serial path is taken unchanged; ``<= 0`` means
    "use all CPUs" (then clamped to the dataset count), matching ``--workers``.
    """
    total = os.cpu_count() or 1
    outer = max(1, min(resolve_worker_count(dataset_workers, dataset_count), total))
    inner_budget = max(1, total // outer)
    if inner_workers is None or inner_workers <= 0:
        resolved_inner = inner_budget
    else:
        resolved_inner = min(int(inner_workers), inner_budget)
    return outer, max(1, resolved_inner)


def _dataset_output_dir(output_root: Path, index: int, dataset: str) -> Path:
    safe = dataset.replace("/", "_").replace("\\", "_")
    return output_root / "_datasets" / f"{index:02d}-{safe}"


def _rebase_manifest_path(value: T.Any, *, from_root: Path, to_root: Path) -> str:
    """Rewrite a per-dataset manifest path so it resolves under ``to_root``.

    Per-dataset manifests store ``image`` as an absolute crop path (or a native
    image path outside the output dir) and ``landmarks`` relative to their own
    output dir. Re-root those onto the combined output so the merged manifest's
    relative paths resolve against the combined ``manifest.json`` directory, and
    leave genuinely external (native-image) paths absolute.
    """
    raw = Path(str(value))
    resolved = raw if raw.is_absolute() else (from_root / raw).resolve()
    try:
        return resolved.relative_to(to_root.resolve()).as_posix()
    except ValueError:
        return str(resolved)


def _rebase_dataset_payload(
    payload: dict[str, T.Any],
    *,
    dataset_output_dir: Path,
    output_root: Path,
) -> dict[str, T.Any]:
    for sample in payload.get("samples", []):
        if not isinstance(sample, dict):
            continue
        for key in ("image", "landmarks", "ground_truth"):
            if sample.get(key):
                sample[key] = _rebase_manifest_path(
                    sample[key],
                    from_root=dataset_output_dir,
                    to_root=output_root,
                )
    return payload


def _dataset_skipped_examples(
    dataset_output_dir: Path, dataset: str, skipped_count: int
) -> list[dict[str, str]]:
    """Return ``skipped_count`` skipped entries for one dataset.

    Real examples are read from the per-dataset ``dataset_audit.json`` (capped at
    50 there); any remainder is padded with a generic pointer so the combined
    manifest's ``skipped_count`` reflects the true sum across datasets.
    """
    if skipped_count <= 0:
        return []
    examples: list[dict[str, str]] = []
    audit = dataset_output_dir / "dataset_audit.json"
    if audit.is_file():
        try:
            payload = read_json(audit)
        except Exception:  # noqa: BLE001
            payload = None
        raw = payload.get("skipped_examples") if isinstance(payload, dict) else None
        if isinstance(raw, list):
            examples = [item for item in raw if isinstance(item, dict)]
    if len(examples) < skipped_count:
        pad = skipped_count - len(examples)
        examples.extend(
            {"sample_id": dataset, "reason": "see per-dataset manifest"}
            for _ in range(pad)
        )
    return examples[:skipped_count]


def _build_one_dataset_for_parallel(
    *,
    dataset_index: int,
    dataset_total: int,
    dataset: str,
    registry: dict[str, T.Any] | None,
    data_root: Path,
    output_root: Path,
    args: argparse.Namespace,
    inner_workers: int,
    log_status: bool = True,
) -> dict[str, T.Any]:
    """Build a single dataset into its own isolated output dir (mode=replace)."""
    started_at = time.time()
    dataset_output_dir = _dataset_output_dir(output_root, dataset_index, dataset)
    dataset_output_dir.mkdir(parents=True, exist_ok=True)

    record: dict[str, T.Any] = {
        "dataset": dataset,
        "source_dir": None,
        "dataset_output_dir": str(dataset_output_dir),
    }

    source, image_root = _resolve_inputs(
        dataset,
        registry,
        data_root,
        args.image_root,
        getattr(args, "prod_dir", None),
    )
    record["source_dir"] = str(source) if source else None

    index_label = f"{dataset_index:02d}/{dataset_total:02d}"
    if log_status:
        log_event(
            "prepare",
            f"{index_label} build {dataset} | mode replace | "
            f"source {_short_build_path(source)}",
            level=Verbosity.INFO,
            dataset=dataset,
            mode="replace",
            source_dir=str(source) if source else None,
            image_root=image_root,
        )

    # Cap the inner worker count for this build so outer * inner stays within the
    # CPU budget; only --workers is overridden, every other arg is preserved.
    build_args = copy.copy(args)
    build_args.workers = inner_workers
    if not log_status:
        build_args.log_level = "quiet"
    manifest_path = _build_dataset(
        dataset,
        source,
        image_root,
        dataset_output_dir,
        mode="replace",
        args=build_args,
    )

    payload = read_json(manifest_path)
    payload = _rebase_dataset_payload(
        payload,
        dataset_output_dir=dataset_output_dir,
        output_root=output_root,
    )

    dataset_samples, total_samples, skipped_count = _manifest_dataset_build_counts(
        manifest_path, dataset
    )
    elapsed = time.time() - started_at
    if log_status:
        log_event(
            "prepare",
            f"{index_label} done {dataset} | samples {fmt_count(dataset_samples)} | "
            f"skipped {fmt_count(skipped_count)} | {elapsed:.1f}s",
            level=Verbosity.INFO,
            dataset=dataset,
            sample_count=dataset_samples,
            manifest_total=total_samples,
            skipped_count=skipped_count,
            duration_seconds=elapsed,
            manifest=str(manifest_path),
        )

    record.update(
        {
            "status": "built",
            "manifest": str(manifest_path),
            "payload": payload,
            "sample_count": dataset_samples,
            "manifest_total": total_samples,
            "skipped_count": skipped_count,
            "skipped_examples": _dataset_skipped_examples(
                dataset_output_dir, dataset, skipped_count
            ),
            "duration_seconds": elapsed,
        }
    )
    return record


@contextlib.contextmanager
def _opencv_single_threaded() -> T.Iterator[None]:
    """Force OpenCV single-threaded inside the block, restoring prior settings.

    Without this, the outer * inner Python workers would each spawn an internal
    OpenCV thread pool and oversubscribe the machine. The previous thread count
    and OpenCL flag are restored on exit so later crop staging (which runs after
    the parallel build returns) and any subsequent build in the same process are
    unaffected.
    """
    try:
        import cv2
    except Exception:  # noqa: BLE001
        yield
        return

    old_threads = cv2.getNumThreads()
    old_opencl: bool | None = None
    with contextlib.suppress(Exception):
        old_opencl = cv2.ocl.useOpenCL()
    cv2.setNumThreads(0)
    with contextlib.suppress(Exception):
        cv2.ocl.setUseOpenCL(False)
    try:
        yield
    finally:
        cv2.setNumThreads(old_threads)
        if old_opencl is not None:
            with contextlib.suppress(Exception):
                cv2.ocl.setUseOpenCL(old_opencl)


def _merge_dedupe_key(sample: dict[str, T.Any]) -> tuple[T.Any, ...]:
    """Stable per-sample identity for merge dedupe.

    Re-running a merge in parallel mode rebuilds the same logical samples under
    fresh ``_datasets/NN-.../`` paths, so the image-path key used by
    ``_write_manifest`` would treat them as new and append duplicates. Keying on
    source identity instead collapses a logical sample to one entry regardless of
    where its crop currently lives.
    """
    source = sample.get("source")
    source_id = source.get("source_id") if isinstance(source, dict) else None
    return (
        sample.get("dataset"),
        source_id or sample.get("sample_id"),
        sample.get("frame_id"),
        sample.get("split"),
    )


def _build_datasets_parallel(
    datasets: list[str],
    *,
    registry: dict[str, T.Any] | None,
    data_root: Path,
    output_root: Path,
    args: argparse.Namespace,
    outer_workers: int,
    inner_workers: int,
) -> tuple[list[dict[str, T.Any]], bool]:
    """Build every dataset concurrently, then merge into the combined manifest.

    Each dataset builds in isolation; only the final serial merge writes to the
    combined ``output_root/manifest.json``. Returns ``(results, built_any)`` with
    the same record shape the serial path produces.
    """
    dataset_total = len(datasets)
    log_event(
        "prepare",
        f"parallel build | datasets {dataset_total} | "
        f"dataset-workers {outer_workers} | workers/dataset {inner_workers}",
        level=Verbosity.INFO,
        datasets=dataset_total,
        dataset_workers=outer_workers,
        inner_workers=inner_workers,
    )

    results: list[dict[str, T.Any]] = []
    errors_to_log: list[tuple[str, Exception]] = []
    # One shared Progress (concurrent build rows + an overall task) when a TTY is
    # available; otherwise per-build bars are suppressed so worker threads don't
    # interleave output and the [prepare] NN/NN lines remain the indicator.
    with (
        concurrent_progress() as build_progress,
        _opencv_single_threaded(),
        ThreadPoolExecutor(max_workers=outer_workers) as executor,
    ):
        overall_task = (
            build_progress.add_task("Datasets", total=dataset_total)
            if build_progress is not None
            else None
        )
        future_to_dataset = {
            executor.submit(
                _build_one_dataset_for_parallel,
                dataset_index=index,
                dataset_total=dataset_total,
                dataset=dataset,
                registry=registry,
                data_root=data_root,
                output_root=output_root,
                args=args,
                inner_workers=inner_workers,
                log_status=build_progress is None,
            ): dataset
            for index, dataset in enumerate(datasets, start=1)
        }
        for future in as_completed(future_to_dataset):
            dataset = future_to_dataset[future]
            if overall_task is not None:
                build_progress.advance(overall_task)
            try:
                results.append(future.result())
            except Exception as err:  # noqa: BLE001
                errors_to_log.append((dataset, err))
                results.append(
                    {
                        "dataset": dataset,
                        "source_dir": None,
                        "status": "error",
                        "error": str(err),
                    }
                )

    for dataset, err in errors_to_log:
        log_error("prepare", f"{dataset}: {err}")

    # Restore the requested dataset order for a deterministic merge and summary.
    order = {dataset: index for index, dataset in enumerate(datasets)}
    results.sort(key=lambda item: order.get(item["dataset"], 10**9))

    new_samples: list[dict[str, T.Any]] = []
    skipped: list[dict[str, str]] = []
    built_any = False
    for record in results:
        if record.get("status") == "error":
            continue
        built_any = True
        payload = record.pop("payload", None) or {}
        for sample in payload.get("samples", []):
            if isinstance(sample, dict):
                new_samples.append(sample)
        skipped.extend(record.get("skipped_examples") or [])

    if built_any:
        # Merge in memory keyed on source identity (not image path) so a repeated
        # --manifest-mode merge in parallel mode never doubles samples, then write
        # once as a replace. _write_manifest still applies its image-path dedupe
        # for cross-dataset overlap when --allow-overlap is not set.
        deduped: dict[tuple[T.Any, ...], dict[str, T.Any]] = {}
        if args.manifest_mode == "merge":
            combined_manifest = output_root / "manifest.json"
            if combined_manifest.is_file():
                existing = read_json(combined_manifest)
                for sample in existing.get("samples", []):
                    if isinstance(sample, dict):
                        deduped[_merge_dedupe_key(sample)] = sample
        # Freshly built samples win on a stable-key collision (current paths).
        for sample in new_samples:
            deduped[_merge_dedupe_key(sample)] = sample

        builder._write_manifest(
            output_root,
            "multi_dataset",
            "default",
            list(deduped.values()),
            mode="replace",
            allow_overlap=args.allow_overlap,
            scenarios=None,
            skipped=skipped,
        )

    return results, built_any


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def _ensure_prepare_logging_defaults(args) -> None:
    """Keep prepare(args) compatible with hand-built argparse.Namespace tests."""

    if not hasattr(args, "log_format"):
        args.log_format = "human"
    if not hasattr(args, "log_level"):
        args.log_level = "info"
    if not hasattr(args, "progress"):
        args.progress = True


def _stage_combined_crops(
    combined_manifest: Path,
    args: argparse.Namespace,
    manifest_payload: T.Mapping[str, T.Any] | None,
) -> T.Mapping[str, T.Any] | None:
    """Stage 256x256 crops in place and return the re-read manifest payload."""

    if not combined_manifest.is_file():
        return manifest_payload
    sample_count = 0
    if isinstance(manifest_payload, dict) and isinstance(
        manifest_payload.get("samples"), list
    ):
        sample_count = len(manifest_payload["samples"])
    log_event(
        "prepare",
        (
            f"stage-crops begin | manifest {combined_manifest} | "
            f"samples {fmt_count(sample_count)}"
        ),
        level=Verbosity.INFO,
        manifest=str(combined_manifest),
        samples=sample_count,
    )
    started_at = time.time()
    stats = stage_crops(
        combined_manifest,
        out_manifest=combined_manifest,
        images_subdir=getattr(args, "stage_crops_subdir", "images"),
        force=getattr(args, "force_stage_crops", False),
        strict=False,
        workers=getattr(args, "workers", 1),
        validate_geometry=True,
        drop_invalid_geometry=True,
        drop_suspicious_geometry=bool(
            getattr(args, "drop_suspicious_stage_geometry", False)
        ),
    )
    elapsed = time.time() - started_at
    identical = stats["staged"] + stats["reused"]
    total = identical + len(stats["mismatches"])
    mismatch_note = (
        f" | left native {fmt_count(len(stats['mismatches']))}"
        if stats["mismatches"]
        else ""
    )
    log_event(
        "prepare",
        (
            f"stage-crops | crops {fmt_count(stats['staged'])} unique | "
            f"reused {fmt_count(stats['reused'])} | "
            f"skipped 256x256 {fmt_count(stats['skipped_already_256'])} | "
            f"bit-identical {fmt_count(identical)}/{fmt_count(total)}{mismatch_note} | "
            f"dir {stats['images_root']} | {elapsed:.1f}s"
        ),
        level=Verbosity.INFO,
        staged=stats["staged"],
        reused=stats["reused"],
        skipped_already_256=stats["skipped_already_256"],
        skipped_no_image=stats["skipped_no_image"],
        mismatched=len(stats["mismatches"]),
        images_root=stats["images_root"],
        duration_seconds=elapsed,
    )
    return read_json(combined_manifest) if combined_manifest.is_file() else None


def prepare(args: argparse.Namespace) -> int:
    _ensure_prepare_logging_defaults(args)
    datasets = _normalize_prepare_datasets(args.datasets)
    if not datasets:
        log_error("prepare", "no datasets requested. Pass --datasets <id> [<id> ...].")
        return 2
    if PRODUCTION_DATASET in datasets and getattr(args, "prod_dir", None) is None:
        log_error("prepare", "dataset 'prod' requires --prod-dir.")
        return 2
    data_root = Path(args.data_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    registry: dict[str, T.Any] | None = None
    if not args.skip_download:
        download_target_names = [
            dataset for dataset in datasets if dataset != PRODUCTION_DATASET
        ]
        if any(d in DATASETS_NEEDING_300W_IMAGES for d in datasets):
            download_target_names.append("300w")
        if any(d in DATASETS_NEEDING_AFLW_IMAGES for d in datasets):
            download_target_names.append("aflw")
        download_targets = downloader.normalize_datasets(download_target_names)
        if download_targets:
            log_event(
                "download",
                f"sources {len(download_targets)} | {_short_list(download_targets)}",
            )
            _, registry = downloader.download_datasets(
                download_targets,
                output_root=data_root,
                extract=True,
                force=args.force,
                skip_checksum=args.skip_checksum,
                keep_going=True,
                # --dataset-workers also fans out the (I/O-bound) download/extract step.
                workers=getattr(args, "dataset_workers", 1),
            )
        else:
            registry = downloader.load_registry(data_root)
            log_event(
                "download",
                "skipping download for local production dataset",
                level=Verbosity.INFO,
            )
    else:
        registry = downloader.load_registry(data_root)

    # Some requested ids (e.g. ``aflw``) are image-only base caches the builder
    # cannot build directly -- they are downloaded above as source layers for
    # other datasets (merl-rav over AFLW). Drop any non-buildable id from the
    # build/merge list so it never reaches the builder's ``--dataset`` choices,
    # where it would otherwise trigger an argparse usage dump and SystemExit.
    non_buildable = [d for d in datasets if d not in PREPARE_BUILDABLE_DATASETS]
    if non_buildable:
        log_event(
            "prepare",
            f"skipping non-buildable source caches | {_short_list(non_buildable)}",
            level=Verbosity.INFO,
            skipped_non_buildable=non_buildable,
        )
        datasets = [d for d in datasets if d in PREPARE_BUILDABLE_DATASETS]
    if not datasets:
        log_error(
            "prepare",
            "no buildable datasets requested (only source-only caches were given).",
        )
        return 2

    results: list[dict[str, T.Any]] = []
    built_any = False
    # A multi-dataset run continues past a single dataset's failure so one bad
    # source cannot abort the rest; a single-dataset run has nothing else to build,
    # so it fails fast unless --keep-going is requested explicitly.
    keep_going = args.keep_going or len(datasets) > 1
    dataset_total = len(datasets)
    # --dataset-workers has priority over --workers; the budget split caps inner
    # workers so outer * inner never exceeds the CPU count. dataset_workers == 1
    # (or a single dataset) keeps outer_workers == 1 and the serial path below;
    # <=0 requests all CPUs.
    outer_workers, inner_workers = _resolve_parallel_budget(
        getattr(args, "dataset_workers", 1), args.workers, dataset_total
    )
    if outer_workers > 1 and dataset_total > 1:
        # Outer parallelism builds each dataset in its own isolated output dir and
        # merges them serially; the serial loop below is skipped (empty iter).
        results, built_any = _build_datasets_parallel(
            datasets,
            registry=registry,
            data_root=data_root,
            output_root=output_root,
            args=args,
            outer_workers=outer_workers,
            inner_workers=inner_workers,
        )
        dataset_iter: list[str] = []
    else:
        dataset_iter = datasets
    for dataset_index, dataset in enumerate(dataset_iter, start=1):
        # First built dataset honors the requested mode; later ones merge into it.
        mode = args.manifest_mode if not built_any else "merge"
        record: dict[str, T.Any] = {"dataset": dataset, "source_dir": None}
        started_at = time.time()
        index_label = f"{dataset_index:02d}/{dataset_total:02d}"

        try:
            # Resolution/staging runs inside the try so a single dataset's missing
            # source or staging error is contained rather than aborting the run.
            source, image_root = _resolve_inputs(
                dataset,
                registry,
                data_root,
                args.image_root,
                getattr(args, "prod_dir", None),
            )
            record["source_dir"] = str(source) if source else None

            log_event(
                "prepare",
                (
                    f"{index_label} build {dataset} | mode {mode} | "
                    f"source {_short_build_path(source)}"
                ),
                level=Verbosity.INFO,
                dataset=dataset,
                mode=mode,
                source_dir=str(source) if source else None,
                image_root=image_root,
            )

            manifest_path = _build_dataset(
                dataset, source, image_root, output_root, mode=mode, args=args
            )
            built_any = True
            record["status"] = "built"
            record["manifest"] = str(manifest_path)

            dataset_samples, total_samples, skipped = _manifest_dataset_build_counts(
                manifest_path, dataset
            )
            elapsed = time.time() - started_at
            log_event(
                "prepare",
                (
                    f"{index_label} done {dataset} | "
                    f"samples {fmt_count(dataset_samples)} | "
                    f"manifest total {fmt_count(total_samples)} | "
                    f"skipped {fmt_count(skipped)} | {elapsed:.1f}s"
                ),
                level=Verbosity.INFO,
                dataset=dataset,
                sample_count=dataset_samples,
                manifest_total=total_samples,
                skipped_count=skipped,
                duration_seconds=elapsed,
                manifest=str(manifest_path),
            )
        except Exception as err:  # noqa: BLE001
            record["status"] = "error"
            record["error"] = str(err)
            elapsed = time.time() - started_at
            log_error(
                "prepare",
                f"{dataset}: {err} | {index_label} failed after {elapsed:.1f}s",
            )
            if not keep_going:
                results.append(record)
                _print_summary(results, None, output_root, datasets)
                return 1
        results.append(record)

    combined_manifest = output_root / "manifest.json"
    combined_payload = (
        read_json(combined_manifest) if combined_manifest.is_file() else None
    )
    if built_any and combined_manifest.is_file():
        combined_payload = _clean_same_image_split_leakage(
            combined_manifest,
            combined_payload,
        )
        combined_payload = _annotate_hard_negative_buckets(
            combined_manifest,
            combined_payload,
        )
    report: dict[str, T.Any] | None = None
    if built_any and not args.skip_validate:
        log_event(
            "prepare",
            f"validate combined manifest begin | manifest {combined_manifest}",
            level=Verbosity.INFO,
            manifest=str(combined_manifest),
        )
        report = _validate(
            combined_manifest,
            require_images=not args.skip_image_exists_check,
            manifest_payload=combined_payload,
            allow_suspicious_geometry=bool(getattr(args, "stage_crops", False)),
            allow_normalized_non_256=bool(getattr(args, "stage_crops", False)),
        )

    # Crop staging runs after validation (so a malformed manifest is never
    # staged) and only when the manifest is valid. It augments the manifest in
    # place; the loader falls back to native decode for any unstaged sample, so
    # it can only speed training up, never change its data.
    if (
        built_any
        and getattr(args, "stage_crops", False)
        and (report is None or report.get("ok", False))
    ):
        combined_payload = _stage_combined_crops(
            combined_manifest, args, combined_payload
        )
        if not args.skip_validate:
            log_event(
                "prepare",
                f"validate staged manifest begin | manifest {combined_manifest}",
                level=Verbosity.INFO,
                manifest=str(combined_manifest),
            )
            report = _validate(
                combined_manifest,
                require_images=not args.skip_image_exists_check,
                manifest_payload=combined_payload,
                allow_suspicious_geometry=bool(getattr(args, "stage_crops", False)),
                allow_normalized_non_256=bool(getattr(args, "stage_crops", False)),
            )

    _print_summary(
        results, report, output_root, datasets, manifest_payload=combined_payload
    )

    errored = [r for r in results if r["status"] == "error"]
    if not built_any:
        return 1
    if report is not None and not report.get("ok", False):
        return 1
    return 1 if errored else 0


def _print_summary(
    results: list[dict[str, T.Any]],
    report: dict[str, T.Any] | None,
    output_root: Path,
    datasets: list[str],
    *,
    manifest_payload: T.Mapping[str, T.Any] | None = None,
) -> None:
    combined_manifest = output_root / "manifest.json"
    per_dataset = (
        _dataset_summary(manifest_payload) if manifest_payload is not None else {}
    )
    errors = [record for record in results if record["status"] == "error"]
    built = [record for record in results if record["status"] != "error"]

    log_event(
        "prepare",
        (
            f"Per-dataset summary | datasets {len(datasets)} | built {len(built)} | "
            f"errors {len(errors)} | names {_short_list(datasets)} | output {output_root}"
        ),
        level=Verbosity.INFO,
        datasets=len(datasets),
        built=len(built),
        errors=len(errors),
        output_root=str(output_root),
    )

    rows: list[list[T.Any]] = []
    for record in results:
        dataset = record["dataset"]
        if record["status"] == "error":
            rows.append([dataset, "ERROR", record.get("error", "")])
            continue
        stats = per_dataset.get(dataset, {})
        count = stats.get("samples", 0)
        schemas = (
            ",".join(f"{k}={v}" for k, v in sorted(stats.get("schemas", {}).items()))
            or "-"
        )
        rows.append([dataset, fmt_count(count), schemas])
    if rows:
        log_table(
            "prepare",
            "per-dataset",
            rows,
            headers=("dataset", "samples", "schemas"),
            level=Verbosity.VERBOSE,
        )

    if report is not None:
        log_event(
            "prepare",
            (
                f"Combined manifest summary | manifest {report['manifest']} | ok {report['ok']} | "
                f"samples {fmt_count(report['valid_samples'])}/"
                f"{fmt_count(report['total_samples'])} | "
                f"schemas {len(report['schemas'])} | "
                f"leakage {report['leakage']['violation_count']}"
            ),
            level=Verbosity.INFO,
            manifest=report["manifest"],
            ok=report["ok"],
            total_samples=report["total_samples"],
            valid_samples=report["valid_samples"],
            schemas=report["schemas"],
            heads=report["heads"],
            leakage=report["leakage"],
        )

    if combined_manifest.is_file():
        log_event(
            "prepare",
            (
                "train command: python tools/run_cdvit_manifest_training_pipeline.py "
                f"--manifest {combined_manifest}"
            ),
            level=Verbosity.INFO,
        )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        required=True,
        metavar="DATASET",
        help="One or more datasets, space- and/or comma-separated (e.g. --datasets wflw-v 300vw,cofw29).",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data/datasets"),
        help="Download/cache root.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/prepared"),
        help="Combined manifest output root.",
    )
    parser.add_argument(
        "--prod-dir",
        "--production-dir",
        dest="prod_dir",
        type=Path,
        default=None,
        help=(
            "Production source directory or .zip containing images and one .fsa "
            "file. Required when --datasets includes prod."
        ),
    )
    parser.add_argument(
        "--image-root",
        default=None,
        help="Override image root (defaults to the 300W cache for annotation-layer datasets).",
    )
    parser.add_argument(
        "--manifest-mode",
        choices=("replace", "merge"),
        default="replace",
        help="Replace (fresh) or merge into an existing combined manifest.",
    )
    parser.add_argument(
        "--allow-overlap",
        action="store_true",
        help="Keep duplicate image paths across datasets.",
    )
    parser.add_argument(
        "--write-overlays",
        action="store_true",
        help="Write visual landmark overlay audit images.",
    )
    parser.add_argument(
        "--samples-per-scenario",
        type=int,
        default=None,
        help=(
            "Maximum samples to keep per scenario/condition when building each dataset. "
            "Forwarded to build_quality_dataset.py as --samples-per-scenario."
        ),
    )
    parser.add_argument("--audit-overlay-limit", type=int, default=50)
    parser.add_argument(
        "--frame-stride", type=int, default=1, help="Frame stride for video datasets."
    )
    parser.add_argument(
        "--max-frames-per-video",
        type=int,
        default=None,
        help="Cap frames per video for video datasets.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=16,
        help="Parallel workers for video frame extraction and overlay rendering (<=0 uses all CPUs).",
    )
    parser.add_argument(
        "--dataset-workers",
        type=int,
        default=1,
        help=(
            "Parallel datasets to build during multidataset prepare runs. 1 keeps "
            "the current serial behavior; <=0 uses all CPUs (clamped to the dataset "
            "count). When >1, each dataset is built in an isolated output dir and "
            "merged; --dataset-workers takes priority over --workers and the two "
            "combined never exceed the CPU count."
        ),
    )
    parser.add_argument(
        "--force", action="store_true", help="Redownload/re-extract existing files."
    )
    parser.add_argument(
        "--skip-checksum",
        action="store_true",
        help="Skip stored checksum verification.",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Reuse already-downloaded/extracted assets from --data-root.",
    )
    parser.add_argument(
        "--skip-validate", action="store_true", help="Skip manifest validation."
    )
    parser.add_argument(
        "--skip-image-exists-check",
        action="store_true",
        help="Do not require manifest images to exist during validation.",
    )
    parser.add_argument(
        "--stage-crops",
        action="store_true",
        help=(
            "After building and validating, stage 256x256 crops for native-image "
            "samples and record them in the combined manifest so training skips "
            "full-resolution decode. Output-neutral: the loader falls back to "
            "native decode for any unstaged sample."
        ),
    )
    parser.add_argument(
        "--stage-crops-subdir",
        default="images",
        help="Crop directory relative to --output-root (default: images).",
    )
    parser.add_argument(
        "--force-stage-crops",
        action="store_true",
        help="Rewrite staged crop PNGs even when they already exist.",
    )
    parser.add_argument(
        "--drop-suspicious-stage-geometry",
        action="store_true",
        help=(
            "During --stage-crops, drop samples with suspicious-but-trainable "
            "geometry. By default only invalid geometry is dropped; suspicious "
            "samples are kept and review overlays are written."
        ),
    )
    parser.add_argument(
        "--keep-going",
        action="store_true",
        help=(
            "Continue after a dataset build fails. Multi-dataset runs already "
            "continue past a single failure; this also keeps single-dataset runs going."
        ),
    )
    parser.add_argument(
        "--progress",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Show download/extraction progress indicators and long-build heartbeat lines.",
    )
    parser.add_argument(
        "--log-level",
        default="info",
        choices=(
            "quiet",
            "info",
            "normal",
            "verbose",
            "debug",
            "warning",
            "error",
            "critical",
            "DEBUG",
            "INFO",
            "WARNING",
            "ERROR",
            "CRITICAL",
        ),
    )
    parser.add_argument("--log-format", default="human", choices=("human", "json"))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    configure_console_logging(
        verbosity_from_name(
            _prepare_log_level_name(getattr(args, "log_level", "info"))
        ),
        getattr(args, "log_format", "human"),
    )
    from lib.datasets.progress import set_progress_enabled

    set_progress_enabled(bool(getattr(args, "progress", True)))
    try:
        return prepare(args)
    except KeyboardInterrupt:
        log_error(
            "prepare",
            "interrupted by user (Ctrl-C). Any partially built manifest in the output root may be incomplete; re-run to finish.",
        )
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
