#!/usr/bin/env python3
"""Path-aware multi-dataset landmark preparation orchestrator.

This command stitches together the existing downloader and manifest builder so a
user can go from "download" to "training-ready manifest" without manually
plumbing source/cache paths between tools. For each requested dataset it will:

* download (or reuse cached) source archives into a default data root,
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
"""

# ruff: noqa: E402
from __future__ import annotations

import argparse
import sys
import time
import typing as T
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lib.io_utils import read_json
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
from tools import download_landmark_datasets as downloader

VIDEO_DATASETS = frozenset({"300vw", "wflw-v"})
# Datasets that are annotation layers over the existing 300W image cache.
DATASETS_NEEDING_300W_IMAGES = frozenset({"jd-landmark", "helen"})
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
        if candidate.is_dir():
            return candidate
    return None


def _find_dir_with_child(root: Path, child: str) -> Path | None:
    if not root.is_dir():
        return None
    if (root / child).is_dir():
        return root
    for candidate in sorted(root.rglob(child)):
        if candidate.is_dir() and candidate.parent.is_dir():
            return candidate.parent
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
    data/datasets/300w/extracted/300w.tar.gz. HELEN/JD builders need the nested
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


def _stage_jd_landmark(
    data_root: Path, registry: dict[str, T.Any] | None
) -> Path | None:
    """Stage Test_data1, Corrected_landmark, and bbox dirs under one source root.

    The JD-landmark builder expects ``<root>/Test_data1`` and
    ``<root>/Corrected_landmark`` plus a discoverable training bbox directory.
    The downloader extracts each archive into its own folder, so we link the
    discovered artifacts into a single staging directory the builder can consume.
    """
    extracted = downloader.resolve_source_dir(registry or {}, "jd-landmark", data_root)
    if extracted is None:
        return None
    staged = Path(data_root) / "jd-landmark" / "staged"
    staged.mkdir(parents=True, exist_ok=True)

    test_data1 = _find_dir_with_child(extracted, "landmark")
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
) -> tuple[Path | None, str | None]:
    """Return the (source_dir, image_root) the builder should use for a dataset."""
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
def _build_dataset(
    dataset: str,
    source: Path | None,
    image_root: str | None,
    output_dir: Path,
    *,
    mode: str,
    args: argparse.Namespace,
) -> Path:
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
) -> dict[str, T.Any]:
    return validate_training_manifest(
        manifest,
        manifest_payload=manifest_payload,
        require_images=require_images,
        raise_on_error=False,
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


def prepare(args: argparse.Namespace) -> int:
    _ensure_prepare_logging_defaults(args)
    datasets = downloader.normalize_datasets(args.datasets)
    if not datasets:
        log_error("prepare", "no datasets requested. Pass --datasets <id> [<id> ...].")
        return 2
    data_root = Path(args.data_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    registry: dict[str, T.Any] | None = None
    if not args.skip_download:
        download_target_names = list(datasets)
        if any(d in DATASETS_NEEDING_300W_IMAGES for d in datasets):
            download_target_names.append("300w")
        if any(d in DATASETS_NEEDING_AFLW_IMAGES for d in datasets):
            download_target_names.append("aflw")
        download_targets = downloader.normalize_datasets(download_target_names)
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
        )
    else:
        registry = downloader.load_registry(data_root)

    results: list[dict[str, T.Any]] = []
    built_any = False
    # A multi-dataset run continues past a single dataset's failure so one bad
    # source cannot abort the rest; a single-dataset run has nothing else to build,
    # so it fails fast unless --keep-going is requested explicitly.
    keep_going = args.keep_going or len(datasets) > 1
    dataset_total = len(datasets)
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
                dataset, registry, data_root, args.image_root
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
    report: dict[str, T.Any] | None = None
    if built_any and not args.skip_validate:
        report = _validate(
            combined_manifest,
            require_images=not args.skip_image_exists_check,
            manifest_payload=combined_payload,
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
