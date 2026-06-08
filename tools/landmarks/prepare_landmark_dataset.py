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

    python tools/landmarks/prepare_landmark_dataset.py \
      --datasets wflw-v \
      --include-google-drive \
      --write-overlays
"""

from __future__ import annotations

import argparse
import json
import sys
import typing as T
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lib.landmarks.datasets.progress import track
from lib.landmarks.manifest.validator import validate_training_manifest
from tools.landmarks import build_quality_dataset as builder
from tools.landmarks import download_landmark_datasets as downloader

VIDEO_DATASETS = frozenset({"300vw", "wflw-v"})
# Datasets that are annotation layers over the existing 300W image cache.
DATASETS_NEEDING_300W_IMAGES = frozenset({"jd-landmark", "helen"})


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


def _symlink(link: Path, target: Path) -> None:
    if link.is_symlink() or link.exists():
        if link.is_symlink() and Path(link).resolve() == target.resolve():
            return
        link.unlink()
    link.symlink_to(target.resolve())


def _stage_jd_landmark(data_root: Path, registry: dict[str, T.Any] | None) -> Path | None:
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
    dataset: str, registry: dict[str, T.Any] | None, data_root: Path, image_root_override: str | None
) -> tuple[Path | None, str | None]:
    """Return the (source_dir, image_root) the builder should use for a dataset."""
    image_root = image_root_override
    if dataset == "jd-landmark":
        source = _stage_jd_landmark(data_root, registry)
    else:
        source = downloader.resolve_source_dir(registry or {}, dataset, data_root)
    if image_root is None and dataset in DATASETS_NEEDING_300W_IMAGES:
        cache_300w = downloader.resolve_source_dir(registry or {}, "300w", data_root)
        if cache_300w is not None:
            image_root = str(cache_300w)
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
    if dataset in VIDEO_DATASETS:
        arglist += ["--frame-stride", str(args.frame_stride)]
        if args.max_frames_per_video is not None:
            arglist += ["--max-frames-per-video", str(args.max_frames_per_video)]
    if args.write_overlays:
        arglist += ["--write-overlays", "--audit-overlay-limit", str(args.audit_overlay_limit)]
    build_args = builder._parser().parse_args(arglist)
    return builder.build(build_args)


def _validate(manifest: Path, *, require_images: bool) -> dict[str, T.Any]:
    return validate_training_manifest(
        manifest,
        require_images=require_images,
        raise_on_error=False,
    )


def _dataset_summary(manifest_path: Path) -> dict[str, dict[str, T.Any]]:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    per_dataset: dict[str, dict[str, T.Any]] = {}
    for sample in payload.get("samples", []):
        if not isinstance(sample, dict):
            continue
        name = str(sample.get("dataset") or sample.get("source", {}).get("dataset") or "unknown")
        entry = per_dataset.setdefault(name, {"samples": 0, "schemas": {}})
        entry["samples"] += 1
        schema = str(sample.get("target_schema") or sample.get("source_schema") or "unknown")
        entry["schemas"][schema] = entry["schemas"].get(schema, 0) + 1
    return per_dataset


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def prepare(args: argparse.Namespace) -> int:
    datasets = downloader.normalize_datasets(args.datasets)
    if not datasets:
        print("No datasets requested. Pass --datasets <id> [<id> ...].", file=sys.stderr)
        return 2
    data_root = Path(args.data_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    registry: dict[str, T.Any] | None = None
    if not args.skip_download:
        download_targets = downloader.normalize_datasets(
            [*datasets, "300w"] if any(d in DATASETS_NEEDING_300W_IMAGES for d in datasets) else datasets
        )
        print(f"Downloading sources for: {', '.join(download_targets)}")
        _, registry = downloader.download_datasets(
            download_targets,
            output_root=data_root,
            include_google_drive=args.include_google_drive,
            extract=True,
            force=args.force,
            skip_checksum=args.skip_checksum,
            keep_going=True,
        )
    else:
        registry = downloader.load_registry(data_root)

    results: list[dict[str, T.Any]] = []
    built_any = False
    dataset_bar = track(datasets, desc="Prepare", total=len(datasets), unit="dataset")
    for dataset in dataset_bar:
        dataset_bar.set_description(f"Prepare {dataset}")
        # First built dataset honors the requested mode; later ones merge into it.
        mode = args.manifest_mode if not built_any else "merge"
        source, image_root = _resolve_inputs(dataset, registry, data_root, args.image_root)
        record: dict[str, T.Any] = {"dataset": dataset, "source_dir": str(source) if source else None}
        try:
            manifest_path = _build_dataset(
                dataset, source, image_root, output_root, mode=mode, args=args
            )
            built_any = True
            record["status"] = "built"
            record["manifest"] = str(manifest_path)
        except Exception as err:  # noqa: BLE001
            record["status"] = "error"
            record["error"] = str(err)
            print(f"ERROR: {dataset}: {err}", file=sys.stderr)
            if not args.keep_going:
                results.append(record)
                _print_summary(results, None, output_root, datasets)
                return 1
        results.append(record)

    combined_manifest = output_root / "manifest.json"
    report: dict[str, T.Any] | None = None
    if built_any and not args.skip_validate:
        report = _validate(combined_manifest, require_images=not args.skip_image_exists_check)

    _print_summary(results, report, output_root, datasets)

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
) -> None:
    combined_manifest = output_root / "manifest.json"
    per_dataset = _dataset_summary(combined_manifest) if combined_manifest.is_file() else {}

    print("\nPer-dataset summary:")
    for record in results:
        dataset = record["dataset"]
        if record["status"] == "error":
            print(f"  {dataset:14s} ERROR: {record['error']}")
            continue
        stats = per_dataset.get(dataset, {})
        count = stats.get("samples", 0)
        schemas = ",".join(f"{k}={v}" for k, v in sorted(stats.get("schemas", {}).items())) or "-"
        print(f"  {dataset:14s} samples={count} schemas={schemas}")

    if report is not None:
        print("\nCombined manifest summary:")
        print(f"  manifest:      {report['manifest']}")
        print(f"  ok:            {report['ok']}")
        print(f"  total_samples: {report['total_samples']}")
        print(f"  valid_samples: {report['valid_samples']}")
        print(f"  schemas:       {report['schemas']}")
        print(f"  heads:         {report['heads']}")
        print(f"  leakage:       {report['leakage']['violation_count']}")

    if combined_manifest.is_file():
        print("\nTraining command:")
        print(
            "  python tools/landmarks/run_cdvit_manifest_training_pipeline.py "
            f"--manifest {combined_manifest}"
        )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--datasets",
        nargs="+",
        required=True,
        metavar="DATASET",
        help="One or more datasets, space- and/or comma-separated (e.g. --datasets wflw-v 300vw,cofw-original).",
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
    parser.add_argument("--image-root", default=None, help="Override image root (defaults to the 300W cache for annotation-layer datasets).")
    parser.add_argument("--manifest-mode", choices=("replace", "merge"), default="replace", help="Replace (fresh) or merge into an existing combined manifest.")
    parser.add_argument("--allow-overlap", action="store_true", help="Keep duplicate image paths across datasets.")
    parser.add_argument("--include-google-drive", action="store_true", help="Download Google Drive assets with gdown when available.")
    parser.add_argument("--write-overlays", action="store_true", help="Write visual landmark overlay audit images.")
    parser.add_argument("--audit-overlay-limit", type=int, default=50)
    parser.add_argument("--frame-stride", type=int, default=1, help="Frame stride for video datasets.")
    parser.add_argument("--max-frames-per-video", type=int, default=None, help="Cap frames per video for video datasets.")
    parser.add_argument("--workers", type=int, default=1, help="Parallel workers for video frame extraction and overlay rendering (<=0 uses all CPUs).")
    parser.add_argument("--force", action="store_true", help="Redownload/re-extract existing files.")
    parser.add_argument("--skip-checksum", action="store_true", help="Skip stored checksum verification.")
    parser.add_argument("--skip-download", action="store_true", help="Reuse already-downloaded/extracted assets from --data-root.")
    parser.add_argument("--skip-validate", action="store_true", help="Skip manifest validation.")
    parser.add_argument("--skip-image-exists-check", action="store_true", help="Do not require manifest images to exist during validation.")
    parser.add_argument("--keep-going", action="store_true", help="Continue after a dataset build fails.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        return prepare(args)
    except KeyboardInterrupt:
        print(
            "\nInterrupted by user (Ctrl-C). Any partially built manifest in the output "
            "root may be incomplete; re-run to finish.",
            file=sys.stderr,
        )
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
