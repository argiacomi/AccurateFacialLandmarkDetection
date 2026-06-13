"""CLI orchestration: source context, build dispatch, argument parsing."""

# ruff: noqa: E402, F403, F405
from __future__ import annotations

import argparse
import contextlib

from lib.datasets.build.core import *  # noqa: F403
from lib.datasets.progress import (
    progress_group,
    set_progress_enabled,
    track as progress_track,
)
from lib.datasets.build.w300 import *  # noqa: F403
from lib.datasets.build.helen import *  # noqa: F403
from lib.datasets.build.lapa import *  # noqa: F403
from lib.datasets.build.jd_landmark import *  # noqa: F403
from lib.datasets.build.ffl import *  # noqa: F403
from lib.datasets.build.subject_session import *  # noqa: F403
from lib.datasets.build.multipie import *  # noqa: F403
from lib.datasets.build.cofw import *  # noqa: F403
from lib.datasets.build.wflw import *  # noqa: F403
from lib.datasets.build.merl_rav import *  # noqa: F403
from lib.datasets.build.video import *  # noqa: F403


@contextlib.contextmanager
def _source_context(
    source_dir: str | None, source_zip: str | None
) -> T.Iterator[Path | None]:
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


def _build_without_progress(args: argparse.Namespace) -> Path:
    dataset = _dataset(args.dataset)
    if dataset not in SUPPORTED_DATASETS:
        raise ValueError(f"unsupported dataset {args.dataset!r}")
    output_dir = Path(args.output_dir)
    scenarios = _parse_csv(args.scenarios)
    limit = None if not args.samples_per_scenario else int(args.samples_per_scenario)
    if args.cofw68_json and dataset != "cofw68":
        raise ValueError("--cofw68-json is only valid with --dataset cofw68")

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
        elif dataset == "cofw68":
            if args.cofw68_json:
                manifest_path = _build_cofw68_json_cropped(
                    Path(args.cofw68_json),
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
                    raise ValueError(
                        "--source-dir or --source-zip is required for cofw68"
                    )
                manifest_path = _build_cofw68(
                    root,
                    output_dir,
                    scenario=args.scenario,
                    scenarios=scenarios,
                    limit=limit,
                    mode=args.manifest_mode,
                    allow_overlap=args.allow_overlap,
                )
        elif dataset in {"multipie", "menpo2d"}:
            if root is None:
                raise ValueError(
                    f"--source-dir or --source-zip is required for {dataset}"
                )
            manifest_path = _build_multipie(
                root,
                output_dir,
                dataset=dataset,
                scenario=args.scenario,
                scenarios=scenarios,
                limit=limit,
                mode=args.manifest_mode,
                allow_overlap=args.allow_overlap,
            )
        elif dataset == "cofw29":
            if root is None:
                raise ValueError(
                    "--source-dir or --source-zip is required for cofw68 original"
                )
            manifest_path = _build_cofw68_original(
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
                raise ValueError(
                    "--source-dir or --source-zip is required for JD-landmark"
                )
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
        elif dataset in {"fll2", "fll3"}:
            if root is None:
                raise ValueError(
                    f"--source-dir or --source-zip is required for {dataset}"
                )
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
                raise ValueError(
                    f"--source-dir or --source-zip is required for {dataset}"
                )
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
        elif dataset == "merl-rav":
            if root is None:
                raise ValueError(
                    "--source-dir or --source-zip is required for MERL-RAV"
                )
            manifest_path = _build_merl_rav(
                root,
                output_dir,
                scenario=args.scenario,
                scenarios=scenarios,
                limit=limit,
                mode=args.manifest_mode,
                allow_overlap=args.allow_overlap,
                image_root=args.image_root,
            )
        elif dataset in {"300vw", "wflw-v"}:
            if root is None and not args.video_root:
                raise ValueError(
                    "--source-dir, --source-zip, or --video-root is required for video datasets"
                )
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
                max_workers=args.workers,
            )
        else:
            if root is None:
                raise ValueError(
                    "--source-dir, --source-zip, --wflw-annotations, or --cofw68-json is required"
                )
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
                workers=args.workers,
            )

    if args.write_overlays:
        _write_visual_audit(
            manifest_path,
            output_dir,
            limit=args.audit_overlay_limit,
            max_workers=args.workers,
        )
    return manifest_path


def build(args: argparse.Namespace) -> Path:
    """Build one dataset and join any parent-owned Rich progress display."""

    dataset = _dataset(args.dataset)
    with (
        progress_group(transient=False),
        progress_track(
            desc=f"Build {dataset} pipeline",
            total=1,
            unit="dataset",
            leave=True,
        ) as pipeline_progress,
    ):
        manifest_path = _build_without_progress(args)
        pipeline_progress.update(1)
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
    parser.add_argument(
        "--manifest-mode", choices=("replace", "merge"), default="replace"
    )
    parser.add_argument("--allow-overlap", action="store_true")
    parser.add_argument("--image-root", default=None)
    parser.add_argument("--video-root", default=None)
    parser.add_argument("--frame-output-dir", default=None)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--max-frames-per-video", type=int, default=None)
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Accepted for compatibility; scans are recursive.",
    )
    parser.add_argument("--wflw-annotations", default=None)
    parser.add_argument("--cofw68-json", default=None)
    parser.add_argument(
        "--write-overlays",
        action="store_true",
        help="Write visual landmark overlay audit images for built samples.",
    )
    parser.add_argument("--audit-overlay-limit", type=int, default=50)
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Parallel workers for video frame extraction and overlay rendering (<=0 uses all CPUs).",
    )
    parser.add_argument(
        "--progress",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Show Rich progress tracking for the dataset build pipeline.",
    )
    parser.add_argument(
        "--no-39pt-profile",
        action="store_true",
        help="Accepted for compatibility; non-68 samples are skipped.",
    )
    parser.add_argument(
        "--include-39pt-profile",
        action="store_true",
        help="Accepted for compatibility; non-68 samples are skipped.",
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="Accepted for compatibility; explicit sources are preferred.",
    )
    parser.add_argument(
        "--download-url",
        default=None,
        help="Accepted for compatibility; explicit sources are preferred.",
    )
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument(
        "--log-level",
        default="info",
        choices=(
            "quiet",
            "info",
            "verbose",
            "debug",
            "DEBUG",
            "INFO",
            "WARNING",
            "ERROR",
            "CRITICAL",
        ),
        help=(
            "Console verbosity. Lowercase values use the shared human-first logger; "
            "legacy stdlib names are accepted for compatibility."
        ),
    )
    parser.add_argument(
        "--log-format",
        default="human",
        choices=("human", "json"),
        help="Console output format: tagged human lines or JSONL events.",
    )
    return parser


def _manifest_completion_summary(manifest_path: Path) -> str:
    """Best-effort ``| samples N | schemas ...`` suffix for the completion line.

    Reads the just-written manifest to report sample and per-schema counts. Any
    failure returns an empty suffix so the completion line still prints.
    """

    try:
        payload = read_json(manifest_path)
    except Exception:  # noqa: BLE001
        return ""
    samples = payload.get("samples") if isinstance(payload, dict) else None
    if not isinstance(samples, list):
        return ""
    schemas: dict[str, int] = {}
    for sample in samples:
        if not isinstance(sample, dict):
            continue
        schema = str(
            sample.get("target_schema") or sample.get("source_schema") or "unknown"
        )
        schemas[schema] = schemas.get(schema, 0) + 1
    suffix = f" | samples {fmt_count(len(samples))}"
    if schemas:
        suffix += f" | schemas {fmt_mapping(schemas)}"
    return suffix


def _quality_log_level_name(value: str | None) -> str:
    """Map legacy stdlib log-level names to shared console verbosity names."""

    key = str(value or "info").lower()
    if key in {"warning", "error", "critical"}:
        return "quiet"
    if key in {"quiet", "info", "verbose", "debug"}:
        return key
    return "info"


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    configure_console_logging(
        verbosity_from_name(_quality_log_level_name(args.log_level)),
        args.log_format,
    )
    set_progress_enabled(bool(args.progress))
    try:
        manifest = build(args)
    except KeyboardInterrupt:
        log_error("manifest", "interrupted by user (Ctrl-C).")
        return 130
    except Exception as err:  # noqa: BLE001
        log_error("manifest", f"manifest build failed: {err}")
        return 1
    log_event(
        "manifest",
        f"wrote {manifest}{_manifest_completion_summary(manifest)}",
        level=Verbosity.INFO,
    )
    return 0


# Re-export every module-level name (including the single-underscore build
# helpers) so `from lib.datasets.build.<mod> import *` resolves bare-name
# calls in sibling modules exactly as they did in the original flat module.
__all__ = [_n for _n in dict(globals()) if not _n.startswith("__")]
