#!/usr/bin/env python3
"""Stage pre-resized 256x256 crops so training skips native-image decode.

Schema-aware manifests can reference native source images (notably MERL-RAV over
native AFLW JPEGs). The training loader decodes the full-resolution image and
resizes it to the 256x256 CD-ViT crop on every ``__getitem__`` -- i.e. every
sample, every epoch -- which dominates throughput on heterogeneous manifests.

This tool decodes each native image once, writes the 256x256 crop as a lossless
BGR PNG, and records ``prepared_image`` + ``prepared_image_orig_hw`` on the
manifest entry. The loader then loads the small PNG and rescales native-space
landmarks with the stored original dimensions instead of touching the native
image.

The crop is provably output-neutral. The native loader path is
``resize(swap(decode))`` where ``swap`` is the BGR->RGB channel reorder and
``resize`` is INTER_LINEAR. ``cv2.resize`` is a per-channel spatial op, so it
commutes with the channel reorder: ``swap(resize(decode))`` equals
``resize(swap(decode))``. We therefore generate the crop as
``resize(decode)`` in BGR, store it losslessly, and the loader reproduces the
native pixels exactly. To guarantee this rather than assume it, the tool
reloads every crop and asserts the full loader output (image AND scaled
landmarks) is bit-identical to the native path; any sample that fails is left
native and reported, so a mismatch can never silently change training data.

Example::

    python tools/stage_prepared_crops.py \
      --manifest data/prepared/manifest.json \
      --out-manifest data/prepared/manifest.staged.json
"""

# ruff: noqa: E402
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import typing as T
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lib.datasets.loader_geometry import (
    landmark_mask_from_entry,
    resolve_loader_source_hw,
    simulate_loader_geometry,
    write_geometry_overlay,
)
from lib.datasets.parallel import parallel_map
from lib.logging_utils import Verbosity, log_event


class _StageJob(T.NamedTuple):
    """One unique native image to stage (the parallelism unit)."""

    image_path: str
    landmarks_path: str
    dataset: str
    image_id: str


class _StageResult(T.NamedTuple):
    """A worker's outcome for one native image; the parent applies it."""

    image_path: str
    rel: str | None
    orig_hw: tuple[int, int] | None
    status: str  # "staged" | "skipped_already_256" | "skipped_no_image" | "mismatch"


def _resolve(base_dir: Path, value: str) -> str:
    raw = str(value or "")
    if not raw:
        return ""
    path = Path(raw)
    if path.is_absolute():
        return str(path)
    return str((base_dir / path).resolve())


def _sanitize(text: str) -> str:
    keep = [c if (c.isalnum() or c in "-_.") else "_" for c in str(text)]
    return "".join(keep).strip("_") or "image"


def _native_image_and_landmarks(image_path: str, landmarks_path: str):
    """Reproduce the loader's native decode path for one sample.

    Mirrors ``LandmarkDataset._load_image_and_landmarks`` exactly so the staged
    crop can be validated against the same pixels and landmarks the trainer
    would otherwise compute.
    """

    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(image_path)
    img = img[:, :, [2, 1, 0]]
    lmk = np.load(landmarks_path).astype(np.float32)[:, :2]
    if float(np.nanmax(lmk)) <= 1.5:
        lmk = lmk * 255.0
    h, w = img.shape[:2]
    if h != 256 or w != 256:
        scale_x = 256.0 / float(w)
        scale_y = 256.0 / float(h)
        img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_LINEAR)
        lmk[:, 0] *= scale_x
        lmk[:, 1] *= scale_y
    return img, lmk, (h, w)


def _prepared_image_and_landmarks(crop_path: str, landmarks_path: str, orig_hw):
    """Reproduce the loader's prepared fast path for one sample."""

    img = cv2.imread(crop_path, cv2.IMREAD_COLOR)
    if img is None or img.shape[0] != 256 or img.shape[1] != 256:
        return None
    img = img[:, :, [2, 1, 0]]
    lmk = np.load(landmarks_path).astype(np.float32)[:, :2]
    if float(np.nanmax(lmk)) <= 1.5:
        lmk = lmk * 255.0
    orig_h, orig_w = int(orig_hw[0]), int(orig_hw[1])
    if orig_h != 256 or orig_w != 256:
        lmk[:, 0] *= 256.0 / float(orig_w)
        lmk[:, 1] *= 256.0 / float(orig_h)
    return img, lmk


def _sample_loader_geometry(
    entry: dict,
    *,
    base_dir: Path,
) -> dict[str, T.Any]:
    landmarks_value = entry.get("landmarks") or entry.get("ground_truth")
    if not landmarks_value:
        return {"ok": False, "reason": "missing_landmarks"}

    landmarks_path = _resolve(base_dir, landmarks_value)
    try:
        points = np.load(landmarks_path).astype(np.float32)[:, :2]
    except Exception as err:  # noqa: BLE001
        return {"ok": False, "reason": f"invalid_landmarks:{err}"}

    hw, source, error = resolve_loader_source_hw(entry, base_dir=base_dir)
    if error or hw is None:
        return {
            "ok": False,
            "reason": error or "missing_loader_geometry_source",
            "geometry_source": source,
        }

    metadata = entry.get("metadata") if isinstance(entry.get("metadata"), dict) else {}
    # Loader parity: use the same mask MakeLMKInsideImage receives so masked-out
    # sentinel coordinates (e.g. MERL-RAV zeroed self-occluded points) are not
    # reported as out-of-frame landmarks.
    diag = simulate_loader_geometry(
        points,
        hw,
        landmark_mask=landmark_mask_from_entry(entry, metadata, int(points.shape[0])),
    )
    diag["geometry_source"] = source
    return diag


def stage_crops(
    manifest_path: str | Path,
    *,
    out_manifest: str | Path | None = None,
    images_subdir: str = "images",
    datasets: T.Iterable[str] | None = None,
    force: bool = False,
    strict: bool = False,
    keep_mismatched_crops: bool = False,
    workers: int | None = 1,
    validate_geometry: bool = False,
    geometry_strict: bool = False,
    drop_invalid_geometry: bool = False,
    drop_suspicious_geometry: bool = False,
    geometry_overlay_dir: str | Path | None = None,
    max_geometry_overlays: int = 200,
) -> dict:
    """Write 256x256 crops and record prepared references on a manifest.

    Returns a stats dict. When ``out_manifest`` equals ``manifest_path`` the
    manifest is augmented in place. Mismatched crops (those that fail to
    reproduce the native pixels/landmarks) are skipped and left on the native
    path unless ``strict`` is set, in which case a ``ValueError`` is raised.
    """

    manifest_path = Path(manifest_path).resolve()
    base_dir = manifest_path.parent
    out_manifest = (
        Path(out_manifest).resolve()
        if out_manifest
        else manifest_path.with_name(f"{manifest_path.stem}.staged.json")
    )
    out_base = out_manifest.parent
    images_root = out_base / images_subdir

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries = payload.get("samples", payload.get("scenarios", []))
    if not isinstance(entries, list):
        raise ValueError(f"manifest {manifest_path} has no samples list")

    _stage_started_at = time.time()
    _stage_total = len(entries)
    log_event(
        "prepare",
        f"stage crops start | samples {_stage_total} | manifest {manifest_path}",
        level=Verbosity.INFO,
        samples=_stage_total,
        manifest=str(manifest_path),
        out_manifest=str(out_manifest),
        images_root=str(images_root),
    )

    dataset_filter = (
        {str(d).strip().lower() for d in datasets if str(d).strip()}
        if datasets
        else None
    )

    staged = skipped_already_256 = skipped_no_image = reused = 0
    mismatches: list[str] = []
    geometry_issues: list[dict[str, T.Any]] = []
    suspicious_geometry: list[dict[str, T.Any]] = []
    invalid_geometry_entries: set[int] = set()
    geometry_dropped = 0
    geometry_overlays_written = 0
    # Flagged samples always get a review overlay; default the directory next
    # to the output manifest unless the caller chooses another location.
    overlay_dir = (
        Path(geometry_overlay_dir)
        if geometry_overlay_dir
        else out_base / "geometry_review"
    )

    def _write_issue_overlay(entry: dict, issue: dict[str, T.Any]) -> None:
        nonlocal geometry_overlays_written
        if geometry_overlays_written >= max_geometry_overlays:
            return
        landmarks_value = entry.get("landmarks") or entry.get("ground_truth")
        diag = issue.get("diagnostics") or {}
        source_hw = diag.get("source_image_hw")
        if not landmarks_value or not source_hw:
            return
        try:
            points = np.load(_resolve(base_dir, landmarks_value)).astype(np.float32)
        except Exception:  # noqa: BLE001
            return
        image_value = entry.get("image")
        metadata = (
            entry.get("metadata") if isinstance(entry.get("metadata"), dict) else {}
        )
        safe_name = (
            str(issue["sample_id"]).replace("/", "_").replace("#", "_") or "sample"
        )
        written = write_geometry_overlay(
            overlay_dir / (issue["dataset"] or "dataset") / f"{safe_name}.png",
            _resolve(base_dir, image_value) if image_value else None,
            points[:, :2],
            (int(source_hw[0]), int(source_hw[1])),
            landmark_mask=landmark_mask_from_entry(
                entry, metadata, int(points.shape[0])
            ),
            diag=diag,
        )
        if written is not None:
            issue["overlay"] = str(written)
            geometry_overlays_written += 1

    if validate_geometry:
        for index, entry in enumerate(entries):
            if not isinstance(entry, dict):
                continue
            dataset = str(entry.get("dataset") or "").strip()
            if dataset_filter is not None and dataset.lower() not in dataset_filter:
                continue
            diag = _sample_loader_geometry(entry, base_dir=base_dir)
            if diag.get("ok") and not diag.get("suspicious"):
                continue
            issue = {
                "index": index,
                "sample_id": entry.get("sample_id") or entry.get("id") or index,
                "dataset": dataset,
                "reason": diag.get("reason") or "invalid_geometry",
                "diagnostics": diag,
            }
            if diag.get("ok"):
                # Trainable, but suspicious loader padding usually means the
                # landmarks are in a wrong coordinate frame. Keep the review
                # overlay, but do not let train-safe manifests retain it.
                suspicious_geometry.append(issue)
                _write_issue_overlay(entry, issue)
                if drop_invalid_geometry or drop_suspicious_geometry:
                    invalid_geometry_entries.add(id(entry))
                continue
            geometry_issues.append(issue)
            invalid_geometry_entries.add(id(entry))
            _write_issue_overlay(entry, issue)
        if suspicious_geometry:
            log_event(
                "prepare",
                (
                    f"stage crops geometry: {len(suspicious_geometry)} suspicious "
                    f"sample(s) {'dropped' if (drop_invalid_geometry or drop_suspicious_geometry) else 'quarantined for review'}; "
                    f"overlays in {overlay_dir}"
                ),
                level=Verbosity.INFO,
                suspicious=len(suspicious_geometry),
                dropped=bool(drop_invalid_geometry or drop_suspicious_geometry),
                overlay_dir=str(overlay_dir),
            )
            if geometry_strict or not (
                drop_invalid_geometry or drop_suspicious_geometry
            ):
                first = suspicious_geometry[0]
                raise ValueError(
                    "stage crop geometry validation failed: "
                    f"{len(suspicious_geometry)} suspicious sample(s); first={first}. "
                    "Use --drop-suspicious-geometry or --drop-invalid-geometry "
                    "to write a train-safe manifest with suspicious samples removed."
                )
        if geometry_issues:
            first = geometry_issues[0]
            if geometry_strict or not drop_invalid_geometry:
                raise ValueError(
                    "stage crop geometry validation failed: "
                    f"{len(geometry_issues)} invalid sample(s); first={first}. "
                    "Use --drop-invalid-geometry to write a train-safe manifest "
                    "with invalid samples removed."
                )
        if invalid_geometry_entries:
            before_drop = len(entries)
            entries[:] = [
                entry for entry in entries if id(entry) not in invalid_geometry_entries
            ]
            geometry_dropped = before_drop - len(entries)

    # Group samples by their resolved native image path. Several samples can
    # share one native image (multiple faces, or MERL-RAV over a single AFLW
    # frame); the loader rescales each sample's own landmarks from the shared
    # crop, so exactly one crop is staged per unique native image and then
    # applied to every sample in the group. The group is the unit of
    # parallelism: it gives one writer per crop path and removes the duplicate
    # decode/resize/validate the per-sample ``crop_for_native`` cache used to
    # avoid serially.
    groups: dict[str, list[dict]] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        if id(entry) in invalid_geometry_entries:
            continue
        dataset = str(entry.get("dataset") or "").strip()
        if dataset_filter is not None and dataset.lower() not in dataset_filter:
            continue
        image_value = entry.get("image")
        landmarks_value = entry.get("landmarks") or entry.get("ground_truth")
        if not image_value or not landmarks_value:
            skipped_no_image += 1
            continue
        image_path = _resolve(base_dir, image_value)
        groups.setdefault(image_path, []).append(entry)

    jobs: list[_StageJob] = []
    for image_path, group in groups.items():
        job_entry = next(
            (entry for entry in group if id(entry) not in invalid_geometry_entries),
            None,
        )
        if job_entry is None:
            continue
        dataset = str(job_entry.get("dataset") or "").strip()
        landmarks_value = job_entry.get("landmarks") or job_entry.get("ground_truth")
        if not landmarks_value:
            skipped_no_image += len(group)
            continue
        landmarks_path = _resolve(base_dir, landmarks_value)
        jobs.append(
            _StageJob(
                image_path=image_path,
                landmarks_path=landmarks_path,
                dataset=dataset,
                image_id=str(job_entry.get("image_id") or Path(image_path).stem),
            )
        )

    def _stage_one(job: _StageJob) -> _StageResult:
        """Stage one unique native image (thread-safe; mutates no shared state).

        Returns the prepared-crop reference and status; the parent applies it to
        every sample in the image's group. The crop filename is derived from the
        image path's digest, so each unique image owns a distinct output file --
        guaranteeing one writer per crop path under the thread pool.
        """

        try:
            native_img, native_lmk, (orig_h, orig_w) = _native_image_and_landmarks(
                job.image_path, job.landmarks_path
            )
        except FileNotFoundError:
            return _StageResult(job.image_path, None, None, "skipped_no_image")

        if orig_h == 256 and orig_w == 256:
            # Native path performs no resize; a crop would add cost without
            # benefit and a 256->256 resize is not guaranteed to be identity.
            return _StageResult(job.image_path, None, None, "skipped_already_256")

        crop_bgr = cv2.resize(
            cv2.imread(job.image_path, cv2.IMREAD_COLOR),
            (256, 256),
            interpolation=cv2.INTER_LINEAR,
        )

        digest = hashlib.sha1(job.image_path.encode("utf-8")).hexdigest()[:8]
        rel = str(
            Path(images_subdir)
            / _sanitize(job.dataset or "dataset")
            / f"{_sanitize(job.image_id)}_{digest}.png"
        )
        crop_path = out_base / rel
        crop_path.parent.mkdir(parents=True, exist_ok=True)
        if force or not crop_path.exists():
            if not cv2.imwrite(str(crop_path), crop_bgr):
                raise RuntimeError(f"failed to write crop {crop_path}")

        prepared = _prepared_image_and_landmarks(
            str(crop_path), job.landmarks_path, (orig_h, orig_w)
        )
        identical = (
            prepared is not None
            and np.array_equal(prepared[0], native_img)
            and np.array_equal(prepared[1], native_lmk)
        )
        if not identical:
            # Leave the samples on the native path; remove the unusable crop.
            if crop_path.exists() and not keep_mismatched_crops:
                crop_path.unlink()
            if strict:
                raise ValueError(
                    f"crop for {job.image_path} did not reproduce native pixels/landmarks"
                )
            return _StageResult(job.image_path, None, None, "mismatch")

        return _StageResult(job.image_path, rel, (orig_h, orig_w), "staged")

    # One crop per unique native image, in parallel; results come back in input
    # order. cv2 decode/resize/encode release the GIL, so threads scale this
    # IO+codec work. workers=1 (default) runs sequentially with identical output.
    results = parallel_map(
        _stage_one,
        jobs,
        workers=workers,
        desc="Stage crops",
        unit="image",
        # One bar over unique native images (the parallel work unit); forced
        # visible and persistent to match the pre-parallel serial loop.
        leave=True,
        disable=False,
    )

    # Parent applies each result to every sample in the image's group and writes
    # the manifest once. Per-sample counters (skips, reuse, mismatches) match the
    # serial code; only the now-deduped work differs.
    for result in results:
        group = groups[result.image_path]
        if result.status == "staged":
            orig_hw = [int(result.orig_hw[0]), int(result.orig_hw[1])]
            valid_group = [
                entry for entry in group if id(entry) not in invalid_geometry_entries
            ]
            for entry in valid_group:
                entry["prepared_image"] = result.rel
                entry["prepared_image_orig_hw"] = orig_hw
            if valid_group:
                staged += 1
                reused += len(valid_group) - 1
        elif result.status == "skipped_already_256":
            skipped_already_256 += len(group)
        elif result.status == "skipped_no_image":
            skipped_no_image += len(group)
        else:  # "mismatch"; strict mode already raised inside the worker
            mismatches.extend([result.image_path] * len(group))

    out_manifest.parent.mkdir(parents=True, exist_ok=True)
    out_manifest.write_text(
        json.dumps(payload, indent=2, sort_keys=False), encoding="utf-8"
    )

    _stage_elapsed = time.time() - _stage_started_at
    log_event(
        "prepare",
        (
            f"stage crops done | crops {staged} unique | reused {reused} | "
            f"skipped 256x256 {skipped_already_256} | "
            f"skipped missing {skipped_no_image} | mismatches {len(mismatches)} | "
            f"geometry invalid {len(geometry_issues)} | "
            f"geometry suspicious {len(suspicious_geometry)} | "
            f"geometry dropped {geometry_dropped} | {_stage_elapsed:.1f}s"
        ),
        level=Verbosity.INFO,
        staged=staged,
        reused=reused,
        skipped_already_256=skipped_already_256,
        skipped_no_image=skipped_no_image,
        mismatches=len(mismatches),
        geometry_invalid=len(geometry_issues),
        geometry_suspicious=len(suspicious_geometry),
        geometry_dropped=geometry_dropped,
        duration_seconds=_stage_elapsed,
        manifest=str(manifest_path),
        out_manifest=str(out_manifest),
        images_root=str(images_root),
    )

    return {
        "manifest": str(manifest_path),
        "out_manifest": str(out_manifest),
        "images_root": str(images_root),
        "staged": staged,
        "reused": reused,
        "skipped_already_256": skipped_already_256,
        "skipped_no_image": skipped_no_image,
        "mismatches": mismatches,
        "geometry_issues": geometry_issues,
        "suspicious_geometry": suspicious_geometry,
        "geometry_dropped": geometry_dropped,
        "geometry_overlay_dir": str(overlay_dir),
        "geometry_overlays_written": geometry_overlays_written,
    }


def stage_manifest(args: argparse.Namespace) -> int:
    stats = stage_crops(
        args.manifest,
        out_manifest=args.out_manifest or None,
        images_subdir=args.images_subdir,
        datasets=args.datasets.split(",") if args.datasets else None,
        force=args.force,
        strict=args.strict,
        keep_mismatched_crops=args.keep_mismatched_crops,
        workers=args.workers,
        validate_geometry=args.validate_geometry,
        geometry_strict=args.geometry_strict,
        drop_invalid_geometry=getattr(args, "drop_invalid_geometry", False),
        drop_suspicious_geometry=getattr(args, "drop_suspicious_geometry", False),
        geometry_overlay_dir=args.geometry_overlay_dir or None,
        max_geometry_overlays=args.max_geometry_overlays,
    )

    staged, reused = stats["staged"], stats["reused"]
    mismatches = stats["mismatches"]
    geometry_issues = stats.get("geometry_issues", [])
    print(f"manifest        : {stats['manifest']}")
    print(f"out manifest    : {stats['out_manifest']}")
    print(f"crops dir       : {stats['images_root']}")
    print(f"staged crops    : {staged} (unique native images)")
    print(f"reused crops    : {reused} (samples sharing a native image)")
    print(
        f"skipped 256x256 : {stats['skipped_already_256']} (native path already cheap)"
    )
    print(f"skipped no image: {stats['skipped_no_image']}")
    print(f"bit-identity OK : {staged + reused}/{staged + reused + len(mismatches)}")
    if mismatches:
        print(f"mismatched (left native): {len(mismatches)}")
        for path in mismatches[:10]:
            print(f"  - {path}")
    if geometry_issues:
        dropped = stats.get("geometry_dropped", 0)
        if dropped:
            print(f"invalid geometry (dropped): {len(geometry_issues)}")
        else:
            print(f"invalid geometry: {len(geometry_issues)}")
        for issue in geometry_issues[:10]:
            print(f"  - {issue['sample_id']}: {issue['reason']}")
    suspicious = stats.get("suspicious_geometry", [])
    if suspicious:
        dropped = stats.get("geometry_dropped", 0)
        status = "dropped" if dropped else "review overlays"
        print(f"suspicious geometry ({status}): {len(suspicious)}")
        for issue in suspicious[:10]:
            pad = (issue.get("diagnostics") or {}).get("padding")
            print(f"  - {issue['sample_id']}: padding={pad}")
    if stats.get("geometry_overlays_written"):
        print(
            f"review overlays : {stats['geometry_overlays_written']} -> "
            f"{stats['geometry_overlay_dir']}"
        )
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--manifest", required=True, help="Input manifest JSON.")
    parser.add_argument(
        "--out-manifest",
        default="",
        help="Output manifest path. Default: <input-stem>.staged.json beside input.",
    )
    parser.add_argument(
        "--images-subdir",
        default="images",
        help="Crop directory relative to the output manifest (default: images).",
    )
    parser.add_argument(
        "--datasets",
        default="",
        help="Comma-separated dataset filter (default: all datasets).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rewrite crop PNGs even when they already exist.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail on the first crop that does not reproduce the native pixels.",
    )
    parser.add_argument(
        "--keep-mismatched-crops",
        action="store_true",
        help="Keep (rather than delete) crop files that failed bit-identity.",
    )
    parser.add_argument(
        "--validate-geometry",
        action="store_true",
        help="Validate every sample's loader geometry before applying staged crops.",
    )
    parser.add_argument(
        "--geometry-strict",
        action="store_true",
        help=(
            "Force failure if --validate-geometry finds invalid samples, "
            "even when --drop-invalid-geometry is also set."
        ),
    )
    parser.add_argument(
        "--drop-invalid-geometry",
        action="store_true",
        help=(
            "When --validate-geometry finds invalid samples, remove them "
            "from the output manifest instead of failing. Also removes "
            "suspicious geometry so the output manifest is train-safe."
        ),
    )
    parser.add_argument(
        "--drop-suspicious-geometry",
        action="store_true",
        help=(
            "When --validate-geometry finds trainable-but-suspicious samples, "
            "remove them from the output manifest instead of failing."
        ),
    )
    parser.add_argument(
        "--geometry-overlay-dir",
        default="",
        help=(
            "Directory for review overlay PNGs of geometry-flagged samples. "
            "Default: <out-manifest dir>/geometry_review."
        ),
    )
    parser.add_argument(
        "--max-geometry-overlays",
        type=int,
        default=200,
        help="Cap on review overlay PNGs written per run (default: 200).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help=(
            "Parallel workers for staging crops (one worker per unique native "
            "image). 1 (default) stages serially; <=0 uses all CPUs."
        ),
    )
    return parser


def main() -> int:
    return stage_manifest(build_arg_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
