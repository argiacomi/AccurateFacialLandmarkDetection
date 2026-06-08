#!/usr/bin/env python3
"""Merge dataset manifests into a ratio-based profile/occlusion hard-negative mix."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import re
import sys
import typing as T
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lib.landmarks.datasets.hard_negative_mining import (
    BUCKET_PRIORITY,
    BUCKET_WEIGHT,
    HardNegativeClass,
    annotate_sample,
    classify_hard_negative,
    source_key,
)
from lib.landmarks.manifest.contract import (
    TRAINING_MANIFEST_CONTRACT,
    TRAINING_MANIFEST_VERSION,
    manifest_summary,
)

logger = logging.getLogger(__name__)

DATASET_DEFAULT_BUCKET: dict[str, str] = {
    "cofw68": "occlusion",
    "cofw6868": "occlusion",
    "300w": "anchor",
    "w300": "anchor",
    "production_validated": "anchor",
    "multipie": "profile",
}

BUCKET_ORDER: tuple[str, ...] = ("profile_occlusion", "profile", "occlusion", "anchor")
DEFAULT_BUCKET_RATIOS: dict[str, float] = {
    "profile_occlusion": 3.0,
    "profile": 2.0,
    "occlusion": 2.0,
    "anchor": 1.0,
}
DEFAULT_TOTAL_SAMPLES = 0

IMAGE_ID_RE = re.compile(r"(image\d+)", re.IGNORECASE)


def _image_ids_from_values(*values: T.Any) -> set[str]:
    out: set[str] = set()
    for value in values:
        if value is None:
            continue
        for match in IMAGE_ID_RE.findall(str(value)):
            out.add(match.lower())
    return out


def _sample_image_ids(sample: T.Mapping[str, T.Any]) -> set[str]:
    source = sample.get("source") if isinstance(sample.get("source"), dict) else {}
    metadata = (
        sample.get("metadata") if isinstance(sample.get("metadata"), dict) else {}
    )

    return _image_ids_from_values(
        sample.get("sample_id"),
        sample.get("id"),
        sample.get("name"),
        sample.get("image"),
        sample.get("image_path"),
        sample.get("landmarks"),
        sample.get("ground_truth"),
        source.get("source_id"),
        source.get("image_id"),
        source.get("sample_id"),
        metadata.get("image_id"),
        metadata.get("merl_image_id"),
        metadata.get("source_landmarks"),
        metadata.get("annotation_file"),
        metadata.get("original_image"),
    )


def _load_excluded_image_ids(path: Path | None) -> set[str]:
    if path is None:
        return set()
    if not path.is_file():
        raise FileNotFoundError(f"exclude image-id file not found: {path}")
    out: set[str] = set()
    for raw in path.read_text(encoding="utf-8").splitlines():
        raw = raw.strip()
        if not raw or raw.startswith("#"):
            continue
        out.update(_image_ids_from_values(raw))
    return out


MANIFEST_ARGS: tuple[tuple[str, str], ...] = (
    ("wflw_manifest", "wflw"),
    ("aflw2000_manifest", "aflw2000-3d"),
    ("merl_rav_manifest", "merl-rav"),
    ("cofw68_manifest", "cofw68"),
    ("menpo2d_manifest", "menpo2d"),
    ("multipie_manifest", "multipie"),
    ("w300_manifest", "300w"),
    ("production_validated_manifest", "production_validated"),
)


def _default_class(dataset: str) -> HardNegativeClass | None:
    bucket = DATASET_DEFAULT_BUCKET.get(dataset.strip().lower())
    if bucket is None:
        return None
    return HardNegativeClass(
        bucket=bucket,
        priority=BUCKET_PRIORITY[bucket],
        weight=BUCKET_WEIGHT[bucket],
        reasons=(f"{dataset}_default",),
    )


def _resolve_manifest_relative_path(manifest_path: Path, value: T.Any) -> str:
    path = Path(str(value))
    if path.is_absolute():
        return str(path.resolve())
    return str((manifest_path.parent / path).resolve())


def _load_samples(manifest_path: Path) -> list[dict[str, T.Any]]:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    samples = payload.get("samples", payload.get("scenarios", []))
    if not isinstance(samples, list):
        raise ValueError(f"manifest {manifest_path} has no 'samples' list")

    resolved: list[dict[str, T.Any]] = []
    for entry in samples:
        if not isinstance(entry, T.Mapping):
            continue
        sample = dict(entry)
        for key in ("image", "landmarks", "ground_truth"):
            value = sample.get(key)
            if value:
                sample[key] = _resolve_manifest_relative_path(manifest_path, value)
        resolved.append(sample)
    return resolved


def _stable_order(samples: T.Sequence[dict[str, T.Any]], *, seed: int) -> list[dict[str, T.Any]]:
    def _hash(sample: dict[str, T.Any]) -> str:
        dataset, source_id = source_key(sample)
        return hashlib.sha256(f"{seed}|{dataset}|{source_id}".encode()).hexdigest()

    return sorted(samples, key=_hash)


def _empty_bucket_map(value: float = 0.0) -> dict[str, float]:
    return {bucket: float(value) for bucket in BUCKET_ORDER}


def _parse_bucket_values(value: str | None, *, normalize_percentages: bool = False) -> dict[str, float]:
    if not value:
        return dict(DEFAULT_BUCKET_RATIOS)

    parsed = _empty_bucket_map()
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"bucket spec must use bucket=value format, got {item!r}")
        bucket, raw_amount = item.split("=", 1)
        bucket = bucket.strip().lower().replace("-", "_")
        if bucket not in BUCKET_ORDER:
            raise ValueError(f"unknown bucket {bucket!r}; choose one of {', '.join(BUCKET_ORDER)}")
        amount = float(raw_amount)
        if amount < 0:
            raise ValueError(f"bucket amount must be non-negative for {bucket}: {amount}")
        parsed[bucket] = amount

    if normalize_percentages:
        total = sum(parsed.values())
        if total <= 0:
            raise ValueError("bucket percentages must sum to a positive value")
        if total > 1.5:
            parsed = {bucket: amount / 100.0 for bucket, amount in parsed.items()}
        percent_total = sum(parsed.values())
        if percent_total <= 0:
            raise ValueError("bucket percentages must sum to a positive value")
        parsed = {bucket: amount / percent_total for bucket, amount in parsed.items()}

    if sum(parsed.values()) <= 0:
        raise ValueError("at least one bucket ratio/percentage must be positive")
    return parsed


def _ratio_targets(*, bucket_ratios: str | None, bucket_percentages: str | None) -> dict[str, float]:
    if bucket_ratios and bucket_percentages:
        raise ValueError("pass only one of --bucket-ratios or --bucket-percentages")
    if bucket_percentages:
        return _parse_bucket_values(bucket_percentages, normalize_percentages=True)
    return _parse_bucket_values(bucket_ratios, normalize_percentages=False)


def _normalize_ratios(ratios: T.Mapping[str, float]) -> dict[str, float]:
    total = sum(max(float(ratios.get(bucket, 0.0)), 0.0) for bucket in BUCKET_ORDER)
    if total <= 0:
        raise ValueError("bucket ratios must sum to a positive value")
    return {bucket: max(float(ratios.get(bucket, 0.0)), 0.0) / total for bucket in BUCKET_ORDER}


def _quota_ceilings(
    *,
    max_profile_occlusion: int | None,
    max_profile: int | None,
    max_occlusion: int | None,
    max_anchors: int | None,
) -> dict[str, int | None]:
    return {
        "profile_occlusion": max_profile_occlusion,
        "profile": max_profile,
        "occlusion": max_occlusion,
        "anchor": max_anchors,
    }


def _integerize_targets(
    float_targets: T.Mapping[str, float], capacities: T.Mapping[str, int], target_total: int
) -> dict[str, int]:
    counts = {bucket: min(int(math.floor(float_targets.get(bucket, 0.0))), capacities[bucket]) for bucket in BUCKET_ORDER}
    remaining = max(target_total - sum(counts.values()), 0)
    while remaining > 0:
        candidates = [bucket for bucket in BUCKET_ORDER if counts[bucket] < capacities[bucket]]
        if not candidates:
            break
        candidates.sort(
            key=lambda bucket: (float_targets.get(bucket, 0.0) - math.floor(float_targets.get(bucket, 0.0))),
            reverse=True,
        )
        progressed = False
        for bucket in candidates:
            if remaining <= 0:
                break
            if counts[bucket] < capacities[bucket]:
                counts[bucket] += 1
                remaining -= 1
                progressed = True
        if not progressed:
            break
    return counts


def _allocate_ratio_counts(
    *,
    available: T.Mapping[str, int],
    ratios: T.Mapping[str, float],
    total_samples: int | None,
    ceilings: T.Mapping[str, int | None],
) -> tuple[dict[str, int], dict[str, int], int]:
    capacities: dict[str, int] = {}
    for bucket in BUCKET_ORDER:
        capacity = max(int(available.get(bucket, 0)), 0)
        ceiling = ceilings.get(bucket)
        if ceiling is not None:
            capacity = min(capacity, max(int(ceiling), 0))
        capacities[bucket] = capacity

    target_pool = [bucket for bucket in BUCKET_ORDER if ratios.get(bucket, 0.0) > 0 and capacities[bucket] > 0]
    feasible_total = sum(capacities[bucket] for bucket in target_pool)
    if feasible_total <= 0:
        return {bucket: 0 for bucket in BUCKET_ORDER}, capacities, 0

    requested_total = feasible_total if total_samples is None or total_samples <= 0 else int(total_samples)
    target_total = min(max(requested_total, 0), feasible_total)

    active = set(target_pool)
    remaining_total = float(target_total)
    float_targets = _empty_bucket_map()
    while active and remaining_total > 0:
        ratio_sum = sum(float(ratios[bucket]) for bucket in active)
        if ratio_sum <= 0:
            break
        proposed = {bucket: remaining_total * float(ratios[bucket]) / ratio_sum for bucket in active}
        saturated = [bucket for bucket, amount in proposed.items() if amount >= capacities[bucket]]
        if not saturated:
            float_targets.update(proposed)
            break
        for bucket in saturated:
            float_targets[bucket] = float(capacities[bucket])
            remaining_total -= float(capacities[bucket])
            active.remove(bucket)

    counts = _integerize_targets(float_targets, capacities, target_total)
    return counts, capacities, target_total


def build_hard_negative_manifest(
    *,
    manifests: T.Mapping[str, Path],
    output_dir: Path,
    total_samples: int | None = DEFAULT_TOTAL_SAMPLES,
    bucket_ratios: str | None = None,
    bucket_percentages: str | None = None,
    max_profile_occlusion: int | None = None,
    max_profile: int | None = None,
    max_occlusion: int | None = None,
    max_anchors: int | None = None,
    allow_overlap: bool = False,
    seed: int = 1337,
    write_audit: bool = False,
    exclude_image_ids_file: Path | None = None,
) -> dict[str, T.Any]:
    ratios = _ratio_targets(bucket_ratios=bucket_ratios, bucket_percentages=bucket_percentages)
    target_percentages = _normalize_ratios(ratios)
    excluded_image_ids = _load_excluded_image_ids(exclude_image_ids_file)
    ceilings = _quota_ceilings(
        max_profile_occlusion=max_profile_occlusion,
        max_profile=max_profile,
        max_occlusion=max_occlusion,
        max_anchors=max_anchors,
    )

    classified: dict[str, list[dict[str, T.Any]]] = {bucket: [] for bucket in BUCKET_ORDER}
    audit: dict[str, dict[str, int]] = {}
    seen_keys: set[tuple[str, str]] = set()

    for dataset, manifest_path in manifests.items():
        dataset_label = dataset.strip().lower()
        dataset_counts: dict[str, int] = dict.fromkeys(
            (
                *BUCKET_ORDER,
                "classified_by_label",
                "dataset_default",
                "skipped",
                "duplicate",
                "excluded_image_id",
            ),
            0,
        )
        for sample in _load_samples(manifest_path):
            sample.setdefault("dataset", dataset_label)

            sample_image_ids = _sample_image_ids(sample)
            if (
                dataset_label == "merl-rav"
                and excluded_image_ids
                and (sample_image_ids & excluded_image_ids)
            ):
                dataset_counts["excluded_image_id"] += 1
                continue

            if sample_image_ids:
                metadata = (
                    dict(sample.get("metadata", {}))
                    if isinstance(sample.get("metadata"), dict)
                    else {}
                )
                metadata.setdefault("source_image_ids", sorted(sample_image_ids))
                if dataset_label == "merl-rav":
                    metadata.setdefault("merl_image_id", sorted(sample_image_ids)[0])
                sample["metadata"] = metadata

            classification = classify_hard_negative(sample)
            classification_source = "classified_by_label"
            if classification is None:
                classification = _default_class(dataset_label)
                classification_source = "dataset_default" if classification is not None else "skipped"
            if classification is None:
                dataset_counts["skipped"] += 1
                continue
            key = source_key(sample)
            if not allow_overlap and key in seen_keys:
                dataset_counts["duplicate"] += 1
                continue
            seen_keys.add(key)
            annotated = annotate_sample(sample, classification)
            annotated.setdefault("dataset", dataset_label)
            classified[classification.bucket].append(annotated)
            dataset_counts[classification.bucket] += 1
            dataset_counts[classification_source] += 1
        audit[dataset_label] = dataset_counts

    available_counts = {bucket: len(classified[bucket]) for bucket in BUCKET_ORDER}
    target_counts, effective_capacities, target_total = _allocate_ratio_counts(
        available=available_counts,
        ratios=ratios,
        total_samples=total_samples,
        ceilings=ceilings,
    )

    selected: list[dict[str, T.Any]] = []
    counts: dict[str, int] = {}
    by_dataset: dict[str, Counter[str]] = defaultdict(Counter)
    for bucket in BUCKET_ORDER:
        ordered = _stable_order(classified[bucket], seed=seed)
        ordered = ordered[: target_counts[bucket]]
        counts[bucket] = len(ordered)
        for sample in ordered:
            dataset_label = str(sample.get("dataset", "")).strip().lower() or "unknown"
            by_dataset[dataset_label][bucket] += 1

            sample_image_ids = _sample_image_ids(sample)
            if dataset_label == "merl-rav" and sample_image_ids:
                split_identity = sorted(sample_image_ids)[0]
            else:
                split_identity = (
                    sample.get("sample_id")
                    or sample.get("image")
                    or sample.get("landmarks")
                )
            split_key = f"{dataset_label}|{split_identity}"
            split_hash = int(hashlib.sha256(split_key.encode()).hexdigest()[:8], 16)
            split = "test" if (split_hash % 100) < 5 else "train"
            sample["split"] = split
            metadata = sample.setdefault("metadata", {})
            if isinstance(metadata, dict):
                metadata["split"] = split
                metadata.setdefault("split_safe_id", split_key)
            sample.setdefault("split_safe_id", split_key)
        selected.extend(ordered)

    total_selected = len(selected)
    actual_percentages = {
        bucket: (counts[bucket] / float(total_selected)) if total_selected else 0.0 for bucket in BUCKET_ORDER
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    selected_summary = manifest_summary(selected)
    (output_dir / "manifest.json").write_text(
        json.dumps(
            {
                "version": TRAINING_MANIFEST_VERSION,
                "manifest_contract": TRAINING_MANIFEST_CONTRACT,
                "landmark_schema": "multi_schema",
                "metadata": {
                    "builder": "AccurateFacialLandmarkDetection.tools.landmarks.build_hard_negative_manifest",
                    "sample_count": total_selected,
                    "seed": seed,
                },
                **selected_summary,
                "samples": selected,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    mix_report = {
        "counts": counts,
        "available_counts": available_counts,
        "target_counts": target_counts,
        "effective_capacities": effective_capacities,
        "target_total": target_total,
        "requested_total": total_samples,
        "target_ratios": {
            bucket: float(ratios.get(bucket, 0.0)) for bucket in BUCKET_ORDER
        },
        "target_percentages": target_percentages,
        "actual_percentages": actual_percentages,
        "ceilings": ceilings,
        "by_dataset": by_dataset,
        "manifest_summary": selected_summary,
        "weights": dict(BUCKET_WEIGHT),
        "dataset_default_buckets": dict(DATASET_DEFAULT_BUCKET),
        "bucket_fill_rates": {
            bucket: counts[bucket] / max(float(target_counts[bucket]), 1.0)
            for bucket in BUCKET_ORDER
        },
        "anchor_count": counts.get("anchor", 0),
        "total": total_selected,
        "seed": seed,
        "allow_overlap": allow_overlap,
        "exclude_image_ids_file": str(exclude_image_ids_file)
        if exclude_image_ids_file
        else None,
        "excluded_image_id_count": len(excluded_image_ids),
    }
    (output_dir / "hard_negative_mix.json").write_text(
        json.dumps(mix_report, indent=2, sort_keys=True), encoding="utf-8"
    )

    if write_audit:
        (output_dir / "dataset_audit.json").write_text(
            json.dumps(audit, indent=2, sort_keys=True), encoding="utf-8"
        )

    logger.info(
        "Wrote %d hard-negative samples to %s (%s)",
        total_selected,
        output_dir / "manifest.json",
        ", ".join(f"{bucket}={counts[bucket]}" for bucket in BUCKET_ORDER),
    )
    return mix_report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wflw-manifest", type=Path)
    parser.add_argument("--aflw2000-manifest", type=Path)
    parser.add_argument("--merl-rav-manifest", type=Path)
    parser.add_argument("--cofw68-manifest", type=Path)
    parser.add_argument("--menpo2d-manifest", type=Path)
    parser.add_argument("--multipie-manifest", type=Path)
    parser.add_argument("--w300-manifest", type=Path)
    parser.add_argument("--production-validated-manifest", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--total-samples",
        type=int,
        default=DEFAULT_TOTAL_SAMPLES,
        help=(
            "Target final sample count. Default 0 uses all feasible samples while preserving "
            "the bucket ratios as much as possible. Pass N to build a bounded balanced subset."
        ),
    )
    parser.add_argument(
        "--bucket-ratios",
        default=None,
        help="Comma-separated bucket=value ratios. Default: profile_occlusion=3,profile=2,occlusion=2,anchor=1",
    )
    parser.add_argument(
        "--bucket-percentages",
        default=None,
        help=(
            "Comma-separated bucket=value percentages/fractions, e.g. "
            "profile_occlusion=37.5,profile=25,occlusion=25,anchor=12.5. Overrides --bucket-ratios."
        ),
    )
    parser.add_argument("--max-profile-occlusion", type=int, default=None, help="Optional hard ceiling for this bucket.")
    parser.add_argument("--max-profile", type=int, default=None, help="Optional hard ceiling for this bucket.")
    parser.add_argument("--max-occlusion", type=int, default=None, help="Optional hard ceiling for this bucket.")
    parser.add_argument("--max-anchors", type=int, default=None, help="Optional hard ceiling for this bucket.")
    parser.add_argument("--allow-overlap", action="store_true")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--write-audit", action="store_true")
    parser.add_argument(
        "--exclude-image-ids-file",
        type=Path,
        default=None,
        help="Drop MERL-RAV samples whose imageNNNNN id appears in this file.",
    )
    parser.add_argument("--log-level", default="INFO")
    return parser


def main(argv: T.Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    logging.basicConfig(level=getattr(logging, str(args.log_level).upper(), logging.INFO))

    manifests: dict[str, Path] = {}
    for attr, dataset_label in MANIFEST_ARGS:
        path = getattr(args, attr)
        if path is not None:
            manifests[dataset_label] = path
    if not manifests:
        _parser().error("at least one dataset manifest is required")

    build_hard_negative_manifest(
        manifests=manifests,
        output_dir=args.output_dir,
        total_samples=args.total_samples,
        bucket_ratios=args.bucket_ratios,
        bucket_percentages=args.bucket_percentages,
        max_profile_occlusion=args.max_profile_occlusion,
        max_profile=args.max_profile,
        max_occlusion=args.max_occlusion,
        max_anchors=args.max_anchors,
        allow_overlap=args.allow_overlap,
        seed=args.seed,
        write_audit=args.write_audit,
        exclude_image_ids_file=args.exclude_image_ids_file,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
