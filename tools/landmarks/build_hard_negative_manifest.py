#!/usr/bin/env python3
"""Merge dataset manifests into a prioritized profile/occlusion hard-negative manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import typing as T
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

logger = logging.getLogger(__name__)

DATASET_DEFAULT_BUCKET: dict[str, str] = {
    "cofw": "occlusion",
    "cofw68": "occlusion",
    "300w": "anchor",
    "w300": "anchor",
}

BUCKET_ORDER: tuple[str, ...] = ("profile_occlusion", "profile", "occlusion", "anchor")

MANIFEST_ARGS: tuple[tuple[str, str], ...] = (
    ("wflw_manifest", "wflw"),
    ("aflw2000_manifest", "aflw2000-3d"),
    ("merl_rav_manifest", "merl-rav"),
    ("cofw_manifest", "cofw"),
    ("menpo2d_manifest", "menpo2d"),
    ("multipie_manifest", "multipie"),
    ("w300_manifest", "300w"),
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


def _load_samples(manifest_path: Path) -> list[dict[str, T.Any]]:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    samples = payload.get("samples", payload.get("scenarios", []))
    if not isinstance(samples, list):
        raise ValueError(f"manifest {manifest_path} has no 'samples' list")
    return [dict(entry) for entry in samples if isinstance(entry, T.Mapping)]


def _stable_order(samples: T.Sequence[dict[str, T.Any]], *, seed: int) -> list[dict[str, T.Any]]:
    def _hash(sample: dict[str, T.Any]) -> str:
        dataset, source_id = source_key(sample)
        return hashlib.sha256(f"{seed}|{dataset}|{source_id}".encode()).hexdigest()

    return sorted(samples, key=_hash)


def build_hard_negative_manifest(
    *,
    manifests: T.Mapping[str, Path],
    output_dir: Path,
    max_profile_occlusion: int | None = None,
    max_profile: int | None = None,
    max_occlusion: int | None = None,
    max_anchors: int | None = None,
    allow_overlap: bool = False,
    seed: int = 1337,
    write_audit: bool = False,
) -> dict[str, T.Any]:
    quotas: dict[str, int | None] = {
        "profile_occlusion": max_profile_occlusion,
        "profile": max_profile,
        "occlusion": max_occlusion,
        "anchor": max_anchors,
    }

    classified: dict[str, list[dict[str, T.Any]]] = {bucket: [] for bucket in BUCKET_ORDER}
    audit: dict[str, dict[str, int]] = {}
    seen_keys: set[tuple[str, str]] = set()

    for dataset, manifest_path in manifests.items():
        dataset_label = dataset.strip().lower()
        dataset_counts: dict[str, int] = dict.fromkeys(
            (*BUCKET_ORDER, "classified_by_label", "dataset_default", "skipped", "duplicate"),
            0,
        )
        for sample in _load_samples(manifest_path):
            sample.setdefault("dataset", dataset_label)
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

    selected: list[dict[str, T.Any]] = []
    counts: dict[str, int] = {}
    by_dataset: dict[str, dict[str, int]] = {}
    for bucket in BUCKET_ORDER:
        ordered = _stable_order(classified[bucket], seed=seed)
        limit = quotas[bucket]
        if limit is not None:
            ordered = ordered[: max(limit, 0)]
        counts[bucket] = len(ordered)
        for sample in ordered:
            dataset_label = str(sample.get("dataset", "")).strip().lower() or "unknown"
            by_dataset.setdefault(dataset_label, {})
            by_dataset[dataset_label][bucket] = by_dataset[dataset_label].get(bucket, 0) + 1
        selected.extend(ordered)

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "manifest.json").write_text(
        json.dumps({"samples": selected}, indent=2, sort_keys=True), encoding="utf-8"
    )

    mix_report = {
        "counts": counts,
        "by_dataset": by_dataset,
        "weights": dict(BUCKET_WEIGHT),
        "dataset_default_buckets": dict(DATASET_DEFAULT_BUCKET),
        "quota_fill_rates": {
            bucket: 1.0 if (quota := quotas[bucket]) is None else counts[bucket] / max(float(quota), 1.0)
            for bucket in BUCKET_ORDER
        },
        "anchor_count": counts.get("anchor", 0),
        "total": len(selected),
        "seed": seed,
        "allow_overlap": allow_overlap,
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
        len(selected),
        output_dir / "manifest.json",
        ", ".join(f"{bucket}={counts[bucket]}" for bucket in BUCKET_ORDER),
    )
    return mix_report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wflw-manifest", type=Path)
    parser.add_argument("--aflw2000-manifest", type=Path)
    parser.add_argument("--merl-rav-manifest", type=Path)
    parser.add_argument("--cofw-manifest", type=Path)
    parser.add_argument("--menpo2d-manifest", type=Path)
    parser.add_argument("--multipie-manifest", type=Path)
    parser.add_argument("--w300-manifest", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-profile-occlusion", type=int, default=None)
    parser.add_argument("--max-profile", type=int, default=None)
    parser.add_argument("--max-occlusion", type=int, default=None)
    parser.add_argument("--max-anchors", type=int, default=None)
    parser.add_argument("--allow-overlap", action="store_true")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--write-audit", action="store_true")
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
        max_profile_occlusion=args.max_profile_occlusion,
        max_profile=args.max_profile,
        max_occlusion=args.max_occlusion,
        max_anchors=args.max_anchors,
        allow_overlap=args.allow_overlap,
        seed=args.seed,
        write_audit=args.write_audit,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
