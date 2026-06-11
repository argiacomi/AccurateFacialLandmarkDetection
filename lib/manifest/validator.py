#!/usr/bin/env python3
"""Strict validator for schema-aware landmark training manifests."""

from __future__ import annotations

import hashlib
import json
import typing as T
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from lib.datasets.loader_geometry import (
    LOADER_IMAGE_SIZE,
    landmark_mask_from_entry,
    points_look_normalized,
    resolve_loader_source_hw,
    simulate_loader_geometry,
    write_geometry_overlay,
)

from lib.core.schema import (
    canonicalize_schema,
    head_name_for_schema,
    infer_schema,
    normalize_landmark_array,
    point_count_for_schema,
)
from lib.manifest.contract import (
    IDENTITY_FIELDS,
    REQUIRED_SAMPLE_FIELDS,
    TRAINING_MANIFEST_CONTRACT,
    TRAINING_MANIFEST_VERSION,
    manifest_summary,
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: T.Mapping[str, T.Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _resolve(manifest_path: Path, value: T.Any) -> Path:
    path = Path(str(value or ""))
    return path if path.is_absolute() else (manifest_path.parent / path).resolve()


def _metadata(sample: T.Mapping[str, T.Any]) -> dict[str, T.Any]:
    return (
        dict(sample.get("metadata", {}))
        if isinstance(sample.get("metadata"), dict)
        else {}
    )


def _source(sample: T.Mapping[str, T.Any]) -> dict[str, T.Any]:
    return (
        dict(sample.get("source", {})) if isinstance(sample.get("source"), dict) else {}
    )


def _value(sample: T.Mapping[str, T.Any], key: str) -> T.Any:
    if sample.get(key) not in (None, ""):
        return sample.get(key)
    source = _source(sample)
    if source.get(key) not in (None, ""):
        return source.get(key)
    metadata = _metadata(sample)
    return metadata.get(key)


def _label(value: T.Any, default: str = "unknown") -> str:
    label = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    while "__" in label:
        label = label.replace("__", "_")
    return label.strip("_") or default


def _example(
    report: dict[str, T.Any], kind: str, payload: T.Any, max_examples: int
) -> None:
    examples = report.setdefault("examples", {}).setdefault(kind, [])
    if len(examples) < max_examples:
        examples.append(payload)


def _load_samples(
    manifest_path: Path,
    manifest_payload: T.Mapping[str, T.Any] | None = None,
) -> tuple[dict[str, T.Any], list[dict[str, T.Any]]]:
    payload = (
        dict(manifest_payload)
        if manifest_payload is not None
        else json.loads(manifest_path.read_text(encoding="utf-8"))
    )
    samples = payload.get("samples", payload.get("scenarios", []))
    if not isinstance(samples, list):
        raise ValueError(f"manifest {manifest_path} must contain a samples list")
    return payload, [sample for sample in samples if isinstance(sample, dict)]


def _detect_schema(points: np.ndarray) -> str:
    array = np.asarray(points)
    if array.ndim == 2 and array.shape[1] >= 2:
        return infer_schema(array[:, :2])
    return infer_schema(array)


def _normalize_schema(raw: T.Any) -> str | None:
    if raw in (None, ""):
        return None
    return canonicalize_schema(raw)


def _validate_projection_audit(
    sample: T.Mapping[str, T.Any],
    *,
    source_schema: str,
    target_schema: str,
    allow_missing_projection_audit: bool,
) -> str | None:
    if source_schema == target_schema:
        return None

    audit = (
        sample.get("mapping_audit")
        or sample.get("projection_audit")
        or _metadata(sample).get("mapping_audit")
        or _metadata(sample).get("projection_audit")
    )
    if not isinstance(audit, dict):
        if allow_missing_projection_audit:
            return None
        return "missing_mapping_or_projection_audit"

    status = _label(audit.get("status"), default="")
    if status not in {
        "ok",
        "projected",
        "mapped",
        "native",
        "legacy_projected",
        "manually_verified",
    }:
        return f"invalid_mapping_or_projection_audit_status:{status or 'missing'}"
    return None


def validate_training_manifest(
    manifest_path: str | Path,
    *,
    report_path: str | Path | None = None,
    manifest_payload: T.Mapping[str, T.Any] | None = None,
    require_images: bool = True,
    allow_legacy_68_projection: bool = False,
    allow_missing_projection_audit: bool = False,
    allow_legacy_missing_contract_fields: bool = False,
    max_examples: int = 25,
    raise_on_error: bool = False,
    geometry_overlay_dir: str | Path | None = None,
    max_geometry_overlays: int = 200,
) -> dict[str, T.Any]:
    """Validate a mixed-schema landmark training manifest.

    A valid native sample has:
    - declared ``source_schema``
    - declared ``target_schema``
    - a landmark .npy shape matching ``target_schema``
    - a ``head_name`` matching ``target_schema``
    - a split-safe identity for train/test leakage checks

    ``allow_legacy_68_projection`` accepts old builder outputs where a
    non-68 source schema was already projected into a 68-point target.
    """

    manifest_path = Path(manifest_path)
    payload, samples = _load_samples(manifest_path, manifest_payload)

    report: dict[str, T.Any] = {
        "manifest": str(manifest_path),
        "manifest_sha256": _sha256_file(manifest_path),
        "contract": payload.get("manifest_contract"),
        "expected_contract": TRAINING_MANIFEST_CONTRACT,
        "version": payload.get("version"),
        "expected_version": TRAINING_MANIFEST_VERSION,
        "total_samples": len(samples),
        "valid_samples": 0,
        "invalid_samples": 0,
        "missing_images": 0,
        "missing_landmarks": 0,
        "invalid_landmarks": 0,
        "missing_required_fields": Counter(),
        "schema_shape_mismatches": 0,
        "head_mismatches": 0,
        "projection_audit_errors": 0,
        "datasets": Counter(),
        "splits": Counter(),
        "schemas": Counter(),
        "source_schemas": Counter(),
        "target_schemas": Counter(),
        "heads": Counter(),
        "hard_negative_buckets": Counter(),
        "examples": {
            "invalid": [],
            "missing_image": [],
            "missing_landmarks": [],
            "schema_shape_mismatch": [],
            "head_mismatch": [],
            "projection_audit": [],
            "leakage": [],
            "landmarks_outside_image": [],
            "unreasonable_loader_padding": [],
            "suspicious_loader_padding": [],
            "normalized_landmarks_non_256_source": [],
            "invalid_geometry": [],
        },
        "geometry": {
            "checked_samples": 0,
            "landmarks_outside_image": 0,
            "unreasonable_loader_padding": 0,
            "suspicious_loader_padding": 0,
            "normalized_landmarks_non_256_source": 0,
            "invalid_geometry": 0,
            "overlays_written": 0,
            "overlay_dir": str(geometry_overlay_dir) if geometry_overlay_dir else None,
        },
        "legacy_68_projection_samples": 0,
        "leakage": {"checked_fields": list(IDENTITY_FIELDS), "violations": []},
    }

    identities: dict[tuple[str, str], dict[str, set[str]]] = defaultdict(
        lambda: defaultdict(set)
    )

    for index, sample in enumerate(samples):
        metadata = _metadata(sample)
        source = _source(sample)
        sample_id = str(sample.get("sample_id") or sample.get("id") or index)
        dataset = _label(
            sample.get("dataset") or source.get("dataset") or metadata.get("dataset")
        )
        split = _label(
            sample.get("split") or metadata.get("split"), default="unspecified"
        )
        bucket = _label(
            sample.get("hard_negative_bucket")
            or metadata.get("hard_negative_bucket")
            or sample.get("condition")
            or metadata.get("condition")
        )

        report["datasets"][dataset] += 1
        report["splits"][split] += 1
        report["hard_negative_buckets"][bucket] += 1

        errors: list[str] = []
        legacy_inferable_contract_fields = {
            "landmark_count",
            "head_name",
            "split_safe_id",
        }
        for field in REQUIRED_SAMPLE_FIELDS:
            if _value(sample, field) not in (None, ""):
                continue
            report["missing_required_fields"][field] += 1
            if (
                field in legacy_inferable_contract_fields
                and allow_legacy_missing_contract_fields
            ):
                continue
            errors.append(f"missing_{field}")

        image_value = sample.get("image") or sample.get("image_path")
        image_path = None
        if image_value and require_images:
            image_path = _resolve(manifest_path, image_value)
            if not image_path.is_file():
                report["missing_images"] += 1
                _example(
                    report,
                    "missing_image",
                    {"sample_id": sample_id, "path": str(image_path)},
                    max_examples,
                )
                errors.append("missing_image")

        landmarks_value = sample.get("landmarks") or sample.get("ground_truth")
        landmarks = None
        detected_schema = None
        if not landmarks_value:
            report["missing_landmarks"] += 1
            _example(
                report, "missing_landmarks", {"sample_id": sample_id}, max_examples
            )
            errors.append("missing_landmarks")
        else:
            landmarks_path = _resolve(manifest_path, landmarks_value)
            if not landmarks_path.is_file():
                report["missing_landmarks"] += 1
                _example(
                    report,
                    "missing_landmarks",
                    {"sample_id": sample_id, "path": str(landmarks_path)},
                    max_examples,
                )
                errors.append("missing_landmarks")
            else:
                try:
                    landmarks = np.load(landmarks_path)
                    detected_schema = _detect_schema(np.asarray(landmarks))
                    normalize_landmark_array(
                        np.asarray(landmarks)[:, :2], schema=detected_schema
                    )
                except Exception as err:  # noqa: BLE001
                    report["invalid_landmarks"] += 1
                    _example(
                        report,
                        "invalid",
                        {
                            "sample_id": sample_id,
                            "path": str(landmarks_path),
                            "error": str(err),
                        },
                        max_examples,
                    )
                    errors.append("invalid_landmarks")

        if landmarks is not None:
            geometry_hw, geometry_source, geometry_error = resolve_loader_source_hw(
                sample,
                base_dir=manifest_path.parent,
            )
            if geometry_error:
                report["geometry"]["invalid_geometry"] += 1
                _example(
                    report,
                    "invalid_geometry",
                    {
                        "sample_id": sample_id,
                        "source": geometry_source,
                        "error": geometry_error,
                    },
                    max_examples,
                )
                errors.append("invalid_geometry")

            if geometry_hw is not None:
                points_xy = np.asarray(landmarks)[:, :2]
                # Loader parity: simulate with the same mask MakeLMKInsideImage
                # receives, so masked-out sentinel coordinates (e.g. MERL-RAV
                # self-occluded points zeroed by the builder) are not reported
                # as out-of-frame landmarks.
                loader_mask = landmark_mask_from_entry(
                    sample, metadata, int(points_xy.shape[0])
                )
                diag = simulate_loader_geometry(
                    points_xy,
                    geometry_hw,
                    landmark_mask=loader_mask,
                )
                report["geometry"]["checked_samples"] += 1
                diag_example = {
                    "sample_id": sample_id,
                    "dataset": dataset,
                    "split": split,
                    "source": geometry_source,
                    "diagnostics": diag,
                }
                needs_overlay = False
                if diag.get("landmarks_outside_image"):
                    report["geometry"]["landmarks_outside_image"] += 1
                    _example(
                        report,
                        "landmarks_outside_image",
                        diag_example,
                        max_examples,
                    )
                if points_look_normalized(points_xy) and tuple(geometry_hw) != (
                    LOADER_IMAGE_SIZE,
                    LOADER_IMAGE_SIZE,
                ):
                    # The loader scales [0,1] points by 255 and assumes the 256
                    # training frame; normalized points on a non-256 source are
                    # silently misplaced even though they stay in-bounds.
                    report["geometry"]["normalized_landmarks_non_256_source"] += 1
                    _example(
                        report,
                        "normalized_landmarks_non_256_source",
                        diag_example,
                        max_examples,
                    )
                    needs_overlay = True
                if not diag.get("ok"):
                    report["geometry"]["invalid_geometry"] += 1
                    reason = str(diag.get("reason") or "invalid_geometry")
                    if reason == "unreasonable_loader_padding":
                        report["geometry"]["unreasonable_loader_padding"] += 1
                        _example(
                            report,
                            "unreasonable_loader_padding",
                            diag_example,
                            max_examples,
                        )
                    else:
                        _example(report, "invalid_geometry", diag_example, max_examples)
                    errors.append(reason)
                    needs_overlay = True
                elif diag.get("suspicious"):
                    # Quarantine, not a hard failure: large-but-trainable
                    # overflow usually means a wrong coordinate frame, but
                    # heavy crops/profiles can legitimately overflow.
                    report["geometry"]["suspicious_loader_padding"] += 1
                    _example(
                        report,
                        "suspicious_loader_padding",
                        diag_example,
                        max_examples,
                    )
                    needs_overlay = True

                if (
                    needs_overlay
                    and geometry_overlay_dir is not None
                    and report["geometry"]["overlays_written"] < max_geometry_overlays
                ):
                    overlay_image = (
                        _resolve(manifest_path, sample.get("prepared_image"))
                        if geometry_source == "prepared_image"
                        else _resolve(manifest_path, image_value)
                        if image_value
                        else None
                    )
                    safe_name = (
                        str(sample_id).replace("/", "_").replace("#", "_") or "sample"
                    )
                    written = write_geometry_overlay(
                        Path(geometry_overlay_dir) / dataset / f"{safe_name}.png",
                        overlay_image,
                        points_xy,
                        geometry_hw,
                        landmark_mask=loader_mask,
                        diag=diag,
                    )
                    if written is not None:
                        report["geometry"]["overlays_written"] += 1
                        diag_example["overlay"] = str(written)

        source_schema = None
        target_schema = None
        if landmarks is not None and detected_schema is not None:
            try:
                source_schema = (
                    _normalize_schema(_value(sample, "source_schema"))
                    or detected_schema
                )
            except ValueError as err:
                errors.append(f"invalid_source_schema:{err}")
                source_schema = detected_schema
            try:
                target_schema = (
                    _normalize_schema(_value(sample, "target_schema")) or source_schema
                )
            except ValueError as err:
                errors.append(f"invalid_target_schema:{err}")
                target_schema = detected_schema

            report["source_schemas"][source_schema] += 1
            report["target_schemas"][target_schema] += 1
            report["schemas"][target_schema] += 1

            observed_count = int(np.asarray(landmarks).shape[0])
            expected_target_count = point_count_for_schema(target_schema)
            declared_count = _value(sample, "landmark_count")
            if declared_count not in (None, ""):
                try:
                    if int(declared_count) != observed_count:
                        errors.append(
                            f"landmark_count_mismatch:{declared_count}!={observed_count}"
                        )
                except (TypeError, ValueError):
                    errors.append(f"invalid_landmark_count:{declared_count!r}")

            legacy_projection = (
                allow_legacy_68_projection
                and target_schema == "2d_68"
                and detected_schema == "2d_68"
                and source_schema != target_schema
            )
            if legacy_projection:
                report["legacy_68_projection_samples"] += 1
            if observed_count != expected_target_count and not legacy_projection:
                report["schema_shape_mismatches"] += 1
                _example(
                    report,
                    "schema_shape_mismatch",
                    {
                        "sample_id": sample_id,
                        "source_schema": source_schema,
                        "target_schema": target_schema,
                        "detected_schema": detected_schema,
                        "shape": list(np.asarray(landmarks).shape),
                    },
                    max_examples,
                )
                errors.append("schema_shape_mismatch")

            expected_head = head_name_for_schema(target_schema)
            declared_head = str(_value(sample, "head_name") or expected_head)
            report["heads"][declared_head] += 1
            if declared_head != expected_head:
                report["head_mismatches"] += 1
                _example(
                    report,
                    "head_mismatch",
                    {
                        "sample_id": sample_id,
                        "target_schema": target_schema,
                        "declared_head": declared_head,
                        "expected_head": expected_head,
                    },
                    max_examples,
                )
                errors.append("head_mismatch")

            projection_error = _validate_projection_audit(
                sample,
                source_schema=source_schema,
                target_schema=target_schema,
                allow_missing_projection_audit=allow_missing_projection_audit
                or legacy_projection,
            )
            if projection_error:
                report["projection_audit_errors"] += 1
                _example(
                    report,
                    "projection_audit",
                    {
                        "sample_id": sample_id,
                        "source_schema": source_schema,
                        "target_schema": target_schema,
                        "error": projection_error,
                    },
                    max_examples,
                )
                errors.append(projection_error)

        for field in IDENTITY_FIELDS:
            value = _value(sample, field)
            if value in (None, "") and field == "image_id":
                value = image_value
            if value in (None, ""):
                continue
            identities[(field, str(value))][split].add(sample_id)

        if errors:
            report["invalid_samples"] += 1
            _example(
                report,
                "invalid",
                {
                    "sample_id": sample_id,
                    "dataset": dataset,
                    "split": split,
                    "errors": errors,
                },
                max_examples,
            )
        else:
            report["valid_samples"] += 1

    for (field, value), split_map in sorted(identities.items()):
        concrete_splits = {
            split for split in split_map if split not in {"", "unspecified"}
        }
        if len(concrete_splits) <= 1:
            continue
        violation = {
            "field": field,
            "value": value,
            "splits": sorted(concrete_splits),
            "sample_ids": {
                split: sorted(ids)[:10] for split, ids in sorted(split_map.items())
            },
        }
        report["leakage"]["violations"].append(violation)
        _example(report, "leakage", violation, max_examples)

    report["missing_required_fields"] = dict(
        sorted(report["missing_required_fields"].items())
    )
    for key in (
        "datasets",
        "splits",
        "schemas",
        "source_schemas",
        "target_schemas",
        "heads",
        "hard_negative_buckets",
    ):
        report[key] = dict(sorted(report[key].items()))

    report["leakage"]["violation_count"] = len(report["leakage"]["violations"])
    report["summary"] = manifest_summary(samples)

    ok = (
        report["valid_samples"] > 0
        and report["invalid_samples"] == 0
        and report["missing_images"] == 0
        and report["missing_landmarks"] == 0
        and report["invalid_landmarks"] == 0
        and report["leakage"]["violation_count"] == 0
    )
    report["ok"] = bool(ok)

    if report_path is not None:
        _write_json(Path(report_path), report)

    if raise_on_error and not ok:
        raise ValueError(
            "training manifest validation failed: "
            f"{report['invalid_samples']} invalid sample(s), "
            f"{report['leakage']['violation_count']} leakage violation(s). "
            f"See {report_path or manifest_path}."
        )

    return report
