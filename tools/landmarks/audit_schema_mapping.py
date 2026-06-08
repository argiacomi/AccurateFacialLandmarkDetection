#!/usr/bin/env python3
"""Audit visual landmark schema mappings for schema-aware CD-ViT training."""

from __future__ import annotations

import argparse
import json
import typing as T
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from lib.landmarks.core.schema import (
    MAP_98_TO_68,
    SCHEMAS_WITHOUT_VERIFIED_FLIP_MAPS,
    canonicalize_schema,
    has_verified_flip_map,
    projection_audit_for_schema,
)


def _load_manifest(path: Path) -> list[dict[str, T.Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    entries = payload.get("samples", payload.get("scenarios", []))
    if not isinstance(entries, list):
        raise ValueError(f"{path} must contain samples or scenarios")
    return [entry for entry in entries if isinstance(entry, dict)]


def _resolve(base: Path, value: T.Any) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else (base / path).resolve()


def _schema(entry: T.Mapping[str, T.Any], points: np.ndarray) -> str:
    metadata = entry.get("metadata") if isinstance(entry.get("metadata"), dict) else {}
    raw = entry.get("source_schema") or metadata.get("source_schema") or f"2d_{points.shape[0]}"
    try:
        return canonicalize_schema(raw)
    except ValueError:
        return str(raw)


def _draw_points(image_path: Path, points: np.ndarray, output_path: Path, *, projected: np.ndarray | None = None) -> None:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        image = np.full((256, 256, 3), 255, dtype=np.uint8)
    h, w = image.shape[:2]
    scale = np.asarray([w - 1, h - 1], dtype=np.float32)
    draw_points = points[:, :2].astype(np.float32)
    if float(np.nanmax(draw_points)) <= 1.5:
        draw_points = draw_points * scale
    for x, y in draw_points:
        cv2.circle(image, (int(round(x)), int(round(y))), 2, (0, 0, 255), -1)
    if projected is not None:
        projected_points = projected[:, :2].astype(np.float32)
        if float(np.nanmax(projected_points)) <= 1.5:
            projected_points = projected_points * scale
        for x, y in projected_points:
            cv2.circle(image, (int(round(x)), int(round(y))), 1, (0, 180, 0), -1)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), image)


def audit_schema_mapping(manifest: Path, output_dir: Path, *, limit: int = 25, write_overlays: bool = False) -> Path:
    entries = _load_manifest(manifest)
    base = manifest.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    report: dict[str, T.Any] = {
        "manifest": str(manifest.resolve()),
        "counts": Counter(),
        "samples": [],
        "map_98_to_68_size": int(MAP_98_TO_68.size),
        "projection_to_68": {},
        "flip_map_audit": {
            "schemas_without_verified_flip_maps": sorted(SCHEMAS_WITHOUT_VERIFIED_FLIP_MAPS),
            "unverified_identity_flip_maps": [],
            "schemas_seen_without_verified_flip_maps": [],
        },
    }

    emitted = 0
    for index, entry in enumerate(entries):
        landmark_value = entry.get("landmarks") or entry.get("ground_truth")
        image_value = entry.get("image")
        if not landmark_value:
            continue
        points = np.load(_resolve(base, landmark_value)).astype(np.float32)
        schema = _schema(entry, points)
        report["counts"][schema] += 1
        try:
            projection_audit = projection_audit_for_schema(schema)
        except ValueError:
            projection_audit = {
                "status": "unknown_schema",
                "source_schema": schema,
                "target_schema": "2d_68",
            }
        projection_status_counts = report["projection_to_68"].setdefault(
            projection_audit["status"],
            0,
        )
        report["projection_to_68"][projection_audit["status"]] = int(projection_status_counts) + 1
        try:
            verified_flip_map = has_verified_flip_map(schema)
        except ValueError:
            verified_flip_map = False
        if not verified_flip_map:
            seen_without_verified = report["flip_map_audit"]["schemas_seen_without_verified_flip_maps"]
            if schema not in seen_without_verified:
                seen_without_verified.append(schema)
        sample_id = str(entry.get("sample_id") or entry.get("id") or index)
        item: dict[str, T.Any] = {
            "sample_id": sample_id,
            "schema": schema,
            "point_count": int(points.shape[0]),
            "projection_to_68": projection_audit,
        }
        projected = None
        if schema == "2d_98":
            projected = points[MAP_98_TO_68, :2]
            item["projected_68_count"] = int(projected.shape[0])
        if write_overlays and image_value and emitted < limit:
            overlay_path = output_dir / "overlays" / f"{schema}_{emitted:04d}.jpg"
            _draw_points(_resolve(base, image_value), points, overlay_path, projected=projected)
            item["overlay"] = str(overlay_path)
            emitted += 1
        report["samples"].append(item)

    output_path = output_dir / "schema_mapping_audit.json"
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return output_path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=25)
    parser.add_argument("--write-overlays", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    output_path = audit_schema_mapping(
        args.manifest,
        args.output_dir,
        limit=args.limit,
        write_overlays=args.write_overlays,
    )
    print(f"Wrote schema mapping audit: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
