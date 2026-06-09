#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np


TARGET_SCHEMAS = ("2d_106", "2d_194", "2d_29")


def _resolve(base: Path, value: Any) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else (base / path).resolve()


def _sample_schema(sample: dict[str, Any]) -> str:
    metadata = (
        sample.get("metadata") if isinstance(sample.get("metadata"), dict) else {}
    )
    return str(
        sample.get("source_schema")
        or metadata.get("source_schema")
        or sample.get("target_schema")
        or metadata.get("target_schema")
        or ""
    )


def _load_manifest(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    samples = payload.get("samples", payload.get("scenarios", []))
    if not isinstance(samples, list):
        raise ValueError(f"{path} does not contain a samples/scenarios list")
    return [s for s in samples if isinstance(s, dict)]


def _scale_points_to_image(points: np.ndarray, image: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[1] < 2:
        raise ValueError(f"expected Nx2 landmarks, got {pts.shape}")

    pts = pts[:, :2].copy()
    h, w = image.shape[:2]

    # Support normalized landmarks if encountered.
    finite = pts[np.isfinite(pts).all(axis=1)]
    if finite.size and float(np.nanmax(finite)) <= 1.5:
        pts[:, 0] *= max(w - 1, 1)
        pts[:, 1] *= max(h - 1, 1)

    return pts


def _draw_labelled_overlay(
    image_path: Path,
    landmarks_path: Path,
    output_path: Path,
    *,
    one_based: bool = False,
    radius: int = 3,
    scale: float = 1.0,
    font_scale: float = 0.35,
) -> None:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        # Fallback canvas when image path is not available/readable.
        image = np.full((256, 256, 3), 255, dtype=np.uint8)

    original_h, original_w = image.shape[:2]

    points = np.load(landmarks_path, allow_pickle=False)
    points = _scale_points_to_image(points, image)

    if scale != 1.0:
        image = cv2.resize(
            image,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_CUBIC if scale > 1.0 else cv2.INTER_AREA,
        )
        points[:, 0] *= scale
        points[:, 1] *= scale

    overlay = image.copy()

    for idx, (x, y) in enumerate(points):
        if not np.isfinite(x) or not np.isfinite(y):
            continue
        xi, yi = int(round(float(x))), int(round(float(y)))
        label = str(idx + 1 if one_based else idx)

        cv2.circle(overlay, (xi, yi), radius, (0, 0, 255), -1)

        # Text with black outline + white fill for readability.
        text_pos = (xi + 4, yi - 4)
        cv2.putText(
            overlay,
            label,
            text_pos,
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            overlay,
            label,
            text_pos,
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), overlay):
        raise OSError(f"failed to write {output_path}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest", type=Path, default=Path("data/prepared/manifest.json")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("runs/schema_index_overlays")
    )
    parser.add_argument(
        "--one-based",
        action="store_true",
        help="draw 1-based labels instead of 0-based",
    )
    parser.add_argument(
        "--scale", type=float, default=1.0, help="output image scale multiplier"
    )
    parser.add_argument("--font-scale", type=float, default=0.35)
    args = parser.parse_args()

    manifest = args.manifest.resolve()
    manifest_base = manifest.parent
    samples = _load_manifest(manifest)

    picked: dict[str, dict[str, Any]] = {}
    for sample in samples:
        schema = _sample_schema(sample)
        if schema in TARGET_SCHEMAS and schema not in picked:
            picked[schema] = sample
        if all(schema in picked for schema in TARGET_SCHEMAS):
            break

    summary: list[dict[str, Any]] = []

    for schema in TARGET_SCHEMAS:
        sample = picked.get(schema)
        if sample is None:
            print(f"missing schema: {schema}")
            continue

        image = _resolve(manifest_base, sample["image"])
        landmarks = _resolve(manifest_base, sample["landmarks"])
        sample_id = str(sample.get("sample_id", "sample"))
        dataset = str(sample.get("dataset", "dataset"))

        output = (
            args.output_dir
            / f"{schema}_{dataset}_{sample_id.replace('/', '_').replace('#', '_')}.jpg"
        )

        _draw_labelled_overlay(
            image,
            landmarks,
            output,
            one_based=args.one_based,
            scale=args.scale,
            font_scale=args.font_scale,
        )

        print(f"{schema}: {dataset} :: {sample_id}")
        print(f"  image:     {image}")
        print(f"  landmarks: {landmarks}")
        print(f"  overlay:   {output}")

        summary.append(
            {
                "schema": schema,
                "dataset": dataset,
                "sample_id": sample_id,
                "image": str(image),
                "landmarks": str(landmarks),
                "overlay": str(output),
                "point_count": int(np.load(landmarks, allow_pickle=False).shape[0]),
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(f"\nwrote summary: {args.output_dir / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
