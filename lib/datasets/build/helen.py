"""helen dataset builder (split from build_quality_dataset)."""

# ruff: noqa: E402, F403, F405
from __future__ import annotations

from lib.datasets.build.core import *  # noqa: F403
from lib.datasets.build.w300 import *  # noqa: F403


def _helen_annotations_path(root: Path) -> Path | None:
    if root.is_file() and root.suffix.lower() == ".json":
        return root
    exact = sorted(root.rglob("annotations.json"), key=lambda item: len(item.parts))
    return exact[0] if exact else None


def _build_helen(
    root: Path,
    output_dir: Path,
    *,
    scenario: str,
    scenarios: tuple[str, ...] | None,
    limit: int | None,
    mode: str,
    allow_overlap: bool,
    image_root: str | None,
) -> Path:
    annotations = _helen_annotations_path(root)
    if annotations is None:
        return _build_expected_schema_dataset(
            root,
            output_dir,
            dataset="helen",
            expected_schema="2d_194",
            parser_name="helen_194",
            scenario=scenario,
            scenarios=scenarios,
            limit=limit,
            mode=mode,
            allow_overlap=allow_overlap,
            image_root=image_root,
        )

    payload = read_json(annotations)
    if not isinstance(payload, list):
        raise ValueError(f"HELEN annotations.json must contain a list: {annotations}")

    helen_roots = _helen_300w_roots(root, image_root)
    if not helen_roots:
        raise ValueError(
            "HELEN dense annotations require a 300W Helen image cache; "
            "pass --image-root pointing to data/300w/300w or its helen subdirectory"
        )
    helen_image_index = _build_combined_image_index(helen_roots)
    samples: list[dict[str, T.Any]] = []
    skipped: list[dict[str, str]] = []
    for index, entry in track(
        enumerate(payload), desc="Build helen", total=len(payload), unit="sample"
    ):
        sample_id = f"annotations/{index:05d}"
        try:
            if isinstance(entry, dict):
                image_name = str(
                    entry.get("image")
                    or entry.get("image_path")
                    or entry.get("filename")
                    or ""
                )
                raw_points = entry.get("landmarks") or entry.get("points")
                width = entry.get("width") or entry.get("image_width")
                height = entry.get("height") or entry.get("image_height")
            elif isinstance(entry, (list, tuple)) and len(entry) >= 2:
                image_info = entry[0]
                raw_points = entry[1]
                if not isinstance(image_info, (list, tuple)) or not image_info:
                    raise ValueError("missing HELEN image info")
                image_name = str(image_info[0])
                width = image_info[1] if len(image_info) > 1 else None
                height = image_info[2] if len(image_info) > 2 else None
            else:
                raise ValueError("unsupported HELEN annotation row")
            if not image_name or raw_points is None:
                raise ValueError("missing image name or landmarks")
            points, detected_schema = _canonical_points(
                raw_points, source_schema="2d_194"
            )
            if detected_schema != "2d_194":
                raise ValueError(f"HELEN expected 2d_194, got {detected_schema}")
            image = _resolve_unique_image(
                helen_roots,
                image_name,
                context="HELEN 300W",
                image_index=helen_image_index,
            )
            # The dense annotations declare the dimensions of the image they
            # were made on; the resolved 300W-cache copy can be a different
            # resolution, leaving valid-looking landmarks outside the resolved
            # image's coordinate frame. Rescale into the resolved frame when
            # declared dims are present, then gate on loader geometry.
            actual_h, actual_w = loader_image_hw(image)
            declared_w = int(width) if width else None
            declared_h = int(height) if height else None
            landmarks_rescaled = False
            if (
                declared_w
                and declared_h
                and (declared_w, declared_h) != (actual_w, actual_h)
            ):
                points = points.copy()
                points[:, 0] *= float(actual_w) / float(declared_w)
                points[:, 1] *= float(actual_h) / float(declared_h)
                landmarks_rescaled = True
            geometry = simulate_loader_geometry(points, (actual_h, actual_w))
            if not geometry.get("ok") or geometry.get("suspicious"):
                _write_geometry_review_overlay(
                    output_dir,
                    dataset="helen",
                    sample_id=f"helen/{Path(image_name).stem}",
                    image_path=image,
                    points=points,
                    source_image_hw=(actual_h, actual_w),
                    diag=geometry,
                )
                raise ValueError(
                    "landmarks do not fit resolved HELEN image: "
                    f"{geometry.get('reason')} (padding={geometry.get('padding')}, "
                    f"declared={declared_w}x{declared_h}, "
                    f"resolved={actual_w}x{actual_h})"
                )
        except Exception as err:  # noqa: BLE001
            skipped.append({"sample_id": sample_id, "reason": str(err)})
            continue

        sample_id = f"helen/{Path(image_name).stem}"
        split = _deterministic_split("helen", sample_id)
        condition, conds = _native_conditions_for_split(scenario, split)
        metadata = {
            "dataset": "helen",
            "dataset_parser": "helen_annotations_json",
            "parser_type": "dataset_specific",
            "annotation_file": str(annotations.resolve()),
            "source_image_name": image_name,
            "resolved_300w_image_path": str(image.resolve()),
            "source_schema": "2d_194",
        }
        if width is not None:
            metadata["image_width"] = int(width)
        if height is not None:
            metadata["image_height"] = int(height)
        metadata["resolved_image_hw"] = [int(actual_h), int(actual_w)]
        if landmarks_rescaled:
            metadata["landmarks_rescaled_from_declared_dims"] = True
            metadata["landmarks_rescale_factors"] = [
                float(actual_w) / float(declared_w),
                float(actual_h) / float(declared_h),
            ]
        metadata["loader_geometry"] = {
            "ok": bool(geometry.get("ok")),
            "suspicious": bool(geometry.get("suspicious")),
            "padding": geometry.get("padding"),
        }
        samples.append(
            _with_split(
                _sample(
                    output_dir=output_dir,
                    dataset="helen",
                    sample_id=sample_id,
                    image=image,
                    points68=points,
                    condition=condition,
                    conditions=conds,
                    source_schema="2d_194",
                    source_id=sample_id,
                    metadata=metadata,
                ),
                split,
            )
        )

    if not samples:
        raise ValueError(f"no HELEN annotation samples built; skipped={skipped[:10]}")
    return _write_manifest(
        output_dir,
        "helen",
        scenario,
        _filter(samples, scenarios, limit),
        mode=mode,
        allow_overlap=allow_overlap,
        scenarios=scenarios,
        skipped=skipped,
    )


# Re-export every module-level name (including the single-underscore build
# helpers) so `from lib.datasets.build.<mod> import *` resolves bare-name
# calls in sibling modules exactly as they did in the original flat module.
__all__ = [_n for _n in dict(globals()) if not _n.startswith("__")]
