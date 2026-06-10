"""wflw dataset builder (split from build_quality_dataset)."""

# ruff: noqa: E402, F403, F405
from __future__ import annotations

from lib.datasets.build.core import *  # noqa: F403


def _parse_wflw_line(
    line: str, line_no: int
) -> tuple[np.ndarray, list[float], dict[str, int], str]:
    parts = line.split()
    if len(parts) < 197:
        raise ValueError(f"WFLW line {line_no} has too few fields")
    points = np.asarray(
        [float(value) for value in parts[:196]], dtype=np.float32
    ).reshape(98, 2)
    bbox: list[float] = []
    if len(parts) >= 201:
        bbox = [float(value) for value in parts[196:200]]
    attrs = dict.fromkeys(WFLW_ATTRIBUTE_NAMES, 0)
    if len(parts) >= 207:
        values = [int(float(value)) for value in parts[200:206]]
        attrs = dict(zip(WFLW_ATTRIBUTE_NAMES, values, strict=True))
        image_rel = " ".join(parts[206:])
    else:
        image_rel = parts[-1]
    return points, bbox, attrs, image_rel


def _find_wflw_annotations(root: Path) -> Path | None:
    for pattern in (
        "list_98pt_rect_attr_train_test.txt",
        "list_98pt_rect_attr_train.txt",
        "list_98pt_rect_attr_test.txt",
        "*98pt*rect*attr*.txt",
    ):
        matches = sorted(root.rglob(pattern), key=lambda item: len(item.parts))
        if matches:
            return matches[0]
    return None


def _find_wflw_images(root: Path) -> Path:
    for name in ("WFLW_images", "images", "Images", "WFLW"):
        matches = [path for path in root.rglob(name) if path.is_dir()]
        if matches:
            return sorted(matches, key=lambda item: len(item.parts))[0]
    return root


def _build_wflw(
    root: Path | None,
    output_dir: Path,
    *,
    annotation_file: str | None,
    image_root: str | None,
    scenario: str,
    scenarios: tuple[str, ...] | None,
    limit: int | None,
    mode: str,
    allow_overlap: bool,
) -> Path:
    if annotation_file:
        annotations = Path(annotation_file)
        root_for_images = annotations.parent
    else:
        root_for_images = root or Path(".")
        annotations = _find_wflw_annotations(root_for_images)
    if annotations is None or not annotations.is_file():
        if root is None:
            raise FileNotFoundError(
                "WFLW annotation file not found; pass --wflw-annotations or --source-dir"
            )
        logger.info(
            "WFLW annotations not found; falling back to generic directory parsing"
        )
        return _build_directory(
            root,
            output_dir,
            dataset="wflw",
            scenario=scenario,
            scenarios=scenarios,
            limit=limit,
            mode=mode,
            allow_overlap=allow_overlap,
            image_root=image_root,
        )

    image_base = Path(image_root) if image_root else _find_wflw_images(root_for_images)
    rows = []
    counts: Counter[str] = Counter()
    for line_no, line in enumerate(
        annotations.read_text(encoding="utf-8", errors="ignore").splitlines(), 1
    ):
        if not line.strip():
            continue
        row = _parse_wflw_line(line, line_no)
        rows.append(row)
        counts[row[3]] += 1

    seen: Counter[str] = Counter()
    samples: list[dict[str, T.Any]] = []
    skipped: list[dict[str, str]] = []
    for points98, bbox, attrs, image_rel in track(
        rows, desc="Build wflw", total=len(rows), unit="sample"
    ):
        seen[image_rel] += 1
        base_id = Path(image_rel).with_suffix("").as_posix()
        sample_id = (
            base_id
            if counts[image_rel] <= 1
            else f"{base_id}#face-{seen[image_rel]:02d}"
        )
        conds = tuple(name for name in WFLW_ATTRIBUTE_NAMES if attrs.get(name)) or (
            _label(scenario),
        )
        image_path = (image_base / image_rel).resolve()
        if not image_path.is_file():
            skipped.append(
                {"sample_id": sample_id, "reason": f"image not found: {image_path}"}
            )
            continue
        points98 = normalize_landmark_array(points98, schema="2d_98")
        crop_image_path, crop_points68, crop_metadata = _crop_sample_image(
            output_dir=output_dir,
            dataset="wflw",
            sample_id=sample_id,
            image_path=image_path,
            points68=points98,
            bbox_xyxy=bbox,
            bbox_source="wflw_rect_attr_bbox",
            pad_ratio=0.25,
        )
        split = _deterministic_split("wflw", sample_id)
        metadata = {
            "bbox": bbox,
            "attributes": attrs,
            "image_id": image_rel,
            "split": split,
        }
        metadata.update(crop_metadata)
        samples.append(
            _with_split(
                _sample(
                    output_dir=output_dir,
                    dataset="wflw",
                    sample_id=sample_id,
                    image=crop_image_path,
                    points68=crop_points68,
                    condition=conds[0],
                    conditions=tuple(
                        dict.fromkeys(
                            (*(_label(item) for item in conds), f"{split}set")
                        )
                    ),
                    source_schema="2d_98",
                    source_id=sample_id,
                    metadata=metadata,
                ),
                split,
            )
        )
    if not samples:
        raise ValueError(f"no WFLW samples built; skipped={skipped[:5]}")
    return _write_manifest(
        output_dir,
        "wflw",
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
