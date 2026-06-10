"""ffl dataset builder (split from build_quality_dataset)."""

# ruff: noqa: E402, F403, F405
from __future__ import annotations

from lib.datasets.build.core import *  # noqa: F403


def _ffl_split_dirs(root: Path, dataset: str) -> list[tuple[Path, str]]:
    if dataset == "fll2":
        candidates = [(root / "train", "train"), (root, "train")]
    else:
        base_candidates = [root / "FLL3_dataset", root]
        candidates = []
        for base in base_candidates:
            candidates.extend(
                (base / split, split) for split in ("train", "val", "test")
            )
    out: list[tuple[Path, str]] = []
    seen: set[Path] = set()
    for split_dir, split in candidates:
        if not (split_dir / "landmark").is_dir():
            continue
        resolved = split_dir.resolve()
        if resolved not in seen:
            seen.add(resolved)
            out.append((split_dir, split))
    return out


def _build_ffl_family(
    root: Path,
    output_dir: Path,
    *,
    dataset: str,
    scenario: str,
    scenarios: tuple[str, ...] | None,
    limit: int | None,
    mode: str,
    allow_overlap: bool,
    image_root: str | None,
) -> Path:
    split_dirs = _ffl_split_dirs(root, dataset)
    if not split_dirs:
        return _build_expected_schema_dataset(
            root,
            output_dir,
            dataset=dataset,
            expected_schema="2d_106",
            parser_name=f"{dataset}_106",
            scenario=scenario,
            scenarios=scenarios,
            limit=limit,
            mode=mode,
            allow_overlap=allow_overlap,
            image_root=image_root,
        )

    samples: list[dict[str, T.Any]] = []
    skipped: list[dict[str, str]] = []
    for split_dir, source_split in split_dirs:
        landmark_dir = split_dir / "landmark"
        image_dir = split_dir / (
            "picture_mask" if (split_dir / "picture_mask").is_dir() else "picture"
        )
        bbox_dir = split_dir / "bbox"
        ffl_landmark_files = sorted(landmark_dir.glob("*.txt"))
        for landmark_path in track(
            ffl_landmark_files,
            desc=f"Build {dataset} ({source_split})",
            total=len(ffl_landmark_files),
            unit="file",
        ):
            try:
                points, detected_schema = _load_landmark_file(landmark_path)
                if detected_schema != "2d_106":
                    raise ValueError(
                        f"{dataset} expected 2d_106, got {detected_schema}"
                    )
                roots = [image_dir]
                if image_root:
                    roots.insert(0, Path(image_root))
                image = _find_named_image(roots, landmark_path.stem)
                if image is None:
                    raise FileNotFoundError(
                        f"{dataset} image not found for {landmark_path.name}"
                    )
            except Exception as err:  # noqa: BLE001
                skipped.append(
                    {"sample_id": landmark_path.as_posix(), "reason": str(err)}
                )
                continue

            bbox_path = _bbox_file_for_landmark(bbox_dir, landmark_path)
            bbox = _read_bbox_file(bbox_path)
            split = _manifest_split_for_source_split(source_split)
            condition, conds = _native_conditions_for_split(scenario, split)
            sample_id = f"{source_split}/{landmark_path.stem}"
            metadata = _path_identity_metadata(
                landmark_path, root=root, dataset=dataset
            )
            metadata.update(
                {
                    "dataset_parser": f"{dataset}_release_106",
                    "parser_type": "dataset_specific",
                    "source_split": source_split,
                    "source_schema": "2d_106",
                    "source_image": str(image.resolve()),
                }
            )
            if bbox_path is not None and bbox is not None:
                metadata["source_bbox"] = str(bbox_path.resolve())
                metadata["bbox_xyxy"] = bbox
            samples.append(
                _with_split(
                    _sample(
                        output_dir=output_dir,
                        dataset=dataset,
                        sample_id=sample_id,
                        image=image,
                        points68=points,
                        condition=condition,
                        conditions=conds,
                        source_schema="2d_106",
                        source_id=sample_id,
                        metadata=metadata,
                    ),
                    split,
                )
            )

    if not samples:
        raise ValueError(
            f"no {dataset} native release samples built; skipped={skipped[:10]}"
        )
    return _write_manifest(
        output_dir,
        dataset,
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
