"""subject_session dataset builder (split from build_quality_dataset)."""

# ruff: noqa: E402, F403, F405
from __future__ import annotations

from lib.datasets.build.core import *  # noqa: F403


def _menpo_list_files(root: Path, dataset: str) -> list[Path]:
    names = {f"{dataset}_train.txt", f"{dataset}_test.txt", f"{dataset}_val.txt"}
    return sorted(path for path in root.rglob("*.txt") if path.name.lower() in names)


def _list_split_from_path(path: Path) -> str:
    lowered = path.stem.lower()
    if "train" in lowered:
        return "train"
    if "val" in lowered or "test" in lowered:
        return "test"
    return "train"


def _menpo_identity_from_image(dataset: str, image_name: str) -> dict[str, str]:
    stem = Path(image_name).stem
    metadata = {"image_id": stem}
    if dataset == "xm2vts":
        parts = stem.split("_")
        if parts:
            metadata["subject_id"] = parts[0]
        if len(parts) > 1:
            metadata["session_id"] = parts[1]
        if len(parts) > 2:
            metadata["capture_id"] = parts[2]
        return metadata
    match = re.match(r"^(?P<subject>\d+)(?P<session>[A-Za-z])(?P<capture>\d+)$", stem)
    if match:
        metadata["subject_id"] = match.group("subject")
        metadata["session_id"] = match.group("session")
        metadata["capture_id"] = match.group("capture")
    else:
        metadata["subject_id"] = stem
    return metadata


def _parse_menpo_list_line(
    line: str,
) -> tuple[str, list[float] | None, list[list[float]] | None, np.ndarray]:
    parts = line.split()
    if len(parts) < 2:
        raise ValueError("empty Menpo-style list row")
    image_name = parts[0]
    values = [float(item) for item in parts[1:]]
    bbox: list[float] | None = None
    coarse: list[list[float]] | None = None
    landmark_values = values
    if len(values) == 150:
        bbox = [float(item) for item in values[:4]]
        coarse = (
            np.asarray(values[4:14], dtype=np.float32)
            .reshape(5, 2)
            .astype(float)
            .tolist()
        )
        landmark_values = values[14:]
    points, detected_schema = _canonical_points(
        np.asarray(landmark_values, dtype=np.float32), source_schema="2d_68"
    )
    if detected_schema != "2d_68":
        raise ValueError(f"Menpo-style list expected 2d_68, got {detected_schema}")
    return image_name, bbox, coarse, points


def _build_subject_session_dataset(
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
    list_files = _menpo_list_files(root, dataset)
    if list_files:
        samples: list[dict[str, T.Any]] = []
        skipped: list[dict[str, str]] = []
        for list_path in list_files:
            source_split = _list_split_from_path(list_path)
            split = _manifest_split_for_source_split(source_split)
            list_lines = list_path.read_text(
                encoding="utf-8", errors="ignore"
            ).splitlines()
            for line_number, line in track(
                enumerate(list_lines, start=1),
                desc=f"Build {dataset} ({source_split})",
                total=len(list_lines),
                unit="line",
            ):
                if not line.strip() or line.lstrip().startswith("#"):
                    continue
                try:
                    image_name, bbox, coarse, points = _parse_menpo_list_line(line)
                    roots = [list_path.parent, root]
                    if image_root:
                        roots.insert(0, Path(image_root))
                    image = _find_named_image(roots, image_name)
                    if image is None:
                        raise FileNotFoundError(
                            f"{dataset} image not found: {image_name}"
                        )
                except Exception as err:  # noqa: BLE001
                    skipped.append(
                        {"sample_id": f"{list_path}:{line_number}", "reason": str(err)}
                    )
                    continue

                sample_id = f"{list_path.stem}/{Path(image_name).stem}"
                condition, conds = _native_conditions_for_split(scenario, split)
                metadata: dict[str, T.Any] = {
                    "dataset": dataset,
                    "dataset_parser": f"{dataset}_menpo_list_68",
                    "parser_type": "dataset_specific",
                    "source_annotation": str(list_path.resolve()),
                    "source_line": line_number,
                    "source_split": source_split,
                    "source_schema": "2d_68",
                    "source_image_name": image_name,
                    "source_image": str(image.resolve()),
                    **_menpo_identity_from_image(dataset, image_name),
                }
                if bbox is not None:
                    metadata["bbox_xyxy"] = bbox
                if coarse is not None:
                    metadata["five_point_landmarks"] = coarse
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
                            source_schema="2d_68",
                            source_id=sample_id,
                            metadata=metadata,
                        ),
                        split,
                    )
                )

        if not samples:
            raise ValueError(
                f"no {dataset} Menpo-style list samples built; skipped={skipped[:10]}"
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

    json_path = _json_source(root)
    if json_path is not None:
        manifest = _build_json(
            json_path,
            output_dir,
            dataset=dataset,
            scenario=scenario,
            scenarios=scenarios,
            limit=limit,
            mode=mode,
            allow_overlap=allow_overlap,
            image_root=image_root,
        )
        payload = read_json(manifest)
        for sample in payload.get("samples", []):
            if not isinstance(sample, dict):
                continue
            metadata = sample.setdefault("metadata", {})
            if isinstance(metadata, dict):
                metadata.setdefault("dataset_parser", f"{dataset}_menpo_style")
                metadata.setdefault("parser_type", "dataset_specific")
        write_json(manifest, payload)
        return manifest

    image_base = Path(image_root) if image_root else root
    image_index = _build_image_index(image_base)
    samples: list[dict[str, T.Any]] = []
    skipped: list[dict[str, str]] = []
    fallback_landmark_files = _landmark_paths(root)
    for landmark_path in track(
        fallback_landmark_files,
        desc=f"Build {dataset}",
        total=len(fallback_landmark_files),
        unit="file",
    ):
        try:
            points, source_schema = _load_landmark_file(landmark_path)
            image = _matching_image(
                landmark_path, root=image_base, image_index=image_index
            )
            if image is None:
                raise FileNotFoundError("matching image not found")
        except Exception as err:  # noqa: BLE001
            skipped.append({"sample_id": landmark_path.as_posix(), "reason": str(err)})
            continue
        rel = landmark_path.relative_to(root)
        sample_id = rel.with_suffix("").as_posix()
        condition, conds = _condition_for_landmark_file(dataset, rel, scenario)
        metadata = _path_identity_metadata(landmark_path, root=root, dataset=dataset)
        metadata.update(
            {
                "dataset_parser": f"{dataset}_menpo_style",
                "parser_type": "dataset_specific",
                "source_schema": source_schema,
            }
        )
        split = _split_from_entry_or_identity(
            {}, metadata, dataset=dataset, sample_id=sample_id
        )
        conds = tuple(dict.fromkeys((*conds, f"{split}set")))
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
                    source_schema=source_schema,
                    source_id=sample_id,
                    metadata=metadata,
                ),
                split,
            )
        )

    if not samples:
        raise ValueError(
            f"no {dataset} Menpo-style samples built; skipped={skipped[:10]}"
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
