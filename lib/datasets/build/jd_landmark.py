"""jd_landmark dataset builder (split from build_quality_dataset)."""

# ruff: noqa: E402, F403, F405
from __future__ import annotations

from lib.datasets.build.core import *  # noqa: F403
from lib.datasets.build.w300 import *  # noqa: F403


def _jd_landmark_sources(
    root: Path,
) -> list[tuple[Path, Path | None, Path | None, str, str]]:
    out: list[tuple[Path, Path | None, Path | None, str, str]] = []
    test_roots = [root / "Test_data1"]
    if root.name == "Test_data1":
        test_roots.insert(0, root)
    for test_root in test_roots:
        landmark_dir = test_root / "landmark"
        if landmark_dir.is_dir():
            out.append(
                (
                    landmark_dir,
                    test_root / "picture",
                    test_root / "rect",
                    "test",
                    "test_data1",
                )
            )

    corrected_roots = [root / "Corrected_landmark"]
    if root.name == "Corrected_landmark":
        corrected_roots.insert(0, root)
    for corrected_root in corrected_roots:
        if corrected_root.is_dir():
            out.append((corrected_root, None, None, "corrected", "corrected_landmark"))
    return out


def _split_hint_from_jd_name(name: str) -> str | None:
    lowered = name.lower()
    if "image_train" in lowered or "_train_" in lowered:
        return "train"
    if "image_test" in lowered or "_test_" in lowered:
        return "test"
    return None


def _build_jd_landmark(
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
    sources = _jd_landmark_sources(root)
    if not sources:
        return _build_expected_schema_dataset(
            root,
            output_dir,
            dataset="jd-landmark",
            expected_schema="2d_106",
            parser_name="jd_landmark_106",
            scenario=scenario,
            scenarios=scenarios,
            limit=limit,
            mode=mode,
            allow_overlap=allow_overlap,
            image_root=image_root,
        )

    test_data1_landmark_names = {
        path.name
        for landmark_dir, _, _, _, source_name in sources
        if source_name == "test_data1"
        for path in landmark_dir.glob("*.txt")
    }
    corrected_by_name = {
        path.name: path
        for corrected_root in (
            root / "Corrected_landmark",
            root if root.name == "Corrected_landmark" else root / "__missing__",
        )
        if corrected_root.is_dir()
        for path in corrected_root.glob("*.txt")
    }
    global_bbox_dirs = _jd_bbox_dirs(root)
    jd_image_index = _build_combined_image_index(
        _candidate_300w_cache_roots(root, image_root)
    )
    samples: list[dict[str, T.Any]] = []
    skipped: list[dict[str, str]] = []
    for landmark_dir, image_dir, bbox_dir, source_split, source_name in sources:
        jd_landmark_files = sorted(landmark_dir.glob("*.txt"))
        for landmark_path in track(
            jd_landmark_files,
            desc=f"Build jd-landmark ({source_name})",
            total=len(jd_landmark_files),
            unit="file",
        ):
            if (
                source_name == "corrected_landmark"
                and landmark_path.name in test_data1_landmark_names
            ):
                skipped.append(
                    {
                        "sample_id": landmark_path.as_posix(),
                        "reason": "superseded by test_data1 corrected override",
                    }
                )
                continue
            image_name = _image_name_from_landmark_name(landmark_path)
            corrected_path = corrected_by_name.get(landmark_path.name)
            annotation_path = (
                corrected_path if corrected_path is not None else landmark_path
            )
            try:
                points, detected_schema = _load_landmark_file(annotation_path)
                if detected_schema != "2d_106":
                    raise ValueError(
                        f"JD-landmark expected 2d_106, got {detected_schema}"
                    )
                try:
                    image = _resolve_jd_300w_image(
                        root, image_root, image_name, image_index=jd_image_index
                    )
                    image_source = "300w_cache"
                except FileNotFoundError:
                    if image_dir is None:
                        raise
                    image = _resolve_unique_image(
                        (image_dir,), image_name, context="JD-landmark Test_data1"
                    )
                    image_source = "test_data1_picture"
            except Exception as err:  # noqa: BLE001
                skipped.append(
                    {"sample_id": landmark_path.as_posix(), "reason": str(err)}
                )
                continue

            bbox_dirs = tuple(
                path for path in (bbox_dir, *global_bbox_dirs) if path is not None
            )
            bbox_path = None
            for candidate_bbox_dir in bbox_dirs:
                bbox_path = _bbox_file_for_landmark(
                    candidate_bbox_dir, landmark_path, image_name=image_name
                )
                if bbox_path is not None:
                    break
            bbox = _read_bbox_file(bbox_path)
            sample_id = f"{source_name}/{Path(image_name).stem}"
            metadata = _path_identity_metadata(
                landmark_path, root=root, dataset="jd-landmark"
            )
            metadata.update(
                {
                    "dataset_parser": "jd_landmark_release_106",
                    "parser_type": "dataset_specific",
                    "source_release": source_name,
                    "source_split": source_split,
                    "source_schema": "2d_106",
                    "source_annotation": str(landmark_path.resolve()),
                    "source_image_name": image_name,
                    "source_image": str(image.resolve()),
                    "resolved_image_source": image_source,
                    "resolved_300w_image_path": str(image.resolve())
                    if image_source == "300w_cache"
                    else None,
                    "base_subset": _jd_300w_base_subset(image_name),
                }
            )
            if corrected_path is not None:
                metadata["corrected_annotation"] = str(corrected_path.resolve())
                metadata["source_landmarks"] = str(corrected_path.resolve())
                metadata["corrected_annotation_applied"] = True
                metadata["corrected_annotation_source_release"] = "corrected_landmark"
            if bbox_path is not None and bbox is not None:
                metadata["source_bbox"] = str(bbox_path.resolve())
                metadata["bbox_xyxy"] = bbox

            split_hint = (
                "test"
                if source_split == "test"
                else _split_hint_from_jd_name(image_name)
            )
            split = _split_from_entry_or_identity(
                {"split": split_hint} if split_hint else {},
                metadata,
                dataset="jd-landmark",
                sample_id=sample_id,
            )
            condition, conds = _native_conditions_for_split(scenario, split)
            samples.append(
                _with_split(
                    _sample(
                        output_dir=output_dir,
                        dataset="jd-landmark",
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
        if limit and scenarios is None and len(samples) >= limit:
            break

    if not samples:
        raise ValueError(
            f"no JD-landmark native release samples built; skipped={skipped[:10]}"
        )
    return _write_manifest(
        output_dir,
        "jd-landmark",
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
