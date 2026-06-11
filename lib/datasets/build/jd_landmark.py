"""jd_landmark dataset builder (split from build_quality_dataset)."""

# ruff: noqa: E402, F403, F405
from __future__ import annotations

from lib.datasets.build.core import *  # noqa: F403

JD_TRAINING_SUBSETS = ("AFW", "HELEN", "IBUG", "LFPW")


def _jd_base_subset(image_name: str) -> str | None:
    prefix = Path(image_name).stem.split("_", 1)[0].lower()
    return prefix if prefix in {"afw", "helen", "lfpw", "ibug"} else None


def _jd_bbox_dirs(root: Path) -> tuple[Path, ...]:
    candidates = [
        root / "Test_data1" / "rect",
        root
        / "training_dataset_face_detection_bounding_box_v1"
        / "training_dataset_face_detection_bounding_box",
    ]
    candidates.extend(
        path
        for path in root.rglob("training_dataset_face_detection_bounding_box")
        if path.is_dir() and "__MACOSX" not in path.parts
    )
    out: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        if not candidate.is_dir():
            continue
        resolved = candidate.resolve()
        if resolved not in seen:
            seen.add(resolved)
            out.append(candidate)
    return tuple(out)


def _jd_training_roots(root: Path) -> list[Path]:
    candidates = [root, root / "Training_data", root / "Training_data.zip"]
    out: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        if not candidate.is_dir():
            continue
        if not any(
            (candidate / subset / "landmark").is_dir() for subset in JD_TRAINING_SUBSETS
        ):
            continue
        resolved = candidate.resolve()
        if resolved not in seen:
            seen.add(resolved)
            out.append(candidate)
    return out


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

    for training_root in _jd_training_roots(root):
        for subset in JD_TRAINING_SUBSETS:
            landmark_dir = training_root / subset / "landmark"
            if landmark_dir.is_dir():
                out.append(
                    (
                        landmark_dir,
                        training_root / subset / "picture",
                        None,
                        "train",
                        "training_data",
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


def _jd_resolve_loader_geometry_path(
    *,
    output_dir: Path,
    sample_id: str,
    image: Path,
    points: np.ndarray,
    bbox: T.Sequence[float] | None,
    bbox_source: str,
    corrected_annotation: bool = False,
) -> tuple[Path, np.ndarray, dict[str, T.Any]]:
    """Choose a JD sample geometry path.

    Policy:
    A. native image + raw landmarks fit the loader -> keep native path
    B. otherwise bbox crop/remap fits the loader -> use crop/remapped landmarks
    C. otherwise raise, and the builder skips the sample
    """

    native = _simulate_loader_geometry(image, points)
    # The loader's 2048 crash guard alone is too lax for path selection: a
    # wrong-coordinate-frame annotation (e.g. corrected landmarks 150px outside
    # a 240px-tall image) pads to ~590px and trains silently. The suspicious
    # threshold sends such samples through the bbox-crop fallback or, failing
    # that, quarantines them with a review overlay.
    native_fits = bool(native.get("ok")) and not native.get("suspicious")
    meta: dict[str, T.Any] = {
        "loader_geometry_policy": "native_fit" if native_fits else "invalid",
        "loader_geometry_native": native,
    }
    if native_fits:
        return image, points.astype(np.float32), meta

    def _quarantine(reason: str) -> ValueError:
        overlay = _write_geometry_review_overlay(
            output_dir,
            dataset="jd-landmark",
            sample_id=sample_id,
            image_path=image,
            points=points,
            source_image_hw=None,
            diag=native,
        )
        message = f"invalid loader geometry: {reason}"
        if overlay is not None:
            message += f"; review overlay: {overlay}"
        return ValueError(message)

    if corrected_annotation and not native_fits:
        raise _quarantine(
            f"native={native.get('reason')}; corrected annotation is not "
            "eligible for bbox fallback"
        )

    if bbox is None:
        raise _quarantine(f"native={native.get('reason')}; no bbox fallback")

    try:
        crop_image, crop_points, crop_meta = _crop_sample_image(
            output_dir=output_dir,
            dataset="jd-landmark",
            sample_id=sample_id,
            image_path=image,
            points68=points,
            bbox_xyxy=bbox,
            bbox_source=bbox_source,
        )
    except Exception as err:  # noqa: BLE001
        raise _quarantine(
            f"native={native.get('reason')}; bbox crop failed: {err}"
        ) from err

    crop_diag = _simulate_loader_geometry(crop_image, crop_points)
    meta["loader_geometry_bbox_crop"] = crop_diag
    if not crop_diag.get("ok") or crop_diag.get("suspicious"):
        with contextlib.suppress(OSError):
            crop_image.unlink()
        raise _quarantine(
            f"native={native.get('reason')}; bbox_crop={crop_diag.get('reason')}"
        )

    meta.update(crop_meta)
    meta["loader_geometry_policy"] = "bbox_crop_fit"
    return crop_image, crop_points.astype(np.float32), meta


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
    training_landmark_names = {
        path.name
        for landmark_dir, _, _, _, source_name in sources
        if source_name == "training_data"
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
    samples: list[dict[str, T.Any]] = []
    skipped: list[dict[str, str]] = []
    for landmark_dir, image_dir, bbox_dir, source_split, source_name in sources:
        source_image_index = (
            _build_image_index(image_dir)
            if image_dir is not None and image_dir.is_dir()
            else None
        )
        jd_landmark_files = sorted(landmark_dir.glob("*.txt"))
        for landmark_path in track(
            jd_landmark_files,
            desc=f"Build jd-landmark ({source_name})",
            total=len(jd_landmark_files),
            unit="file",
        ):
            if source_name == "corrected_landmark":
                superseded_by = (
                    "test_data1"
                    if landmark_path.name in test_data1_landmark_names
                    else "training_data"
                    if landmark_path.name in training_landmark_names
                    else None
                )
                if superseded_by is not None:
                    skipped.append(
                        {
                            "sample_id": landmark_path.as_posix(),
                            "reason": f"superseded by {superseded_by} corrected override",
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
                if image_dir is None:
                    raise FileNotFoundError(
                        f"JD-landmark {source_name} ships no picture directory for "
                        f"{image_name}; corrected landmarks only apply as overrides "
                        "to Training_data/Test_data1 samples"
                    )
                image = _resolve_unique_image(
                    (image_dir,),
                    image_name,
                    context=f"JD-landmark {source_name}",
                    image_index=source_image_index,
                )
                image_source = f"{source_name}_picture"
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
                    "base_subset": _jd_base_subset(image_name),
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
            try:
                selected_image, selected_points, geometry_metadata = (
                    _jd_resolve_loader_geometry_path(
                        output_dir=output_dir,
                        sample_id=sample_id,
                        image=image,
                        points=points,
                        bbox=bbox,
                        bbox_source=str(bbox_path.resolve())
                        if bbox_path is not None
                        else "jd_bbox",
                        corrected_annotation=corrected_path is not None,
                    )
                )
            except ValueError as err:
                skipped.append(
                    {"sample_id": landmark_path.as_posix(), "reason": str(err)}
                )
                continue

            metadata.update(geometry_metadata)
            samples.append(
                _with_split(
                    _sample(
                        output_dir=output_dir,
                        dataset="jd-landmark",
                        sample_id=sample_id,
                        image=selected_image,
                        points68=selected_points,
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
