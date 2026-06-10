"""merl_rav dataset builder (split from build_quality_dataset)."""

# ruff: noqa: E402, F403, F405
from __future__ import annotations

from lib.datasets.build.core import *  # noqa: F403


def _merl_rav_landmark_files(root: Path) -> list[Path]:
    return sorted(
        path
        for path in root.rglob("*.pts")
        if path.is_file() and not path.name.startswith(".")
    )


def _merl_rav_image_identity(sample_stem: str) -> tuple[str, int | None]:
    """Return the base AFLW image id and optional MERL-RAV face index.

    MERL-RAV labels can be named like image00070_2.pts, where the suffix is a
    face/annotation index in the same AFLW image. AFLW image lookup should use
    image00070, while sample ids should keep image00070_2.
    """

    stem = str(sample_stem)
    base, sep, tail = stem.rpartition("_")
    if sep and tail.isdigit() and re.fullmatch(r"image\d+", base, flags=re.IGNORECASE):
        return base, int(tail)
    return stem, None


def _merl_rav_image_name_candidates(sample_stem: str) -> tuple[str, ...]:
    image_id, _ = _merl_rav_image_identity(sample_stem)
    return tuple(dict.fromkeys((sample_stem, image_id)))


def _merl_rav_landmark_files(root: Path) -> list[Path]:
    return sorted(
        path
        for path in root.rglob("*.pts")
        if path.is_file() and not path.name.startswith(".")
    )


def _parse_merl_rav_pts_signed(path: Path) -> np.ndarray:
    """Parse a MERL-RAV ``.pts`` file preserving signed coordinates.

    MERL-RAV signed-coordinate semantics:
      * positive ``x y``: visible landmark
      * negative ``-x -y``: externally occluded landmark estimated at ``abs(x), abs(y)``
      * ``-1 -1``: self-occluded landmark with no usable coordinate
    """

    rows: list[tuple[float, float]] = []
    in_block = False
    saw_brace = False
    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8", errors="ignore").splitlines(), start=1
    ):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line == "{":
            in_block = True
            saw_brace = True
            continue
        if line == "}":
            break
        if saw_brace and not in_block:
            continue
        if ":" in line and not re.match(r"^[+-]?\d", line):
            continue
        if not in_block and any(
            line.lower().startswith(prefix) for prefix in ("version", "n_points")
        ):
            continue

        parts = line.replace(",", " ").split()
        if len(parts) < 2:
            continue
        try:
            rows.append((float(parts[0]), float(parts[1])))
        except ValueError as err:
            raise ValueError(
                f"invalid MERL-RAV .pts row {line_number} in {path}: {line}"
            ) from err

    if len(rows) != 68:
        raise ValueError(
            f"MERL-RAV .pts file must contain 68 points, got {len(rows)}: {path}"
        )
    return np.asarray(rows, dtype=np.float32)


def _merl_rav_decode_signed_points(
    signed_xy: np.ndarray,
    *,
    image_hw: tuple[int, int],
) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Decode MERL-RAV signed landmarks into safe coordinates and masks.

    Returns:
      labels:
        ``visible``, ``externally_occluded``, or ``self_occluded`` per point.
      safe_points:
        Coordinates to save for training. Invalid/out-of-image points are set to 0.
      source_valid:
        Points with a coordinate estimate in MERL-RAV source space.
      in_image:
        Source-valid points that lie inside the matched AFLW image.
      coordinate_valid:
        Points usable for heatmap/coordinate loss. Currently same as ``in_image``.
      score_visible:
        Points visible/scorable by default. Externally/self occluded are false.
    """

    arr = np.asarray(signed_xy, dtype=np.float32)
    if arr.shape != (68, 2):
        raise ValueError(
            f"MERL-RAV signed landmarks must have shape (68, 2), got {arr.shape}"
        )

    height, width = int(image_hw[0]), int(image_hw[1])
    if height <= 0 or width <= 0:
        raise ValueError(f"invalid image size for MERL-RAV decode: {image_hw}")

    labels: list[str] = []
    decoded = np.zeros((68, 2), dtype=np.float32)
    source_valid = np.zeros((68,), dtype=bool)
    score_visible = np.zeros((68,), dtype=bool)

    for idx, (x_value, y_value) in enumerate(arr):
        x = float(x_value)
        y = float(y_value)

        if not np.isfinite([x, y]).all():
            labels.append("invalid")
            continue

        if x == -1.0 and y == -1.0:
            labels.append("self_occluded")
            continue

        if x < 0.0 or y < 0.0:
            labels.append("externally_occluded")
            decoded[idx] = (abs(x), abs(y))
            source_valid[idx] = True
            score_visible[idx] = False
            continue

        labels.append("visible")
        decoded[idx] = (x, y)
        source_valid[idx] = True
        score_visible[idx] = True

    finite = np.isfinite(decoded).all(axis=1)
    in_image = (
        source_valid
        & finite
        & (decoded[:, 0] >= 0.0)
        & (decoded[:, 0] < float(width))
        & (decoded[:, 1] >= 0.0)
        & (decoded[:, 1] < float(height))
    )
    coordinate_valid = in_image.copy()
    score_visible = score_visible & coordinate_valid

    safe_points = decoded.astype(np.float32, copy=True)
    safe_points[~coordinate_valid] = 0.0

    if not coordinate_valid.any():
        raise ValueError(
            "MERL-RAV sample has no coordinate-valid in-image landmarks after "
            "signed-coordinate decoding"
        )

    return labels, safe_points, source_valid, in_image, coordinate_valid, score_visible


def _merl_rav_conditions(
    scenario: str,
    split: str,
    visibility: tuple[bool, ...] | None,
) -> tuple[str, tuple[str, ...]]:
    base_condition, base_conditions = _native_conditions_for_split(scenario, split)
    extra: list[str] = []
    if visibility is not None:
        if any(not bool(value) for value in visibility):
            extra.append("occlusion")
        else:
            extra.append("clean")
    conditions = tuple(dict.fromkeys((*extra, *base_conditions)))
    return conditions[0] if conditions else base_condition, conditions


def _merl_rav_path_conditions(landmark_path: Path) -> tuple[str, ...]:
    labels = [_label(part) for part in landmark_path.parts]
    out: list[str] = []

    for token in (
        "frontal",
        "profile",
        "semiprofile",
        "semi_profile",
        "occlusion",
        "occluded",
        "expression",
        "illumination",
        "blur",
    ):
        if token in labels and token not in out:
            out.append("occlusion" if token == "occluded" else token)

    return tuple(out)


def _merge_condition_labels(*groups: tuple[str, ...]) -> tuple[str, ...]:
    out: list[str] = []
    for group in groups:
        for item in group:
            label = _label(item)
            if label and label not in out:
                out.append(label)
    return tuple(out or ("default",))


def _build_merl_rav(
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
    landmark_paths = _merl_rav_landmark_files(root)
    if not landmark_paths:
        raise ValueError(f"no MERL-RAV .pts files found below {root}")

    image_roots = (Path(image_root),) if image_root else (root,)
    image_index = _build_combined_image_index(image_roots)

    samples: list[dict[str, T.Any]] = []
    skipped: list[dict[str, str]] = []
    for landmark_path in track(
        landmark_paths,
        desc="Build merl-rav",
        total=len(landmark_paths),
        unit="file",
    ):
        sample_stem = landmark_path.stem
        image_id, face_index = _merl_rav_image_identity(sample_stem)
        sample_id = f"merl-rav/{sample_stem}"

        try:
            signed_points = _parse_merl_rav_pts_signed(landmark_path)

            image = None
            for candidate_name in _merl_rav_image_name_candidates(sample_stem):
                image = _find_named_image(
                    image_roots,
                    candidate_name,
                    image_index=image_index,
                )
                if image is not None:
                    break
            if image is None:
                raise FileNotFoundError(
                    f"AFLW image not found for {sample_stem} "
                    f"(tried base image id {image_id})"
                )

            image_bgr = cv2.imread(str(image), cv2.IMREAD_COLOR)
            if image_bgr is None:
                raise FileNotFoundError(f"could not read AFLW image: {image}")
            image_hw = image_bgr.shape[:2]

            (
                visibility_labels,
                points,
                source_valid,
                in_image,
                coordinate_valid,
                score_visible,
            ) = _merl_rav_decode_signed_points(signed_points, image_hw=image_hw)

            detected_schema = "2d_68"
            points = normalize_landmark_array(points, schema=detected_schema)

            bbox_points = points[coordinate_valid]
            face_bbox = _bbox_from_points_xyxy(bbox_points)
            normalizer = _normalizer_from_bbox(face_bbox)

        except Exception as err:  # noqa: BLE001
            skipped.append({"sample_id": sample_id, "reason": str(err)})
            continue

        # Split by source image, not face index, so multiple faces from the same
        # AFLW image cannot leak across train/test.
        split = _deterministic_split("merl-rav", image_id)

        score_visibility_tuple = tuple(bool(value) for value in score_visible.tolist())
        condition, conds = _merl_rav_conditions(
            scenario,
            split,
            score_visibility_tuple,
        )
        path_conds = _merl_rav_path_conditions(landmark_path)
        conds = _merge_condition_labels(path_conds, conds)
        condition = conds[0]

        metadata = _path_identity_metadata(landmark_path, root=root, dataset="merl-rav")
        metadata.update(
            {
                "dataset_parser": "merl_rav_native_aflw_signed_pts",
                "parser_type": "dataset_specific",
                "source_schema": detected_schema,
                "source_image": str(image.resolve()),
                "source_image_name": image.name,
                "source_condition": path_conds[0] if path_conds else None,
                "source_conditions": list(path_conds),
                "image_id": image_id,
                "merl_rav_label_id": sample_stem,
                "merl_rav_visibility_labels": visibility_labels,
                "visibility_target_source": "merl_rav_signed_coordinates_score_visible",
                "coordinate_validity_source": "merl_rav_signed_coordinates_in_image",
                "landmark_source_valid_mask": [
                    bool(value) for value in source_valid.tolist()
                ],
                "landmark_in_image_mask": [bool(value) for value in in_image.tolist()],
                "landmark_coordinate_valid_mask": [
                    bool(value) for value in coordinate_valid.tolist()
                ],
                "landmark_score_visibility_mask": [
                    bool(value) for value in score_visible.tolist()
                ],
                "landmark_source_valid_count": int(source_valid.sum()),
                "landmark_in_image_count": int(in_image.sum()),
                "landmark_coordinate_valid_count": int(coordinate_valid.sum()),
                "landmark_score_visible_count": int(score_visible.sum()),
                "visible_landmark_count": int(score_visible.sum()),
                "occluded_landmark_count": int(
                    (coordinate_valid & ~score_visible).sum()
                ),
                "self_occluded_landmark_count": int(
                    sum(label == "self_occluded" for label in visibility_labels)
                ),
                "externally_occluded_landmark_count": int(
                    sum(label == "externally_occluded" for label in visibility_labels)
                ),
                "face_bbox": face_bbox,
                "face_bbox_source": "merl_rav_coordinate_valid_landmark_bbox",
                "normalizer_source": "merl_rav_coordinate_valid_landmark_bbox_max_side",
                "aflw_image_source": "aflw_native",
            }
        )
        if face_index is not None:
            metadata["face_index"] = face_index

        samples.append(
            _with_split(
                _sample(
                    output_dir=output_dir,
                    dataset="merl-rav",
                    sample_id=sample_id,
                    image=image,
                    points68=points,
                    condition=condition,
                    conditions=conds,
                    source_schema=detected_schema,
                    source_id=sample_id,
                    metadata=metadata,
                    visibility=score_visibility_tuple,
                    normalizer=normalizer,
                ),
                split,
            )
        )
        if _limit_reached_for_build(samples, scenarios, limit):
            break

    if not samples:
        raise ValueError(f"no MERL-RAV samples built; skipped={skipped[:10]}")

    return _write_manifest(
        output_dir,
        "merl-rav",
        scenario,
        _filter(samples, scenarios, limit),
        mode=mode,
        # MERL-RAV can contain multiple face labels for one AFLW image. These are
        # distinct samples, so do not dedupe by shared image path.
        allow_overlap=True,
        scenarios=scenarios,
        skipped=skipped,
    )


# Re-export every module-level name (including the single-underscore build
# helpers) so `from lib.datasets.build.<mod> import *` resolves bare-name
# calls in sibling modules exactly as they did in the original flat module.
__all__ = [_n for _n in dict(globals()) if not _n.startswith("__")]
