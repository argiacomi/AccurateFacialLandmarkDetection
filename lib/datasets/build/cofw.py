"""cofw dataset builder (split from build_quality_dataset)."""

# ruff: noqa: E402, F403, F405
from __future__ import annotations

from lib.datasets.build.core import *  # noqa: F403


def _cofw68_original_mat_files(root: Path) -> list[tuple[Path, str]]:
    out: list[tuple[Path, str]] = []
    for path in sorted(root.rglob("*.mat"), key=lambda item: len(item.parts)):
        name = path.name.lower()
        if "cofw" not in name or "color" not in name:
            continue
        if "train" in name:
            out.append((path, "train"))
        elif "test" in name:
            out.append((path, "test"))
    return out


def _mat_first_key(payload: T.Mapping[str, T.Any], names: tuple[str, ...]) -> T.Any:
    lowered = {key.lower(): key for key in payload if not key.startswith("__")}
    for name in names:
        if name.lower() in lowered:
            return payload[lowered[name.lower()]]
    for key in payload:
        key_l = key.lower()
        if key.startswith("__"):
            continue
        if any(name.lower() in key_l for name in names):
            return payload[key]
    return None


def _cofw68_original_points_array(value: T.Any) -> list[np.ndarray]:
    if value is None:
        return []
    arr = np.asarray(value)
    if arr.dtype == object:
        out = []
        for item in arr.reshape(-1):
            try:
                points, _ = _canonical_points(item, source_schema="2d_29")
            except Exception:
                continue
            out.append(points)
        return out
    arr = np.asarray(value, dtype=np.float32)
    if arr.ndim == 3 and arr.shape[-2:] == (29, 2):
        return [arr[index].astype(np.float32) for index in range(arr.shape[0])]
    if arr.ndim == 3 and arr.shape[:2] == (29, 2):
        return [arr[:, :, index].astype(np.float32) for index in range(arr.shape[2])]
    if arr.ndim == 2:
        if arr.shape[0] == 87:
            return [
                np.stack((arr[:29, index], arr[29:58, index]), axis=1).astype(
                    np.float32
                )
                for index in range(arr.shape[1])
            ]
        if arr.shape[1] == 87:
            return [
                np.stack((arr[index, :29], arr[index, 29:58]), axis=1).astype(
                    np.float32
                )
                for index in range(arr.shape[0])
            ]
        if arr.shape[1] == 58:
            return [row.reshape(29, 2).astype(np.float32) for row in arr]
        if arr.shape[0] == 58:
            return [
                arr[:, index].reshape(29, 2).astype(np.float32)
                for index in range(arr.shape[1])
            ]
        if arr.shape == (29, 2):
            return [arr.astype(np.float32)]
    return []


def _cofw68_original_image_array(value: T.Any) -> list[np.ndarray]:
    if value is None:
        return []
    arr = np.asarray(value)
    if arr.dtype == object:
        return [np.asarray(item) for item in arr.reshape(-1)]
    if arr.ndim == 4:
        if arr.shape[-1] in (1, 3, 4):
            return [arr[index] for index in range(arr.shape[0])]
        if arr.shape[0] in (1, 3, 4):
            return [
                np.moveaxis(arr[:, :, :, index], 0, -1)
                for index in range(arr.shape[-1])
            ]
        return [arr[:, :, :, index] for index in range(arr.shape[-1])]
    if arr.ndim in (2, 3):
        return [arr]
    return []


def _cofw68_original_visibility(value: T.Any, count: int) -> list[list[bool]]:
    if value is None:
        return [[True] * 29 for _ in range(count)]
    arr = np.asarray(value)
    if arr.dtype == object:
        rows = [np.asarray(item).reshape(-1) for item in arr.reshape(-1)]
    elif arr.ndim == 2 and arr.shape[1] == 29:
        rows = [arr[index] for index in range(arr.shape[0])]
    elif arr.ndim == 2 and arr.shape[0] == 29:
        rows = [arr[:, index] for index in range(arr.shape[1])]
    elif arr.size == 29:
        rows = [arr.reshape(-1)]
    else:
        rows = []
    out: list[list[bool]] = []
    for row in rows[:count]:
        # cofw68 stores occlusion flags in common releases: 1 means occluded.
        out.append([not bool(item) for item in np.asarray(row).reshape(-1)[:29]])
    while len(out) < count:
        out.append([True] * 29)
    return out


def _write_cofw68_original_image(
    output_dir: Path, sample_id: str, image: np.ndarray
) -> Path:
    path = (
        output_dir / "images" / "cofw29" / f"{safe_id(sample_id).replace('/', '_')}.png"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(image)
    if arr.ndim == 3 and arr.shape[0] in (1, 3, 4) and arr.shape[-1] not in (1, 3, 4):
        arr = np.moveaxis(arr, 0, -1)
    if arr.ndim == 2:
        write_arr = arr
    else:
        if arr.shape[-1] == 1:
            arr = arr[:, :, 0]
            write_arr = arr
        else:
            write_arr = arr[:, :, [2, 1, 0]] if arr.shape[-1] >= 3 else arr
    write_arr = np.clip(write_arr, 0, 255).astype(np.uint8)
    ok = cv2.imwrite(str(path), write_arr)
    if not ok:
        raise OSError(f"failed to write cofw68 original image: {path}")
    return path


def _is_hdf5_mat(path: Path) -> bool:
    try:
        with path.open("rb") as handle:
            header = handle.read(128)
        return (
            header.startswith(b"\x89HDF\r\n\x1a\n") or b"MATLAB 7.3 MAT-file" in header
        )
    except OSError:
        return False


def _cofw68_original_hdf5_arrays(
    path: Path,
    declared_split: str,
) -> tuple[
    list[np.ndarray], list[np.ndarray], list[list[bool]], list[list[float] | None]
]:
    try:
        import h5py
    except ImportError as err:
        raise RuntimeError(
            "h5py is required to read cofw68 original MATLAB v7.3 files"
        ) from err

    with h5py.File(path, "r") as handle:
        trainish = declared_split == "train"
        phis_key = (
            "phisTr"
            if trainish and "phisTr" in handle
            else "phisT"
            if "phisT" in handle
            else "phisTr"
        )
        images_key = (
            "IsTr"
            if trainish and "IsTr" in handle
            else "IsT"
            if "IsT" in handle
            else "IsTr"
        )
        bboxes_key = (
            "bboxesTr"
            if trainish and "bboxesTr" in handle
            else "bboxesT"
            if "bboxesT" in handle
            else None
        )
        phis = np.asarray(handle[phis_key], dtype=np.float32)
        if phis.ndim != 2:
            raise ValueError(f"cofw68 original phis must be 2D, got {phis.shape}")
        if phis.shape[0] == 87:
            columns = [phis[:, index] for index in range(phis.shape[1])]
        elif phis.shape[1] == 87:
            columns = [phis[index, :] for index in range(phis.shape[0])]
        else:
            raise ValueError(
                f"cofw68 original phis must have 87 rows/columns, got {phis.shape}"
            )

        points_rows = [
            np.stack((column[:29], column[29:58]), axis=1).astype(np.float32)
            for column in columns
        ]
        visibility_rows = [
            [not bool(item) for item in np.asarray(column[58:87]).reshape(-1)[:29]]
            for column in columns
        ]

        images: list[np.ndarray] = []
        image_refs = handle[images_key]
        for index in range(len(points_rows)):
            ref = (
                image_refs[0, index]
                if image_refs.ndim == 2 and image_refs.shape[0] == 1
                else image_refs[index]
            )
            # Reorient to the annotation frame so 29-point landmarks/bboxes align
            # (the cofw6868 reader applies the same transpose).
            images.append(_orient_cofw68_hdf5_image(np.asarray(handle[ref])))

        bbox_rows: list[list[float] | None] = [None] * len(points_rows)
        if bboxes_key and bboxes_key in handle:
            bboxes = np.asarray(handle[bboxes_key], dtype=np.float32)
            if bboxes.ndim == 2 and bboxes.shape[0] == 4:
                bbox_rows = [
                    [float(value) for value in bboxes[:, index]]
                    for index in range(bboxes.shape[1])
                ]
            elif bboxes.ndim == 2 and bboxes.shape[1] == 4:
                bbox_rows = [
                    [float(value) for value in bboxes[index, :]]
                    for index in range(bboxes.shape[0])
                ]
            bbox_rows = (bbox_rows + [None] * len(points_rows))[: len(points_rows)]

    return points_rows, images, visibility_rows, bbox_rows


def _is_matlab_hdf_reader_error(err: Exception) -> bool:
    message = str(err)
    return "HDF reader" in message or "MATLAB 7.3" in message


def _build_cofw68_original(
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
    mat_files = _cofw68_original_mat_files(root)
    if not mat_files:
        return _build_expected_schema_dataset(
            root,
            output_dir,
            dataset="cofw29",
            expected_schema="2d_29",
            parser_name="cofw_original_29",
            scenario=scenario,
            scenarios=scenarios,
            limit=limit,
            mode=mode,
            allow_overlap=allow_overlap,
            image_root=image_root,
        )

    samples: list[dict[str, T.Any]] = []
    skipped: list[dict[str, str]] = []
    image_index = _build_combined_image_index(
        [Path(image_root) if image_root else root]
    )
    sio: T.Any | None = None
    for mat_path, declared_split in mat_files:
        try:
            if _is_hdf5_mat(mat_path):
                points_rows, images, visibility_rows, bbox_rows = (
                    _cofw68_original_hdf5_arrays(mat_path, declared_split)
                )
            else:
                if sio is None:
                    try:
                        import scipy.io as sio_module
                    except ImportError as err:
                        raise RuntimeError(
                            "scipy is required to read COFW original .mat files"
                        ) from err
                    sio = sio_module

                try:
                    payload = sio.loadmat(mat_path)
                except (NotImplementedError, ValueError) as err:
                    if not _is_matlab_hdf_reader_error(err):
                        raise
                    points_rows, images, visibility_rows, bbox_rows = (
                        _cofw68_original_hdf5_arrays(mat_path, declared_split)
                    )
                else:
                    points_rows = _cofw68_original_points_array(
                        _mat_first_key(
                            payload, ("phisTr", "phisT", "phis", "points", "landmarks")
                        )
                    )
                    images = _cofw68_original_image_array(
                        _mat_first_key(payload, ("IsTr", "IsT", "images", "image"))
                    )
                    visibility_rows = _cofw68_original_visibility(
                        _mat_first_key(
                            payload,
                            (
                                "occlusionsTr",
                                "occlusionsT",
                                "occlusion",
                                "occ",
                                "occluded",
                            ),
                        ),
                        len(points_rows),
                    )
                    bbox_rows = [None] * len(points_rows)
        except Exception as err:  # noqa: BLE001
            skipped.append({"sample_id": mat_path.as_posix(), "reason": str(err)})
            continue
        for index, points in track(
            enumerate(points_rows),
            desc="Build cofw29",
            total=len(points_rows),
            unit="sample",
        ):
            sample_id = f"cofw68_original/{declared_split}/{mat_path.stem}_{index:04d}"
            try:
                points29 = normalize_landmark_array(points, schema="2d_29")
                if index < len(images):
                    image_path = _write_cofw68_original_image(
                        output_dir, sample_id, images[index]
                    )
                else:
                    image = _matching_image(
                        mat_path,
                        root=Path(image_root) if image_root else root,
                        image_index=image_index,
                    )
                    if image is None:
                        raise FileNotFoundError(
                            "cofw68 original image not found in MAT or image root"
                        )
                    image_path = image
            except Exception as err:  # noqa: BLE001
                skipped.append({"sample_id": sample_id, "reason": str(err)})
                continue
            visibility = (
                visibility_rows[index] if index < len(visibility_rows) else [True] * 29
            )
            metadata = {
                "dataset": "cofw29",
                "dataset_parser": "cofw_original_29",
                "parser_type": "dataset_specific",
                "annotation_file": str(mat_path.resolve()),
                "source_schema": "2d_29",
                "split": declared_split,
                "cofw68_original_index": index,
                "occlusion_mask": [not bool(item) for item in visibility],
                "landmark_score_visibility_mask": visibility,
            }
            if index < len(bbox_rows) and bbox_rows[index] is not None:
                metadata["bbox_xyxy"] = bbox_rows[index]
            condition = (
                "occlusion" if any(not bool(item) for item in visibility) else scenario
            )
            samples.append(
                _with_split(
                    _sample(
                        output_dir=output_dir,
                        dataset="cofw29",
                        sample_id=sample_id,
                        image=image_path,
                        points68=points29,
                        condition=condition,
                        conditions=tuple(
                            dict.fromkeys((_label(condition), f"{declared_split}set"))
                        ),
                        source_schema="2d_29",
                        source_id=sample_id,
                        metadata=metadata,
                        visibility=visibility,
                    ),
                    declared_split,
                )
            )

    if not samples:
        raise ValueError(
            f"no cofw68 original 29-point samples built; skipped={skipped[:10]}"
        )
    return _write_manifest(
        output_dir,
        "cofw29",
        scenario,
        _filter(samples, scenarios, limit),
        mode=mode,
        allow_overlap=allow_overlap,
        scenarios=scenarios,
        skipped=skipped,
    )


def _cofw6868_annotation_paths(root: Path) -> list[Path]:
    return sorted(
        path
        for path in root.rglob("*_points.mat")
        if path.is_file() and "test_annotations" in path.as_posix()
    )


def _cofw68_test_color_mat(root: Path) -> Path:
    matches = sorted(
        root.rglob("COFW_test_color.mat"), key=lambda item: len(item.parts)
    )
    if not matches:
        raise FileNotFoundError(f"COFW_test_color.mat not found below {root}")
    return matches[0]


def _cofw68_test_bboxes(root: Path) -> np.ndarray | None:
    matches = sorted(
        root.rglob("cofw6868_test_bboxes.mat"), key=lambda item: len(item.parts)
    )
    if not matches:
        return None
    try:
        import scipy.io as sio

        payload = sio.loadmat(matches[0])
        boxes = np.asarray(payload.get("bboxes"), dtype=np.float32)
        return boxes if boxes.ndim == 2 and boxes.shape[1] == 4 else None
    except (ImportError, OSError, TypeError, ValueError, NotImplementedError):
        return None


def _cofw68_annotation_index(path: Path) -> int:
    text = path.stem.replace("_points", "")
    return int(text) - 1


def _cofw68_points_and_occ(
    path: Path,
) -> tuple[np.ndarray, list[bool], dict[str, T.Any]]:
    import scipy.io as sio

    payload = sio.loadmat(path)
    if "Points" not in payload:
        raise ValueError(f"cofw6868 annotation missing Points: {path}")
    points68, schema = _canonical_points(payload["Points"], source_schema="2d_68")

    occ_raw = payload.get("Occ")
    occ_mask: list[bool] = []
    visibility: list[bool] = []
    if occ_raw is not None:
        occ_arr = np.asarray(occ_raw).reshape(-1)
        occ_mask = [bool(x) for x in occ_arr[:68]]
        visibility = [not bool(x) for x in occ_arr[:68]]
    if len(visibility) != 68:
        visibility = [True] * 68

    metadata = {
        "source_schema": schema,
        "occlusion_mask": occ_mask,
        "landmark_score_visibility_mask": visibility,
    }
    return points68, visibility, metadata


def _orient_cofw68_hdf5_image(arr: np.ndarray) -> np.ndarray:
    """Normalize a cofw68 HDF5 image plane to the annotation coordinate frame.

    cofw68 MATLAB v7.3 (HDF5) stores image planes channel-first and with H/W
    swapped relative to the landmark/bbox frame. Points and bboxes only align
    once the channels are moved last and the spatial axes are transposed. This
    applies to every cofw68 HDF5 image, so both the cofw6868 and cofw29
    readers must use it.
    """
    arr = np.asarray(arr)
    # cofw68 HDF5 images are usually channel-first: C,H,W.
    if arr.ndim == 3 and arr.shape[0] in (1, 3, 4):
        arr = np.moveaxis(arr, 0, -1)
    if arr.ndim == 3:
        arr = np.transpose(arr, (1, 0, 2))
    elif arr.ndim == 2:
        arr = arr.T
    return arr


def _cofw68_hdf5_image_by_index(mat_path: Path, index: int) -> np.ndarray:
    import h5py

    with h5py.File(mat_path, "r") as h5:
        refs = h5["IsT"][()]
        ref = refs.reshape(-1)[index]
        arr = np.asarray(h5[ref])

    arr = _orient_cofw68_hdf5_image(arr)

    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr[:, :, 0]
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(arr)


def _write_cofw68_image(output_dir: Path, index: int, image: np.ndarray) -> Path:
    from PIL import Image

    # This is an intermediate full-resolution decode used only as crop input.
    # Keep it out of output_dir/images so the prepared image tree contains only
    # final manifest images such as images/cofw68/*.jpg.
    path = output_dir / "source_images" / "cofw68" / f"cofw68_test_{index + 1:04d}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.is_file():
        Image.fromarray(image).save(path)
    return path


def _build_cofw68(
    root: Path,
    output_dir: Path,
    *,
    scenario: str,
    scenarios: tuple[str, ...] | None,
    limit: int | None,
    mode: str,
    allow_overlap: bool,
) -> Path:
    color_mat = _cofw68_test_color_mat(root)
    annotations = _cofw6868_annotation_paths(root)
    boxes = _cofw68_test_bboxes(root)

    samples: list[dict[str, T.Any]] = []
    skipped: list[dict[str, str]] = []

    for ann in track(
        annotations, desc="Build cofw68", total=len(annotations), unit="sample"
    ):
        try:
            idx = _cofw68_annotation_index(ann)
            sample_id = f"cofw68_test_{idx + 1:04d}"
            split = _deterministic_split("cofw68", sample_id)
            points68, visibility, metadata = _cofw68_points_and_occ(ann)
            image_arr = _cofw68_hdf5_image_by_index(color_mat, idx)
            image_path = _write_cofw68_image(output_dir, idx, image_arr)

            raw_bbox = None
            if boxes is not None and 0 <= idx < len(boxes):
                raw_bbox = [float(x) for x in boxes[idx].tolist()]
                x, y, width, height = raw_bbox
                metadata["face_bbox_raw"] = raw_bbox
                metadata["face_bbox_raw_format"] = "xywh"
                metadata["face_bbox_raw_source"] = "cofw6868_test_bboxes"
                metadata["face_bbox"] = [x, y, x + width, y + height]
                metadata["face_bbox_format"] = "ltrb"
                metadata["face_bbox_source"] = "cofw6868_test_bboxes"

            metadata.update(
                {
                    "annotation_file": str(ann.resolve()),
                    "cofw68_index": idx + 1,
                    "split": split,
                    "image_source_mat": str(color_mat.resolve()),
                    "source_schema": "2d_68",
                }
            )

            entry_for_crop = {"visibility": visibility}
            visible_mask, visible_mask_source = _cofw68_visibility_mask_and_source(
                entry_for_crop, metadata
            )
            bbox_ltrb, bbox_source = _cofw68_choose_crop_bbox(
                entry_for_crop,
                metadata,
                image_path,
                points68,
                visible_mask,
            )
            crop_image_path, crop_points68, crop_metadata = _crop_sample_image(
                output_dir=output_dir,
                dataset="cofw68",
                sample_id=sample_id,
                image_path=image_path,
                points68=points68,
                bbox_xyxy=bbox_ltrb,
                bbox_source=bbox_source,
                pad_ratio=0.25,
            )
        except Exception as err:  # noqa: BLE001
            skipped.append({"sample_id": ann.as_posix(), "reason": str(err)})
            continue

        metadata.update(crop_metadata)
        metadata["face_bbox"] = [float(v) for v in bbox_ltrb]
        metadata["face_bbox_format"] = "ltrb"
        metadata["face_bbox_source"] = bbox_source
        metadata["crop_visibility_mask_source"] = visible_mask_source
        metadata["crop_visible_landmark_count"] = int(
            np.asarray(visible_mask, dtype=bool).sum()
        )
        metadata["visibility"] = visibility

        samples.append(
            _with_split(
                _sample(
                    output_dir=output_dir,
                    dataset="cofw68",
                    sample_id=sample_id,
                    image=crop_image_path,
                    points68=crop_points68,
                    condition="occlusion",
                    conditions=("occlusion", f"{split}set"),
                    source_schema="2d_68",
                    source_id=sample_id,
                    metadata=metadata,
                    visibility=visibility,
                ),
                split,
            )
        )

    if not samples:
        raise ValueError(f"no cofw6868 test samples built; skipped={skipped[:5]}")

    return _write_manifest(
        output_dir,
        "cofw68",
        scenario,
        _filter(samples, scenarios, limit),
        mode=mode,
        allow_overlap=allow_overlap,
        scenarios=scenarios,
        skipped=skipped,
    )


# ---------------------------------------------------------------------------
# cofw68 bbox helpers.
#
# Some local cofw68 materializations mark benchmark boxes as ltrb even when the
# values are effectively xywh. Choose a bbox by checking whether it contains the
# visible/valid landmarks. Fall back to visible-landmark bbox when the benchmark
# bbox is inconsistent.
# ---------------------------------------------------------------------------
def _cofw68_visibility_mask_and_source(entry, metadata):
    raw = entry.get("visibility", metadata.get("visibility"))
    if isinstance(raw, (list, tuple)) and len(raw) == 68:
        return np.asarray([bool(v) for v in raw], dtype=bool), "visibility"

    raw = metadata.get("landmark_score_visibility_mask")
    if isinstance(raw, (list, tuple)) and len(raw) == 68:
        return np.asarray(
            [bool(v) for v in raw], dtype=bool
        ), "landmark_score_visibility_mask"

    # cofw68 Occ is occluded=True. If present, invert it.
    occ = metadata.get("occlusion", entry.get("occlusion"))
    if not isinstance(occ, (list, tuple)):
        occ = metadata.get("occlusion_mask")
    if isinstance(occ, (list, tuple)) and len(occ) == 68:
        return np.asarray([not bool(v) for v in occ], dtype=bool), "occlusion_mask"

    return np.ones((68,), dtype=bool), "all_landmarks_fallback"


def _cofw68_visibility_mask_for_crop(entry, metadata):
    return _cofw68_visibility_mask_and_source(entry, metadata)[0]


def _cofw68_bbox_candidates(entry, metadata):
    candidates = []

    def add(label, bbox, fmt):
        if bbox is None:
            return
        try:
            vals = [float(v) for v in list(bbox)[:4]]
        except Exception:
            return
        if len(vals) != 4 or not all(np.isfinite(vals)):
            return
        x, y, a, b = vals
        if fmt == "xywh":
            if a > 0 and b > 0:
                candidates.append((label + "_xywh", [x, y, x + a, y + b]))
        elif fmt == "ltrb":
            if a > x and b > y:
                candidates.append((label + "_ltrb", [x, y, a, b]))
        else:
            # Include both interpretations; the scorer will choose.
            if a > x and b > y:
                candidates.append((label + "_as_ltrb", [x, y, a, b]))
            if a > 0 and b > 0:
                candidates.append((label + "_as_xywh", [x, y, x + a, y + b]))

    source = str(
        entry.get("face_bbox_source")
        or entry.get("bbox_source")
        or metadata.get("face_bbox_source")
        or metadata.get("bbox_source")
        or ""
    ).lower()
    raw_fmt = str(
        entry.get("face_bbox_raw_format")
        or entry.get("bbox_raw_format")
        or metadata.get("face_bbox_raw_format")
        or metadata.get("bbox_raw_format")
        or ""
    ).lower()
    fmt = str(
        entry.get("face_bbox_format")
        or entry.get("bbox_format")
        or metadata.get("face_bbox_format")
        or metadata.get("bbox_format")
        or ""
    ).lower()

    # Prefer raw benchmark bbox if available.
    raw_bbox = (
        entry.get("face_bbox_raw")
        or entry.get("bbox_raw")
        or metadata.get("face_bbox_raw")
        or metadata.get("bbox_raw")
    )
    if raw_bbox is not None:
        add("face_bbox_raw", raw_bbox, raw_fmt or "xywh")

    bbox = (
        entry.get("face_bbox")
        or entry.get("bbox")
        or metadata.get("face_bbox")
        or metadata.get("bbox")
    )
    if bbox is not None:
        if "cofw68" in source and raw_bbox is None:
            # The local builder has shown stale/misleading "ltrb" metadata for
            # cofw68. For cofw68 benchmark boxes, consider xywh first.
            add("face_bbox", bbox, "xywh")
            add("face_bbox", bbox, "ltrb")
        else:
            add("face_bbox", bbox, fmt or None)

    # Deduplicate.
    out = []
    seen = set()
    for label, box in candidates:
        key = tuple(round(float(v), 4) for v in box)
        if key not in seen:
            seen.add(key)
            out.append((label, box))
    return out


def _cofw68_score_bbox_candidate(bbox_ltrb, points68, visible_mask, image_hw):
    try:
        left, top, right, bottom = _bbox_to_square_with_padding(
            bbox_ltrb,
            image_hw=image_hw,
            pad_ratio=0.25,
        )
    except Exception:
        return -1, float("inf")

    pts = np.asarray(points68, dtype=np.float32)
    mask = np.asarray(visible_mask, dtype=bool)
    if not mask.any():
        mask = np.ones((68,), dtype=bool)

    valid = pts[mask]
    inside = (
        (valid[:, 0] >= left - 2)
        & (valid[:, 0] <= right + 2)
        & (valid[:, 1] >= top - 2)
        & (valid[:, 1] <= bottom + 2)
    )
    count_inside = int(inside.sum())
    area = float(max(right - left, 1.0) * max(bottom - top, 1.0))
    return count_inside, area


def _cofw68_choose_crop_bbox(entry, metadata, image_path, points68, visible_mask):
    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError(f"could not read cofw68 image: {image_path}")
    image_hw = image_bgr.shape[:2]

    visible_mask = np.asarray(visible_mask, dtype=bool)
    if not visible_mask.any():
        visible_mask = np.ones((68,), dtype=bool)
    required = int(visible_mask.sum())
    best = None

    for label, bbox in _cofw68_bbox_candidates(entry, metadata):
        score, area = _cofw68_score_bbox_candidate(
            bbox, points68, visible_mask, image_hw
        )
        if best is None or score > best[0] or (score == best[0] and area < best[1]):
            best = (score, area, label, bbox)

    if best is not None and best[0] >= max(1, int(0.95 * required)):
        return best[3], f"cofw68_bbox_v2:{best[2]}"

    # Benchmark bbox is inconsistent with visible landmarks. Use visible
    # landmarks to derive the crop. This is safer for CD-ViT than training on
    # exploded coordinates.
    pts = np.asarray(points68, dtype=np.float32)
    return _bbox_from_points_xyxy(
        pts[visible_mask]
    ), "cofw68_bbox_v2:visible_landmark_bbox_fallback"


def _cofw68_bbox4(value: T.Any) -> list[float] | None:
    if value is None:
        return None
    try:
        values = [float(v) for v in list(value)[:4]]
    except Exception:
        return None
    if len(values) != 4 or not all(np.isfinite(values)):
        return None
    return values


def _cofw68_entry_is_materialized_crop(
    entry: T.Mapping[str, T.Any], metadata: T.Mapping[str, T.Any]
) -> bool:
    crop_bbox = entry.get("crop_bbox_xyxy") or metadata.get("crop_bbox_xyxy")
    crop_output_size = entry.get("crop_output_size") or metadata.get("crop_output_size")
    original_image = entry.get("original_image") or metadata.get("original_image")
    try:
        output_size = int(crop_output_size)
    except (TypeError, ValueError):
        output_size = None
    return crop_bbox is not None or output_size == 256 or original_image is not None


def _build_cofw68_json_cropped(
    path: Path,
    output_dir: Path,
    *,
    scenario: str,
    scenarios: tuple[str, ...] | None,
    limit: int | None,
    mode: str,
    allow_overlap: bool,
    image_root: str | None,
) -> Path:
    payload = read_json(path)
    entries = (
        payload.get("samples", payload.get("entries", payload))
        if isinstance(payload, dict)
        else payload
    )
    if not isinstance(entries, list):
        raise ValueError(
            f"cofw68 JSON source must contain list, entries, or samples list: {path}"
        )

    image_base = Path(image_root) if image_root else path.parent
    samples: list[dict[str, T.Any]] = []
    skipped: list[dict[str, str]] = []

    for idx, entry in track(
        enumerate(entries), desc="Build cofw68", total=len(entries), unit="sample"
    ):
        if not isinstance(entry, dict):
            continue

        metadata = (
            dict(entry.get("metadata", {}))
            if isinstance(entry.get("metadata"), dict)
            else {}
        )
        image_value = entry.get("image") or entry.get("image_path") or entry.get("path")
        landmark_value = (
            entry.get("landmarks")
            or entry.get("points")
            or entry.get("ground_truth")
            or entry.get("pts")
        )
        sample_id = str(
            entry.get("sample_id")
            or entry.get("id")
            or entry.get("name")
            or f"cofw68/{idx:04d}"
        )

        if _cofw68_entry_is_materialized_crop(entry, metadata):
            raise ValueError(
                "--cofw68-json points to an already-cropped manifest entry "
                f"{sample_id!r}; use raw cofw68 JSON/source instead"
            )

        if image_value is None or landmark_value is None:
            skipped.append(
                {"sample_id": sample_id, "reason": "missing image or landmarks"}
            )
            continue

        try:
            image_path = _resolve_path(image_value, base_dir=image_base)
            source_schema = (
                str(entry.get("source_schema") or metadata.get("source_schema") or "")
                or None
            )
            points68, detected_schema = _load_points(
                landmark_value,
                base_dir=path.parent,
                source_schema=source_schema,
            )
            visibility = entry.get("visibility", metadata.get("visibility"))
            visible_mask, visible_mask_source = _cofw68_visibility_mask_and_source(
                entry, metadata
            )
            bbox_ltrb, bbox_source = _cofw68_choose_crop_bbox(
                entry, metadata, image_path, points68, visible_mask
            )
            crop_image_path, crop_points68, crop_metadata = _crop_sample_image(
                output_dir=output_dir,
                dataset="cofw68",
                sample_id=sample_id,
                image_path=image_path,
                points68=points68,
                bbox_xyxy=bbox_ltrb,
                bbox_source=bbox_source,
                pad_ratio=0.25,
            )
        except Exception as err:  # noqa: BLE001
            skipped.append({"sample_id": sample_id, "reason": str(err)})
            continue

        is_occluded = True
        if isinstance(visibility, (list, tuple)) and visibility:
            is_occluded = any(not bool(v) for v in visibility)

        explicit_split = _label(entry.get("split") or metadata.get("split") or "")
        split = (
            explicit_split
            if explicit_split in {"train", "test"}
            else _deterministic_split("cofw68", sample_id)
        )

        conds = _conditions(entry, "occlusion" if is_occluded else scenario)
        if is_occluded and "occlusion" not in conds:
            conds = tuple(dict.fromkeys((*conds, "occlusion")))
        split_condition = f"{split}set"
        if split_condition not in conds:
            conds = tuple(dict.fromkeys((*conds, split_condition)))

        merged_metadata = dict(metadata)
        input_bbox = _cofw68_bbox4(
            entry.get("face_bbox")
            or entry.get("bbox")
            or metadata.get("face_bbox")
            or metadata.get("bbox")
        )
        if input_bbox is not None:
            merged_metadata.setdefault("face_bbox_input", input_bbox)
            input_format = str(
                entry.get("face_bbox_format")
                or entry.get("bbox_format")
                or metadata.get("face_bbox_format")
                or metadata.get("bbox_format")
                or ""
            ).strip()
            if input_format:
                merged_metadata.setdefault("face_bbox_input_format", input_format)
            input_source = str(
                entry.get("face_bbox_source")
                or entry.get("bbox_source")
                or metadata.get("face_bbox_source")
                or metadata.get("bbox_source")
                or "cofw68_json"
            )
            merged_metadata.setdefault("face_bbox_input_source", input_source)

        raw_bbox = _cofw68_bbox4(
            entry.get("face_bbox_raw")
            or entry.get("bbox_raw")
            or metadata.get("face_bbox_raw")
            or metadata.get("bbox_raw")
        )
        if raw_bbox is not None:
            merged_metadata.setdefault("face_bbox_raw", raw_bbox)
            raw_format = str(
                entry.get("face_bbox_raw_format")
                or entry.get("bbox_raw_format")
                or metadata.get("face_bbox_raw_format")
                or metadata.get("bbox_raw_format")
                or "xywh"
            )
            merged_metadata.setdefault("face_bbox_raw_format", raw_format)
            raw_source = str(
                entry.get("face_bbox_raw_source")
                or entry.get("bbox_raw_source")
                or metadata.get("face_bbox_raw_source")
                or metadata.get("bbox_raw_source")
                or "cofw68_json"
            )
            merged_metadata.setdefault("face_bbox_raw_source", raw_source)

        merged_metadata.update(crop_metadata)
        merged_metadata["face_bbox"] = [float(v) for v in bbox_ltrb]
        merged_metadata["face_bbox_format"] = "ltrb"
        merged_metadata["face_bbox_source"] = bbox_source
        merged_metadata["crop_visibility_mask_source"] = visible_mask_source
        merged_metadata["crop_visible_landmark_count"] = int(
            np.asarray(visible_mask, dtype=bool).sum()
        )
        merged_metadata.setdefault("source_schema", detected_schema)
        if visibility is not None:
            merged_metadata["visibility"] = visibility

        samples.append(
            _with_split(
                _sample(
                    output_dir=output_dir,
                    dataset="cofw68",
                    sample_id=sample_id,
                    image=crop_image_path,
                    points68=crop_points68,
                    condition="occlusion"
                    if is_occluded
                    else str(entry.get("condition") or conds[0]),
                    conditions=tuple(_label(item) for item in conds),
                    source_schema=source_schema or detected_schema,
                    source_id=str(entry.get("source_id") or sample_id),
                    metadata=merged_metadata,
                    visibility=visibility,
                ),
                split,
            )
        )

    if not samples:
        raise ValueError(
            f"no cropped cofw68 JSON samples built; skipped={skipped[:10]}"
        )

    return _write_manifest(
        output_dir,
        "cofw68",
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
