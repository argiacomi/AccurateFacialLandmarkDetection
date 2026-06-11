"""video dataset builder (split from build_quality_dataset)."""

# ruff: noqa: E402, F403, F405
from __future__ import annotations

from lib.datasets.build.core import *  # noqa: F403


def _candidate_frame_stems(frame_index: int) -> tuple[str, ...]:
    one_based = int(frame_index) + 1
    return tuple(
        dict.fromkeys(
            (
                f"frame_{frame_index:06d}",
                f"{frame_index:06d}",
                f"{frame_index:05d}",
                f"{frame_index:04d}",
                str(frame_index),
                f"frame_{one_based:06d}",
                f"{one_based:06d}",
                f"{one_based:05d}",
                f"{one_based:04d}",
                str(one_based),
            )
        )
    )


def _candidate_frame_indices_from_stem(stem: str) -> tuple[int, ...]:
    """Return zero-based frame indices a landmark filename could represent."""
    values: list[int] = []
    for token in reversed(re.findall(r"\d+", stem)):
        raw = int(token)
        for candidate in (raw, raw - 1):
            if candidate >= 0 and candidate not in values:
                values.append(candidate)
    return tuple(values)


def _directory_frame_numbering_bases(paths: T.Sequence[Path]) -> dict[Path, int]:
    """Detect each annotation directory's frame-numbering base (0- or 1-based).

    Registering every file under both ``raw`` and ``raw - 1`` lets ``setdefault``
    collision order decide the mapping, which pairs 1-based annotation releases
    (300VW ships ``000001.pts`` for the first frame) with the *previous* video
    frame for every frame except 0. When a directory's numeric stems clearly
    start at 0 or 1, commit to that base so extracted zero-based frame ``i``
    maps to file ``i + base`` exactly; directories without a clear convention
    keep the permissive dual registration.
    """

    by_dir: dict[Path, list[int | None]] = {}
    for path in paths:
        tokens = re.findall(r"\d+", path.stem)
        by_dir.setdefault(path.parent, []).append(int(tokens[-1]) if tokens else None)
    bases: dict[Path, int] = {}
    for parent, values in by_dir.items():
        if len(values) < 2 or any(value is None for value in values):
            continue
        base = min(T.cast("list[int]", values))
        if base in (0, 1):
            bases[parent] = base
    return bases


def _frame_indices_for_landmark_path(
    path: Path, directory_bases: T.Mapping[Path, int]
) -> tuple[int, ...]:
    base = directory_bases.get(path.parent)
    if base is None:
        return _candidate_frame_indices_from_stem(path.stem)
    tokens = re.findall(r"\d+", path.stem)
    if not tokens:
        return ()
    frame_index = int(tokens[-1]) - base
    return (frame_index,) if frame_index >= 0 else ()


def _frame_landmark_files(root: Path) -> T.Iterator[Path]:
    for suffix in LANDMARK_EXTS:
        yield from root.rglob(f"*{suffix}")


def _add_frame_landmark_index_entry(
    index: dict[tuple[str, int], Path],
    *,
    video_id: str,
    frame_index: int,
    path: Path,
) -> None:
    normalized_video_id = str(video_id).replace("\\", "/").strip("/")
    if not normalized_video_id:
        return
    index.setdefault((normalized_video_id, int(frame_index)), path)


def _frame_landmark_video_id_aliases(parts: T.Sequence[str]) -> tuple[str, ...]:
    """Return video-id aliases for a frame-landmark file path.

    Handles layouts such as:
      WFLW_V_release/annotations/<video_id>/<frame>.pts
      WFLW_V_release/landmarks/<video_id>/<frame>.pts
      300VW/<seq>/annot/<frame>.pts

    The extracted video id is based on the video path, usually replacing the
    annotation directory with videos/ or vid/.
    """

    if len(parts) <= 1:
        return ()

    parent_parts = list(parts[:-1])
    structured_roots = {
        "annot",
        "annotation",
        "annotations",
        "landmark",
        "landmarks",
        "label",
        "labels",
    }
    aliases: list[str] = []

    def add(seq: T.Sequence[str]) -> None:
        clean = [str(item).strip("/") for item in seq if str(item).strip("/")]
        if not clean:
            return
        value = "/".join(clean)
        if value not in aliases:
            aliases.append(value)

    # Literal parent path fallback.
    add(parent_parts)

    for index, part in enumerate(parent_parts):
        lowered = part.lower()
        if lowered not in structured_roots:
            continue

        # Bare id after annotations/<video_id>/...
        add(parent_parts[index + 1 :])

        # Same archive path, replacing annotations/landmarks with video roots.
        replacements = ("videos", "video", "frames", "images")
        if lowered in {"annot", "annotation", "annotations"}:
            replacements = ("videos", "video", "vid", "frames", "images")

        for replacement in replacements:
            replaced = parent_parts.copy()
            replaced[index] = replacement
            add(replaced)

    return tuple(aliases)


def _build_frame_landmark_index(root: Path) -> dict[tuple[str, int], Path]:
    """Build a video_id/frame_index -> landmark path index with one tree walk.

    This replaces the old per-frame ``root.rglob(...)`` fallback in
    ``_find_frame_landmark_file``. Structured layouts are still favored by
    sorting short paths first and by indexing annotations/, landmarks/, labels/
    directories before generic filename-prefix fallbacks.
    """
    index: dict[tuple[str, int], Path] = {}
    if not root.is_dir():
        return index

    structured_roots = {"annotations", "landmarks", "labels"}
    landmark_files = [
        path
        for path in sorted(
            _frame_landmark_files(root),
            key=lambda item: (len(item.parts), item.as_posix()),
        )
        if path.is_file()
    ]
    directory_bases = _directory_frame_numbering_bases(landmark_files)
    for path in landmark_files:
        try:
            rel = path.relative_to(root)
        except ValueError:
            rel = path
        parts = rel.parts
        frame_indices = _frame_indices_for_landmark_path(path, directory_bases)
        if not frame_indices:
            continue

        for video_id_alias in _frame_landmark_video_id_aliases(parts):
            for frame_index in frame_indices:
                _add_frame_landmark_index_entry(
                    index,
                    video_id=video_id_alias,
                    frame_index=frame_index,
                    path=path,
                )

        # Fast structured layouts: annotations/<video_id>/<frame>.npy and peers.
        if len(parts) > 2 and parts[0] in structured_roots:
            video_id = "/".join(parts[1:-1])
            for frame_index in frame_indices:
                _add_frame_landmark_index_entry(
                    index, video_id=video_id, frame_index=frame_index, path=path
                )

        # 300VW layout: <sequence>/annot/<frame>.pts next to <sequence>/vid.avi.
        if len(parts) > 2 and parts[-2] == "annot":
            video_id = "/".join((*parts[:-2], "vid"))
            for frame_index in frame_indices:
                _add_frame_landmark_index_entry(
                    index, video_id=video_id, frame_index=frame_index, path=path
                )

        # Generic nested layout: <video_id>/<frame>.npy.
        if len(parts) > 1 and parts[0] not in structured_roots:
            video_id = "/".join(parts[:-1])
            for frame_index in frame_indices:
                _add_frame_landmark_index_entry(
                    index, video_id=video_id, frame_index=frame_index, path=path
                )

        # Flat fallback layout: <video_id>_<frame>.npy or <video_id>-frame_000001.npy.
        for frame_index in frame_indices:
            for candidate_stem in _candidate_frame_stems(frame_index):
                if path.stem == candidate_stem:
                    continue
                for separator in ("_", "-", ".", " ", ""):
                    suffix = f"{separator}{candidate_stem}"
                    if not path.stem.endswith(suffix):
                        continue
                    video_id = path.stem[: -len(suffix)].strip("_.- /")
                    _add_frame_landmark_index_entry(
                        index, video_id=video_id, frame_index=frame_index, path=path
                    )
    return index


def _find_frame_landmark_file(
    landmark_index: T.Mapping[tuple[str, int], Path],
    video_id: str,
    frame_index: int,
) -> Path | None:
    safe_video_id = str(video_id).replace("\\", "/").strip("/")
    return landmark_index.get((safe_video_id, int(frame_index)))


@dataclass(frozen=True, slots=True)
class _VideoFrameTask:
    """Inputs for decoding one video's frames in a worker."""

    video_path: Path
    video_id: str
    frame_root: Path
    frame_stride: int
    max_frames_per_video: int | None


def _extract_video_frames_task(
    task: _VideoFrameTask,
) -> tuple[str, list[dict[str, T.Any]] | None, str | None]:
    """Decode one video; return (video_id, frame_records, error)."""
    try:
        records = extract_video_frames(
            task.video_path,
            task.frame_root,
            stride=task.frame_stride,
            max_frames=task.max_frames_per_video,
            video_id=task.video_id,
            progress=False,
        )
        return task.video_id, records, None
    except Exception as err:  # noqa: BLE001
        return task.video_id, None, str(err)


def _wflwv_npy_kind(path: Path) -> str | None:
    lowered = path.as_posix().lower()
    if "bbox" in lowered or "bboxes" in lowered or "box" in lowered:
        return "bbox"
    if (
        "landmark" in lowered
        or "landmarks" in lowered
        or "point" in lowered
        or "points" in lowered
        or "/pts" in lowered
        or "keypoint" in lowered
    ):
        return "landmarks"

    try:
        arr = np.load(path, mmap_mode="r", allow_pickle=False)
    except Exception:  # noqa: BLE001
        return None

    shape = tuple(int(v) for v in getattr(arr, "shape", ()))
    if len(shape) >= 3 and shape[-1] >= 2 and shape[-2] in {68, 98, 106, 194}:
        return "landmarks"
    if len(shape) == 2 and shape[1] in {136, 196, 212, 388}:
        return "landmarks"
    if len(shape) >= 2 and shape[-1] == 4:
        return "bbox"
    return None


def _wflwv_sequence_video_id_aliases(root: Path, path: Path) -> tuple[str, ...]:
    try:
        rel = path.relative_to(root)
    except ValueError:
        rel = path
    parts = list(rel.parts)
    if not parts:
        return ()

    stem = path.stem
    parent_parts = parts[:-1]
    aliases: list[str] = []

    def add(seq: T.Sequence[str]) -> None:
        clean = [
            str(item).replace("\\", "/").strip("/")
            for item in seq
            if str(item).strip("/")
        ]
        if not clean:
            return
        value = "/".join(clean)
        if value not in aliases:
            aliases.append(value)

    add((stem,))
    add((*parent_parts, stem))

    structured_tokens = {
        "bbox",
        "bboxes",
        "box",
        "boxes",
        "landmark",
        "landmarks",
        "point",
        "points",
        "pts",
        "annotation",
        "annotations",
        "label",
        "labels",
    }

    for index, part in enumerate(parent_parts):
        if part.lower() not in structured_tokens:
            continue

        for replacement in ("videos", "video"):
            replaced = parent_parts.copy()
            replaced[index] = replacement
            add((*replaced, stem))

        add((*parent_parts[index + 1 :], stem))

    return tuple(aliases)


def _build_wflwv_sequence_index(root: Path) -> dict[str, dict[str, Path]]:
    index: dict[str, dict[str, Path]] = {"landmarks": {}, "bbox": {}}
    if not root.is_dir():
        return index

    for npy_path in sorted(root.rglob("*.npy"), key=lambda item: item.as_posix()):
        if not npy_path.is_file():
            continue
        kind = _wflwv_npy_kind(npy_path)
        if kind not in index:
            continue
        for alias in _wflwv_sequence_video_id_aliases(root, npy_path):
            index[kind].setdefault(alias, npy_path)
    return index


def _wflwv_payload_array(payload: T.Any, *, kind: str) -> np.ndarray:
    if (
        isinstance(payload, np.ndarray)
        and payload.dtype == object
        and payload.shape == ()
    ):
        payload = payload.item()

    if isinstance(payload, dict):
        keys = (
            ("landmarks", "landmark", "points", "pts", "keypoints")
            if kind == "landmarks"
            else ("bbox", "bboxes", "boxes", "face_bbox")
        )
        for key in keys:
            if key in payload:
                return np.asarray(payload[key])
        raise ValueError(f"WFLW-V {kind} npy dict does not contain expected keys")

    return np.asarray(payload)


def _wflwv_load_npy_array(path: Path, *, kind: str) -> np.ndarray:
    payload = np.load(path, allow_pickle=True)
    return _wflwv_payload_array(payload, kind=kind)


def _wflwv_frame_row(path: Path, frame_index: int, *, kind: str) -> np.ndarray:
    arr = _wflwv_load_npy_array(path, kind=kind)
    if arr.ndim == 0:
        raise ValueError(f"WFLW-V {kind} array is scalar: {path}")

    frame_index = int(frame_index)
    if frame_index < 0 or frame_index >= int(arr.shape[0]):
        raise IndexError(
            f"WFLW-V {kind} frame {frame_index} out of range for {path} "
            f"with shape {arr.shape}"
        )
    return np.asarray(arr[frame_index])


def _wflwv_sequence_frame(
    index: T.Mapping[str, T.Mapping[str, Path]],
    video_id: str,
    frame_index: int,
) -> tuple[np.ndarray, str, Path, Path | None, list[float] | None] | None:
    keys = tuple(
        dict.fromkeys(
            (
                str(video_id).replace("\\", "/").strip("/"),
                Path(str(video_id)).stem,
                Path(str(video_id)).name,
            )
        )
    )

    landmark_path = None
    for key in keys:
        landmark_path = index.get("landmarks", {}).get(key)
        if landmark_path is not None:
            break
    if landmark_path is None:
        return None

    raw_points = _wflwv_frame_row(landmark_path, frame_index, kind="landmarks")
    points, source_schema = _canonical_points(raw_points, source_schema=None)

    bbox_path = None
    bbox_xyxy = None
    for key in keys:
        bbox_path = index.get("bbox", {}).get(key)
        if bbox_path is not None:
            break
    if bbox_path is not None:
        raw_bbox = np.asarray(
            _wflwv_frame_row(bbox_path, frame_index, kind="bbox"),
            dtype=np.float32,
        ).reshape(-1)
        if raw_bbox.size >= 4 and np.all(np.isfinite(raw_bbox[:4])):
            bbox_xyxy = [float(value) for value in raw_bbox[:4]]

    return points, source_schema, landmark_path, bbox_path, bbox_xyxy


def _video_dataset_source_metadata(dataset: str, video_id: str) -> dict[str, T.Any]:
    dataset = _dataset(dataset)
    raw_parts = [part for part in str(video_id).replace("\\", "/").split("/") if part]
    parts = [_label(part) for part in raw_parts]
    metadata: dict[str, T.Any] = {}

    if dataset == "300vw":
        # In the unpacked 300VW layout, directories like 001/002/003 are sequence
        # ids, not challenge-category ids. Only infer category from explicit
        # category/scenario tokens. Otherwise use a general 300vw bucket.
        sequence_id = None
        if "vid" in parts:
            vid_index = parts.index("vid")
            if vid_index > 0:
                sequence_id = raw_parts[vid_index - 1]
        elif raw_parts:
            sequence_id = raw_parts[-1]

        category: int | None = None
        for part in parts:
            match = re.fullmatch(r"(?:category|cat|scenario|challenge)_?([123])", part)
            if match:
                category = int(match.group(1))
                break

        if sequence_id is not None:
            metadata["sequence_id"] = str(sequence_id)

        if category is not None:
            difficulty = {
                1: "well_lit",
                2: "mild_unconstrained",
                3: "challenging",
            }[category]
            metadata["video_dataset_category"] = category
            metadata["video_difficulty"] = difficulty
            metadata["source_condition"] = f"300vw_category_{category}"
            metadata["source_conditions"] = [f"300vw_category_{category}", difficulty]
        else:
            metadata["source_condition"] = "300vw"
            metadata["source_conditions"] = ["300vw"]

    elif dataset == "wflw-v":
        metadata["source_condition"] = "wflw_v"
        metadata["source_conditions"] = ["wflw_v"]

    return metadata


def _video_dataset_source_conditions(
    dataset: str,
    scenario: str,
    split: str,
    video_id: str,
) -> tuple[str, tuple[str, ...]]:
    metadata = _video_dataset_source_metadata(dataset, video_id)
    conditions: list[str] = []

    source_conditions = metadata.get("source_conditions")
    if isinstance(source_conditions, list):
        conditions.extend(str(item) for item in source_conditions)

    conditions.append("video_frame")

    scenario_label = _label(scenario)
    if scenario_label != "default":
        conditions.append(scenario_label)

    conditions.append(f"{split}set")
    conditions = list(dict.fromkeys(_label(item) for item in conditions))
    return conditions[0], tuple(conditions)


def _build_video_dataset(
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
    video_root: str | None,
    frame_output_dir: str | None,
    frame_stride: int,
    max_frames_per_video: int | None,
    max_workers: int = 1,
) -> Path:
    json_path = _json_source(root)
    if json_path is not None:
        return _build_json(
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

    videos_root = Path(video_root) if video_root else root
    videos = video_files(videos_root)
    if not videos:
        return _build_directory(
            root,
            output_dir,
            dataset=dataset,
            scenario=scenario,
            scenarios=scenarios,
            limit=limit,
            mode=mode,
            allow_overlap=allow_overlap,
            image_root=image_root,
        )

    frame_root = (
        Path(frame_output_dir) if frame_output_dir else output_dir / "frames" / dataset
    )
    samples: list[dict[str, T.Any]] = []
    skipped: list[dict[str, str]] = []
    frame_landmark_index = _build_frame_landmark_index(root)
    wflwv_sequence_index = (
        _build_wflwv_sequence_index(root) if dataset == "wflw-v" else None
    )

    # Decode every video in parallel (OpenCV releases the GIL); the per-frame
    # sample assembly below stays sequential to keep manifest ordering and
    # split assignment deterministic regardless of worker count.
    tasks = [
        _VideoFrameTask(
            video_path=video_path,
            video_id=video_path.resolve()
            .relative_to(videos_root.resolve())
            .with_suffix("")
            .as_posix(),
            frame_root=frame_root,
            frame_stride=frame_stride,
            max_frames_per_video=max_frames_per_video,
        )
        for video_path in videos
    ]
    extracted = parallel_map(
        _extract_video_frames_task,
        tasks,
        workers=max_workers,
        desc=f"Videos {dataset}",
        unit="video",
    )

    for task, (video_id, frame_records, error) in track(
        zip(tasks, extracted),
        desc=f"Build {dataset}",
        total=len(tasks),
        unit="video",
    ):
        if error is not None:
            skipped.append({"sample_id": video_id, "reason": error})
            continue
        video_path = task.video_path
        split = _deterministic_split(dataset, video_id)
        for record in frame_records:
            frame_index = int(record["frame_index"])
            sample_id = f"{dataset}/{video_id}/frame_{frame_index:06d}"
            bbox_path = None
            bbox_xyxy = None
            sequence_record = None
            if wflwv_sequence_index is not None:
                try:
                    sequence_record = _wflwv_sequence_frame(
                        wflwv_sequence_index,
                        video_id,
                        frame_index,
                    )
                except Exception as err:  # noqa: BLE001
                    skipped.append({"sample_id": sample_id, "reason": str(err)})
                    continue

            if sequence_record is not None:
                points, source_schema, landmark_path, bbox_path, bbox_xyxy = (
                    sequence_record
                )
            else:
                landmark_path = _find_frame_landmark_file(
                    frame_landmark_index,
                    video_id,
                    frame_index,
                )
                if landmark_path is None:
                    skipped.append(
                        {
                            "sample_id": sample_id,
                            "reason": "matching frame landmarks not found",
                        }
                    )
                    continue
                try:
                    points, source_schema = _load_landmark_file(landmark_path)
                except Exception as err:  # noqa: BLE001
                    skipped.append({"sample_id": sample_id, "reason": str(err)})
                    continue

            metadata = {
                "dataset": dataset,
                "video_id": video_id,
                "frame_index": frame_index,
                "frame_id": record["frame_id"],
                "split": split,
                "split_safe_id": video_id,
                "source_video": str(video_path.resolve()),
                "source_landmarks": str(landmark_path.resolve()),
            }
            metadata.update(_video_dataset_source_metadata(dataset, video_id))
            if bbox_path is not None and bbox_xyxy is not None:
                metadata["source_bbox"] = str(bbox_path.resolve())
                metadata["bbox_xyxy"] = bbox_xyxy
                metadata["bbox_source"] = "wflw_v_bbox_npy"
            condition, conditions = _video_dataset_source_conditions(
                dataset, scenario, split, video_id
            )
            samples.append(
                _with_split(
                    _sample(
                        output_dir=output_dir,
                        dataset=dataset,
                        sample_id=sample_id,
                        image=Path(record["image"]),
                        points68=points,
                        condition=condition,
                        conditions=conditions,
                        source_schema=source_schema,
                        source_id=sample_id,
                        metadata=metadata,
                    ),
                    split,
                )
            )

    if not samples:
        raise ValueError(
            f"no {dataset} video-frame samples built; skipped={skipped[:10]}"
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
