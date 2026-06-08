"""Reusable deterministic video-to-frame extraction helpers."""

from __future__ import annotations

import typing as T
from pathlib import Path

import cv2

from lib.landmarks.datasets.progress import track

VIDEO_EXTS = (".mp4", ".avi", ".mov", ".mkv", ".mpg", ".mpeg", ".webm")


def video_files(root: Path) -> list[Path]:
    """Return supported video files below a root in deterministic order."""

    return sorted(
        path
        for suffix in VIDEO_EXTS
        for path in root.rglob(f"*{suffix}")
        if path.is_file()
    )


def selected_frame_indices(
    frame_count: int,
    *,
    stride: int = 1,
    max_frames: int | None = None,
) -> list[int]:
    """Select frame indices using a deterministic stride plus optional thinning."""

    if frame_count <= 0:
        return []
    stride = max(1, int(stride or 1))
    indices = list(range(0, int(frame_count), stride))
    if max_frames is None or max_frames <= 0 or len(indices) <= max_frames:
        return indices
    if max_frames == 1:
        return [indices[0]]

    step = (len(indices) - 1) / float(max_frames - 1)
    selected = []
    seen = set()
    for offset in range(max_frames):
        index = indices[int(round(offset * step))]
        if index not in seen:
            seen.add(index)
            selected.append(index)
    return selected


def _safe_video_id(video_path: Path, video_id: str | None = None) -> str:
    text = str(video_id or video_path.stem).strip().replace("\\", "/").strip("/")
    return (
        "".join(ch if ch.isalnum() or ch in "._-/#" else "_" for ch in text) or "video"
    )


def extract_video_frames(
    video_path: Path,
    output_root: Path,
    *,
    stride: int = 1,
    max_frames: int | None = None,
    video_id: str | None = None,
    image_ext: str = ".jpg",
    progress: bool = True,
) -> list[dict[str, T.Any]]:
    """Extract selected video frames into a stable derived-data directory.

    The returned records include ``video_id`` and ``frame_index`` so callers can
    use the video id as their split-safe leakage identity.
    """

    video_path = Path(video_path)
    output_root = Path(output_root)
    video_key = _safe_video_id(video_path, video_id)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"could not open video: {video_path}")

    try:
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        indices = selected_frame_indices(
            frame_count,
            stride=stride,
            max_frames=max_frames,
        )
        records: list[dict[str, T.Any]] = []
        for frame_index in track(
            indices,
            desc=f"Frames {video_key}",
            total=len(indices),
            unit="frame",
            disable=None if progress else True,
        ):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            frame_path = output_root / video_key / f"frame_{frame_index:06d}{image_ext}"
            frame_path.parent.mkdir(parents=True, exist_ok=True)
            written = cv2.imwrite(str(frame_path), frame)
            if not written:
                raise OSError(f"failed to write video frame: {frame_path}")
            records.append(
                {
                    "video_id": video_key,
                    "frame_index": int(frame_index),
                    "frame_id": f"{video_key}:{frame_index:06d}",
                    "image": frame_path,
                    "video_path": video_path,
                }
            )
    finally:
        cap.release()

    if not records:
        raise ValueError(f"no frames extracted from {video_path}")
    return records
