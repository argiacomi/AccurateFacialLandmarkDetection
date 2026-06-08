"""Shared tqdm-based progress helper for the landmark dataset pipeline.

Centralizes progress-bar configuration so download, extraction, frame
extraction, manifest building, and orchestration all report long-running work
consistently. Bars auto-disable when stderr is not a TTY (tests, pipes,
redirected logs), so they only appear during interactive runs.
"""

from __future__ import annotations

import typing as T

from tqdm import tqdm

_T = T.TypeVar("_T")


def track(
    iterable: T.Iterable[_T] | None = None,
    *,
    desc: str,
    total: int | None = None,
    unit: str = "it",
    unit_scale: bool = False,
    leave: bool = False,
    disable: bool | None = None,
) -> "tqdm[_T]":
    """Wrap an iterable (or create a manual bar) with consistent tqdm settings.

    Pass ``disable=None`` (the default) to auto-disable on non-interactive
    output. The returned object is a ``tqdm`` instance, so callers can iterate
    it and still call ``set_description`` / ``update`` as needed.
    """
    return tqdm(
        iterable,
        desc=desc,
        total=total,
        unit=unit,
        unit_scale=unit_scale,
        unit_divisor=1024,
        leave=leave,
        disable=disable,
    )
