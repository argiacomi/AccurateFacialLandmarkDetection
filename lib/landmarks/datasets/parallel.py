"""Bounded parallel execution helper for dataset-preparation bottlenecks.

The heavy steps in the preparation pipeline -- video frame decoding and overlay
rendering -- are dominated by OpenCV/native work and disk IO, both of which
release the GIL. A thread pool therefore yields real parallel speedups without
the pickling and spawn-import fragility of a process pool, while preserving
deterministic, input-ordered results and identical error semantics.
"""

from __future__ import annotations

import os
import typing as T
from collections.abc import Callable, Iterable
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

from lib.landmarks.datasets.progress import track

_T = T.TypeVar("_T")
_R = T.TypeVar("_R")


def resolve_worker_count(workers: int | None, item_count: int) -> int:
    """Clamp a requested worker count to a sane range for ``item_count`` items.

    ``None`` or a non-positive value means "use all available CPUs". The result
    is at least 1 and never exceeds the number of items.
    """
    if workers is None or workers <= 0:
        workers = os.cpu_count() or 1
    return max(1, min(int(workers), max(1, item_count)))


def parallel_map(
    func: Callable[[_T], _R],
    items: Iterable[_T],
    *,
    workers: int | None,
    desc: str,
    unit: str = "it",
    use_processes: bool = False,
) -> list[_R]:
    """Map ``func`` over ``items`` with a bounded pool, preserving input order.

    Falls back to a sequential map when only one worker (or one item) is
    effective, so callers get identical results and error semantics regardless
    of the worker count. Exceptions raised by ``func`` propagate to the caller.

    Threads are used by default because the pipeline's parallel steps spend
    their time in GIL-releasing native code (OpenCV) and disk IO; pass
    ``use_processes=True`` only for CPU-bound, picklable work.
    """
    items = list(items)
    worker_count = resolve_worker_count(workers, len(items))
    if worker_count <= 1 or len(items) <= 1:
        return [
            func(item) for item in track(items, desc=desc, total=len(items), unit=unit)
        ]

    executor_cls = ProcessPoolExecutor if use_processes else ThreadPoolExecutor
    results: list[_R] = [T.cast(_R, None)] * len(items)
    with executor_cls(max_workers=worker_count) as executor:
        future_to_index = {
            executor.submit(func, item): index for index, item in enumerate(items)
        }
        for future in track(
            as_completed(future_to_index), desc=desc, total=len(items), unit=unit
        ):
            results[future_to_index[future]] = future.result()
    return results
