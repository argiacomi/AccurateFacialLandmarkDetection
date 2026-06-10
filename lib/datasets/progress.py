"""Shared progress helper for landmark dataset tools.

Uses Rich for interactive terminals, degrades to a plain iterable in captured
logs/non-TTY output, and preserves the old tqdm-like behavior where
``disable=False`` forces visible progress output for tests and scripts.
"""

from __future__ import annotations

import contextlib
import sys
import typing as T

from lib.logging_utils import rich_available

try:
    from rich.console import Console
    from rich.progress import (
        BarColumn,
        MofNCompleteColumn,
        Progress,
        TaskProgressColumn,
        TextColumn,
        TimeElapsedColumn,
        TimeRemainingColumn,
    )
except Exception:  # noqa: BLE001
    Console = None  # type: ignore[assignment]
    Progress = None  # type: ignore[assignment]
    BarColumn = None  # type: ignore[assignment]
    MofNCompleteColumn = None  # type: ignore[assignment]
    TaskProgressColumn = None  # type: ignore[assignment]
    TextColumn = None  # type: ignore[assignment]
    TimeElapsedColumn = None  # type: ignore[assignment]
    TimeRemainingColumn = None  # type: ignore[assignment]

_T = T.TypeVar("_T")
_PROGRESS_ENABLED = True
# When a parent owns a single shared Progress (see ``progress_group``), every
# track() call adds a task (row) to it instead of spawning its own live display.
# This is what lets concurrent build loops render side by side without fighting
# over the terminal. ``None`` means "no group active" (the normal serial path).
_SHARED_PROGRESS: T.Any = None


def set_progress_enabled(enabled: bool) -> None:
    global _PROGRESS_ENABLED
    _PROGRESS_ENABLED = bool(enabled)


def _stderr_is_tty() -> bool:
    try:
        return bool(sys.stderr.isatty())
    except Exception:  # noqa: BLE001
        return False


class _PlainTrack(T.Generic[_T]):
    """Tiny tqdm-compatible fallback used for non-TTY and forced test output."""

    def __init__(
        self,
        iterable: T.Iterable[_T] | None = None,
        *,
        desc: str = "",
        total: int | None = None,
        render: bool = False,
    ) -> None:
        self._iterable = iterable
        self._desc = desc
        self.total = total
        self.n: int | float = 0
        self._render = render
        self._rendered = False

    def _maybe_render(self) -> None:
        if self._render and not self._rendered and self._desc:
            print(self._desc, file=sys.stderr, flush=True)
            self._rendered = True

    def __iter__(self) -> T.Iterator[_T]:
        self._maybe_render()
        if self._iterable is None:
            return iter(())

        def _generator() -> T.Iterator[_T]:
            for item in self._iterable or ():
                yield item
                self.update(1)

        return _generator()

    def __enter__(self) -> "_PlainTrack[_T]":
        self._maybe_render()
        return self

    def __exit__(self, *_exc: T.Any) -> None:
        self.close()

    def set_description(self, desc: str) -> None:
        self._desc = desc

    def update(self, n: int | float = 1) -> None:
        self.n += n

    def close(self) -> None:
        return None


class _RichTrack(T.Generic[_T]):
    def __init__(
        self,
        iterable: T.Iterable[_T] | None,
        *,
        desc: str,
        total: int | None,
        unit: str,
        unit_scale: bool,
        leave: bool,
        force_terminal: bool,
    ) -> None:
        self._iterable = iterable
        self._desc = desc
        self.total = total
        self.n: int | float = 0
        self._leave = leave
        self._unit = unit
        self._unit_scale = unit_scale
        self._force_terminal = force_terminal
        self._progress: T.Any | None = None
        self._task_id: T.Any | None = None

    def _start(self) -> None:
        if self._progress is not None:
            return
        assert Console is not None
        assert Progress is not None
        assert TextColumn is not None
        assert BarColumn is not None
        assert TimeElapsedColumn is not None
        assert TimeRemainingColumn is not None

        percent_column = (
            TextColumn("{task.percentage:>5.1f}%")
            if self.total is not None
            else TextColumn("")
        )
        count_column = (
            TextColumn("{task.completed}/{task.total}")
            if self.total is not None
            else TextColumn("{task.completed}")
        )
        self._progress = Progress(
            TextColumn("[bold blue]{task.description}[/bold blue]"),
            BarColumn(bar_width=None),
            percent_column,
            count_column,
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=Console(
                file=sys.stderr,
                highlight=False,
                soft_wrap=True,
                force_terminal=self._force_terminal,
            ),
            transient=not self._leave,
            expand=True,
        )
        self._progress.start()
        self._task_id = self._progress.add_task(self._desc, total=self.total)

    def __iter__(self) -> T.Iterator[_T]:
        if self._iterable is None:
            return iter(())
        self._start()

        def _generator() -> T.Iterator[_T]:
            try:
                for item in self._iterable or ():
                    yield item
                    self.update(1)
            finally:
                self.close()

        return _generator()

    def __enter__(self) -> "_RichTrack[_T]":
        self._start()
        return self

    def __exit__(self, *_exc: T.Any) -> None:
        self.close()

    def set_description(self, desc: str) -> None:
        self._desc = desc
        if self._progress is not None and self._task_id is not None:
            self._progress.update(self._task_id, description=desc)

    def update(self, n: int | float = 1) -> None:
        self.n += n
        if self._progress is not None and self._task_id is not None:
            self._progress.advance(self._task_id, n)

    def close(self) -> None:
        if self._progress is not None:
            self._progress.stop()
            self._progress = None
            self._task_id = None


class _SharedTaskTrack(T.Generic[_T]):
    """One task (row) inside a parent-owned shared Progress.

    Adds its task on first use and removes it on close so concurrent loops come
    and go as separate rows in a single live display. All Rich task operations
    are guarded by the Progress' internal lock, so updates from worker threads
    are safe without an explicit callback or queue.
    """

    def __init__(
        self,
        iterable: T.Iterable[_T] | None,
        *,
        desc: str,
        total: int | None,
        progress: T.Any,
    ) -> None:
        self._iterable = iterable
        self._desc = desc
        self.total = total
        self.n: int | float = 0
        self._progress = progress
        self._task_id: T.Any | None = None

    def _start(self) -> None:
        if self._task_id is None:
            self._task_id = self._progress.add_task(self._desc, total=self.total)

    def __iter__(self) -> T.Iterator[_T]:
        if self._iterable is None:
            return iter(())
        self._start()

        def _generator() -> T.Iterator[_T]:
            try:
                for item in self._iterable or ():
                    yield item
                    self.update(1)
            finally:
                self.close()

        return _generator()

    def __enter__(self) -> "_SharedTaskTrack[_T]":
        self._start()
        return self

    def __exit__(self, *_exc: T.Any) -> None:
        self.close()

    def set_description(self, desc: str) -> None:
        self._desc = desc
        if self._task_id is not None:
            with contextlib.suppress(Exception):
                self._progress.update(self._task_id, description=desc)

    def update(self, n: int | float = 1) -> None:
        self.n += n
        if self._task_id is not None:
            with contextlib.suppress(Exception):
                self._progress.advance(self._task_id, n)

    def close(self) -> None:
        if self._task_id is not None:
            with contextlib.suppress(Exception):
                self._progress.remove_task(self._task_id)
            self._task_id = None


@contextlib.contextmanager
def progress_group(*, transient: bool = True) -> T.Iterator[T.Any]:
    """Own a single Rich Progress for a block of concurrent work.

    While active, every ``track()`` call adds a task (row) to this one Progress
    instead of creating its own live display, so concurrent loops -- e.g. several
    dataset builds running on a thread pool -- render as separate rows without
    fighting over the terminal cursor. Yields the Progress (so the caller can add
    its own parent task) or ``None`` when no shared display can be owned
    (progress disabled, Rich unavailable, non-TTY, or already inside a group); in
    that case ``track()`` keeps its normal per-call behavior.
    """
    global _SHARED_PROGRESS
    if (
        not _PROGRESS_ENABLED
        or _SHARED_PROGRESS is not None
        or not rich_available()
        or not _stderr_is_tty()
    ):
        yield _SHARED_PROGRESS
        return

    assert Progress is not None and Console is not None
    progress = Progress(
        TextColumn("[bold blue]{task.description}[/bold blue]"),
        BarColumn(bar_width=None),
        TaskProgressColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=Console(file=sys.stderr, highlight=False, soft_wrap=True),
        transient=transient,
        expand=True,
    )
    progress.start()
    _SHARED_PROGRESS = progress
    try:
        yield progress
    finally:
        _SHARED_PROGRESS = None
        progress.stop()


def track(
    iterable: T.Iterable[_T] | None = None,
    *,
    desc: str,
    total: int | None = None,
    unit: str = "it",
    unit_scale: bool = False,
    leave: bool = False,
    disable: bool | None = None,
) -> T.Any:
    """Wrap an iterable or create a manual progress object.

    ``disable=None`` keeps logs clean by disabling progress outside TTYs.
    ``disable=True`` disables progress.
    ``disable=False`` forces visible output, matching tqdm's historical behavior.
    """

    forced = disable is False
    disabled = disable is True
    stderr_is_tty = _stderr_is_tty()

    if disabled or not _PROGRESS_ENABLED:
        return _PlainTrack(iterable, desc=desc, total=total)

    # A parent owns the live display: add a row to it rather than starting a
    # competing one. This is the concurrent-build path; it takes priority over
    # the per-call TTY/force heuristics below.
    if _SHARED_PROGRESS is not None:
        return _SharedTaskTrack(
            iterable, desc=desc, total=total, progress=_SHARED_PROGRESS
        )

    if not forced and not stderr_is_tty:
        return _PlainTrack(iterable, desc=desc, total=total)

    if not rich_available():
        return _PlainTrack(iterable, desc=desc, total=total, render=forced)

    # disable=False should force visible progress. Some terminals and wrappers
    # used by IDEs report stderr as non-TTY even though Rich output is useful.
    # In that case, force Rich terminal rendering instead of falling back to a
    # one-line plain label.
    return _RichTrack(
        iterable,
        desc=desc,
        total=total,
        unit=unit,
        unit_scale=unit_scale,
        leave=leave,
        force_terminal=forced or not stderr_is_tty,
    )
