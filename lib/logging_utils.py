"""Shared console logging for the CD-ViT trainer, pipeline, and dataset tools.

Goals:

* One consistent, human-first console format with short lowercase ``[tag]``
  prefixes that are easy to skim and grep.
* A ``--log-format json`` mode that emits the same events as JSONL for CI/debug.
* Verbosity tiers (``--quiet`` / default / ``--verbose`` / ``--debug``) so a run
  can be terse or detailed without changing ``--log-every``.

The detailed, machine-readable record of a run still lives in the structured
files (``runtime_metrics.jsonl``, ``pipeline_progress.jsonl``,
``eval_report.json``, ``dataset_audit.json``). The console only summarizes and
points at those files; these helpers intentionally trade precision for
readability.
"""

from __future__ import annotations

import enum
import json
import logging
import math
import sys
import typing as T


class Verbosity(enum.IntEnum):
    """Console detail tiers, ordered from terse (QUIET) to noisy (DEBUG).

    A message is tagged with the lowest tier at which it should appear and is
    shown when the active verbosity is at least that tier (see :func:`is_enabled`).
    """

    QUIET = 0  # only stage/epoch/eval summaries and errors
    INFO = 1  # default: per-batch train/eval lines
    VERBOSE = 2  # adds head diagnostics, sampler detail, checkpoint writes
    DEBUG = 3  # adds full structures and stack traces


_STATE: dict[str, T.Any] = {
    "verbosity": Verbosity.INFO,
    "log_format": "human",
}


# --------------------------------------------------------------------------- #
# Configuration / state
# --------------------------------------------------------------------------- #
def configure_console_logging(
    verbosity: Verbosity | int = Verbosity.INFO,
    log_format: str = "human",
    *,
    configure_stdlib: bool = True,
) -> None:
    """Set the process-wide console verbosity and output format.

    ``configure_stdlib`` also points the standard ``logging`` module at the
    console with a bare ``%(message)s`` format so dataset tools that use
    ``logging.getLogger`` share the same clean look (no ``QUIET:name:`` prefix).
    """

    _STATE["verbosity"] = Verbosity(int(verbosity))
    _STATE["log_format"] = "json" if str(log_format).lower() == "json" else "human"
    if configure_stdlib:
        std_level = (
            logging.DEBUG if _STATE["verbosity"] >= Verbosity.DEBUG else logging.QUIET
        )
        logging.basicConfig(level=std_level, format="%(message)s", force=True)


def get_verbosity() -> Verbosity:
    return _STATE["verbosity"]


def get_log_format() -> str:
    return _STATE["log_format"]


def is_enabled(level: Verbosity | int) -> bool:
    """Whether a message at ``level`` should be shown at the active verbosity."""

    return _STATE["verbosity"] >= Verbosity(int(level))


def is_verbose() -> bool:
    return is_enabled(Verbosity.VERBOSE)


def is_debug() -> bool:
    return is_enabled(Verbosity.DEBUG)


#: Accepted ``--log-level`` names, ordered terse -> noisy.
LOG_LEVEL_NAMES: tuple[str, ...] = ("quiet", "info", "verbose", "debug")

_VERBOSITY_BY_NAME = {
    "quiet": Verbosity.QUIET,
    "info": Verbosity.INFO,
    "verbose": Verbosity.VERBOSE,
    "debug": Verbosity.DEBUG,
}


def verbosity_from_name(name: str | None) -> Verbosity:
    """Resolve a ``--log-level`` name to a :class:`Verbosity` tier.

    Unknown or missing names fall back to ``INFO`` so callers never crash on a
    stray value.
    """

    if not name:
        return Verbosity.INFO
    return _VERBOSITY_BY_NAME.get(str(name).lower(), Verbosity.INFO)


# --------------------------------------------------------------------------- #
# Emit
# --------------------------------------------------------------------------- #
def log_event(
    tag: str,
    message: str = "",
    *,
    level: Verbosity | int = Verbosity.INFO,
    stream: T.TextIO | None = None,
    **fields: T.Any,
) -> None:
    """Emit one console event.

    Human mode prints ``[tag] message``. JSON mode prints a single-line JSON
    object carrying ``tag``, ``message``, and any structured ``fields``. Nothing
    is printed when ``level`` exceeds the active verbosity.
    """

    if not is_enabled(level):
        return
    out = stream if stream is not None else sys.stdout
    if _STATE["log_format"] == "json":
        payload = {"tag": tag, "message": message, **fields}
        print(json.dumps(payload, sort_keys=True, default=str), file=out, flush=True)
    else:
        text = f"[{tag}] {message}" if message else f"[{tag}]"
        print(text, file=out, flush=True)


def log_error(tag: str, message: str, **fields: T.Any) -> None:
    """Emit an error line. Always shown (even under ``--quiet``) and goes to stderr."""

    log_event(tag, message, level=Verbosity.QUIET, stream=sys.stderr, **fields)


def log_table(
    tag: str,
    title: str,
    rows: T.Sequence[T.Sequence[T.Any]],
    *,
    level: Verbosity | int = Verbosity.INFO,
    headers: T.Sequence[str] | None = None,
) -> None:
    """Emit an aligned key/value style table under a ``[tag] title`` header.

    In JSON mode the rows are emitted as a structured payload instead.
    """

    if not is_enabled(level):
        return
    if _STATE["log_format"] == "json":
        log_event(
            tag,
            title,
            level=level,
            rows=[list(row) for row in rows],
            headers=list(headers) if headers else None,
        )
        return
    str_rows = [[str(cell) for cell in row] for row in rows]
    widths: list[int] = []
    for row in ([list(headers)] if headers else []) + str_rows:
        for col, cell in enumerate(row):
            if col >= len(widths):
                widths.append(len(cell))
            else:
                widths[col] = max(widths[col], len(cell))
    print(f"[{tag}] {title}", flush=True)
    if headers:
        print("  " + "  ".join(h.ljust(widths[i]) for i, h in enumerate(headers)))
    for row in str_rows:
        print("  " + "  ".join(cell.ljust(widths[i]) for i, cell in enumerate(row)))


# --------------------------------------------------------------------------- #
# Value formatters (human-first; trade precision for readability)
# --------------------------------------------------------------------------- #
def fmt_num(value: T.Any, precision: int = 4) -> str:
    """Format a scalar with fixed precision; ``None`` -> ``n/a``, NaN/Inf named."""

    if value is None:
        return "n/a"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if math.isnan(number):
        return "nan"
    if math.isinf(number):
        return "inf" if number > 0 else "-inf"
    return f"{number:.{precision}f}"


def fmt_count(value: T.Any) -> str:
    """Format an integer count with thousands separators (``40000`` -> ``40,000``)."""

    try:
        return f"{int(value):,}"
    except (TypeError, ValueError):
        return str(value)


def fmt_duration(seconds: T.Any) -> str:
    """Human-readable elapsed time: ``850ms``, ``3.45s``, ``2m05s``, ``1h02m03s``."""

    if seconds is None:
        return "n/a"
    total = float(seconds)
    if math.isnan(total) or math.isinf(total):
        return fmt_num(total)
    if total < 1.0:
        return f"{total * 1000:.0f}ms"
    if total < 60.0:
        return f"{total:.2f}s"
    minutes, secs = divmod(int(round(total)), 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{secs:02d}s"
    return f"{minutes}m{secs:02d}s"


def fmt_progress(done: T.Any, total: T.Any) -> str:
    """Format progress as ``done/total (pct%)`` with grouped digits."""

    done_int = int(done)
    total_int = int(total)
    if total_int > 0:
        pct = 100.0 * done_int / total_int
        return f"{done_int:,}/{total_int:,} ({pct:5.1f}%)"
    return f"{done_int:,}"


def _is_zero(value: T.Any) -> bool:
    if isinstance(value, bool):
        return False
    try:
        return float(value) == 0.0
    except (TypeError, ValueError):
        return False


def _as_float(value: T.Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float("-inf")
    return number if math.isfinite(number) else float("-inf")


def fmt_mapping(
    mapping: T.Mapping[str, T.Any] | None,
    precision: int = 4,
    *,
    keys: T.Sequence[str] | None = None,
    omit_zero: bool = False,
    max_items: int | None = None,
) -> str:
    """Format a ``name -> number`` mapping as ``name=val name2=val2``.

    Integer values keep thousands separators; floats use ``precision``. An empty
    or missing mapping renders as ``-`` so it stays compact in a log line.

    * ``keys`` pins display order (and restricts to those keys) for stable
      columns across steps.
    * ``omit_zero`` drops exactly-zero entries to keep wide lines short.
    * ``max_items`` caps the entries shown, appending ``+N more``.
    """

    if not mapping:
        return "-"
    if keys is not None:
        order = {key: index for index, key in enumerate(keys)}
        items = [(key, mapping[key]) for key in keys if key in mapping]
    else:
        items = list(mapping.items())
        order = None
    if omit_zero:
        items = [(key, value) for key, value in items if not _is_zero(value)]
    if order is not None:
        items.sort(key=lambda kv: order[kv[0]])
    extra = 0
    if max_items is not None and len(items) > max_items:
        extra = len(items) - max_items
        items = items[:max_items]
    if not items:
        return "-"
    parts: list[str] = []
    for key, value in items:
        if isinstance(value, bool):
            parts.append(f"{key}={value}")
        elif isinstance(value, int):
            parts.append(f"{key}={value:,}")
        else:
            parts.append(f"{key}={fmt_num(value, precision)}")
    text = " ".join(parts)
    if extra > 0:
        text += f" +{extra} more"
    return text


def summarize_mapping(
    mapping: T.Mapping[str, T.Any] | None,
    *,
    top_n: int = 5,
    precision: int = 1,
    as_percent: bool = False,
    total: float | None = None,
) -> str:
    """Summarize a mapping as its largest ``top_n`` entries, ``name=value``.

    With ``as_percent`` the values are rendered as a share of ``total`` (or the
    sum of all values). Remaining entries collapse into a ``+N more`` suffix.
    Empty/missing mappings render as ``-``.
    """

    if not mapping:
        return "-"
    items = sorted(mapping.items(), key=lambda kv: _as_float(kv[1]), reverse=True)
    shown = items[: max(int(top_n), 0)]
    if as_percent:
        denom = float(total) if total else sum(_as_float(v) for _, v in items)
        denom = denom if denom else 1.0
        parts = [
            f"{key}={100.0 * _as_float(value) / denom:.{precision}f}%"
            for key, value in shown
        ]
    else:
        parts = [f"{key}={fmt_num(value, precision)}" for key, value in shown]
    text = " ".join(parts) if parts else "-"
    extra = len(items) - len(shown)
    if extra > 0:
        text += f" +{extra} more"
    return text


__all__ = [
    "LOG_LEVEL_NAMES",
    "Verbosity",
    "configure_console_logging",
    "fmt_count",
    "fmt_duration",
    "fmt_mapping",
    "fmt_num",
    "fmt_progress",
    "get_log_format",
    "get_verbosity",
    "is_debug",
    "is_enabled",
    "is_verbose",
    "log_error",
    "log_event",
    "log_table",
    "summarize_mapping",
    "verbosity_from_name",
]
