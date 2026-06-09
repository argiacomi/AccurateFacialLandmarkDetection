"""Console log formatting helpers for readable training output.

These produce compact, aligned, human-scannable fragments for the trainer's
stdout. The machine-readable record of a run is ``runtime_metrics.jsonl`` (see
:func:`lib.training.profiling.append_runtime_metrics`); these helpers are for the
console only and intentionally trade precision for readability.

Convention: lines are prefixed with a short lowercase ``[tag]`` (``[train]``,
``[epoch]``, ``[eval]``, ``[data]``, ...) so a run is easy to skim and grep.
"""

from __future__ import annotations

import math
import typing as T


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


def fmt_mapping(mapping: T.Mapping[str, T.Any] | None, precision: int = 4) -> str:
    """Format a ``name -> number`` mapping as ``name=val name2=val2``.

    Integer values keep thousands separators; floats use ``precision``. An empty
    or missing mapping renders as ``-`` so it stays compact inside a log line.
    """

    if not mapping:
        return "-"
    parts: list[str] = []
    for key, value in mapping.items():
        if isinstance(value, bool):
            parts.append(f"{key}={value}")
        elif isinstance(value, int):
            parts.append(f"{key}={value:,}")
        else:
            parts.append(f"{key}={fmt_num(value, precision)}")
    return " ".join(parts)


__all__ = [
    "fmt_count",
    "fmt_duration",
    "fmt_mapping",
    "fmt_num",
    "fmt_progress",
]
