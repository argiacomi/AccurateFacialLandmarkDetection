"""Back-compat shim: console formatters now live in :mod:`lib.logging_utils`.

The formatting helpers were promoted to the shared, top-level logging module so
the trainer, pipeline, and dataset tools import them from one place. This module
re-exports them so existing ``from lib.training.log_format import ...`` imports
keep working.
"""

from __future__ import annotations

from lib.logging_utils import (
    fmt_count,
    fmt_duration,
    fmt_mapping,
    fmt_num,
    fmt_progress,
)

__all__ = [
    "fmt_count",
    "fmt_duration",
    "fmt_mapping",
    "fmt_num",
    "fmt_progress",
]
