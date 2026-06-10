"""Tests for console log formatting helpers."""

from __future__ import annotations

import math

from lib.training.log_format import (
    fmt_count,
    fmt_duration,
    fmt_mapping,
    fmt_num,
    fmt_progress,
)


def test_fmt_num_precision_and_specials():
    assert fmt_num(2.2378997802734375) == "2.2379"
    assert fmt_num(2.2378997802734375, 2) == "2.24"
    assert fmt_num(0) == "0.0000"
    assert fmt_num(None) == "n/a"
    assert fmt_num(float("nan")) == "nan"
    assert fmt_num(float("inf")) == "inf"
    assert fmt_num(float("-inf")) == "-inf"
    # Non-numeric input falls back to its string form rather than raising.
    assert fmt_num("ema") == "ema"


def test_fmt_count_groups_digits():
    assert fmt_count(40000) == "40,000"
    assert fmt_count(5440) == "5,440"
    assert fmt_count(0) == "0"


def test_fmt_duration_scales_units():
    assert fmt_duration(0.085) == "85ms"
    assert fmt_duration(3.456) == "3.46s"
    assert fmt_duration(65) == "1m05s"
    assert fmt_duration(3723) == "1h02m03s"
    assert fmt_duration(None) == "n/a"


def test_fmt_progress_reports_percentage():
    assert fmt_progress(5440, 40000) == "5,440/40,000 ( 13.6%)"
    assert fmt_progress(40000, 40000) == "40,000/40,000 (100.0%)"
    # No total -> just the count, no divide-by-zero.
    assert fmt_progress(7, 0) == "7"


def test_fmt_mapping_mixed_types():
    assert fmt_mapping({"loc": 0.1234, "heat": 1.842}) == "loc=0.1234 heat=1.8420"
    # Integer values keep grouping; floats use precision.
    assert fmt_mapping({"head_68": 1200, "head_98": 6}) == "head_68=1,200 head_98=6"
    assert fmt_mapping({}) == "-"
    assert fmt_mapping(None) == "-"
    assert fmt_mapping({"ok": True}) == "ok=True"


def test_fmt_num_matches_round_trip_for_typical_loss():
    # The per-batch trainer log formats loss.item() values; ensure no scientific
    # notation or runaway precision leaks through.
    text = fmt_num(math.pi, 4)
    assert text == "3.1416"
    assert "e" not in text
