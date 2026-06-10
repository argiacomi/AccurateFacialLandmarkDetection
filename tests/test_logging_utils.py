"""Tests for the shared console logging helpers."""

from __future__ import annotations

import json

import pytest

from lib.logging_utils import (
    Verbosity,
    configure_console_logging,
    fmt_mapping,
    log_error,
    log_event,
    log_table,
    summarize_mapping,
    verbosity_from_name,
)


@pytest.fixture(autouse=True)
def _reset_logging_state():
    """Keep global console-logging state from leaking across tests."""

    configure_console_logging(Verbosity.INFO, "human", configure_stdlib=False)
    yield
    configure_console_logging(Verbosity.INFO, "human", configure_stdlib=False)


# --------------------------------------------------------------------------- #
# fmt_mapping extensions
# --------------------------------------------------------------------------- #
def test_fmt_mapping_omit_zero_drops_zero_components():
    mapping = {"loc": 0.12, "heat": 1.84, "star": 0.0, "vis": 0.0, "aux": 0.13}
    assert fmt_mapping(mapping, omit_zero=True) == "loc=0.1200 heat=1.8400 aux=0.1300"


def test_fmt_mapping_keys_pins_order_and_filters():
    mapping = {"aux": 0.13, "loc": 0.12, "heat": 1.84}
    out = fmt_mapping(mapping, keys=("loc", "heat", "star", "aux"))
    assert out == "loc=0.1200 heat=1.8400 aux=0.1300"


def test_fmt_mapping_max_items_appends_more():
    mapping = {"a": 5, "b": 4, "c": 3, "d": 2}
    assert fmt_mapping(mapping, max_items=2) == "a=5 b=4 +2 more"


def test_fmt_mapping_all_zero_omitted_renders_dash():
    assert fmt_mapping({"x": 0.0, "y": 0}, omit_zero=True) == "-"


# --------------------------------------------------------------------------- #
# summarize_mapping
# --------------------------------------------------------------------------- #
def test_summarize_mapping_top_n_by_value_with_more():
    mapping = {"a": 1, "b": 9, "c": 5, "d": 2}
    assert summarize_mapping(mapping, top_n=2) == "b=9.0 c=5.0 +2 more"


def test_summarize_mapping_as_percent():
    mapping = {"profile": 250, "anchor": 750}
    assert summarize_mapping(mapping, top_n=2, as_percent=True) == "anchor=75.0% profile=25.0%"


def test_summarize_mapping_empty():
    assert summarize_mapping({}) == "-"
    assert summarize_mapping(None) == "-"


# --------------------------------------------------------------------------- #
# verbosity
# --------------------------------------------------------------------------- #
def test_verbosity_from_name():
    assert verbosity_from_name("quiet") is Verbosity.QUIET
    assert verbosity_from_name("info") is Verbosity.INFO
    assert verbosity_from_name("verbose") is Verbosity.VERBOSE
    assert verbosity_from_name("debug") is Verbosity.DEBUG
    # Unknown / missing -> INFO.
    assert verbosity_from_name("nope") is Verbosity.INFO
    assert verbosity_from_name(None) is Verbosity.INFO


# --------------------------------------------------------------------------- #
# log_event human / json / verbosity gating
# --------------------------------------------------------------------------- #
def test_log_event_human_prefixes_tag(capsys):
    log_event("train", "epoch 1 | loss 0.5", level=Verbosity.INFO)
    assert capsys.readouterr().out == "[train] epoch 1 | loss 0.5\n"


def test_log_event_suppressed_below_active_verbosity(capsys):
    configure_console_logging(Verbosity.QUIET, "human", configure_stdlib=False)
    log_event("train", "per-batch", level=Verbosity.INFO)  # hidden under --quiet
    log_event("epoch", "summary", level=Verbosity.QUIET)  # shown
    out = capsys.readouterr().out
    assert "per-batch" not in out
    assert "[epoch] summary" in out


def test_log_event_json_mode_emits_structured_payload(capsys):
    configure_console_logging(Verbosity.INFO, "json", configure_stdlib=False)
    log_event("eval", "done", level=Verbosity.QUIET, nme=3.45, n=2048)
    payload = json.loads(capsys.readouterr().out.strip())
    assert payload == {"tag": "eval", "message": "done", "nme": 3.45, "n": 2048}


def test_log_error_goes_to_stderr_even_under_quiet(capsys):
    configure_console_logging(Verbosity.QUIET, "human", configure_stdlib=False)
    log_error("pipeline", "boom")
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "[pipeline] boom" in captured.err


def test_log_table_aligns_columns(capsys):
    log_table("eval", "slices", [["frontal", "3.21"], ["profile", "5.0"]])
    out = capsys.readouterr().out
    assert "[eval] slices" in out
    # Second column is left-aligned under a shared width.
    assert "  frontal  3.21" in out
    assert "  profile  5.0" in out


# --------------------------------------------------------------------------- #
# Domain artifacts that build on the helpers
# --------------------------------------------------------------------------- #
def test_sampler_summary_line_is_compact():
    from lib.training.heatmap_stage import _sampler_summary_line

    diagnostics = {
        "actual_mix": {
            "bucket": {
                "profile_occlusion": 372,
                "profile": 250,
                "occlusion": 248,
                "anchor": 130,
            }
        },
        "fallback_counts": {"exact": 100, "exact_to_bucket": 10, "bucket_to_any": 8},
        "missing_targets": {"bucket": [], "dataset": ["x", "y"], "schema": []},
    }
    line = _sampler_summary_line(12, diagnostics)
    assert line.startswith("e012 domain mix |")
    assert "profile_occlusion=37.2%" in line
    assert "fallback 18" in line  # non-exact fallbacks summed
    assert "missing bucket=0 dataset=2 schema=0" in line
    # No raw dict braces leak into the console line.
    assert "{" not in line


def test_pipeline_stage_summary_line():
    from tools.run_cdvit_manifest_training_pipeline import (
        StageResult,
        _stage_summary_line,
    )

    result = StageResult(
        name="validate_cdvit_manifest",
        status="ok",
        duration_seconds=12.3,
        outputs=["a.json", "b.json"],
        notes=["valid 418,220 / 421,800", "schemas 6"],
    )
    line = _stage_summary_line(result)
    assert line == (
        "validate_cdvit_manifest ok | 12.30s | outputs 2 | "
        "valid 418,220 / 421,800 | schemas 6"
    )

    errored = StageResult(
        name="train_cdvit", status="error", duration_seconds=1.0, error="boom"
    )
    assert _stage_summary_line(errored).endswith("| error boom")
