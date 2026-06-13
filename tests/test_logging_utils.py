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
    assert (
        summarize_mapping(mapping, top_n=2, as_percent=True)
        == "anchor=75.0% profile=25.0%"
    )


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


def test_configure_console_logging_default_stdlib_does_not_crash():
    configure_console_logging(Verbosity.INFO, "human")


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
    # Guard against accidentally using pprint(), which quotes table rows.
    assert "'  frontal" not in out
    assert "'  profile" not in out


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


def test_batch_mix_summary_line_is_compact():
    from lib.training.heatmap_stage import _batch_mix_summary_line

    mix = {
        "bucket": {
            "occlusion": 10,
            "anchor": 9,
            "blur": 1,
            "semifrontal": 1,
            "profile": 9,
            "expression": 2,
        },
        "dataset": {
            "cofw68": 6,
            "aflw2000": 8,
            "wflw": 3,
            "multipie": 5,
            "merl_rav": 5,
            "cofw29": 1,
            "menpo2d": 2,
            "w300": 2,
        },
        "schema": {"2d_68": 22, "2d_98": 3, "2d_39": 6, "2d_29": 1},
    }

    line = _batch_mix_summary_line(mix)

    assert "bucket occlusion=31.2%" in line
    assert "schema 2d_68=68.8%" in line
    assert "dataset aflw2000=25.0%" in line
    assert "+2 more" in line or "+5 more" in line
    assert "{" not in line
    assert "}" not in line


def test_sampler_targets_summary_is_compact():
    from lib.training.loaders import _sampler_targets_summary

    targets = {
        "bucket": {
            "anchor": 0.25,
            "occlusion": 0.25,
            "profile": 0.25,
            "profile_occlusion": 0.25,
        },
        "dataset": {
            "300vw": 1.0,
            "aflw2000": 1.0,
            "cofw29": 1.0,
            "cofw68": 1.0,
            "fll2": 1.0,
        },
        "schema": {
            "2d_106": 1.0,
            "2d_194": 1.0,
            "2d_29": 1.0,
            "2d_39": 1.0,
            "2d_68": 1.0,
        },
    }

    line = _sampler_targets_summary(targets)

    assert line.startswith("bucket ")
    assert "dataset " in line
    assert "schema " in line
    assert "+1 more" in line
    assert "{" not in line
    assert "}" not in line


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


def test_pipeline_log_level_accepts_quiet_and_normal_alias():
    from tools.run_cdvit_manifest_training_pipeline import _build_arg_parser

    parser = _build_arg_parser()
    assert parser.parse_args(["--log-level", "quiet"]).log_level == "quiet"
    assert parser.parse_args(["--log-level", "normal"]).log_level == "normal"


def test_pipeline_normal_log_level_maps_to_trainer_info():
    from tools.run_cdvit_manifest_training_pipeline import (
        _trainer_log_level_for_pipeline,
    )

    assert _trainer_log_level_for_pipeline("normal") == "info"
    assert _trainer_log_level_for_pipeline("quiet") == "quiet"
    assert _trainer_log_level_for_pipeline("verbose") == "verbose"


def test_pipeline_production_command_forwards_logging_flags():
    from argparse import Namespace
    from pathlib import Path

    from tools.run_cdvit_manifest_training_pipeline import (
        PipelinePaths,
        _production_build_command,
    )

    args = Namespace(
        python_executable="python",
        prod_dir=Path("prod"),
        log_format="json",
        log_level="normal",
        production_build_arg=[],
    )
    paths = PipelinePaths(output_root=Path("runs"), run_name="demo")

    command = _production_build_command(args, paths)

    assert command is not None
    assert command[command.index("--log-format") + 1] == "json"
    assert command[command.index("--log-level") + 1] == "info"


def test_production_manifest_parser_accepts_shared_logging_flags():
    from tools.build_production_validated_manifest import _parser

    args = _parser().parse_args(
        [
            "--prod-dir",
            "prod",
            "--output-dir",
            "out",
            "--log-format",
            "json",
            "--log-level",
            "quiet",
        ]
    )

    assert args.log_format == "json"
    assert args.log_level == "quiet"


def test_quality_manifest_parser_accepts_progress_flag():
    from tools.build_quality_dataset import _parser

    args = _parser().parse_args(
        [
            "--dataset",
            "wflw",
            "--output-dir",
            "out",
            "--no-progress",
        ]
    )

    assert args.progress is False


def test_hard_negative_parser_accepts_shared_logging_flags():
    from tools.build_hard_negative_manifest import _parser

    args = _parser().parse_args(
        [
            "--w300-manifest",
            "manifest.json",
            "--output-dir",
            "out",
            "--log-format",
            "json",
            "--log-level",
            "quiet",
        ]
    )

    assert args.log_format == "json"
    assert args.log_level == "quiet"


def test_prepare_parser_accepts_shared_logging_flags():
    from tools.prepare_landmark_dataset import _parser

    args = _parser().parse_args(
        [
            "--datasets",
            "wflw",
            "--log-format",
            "json",
            "--log-level",
            "verbose",
            "--no-progress",
            "--skip-build",
        ]
    )

    assert args.log_format == "json"
    assert args.log_level == "verbose"
    assert args.progress is False
    assert args.skip_build is True


def test_dataset_track_disabled_under_capture():
    from lib.datasets.progress import set_progress_enabled, track

    set_progress_enabled(True)
    values = list(track([1, 2, 3], desc="Datasets", total=3))

    assert values == [1, 2, 3]


def test_progress_group_noop_when_disabled(monkeypatch):
    import lib.datasets.progress as prog

    monkeypatch.setattr(prog, "_PROGRESS_ENABLED", False)
    with prog.progress_group() as group:
        assert group is None


def test_track_routes_concurrent_loops_into_one_shared_progress(monkeypatch):
    import lib.datasets.progress as prog

    class FakeProgress:
        def __init__(self):
            self.tasks = {}
            self._next = 0
            self.removed = []

        def add_task(self, desc, total=None):
            tid = self._next
            self._next += 1
            self.tasks[tid] = {"desc": desc, "total": total, "completed": 0}
            return tid

        def advance(self, tid, n=1):
            self.tasks[tid]["completed"] += n

        def update(self, tid, **kwargs):
            self.tasks[tid].update(kwargs)

        def remove_task(self, tid):
            self.removed.append(tid)

    fake = FakeProgress()
    monkeypatch.setattr(prog, "_PROGRESS_ENABLED", True)
    monkeypatch.setattr(prog, "_SHARED_PROGRESS", fake)

    # Two loops (as if from two worker threads) each become a distinct row in the
    # single shared Progress instead of starting competing live displays.
    out_a = list(prog.track([1, 2, 3], desc="Build a", total=3))
    out_b = list(prog.track([1, 2], desc="Build b", total=2))

    assert out_a == [1, 2, 3]
    assert out_b == [1, 2]
    assert {t["desc"] for t in fake.tasks.values()} == {"Build a", "Build b"}
    assert fake.tasks[0]["completed"] == 3
    assert fake.tasks[1]["completed"] == 2
    # Each task is removed when its loop finishes so rows do not accumulate.
    assert set(fake.removed) == {0, 1}


def test_quality_build_joins_parent_progress_display(monkeypatch, tmp_path):
    from argparse import Namespace

    import lib.datasets.build.orchestrator as orchestrator
    import lib.datasets.progress as prog

    class FakeProgress:
        def __init__(self):
            self.tasks = {}
            self._next = 0
            self.removed = []

        def add_task(self, desc, total=None, **kwargs):
            tid = self._next
            self._next += 1
            self.tasks[tid] = {"desc": desc, "total": total, "completed": 0}
            return tid

        def advance(self, tid, n=1):
            self.tasks[tid]["completed"] += n

        def update(self, tid, **kwargs):
            self.tasks[tid].update(kwargs)

        def remove_task(self, tid):
            self.removed.append(tid)

    fake = FakeProgress()
    manifest = tmp_path / "manifest.json"
    monkeypatch.setattr(prog, "_PROGRESS_ENABLED", True)
    monkeypatch.setattr(prog, "_SHARED_PROGRESS", fake)
    monkeypatch.setattr(
        orchestrator,
        "_build_without_progress",
        lambda args: manifest,
    )

    result = orchestrator.build(Namespace(dataset="wflw"))

    assert result == manifest
    assert prog._SHARED_PROGRESS is fake
    assert list(fake.tasks.values()) == [
        {
            "desc": "Build wflw pipeline",
            "total": 1,
            "completed": 1,
        }
    ]
    assert fake.removed == [0]
