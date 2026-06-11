from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest

SCRIPT = (
    Path(__file__).resolve().parents[1] / "tools" / "tune_training_hyperparameters.py"
)
spec = importlib.util.spec_from_file_location("tune_training_hyperparameters", SCRIPT)
tuner = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(tuner)


def test_objective_score_uses_hard_slices_and_reports_missing():
    metrics = {
        "heldout_68_nme": 0.04,
        "profile_nme": 0.06,
        "occlusion_nme": 0.05,
    }

    score, diagnostics = tuner.objective_score(
        metrics,
        hard_slice_weight=0.25,
        required_slices=("profile_nme", "occlusion_nme", "profile_occlusion_nme"),
    )

    assert score == pytest.approx(0.04 + 0.25 * 0.06 + 0.25 * 0.05)
    assert diagnostics["missing_slices"] == ["profile_occlusion_nme"]
    assert diagnostics["used_slices"] == {"profile_nme": 0.06, "occlusion_nme": 0.05}


def test_objective_score_penalizes_overall_and_frontal_regressions():
    score, diagnostics = tuner.objective_score(
        {"heldout_68_nme": 0.045, "frontal_nme": 0.035},
        baseline_metrics={"heldout_68_nme": 0.04, "frontal_nme": 0.03},
        hard_slice_weight=0.25,
        regression_penalty_weight=2.0,
        required_slices=(),
    )

    assert diagnostics["regressions"] == {
        "overall_68_nme": pytest.approx(0.005),
        "frontal_nme": pytest.approx(0.005),
    }
    assert score == pytest.approx(0.045 + 2.0 * 0.005 + 2.0 * 0.005)


def test_generate_star_bracket_keeps_non_star_values_fixed(tmp_path):
    args = tuner.build_arg_parser().parse_args(
        [
            "--output-dir",
            str(tmp_path),
            "--star-bracket",
            "0,0.01",
        ]
    )
    base = tuner.baseline_config(args)
    runs = tuner.generate_star_bracket(args, base)

    assert [run["params"]["star_loss_weight"] for run in runs] == [0.0, 0.01]
    assert {run["params"]["schema_consistency_weight"] for run in runs} == {
        base["schema_consistency_weight"]
    }
    assert {run["params"]["auxiliary_loss_weight"] for run in runs} == {
        base["auxiliary_loss_weight"]
    }
    assert {run["params"]["lr"] for run in runs} == {base["lr"]}


def test_rank_results_and_multiseed_aggregation():
    results = [
        {"id": "a-1", "parent_trial": "a", "score": 0.5, "params": {"lr": 1e-4}},
        {"id": "a-2", "parent_trial": "a", "score": 0.7, "params": {"lr": 1e-4}},
        {"id": "b-1", "parent_trial": "b", "score": 0.4, "params": {"lr": 2e-4}},
        {"id": "b-2", "parent_trial": "b", "score": 0.6, "params": {"lr": 2e-4}},
    ]

    ranked = tuner.rank_results(results, top_k=2)
    assert [item["id"] for item in ranked] == ["b-1", "a-1"]

    summaries = tuner.aggregate_by_parent(results, top_k=2)
    assert summaries[0]["parent_trial"] == "b"
    assert summaries[0]["mean_score"] == pytest.approx(0.5)
    assert summaries[1]["parent_trial"] == "a"
    assert summaries[1]["mean_score"] == pytest.approx(0.6)


def test_lr_sweep_freezes_loss_weights(tmp_path):
    args = tuner.build_arg_parser().parse_args(
        [
            "--output-dir",
            str(tmp_path),
            "--lr-sweep",
            "0.0001,0.0002",
        ]
    )
    selected = {
        "star_loss_weight": 0.01,
        "schema_consistency_weight": 0.04,
        "auxiliary_loss_weight": 0.05,
        "locw": 1.0,
        "hw": 10.0,
        "lr": 1e-4,
    }
    runs = tuner.generate_lr_sweep(args, selected)

    assert [run["params"]["lr"] for run in runs] == [1e-4, 2e-4]
    assert {run["params"]["star_loss_weight"] for run in runs} == {0.01}
    assert {run["params"]["schema_consistency_weight"] for run in runs} == {0.04}
    assert {run["params"]["auxiliary_loss_weight"] for run in runs} == {0.05}


def test_build_train_command_uses_eval_report_and_runtime_jsonl(tmp_path):
    args = tuner.build_arg_parser().parse_args(
        [
            "--output-dir",
            str(tmp_path),
            "--train-command",
            "python TrainHeatmapStageFP16.py",
        ]
    )
    run = tuner.make_run_config(
        stage="baseline",
        params=tuner.baseline_config(args),
        seed=17,
        index=0,
    )
    command = tuner.build_train_command(args, run, tmp_path / "runs" / run["id"])

    assert "--eval-report-json" in command
    assert command[command.index("--eval-report-json") + 1].endswith("metrics.json")
    assert "--runtime-metrics-jsonl" in command
    assert command[command.index("--runtime-metrics-jsonl") + 1].endswith(
        "runtime_metrics.jsonl"
    )
    assert "--runtime-metrics-path" not in command


def test_normalize_metrics_flattens_trainer_eval_report():
    metrics = tuner.normalize_metrics(
        {
            "model": {
                "overall": {"nme": 0.04},
                "by_hard_negative_bucket": {
                    "profile": {"nme": 0.06},
                    "occlusion": {"nme": 0.05},
                    "profile_occlusion": {"nme": 0.08},
                    "anchor": {"nme": 0.035},
                },
                "by_face_size": {"small": {"nme": 0.07}},
            }
        }
    )

    assert metrics["heldout_68_nme"] == pytest.approx(0.04)
    assert metrics["profile_nme"] == pytest.approx(0.06)
    assert metrics["occlusion_nme"] == pytest.approx(0.05)
    assert metrics["profile_occlusion_nme"] == pytest.approx(0.08)
    assert metrics["blur_nme"] == pytest.approx(0.07)
    assert metrics["frontal_nme"] == pytest.approx(0.035)


def test_write_recommendation_outputs_flags_and_json(tmp_path):
    args = tuner.build_arg_parser().parse_args(["--output-dir", str(tmp_path)])
    selected_loss = {
        "parent_trial": "loss-a",
        "mean_score": 0.05,
        "std_score": 0.001,
        "params": {
            "star_loss_weight": 0.01,
            "schema_consistency_weight": 0.04,
            "auxiliary_loss_weight": 0.05,
            "locw": 1.0,
            "hw": 10.0,
            "lr": 1e-4,
        },
    }
    selected_lr = {
        "parent_trial": "lr-b",
        "mean_score": 0.045,
        "std_score": 0.0005,
        "params": {
            "star_loss_weight": 0.01,
            "schema_consistency_weight": 0.04,
            "auxiliary_loss_weight": 0.05,
            "locw": 1.0,
            "hw": 10.0,
            "lr": 2e-4,
        },
    }

    recommendation = tuner.write_recommendation(
        args,
        baseline_result={"score": 0.06},
        selected_loss_summary=selected_loss,
        selected_lr_summary=selected_lr,
    )

    assert recommendation["recommended"] == {
        "star_loss_weight": 0.01,
        "schema_consistency_weight": 0.04,
        "auxiliary_loss_weight": 0.05,
        "locw": 1.0,
        "hw": 10.0,
        "lr": 2e-4,
    }
    assert "--lr" in recommendation["training_flags"]
    assert (tmp_path / "best_training_hyperparameters.json").exists()


class _FakeTrial:
    def __init__(self, number: int):
        self.number = number
        self.params = {}

    def suggest_float(self, name, low, high, log=False):
        value = (float(low) + float(high)) / 2.0
        self.params[name] = value
        return value


class _FakeStudy:
    def __init__(self):
        self.asked = []
        self.told = []
        self.attrs = {}

    def ask(self):
        trial = _FakeTrial(len(self.asked))
        self.asked.append(trial)
        return trial

    def tell(self, number, value):
        self.told.append((int(number), float(value)))

    def get_trials(self, deepcopy=False):
        return []

    def set_user_attr(self, name, value):
        self.attrs[name] = value


class _FakeOptuna:
    def __init__(self):
        self.study = _FakeStudy()
        self.created_kwargs = None
        self.samplers = types.SimpleNamespace(
            TPESampler=lambda seed=None: {"seed": seed}
        )
        self.pruners = types.SimpleNamespace(
            MedianPruner=lambda n_startup_trials=0, n_warmup_steps=0: {
                "startup": n_startup_trials,
                "warmup": n_warmup_steps,
            },
            SuccessiveHalvingPruner=lambda min_resource=1, reduction_factor=3: {
                "min_resource": min_resource,
                "reduction_factor": reduction_factor,
            },
            HyperbandPruner=lambda min_resource=1, reduction_factor=3: {
                "min_resource": min_resource,
                "reduction_factor": reduction_factor,
            },
            NopPruner=lambda: {"nop": True},
        )

    def create_study(self, **kwargs):
        self.created_kwargs = kwargs
        return self.study


def test_generate_loss_search_uses_optuna_study_when_available(tmp_path, monkeypatch):
    fake_optuna = _FakeOptuna()
    monkeypatch.setattr(
        tuner.importlib,
        "import_module",
        lambda name: fake_optuna if name == "optuna" else None,
    )
    args = tuner.build_arg_parser().parse_args(
        [
            "--output-dir",
            str(tmp_path),
            "--optuna-trials",
            "2",
        ]
    )

    runs = tuner.generate_loss_search(args, tuner.baseline_config(args))

    assert fake_optuna.created_kwargs["study_name"] == args.optuna_study_name
    assert fake_optuna.created_kwargs["direction"] == "minimize"
    assert fake_optuna.created_kwargs["load_if_exists"] is True
    assert str(tmp_path / "optuna_study.db") in fake_optuna.created_kwargs["storage"]
    assert len(fake_optuna.study.asked) == 2
    assert [run["optuna_source"] for run in runs] == ["optuna", "optuna"]
    assert runs[0]["params"]["star_loss_weight"] == pytest.approx(0.015)
    assert (tmp_path / "optuna_trial_plan.json").exists()


def test_require_optuna_fails_when_optuna_missing(tmp_path, monkeypatch):
    def missing_import(name):
        if name == "optuna":
            raise ImportError("missing optuna")
        raise AssertionError(name)

    monkeypatch.setattr(tuner.importlib, "import_module", missing_import)
    args = tuner.build_arg_parser().parse_args(
        [
            "--output-dir",
            str(tmp_path),
            "--optuna-trials",
            "1",
            "--require-optuna",
        ]
    )

    with pytest.raises(tuner.TuningError, match="Optuna is required"):
        tuner.generate_loss_search(args, tuner.baseline_config(args))


def test_tell_optuna_result_reports_completed_score(tmp_path, monkeypatch):
    fake_optuna = _FakeOptuna()
    monkeypatch.setattr(
        tuner.importlib,
        "import_module",
        lambda name: fake_optuna if name == "optuna" else None,
    )
    args = tuner.build_arg_parser().parse_args(["--output-dir", str(tmp_path)])
    result = {
        "stage": "optuna_loss_search",
        "optuna_source": "optuna",
        "optuna_trial_number": 3,
        "score": 0.123,
    }

    tuner.tell_optuna_result(args, result)

    assert fake_optuna.study.told == [(3, 0.123)]


def test_legacy_pipeline_mock_metrics_writes_recommendation_and_run_artifacts(tmp_path):
    args = tuner.build_arg_parser().parse_args(
        [
            "--output-dir",
            str(tmp_path),
            "--dry-run",
            "--mock-metrics",
            "--disable-optuna",
            "--legacy-staged-search",
            "--star-bracket",
            "0,0.01",
            "--optuna-trials",
            "2",
            "--loss-top-k",
            "1",
            "--loss-finalist-seeds",
            "17,29",
            "--lr-sweep",
            "0.0001,0.0002",
            "--lr-top-k",
            "1",
            "--lr-finalist-seeds",
            "17,29",
        ]
    )

    recommendation = tuner.run_pipeline(args)

    assert recommendation["recommended"]["star_loss_weight"] >= 0.0
    assert (tmp_path / "baseline_config.json").exists()
    assert (tmp_path / "ranked_loss_candidates.json").exists()
    assert (tmp_path / "loss_finalist_summary.json").exists()
    assert (tmp_path / "ranked_lr_candidates.json").exists()
    assert (tmp_path / "lr_finalist_summary.json").exists()
    assert (tmp_path / "best_training_hyperparameters.json").exists()
    assert any((tmp_path / "runs").iterdir())
    assert (tmp_path / "optuna_study.json").exists()


def test_joint_pipeline_mock_metrics_optimizes_lr_and_loss(tmp_path):
    args = tuner.build_arg_parser().parse_args(
        [
            "--output-dir",
            str(tmp_path),
            "--dry-run",
            "--mock-metrics",
            "--disable-optuna",
            "--optuna-trials",
            "6",
            "--loss-top-k",
            "1",
            "--loss-finalist-seeds",
            "17,29",
        ]
    )

    recommendation = tuner.run_pipeline(args)

    # The joint search owns the learning rate, so the recommendation carries an
    # lr and the separate LR-sweep artifacts are not produced.
    assert "lr" in recommendation["recommended"]
    assert recommendation["recommended"]["lr"] > 0.0
    assert (tmp_path / "ranked_loss_candidates.json").exists()
    assert (tmp_path / "loss_finalist_summary.json").exists()
    assert not (tmp_path / "ranked_lr_candidates.json").exists()
    assert not (tmp_path / "lr_finalist_summary.json").exists()
    assert (tmp_path / "best_training_hyperparameters.json").exists()


def test_interactive_search_interleaves_ask_run_tell(tmp_path):
    args = tuner.build_arg_parser().parse_args(
        [
            "--output-dir",
            str(tmp_path),
            "--dry-run",
            "--mock-metrics",
            "--optuna-trials",
            "3",
        ]
    )
    study = _FakeStudy()
    base = tuner.baseline_config(args)

    tuner._run_interactive_search(
        args, base, None, study=study, ranges=tuner.optuna_ranges(args)
    )

    # One ask and one tell per trial, with the score told before the next ask.
    assert len(study.asked) == 3
    assert len(study.told) == 3
    assert tuner._finished_optuna_trials(tmp_path) == 3


def test_optuna_ranges_includes_lr_by_default_and_excludes_in_legacy(tmp_path):
    joint = tuner.build_arg_parser().parse_args(["--output-dir", str(tmp_path)])
    legacy = tuner.build_arg_parser().parse_args(
        ["--output-dir", str(tmp_path), "--legacy-staged-search"]
    )

    assert "lr" in tuner.optuna_ranges(joint)
    assert "lr" not in tuner.optuna_ranges(legacy)


def test_search_epochs_only_budgets_search_stage_runs(tmp_path):
    args = tuner.build_arg_parser().parse_args(
        ["--output-dir", str(tmp_path), "--search-epochs", "25"]
    )
    params = tuner.baseline_config(args)

    search_run = tuner.make_run_config(
        stage="optuna_loss_search", params=params, seed=17, index=0
    )
    finalist_run = tuner.make_run_config(
        stage="loss_finalist_seed", params=params, seed=17, index=0
    )
    search_cmd = tuner.build_train_command(
        args, search_run, tmp_path / "runs" / search_run["id"]
    )
    finalist_cmd = tuner.build_train_command(
        args, finalist_run, tmp_path / "runs" / finalist_run["id"]
    )

    assert "--epoch" in search_cmd
    assert search_cmd[search_cmd.index("--epoch") + 1] == "25"
    assert "--epoch" not in finalist_cmd


def test_intermediate_score_reads_per_epoch_eval(tmp_path):
    args = tuner.build_arg_parser().parse_args(["--output-dir", str(tmp_path)])
    metrics_path = tmp_path / "metrics.json"
    metrics_path.write_text(
        '{"epoch": 7, "model": {"overall": {"nme": 0.05}}}', encoding="utf-8"
    )

    observation = tuner._intermediate_score(args, metrics_path, baseline_metrics=None)

    assert observation is not None
    epoch, score = observation
    assert epoch == 7
    # Slice-free fast eval scores on overall NME alone.
    assert score == pytest.approx(0.05)


def test_intermediate_score_handles_missing_and_partial_files(tmp_path):
    args = tuner.build_arg_parser().parse_args(["--output-dir", str(tmp_path)])
    missing = tmp_path / "nope.json"
    assert tuner._intermediate_score(args, missing, baseline_metrics=None) is None

    no_epoch = tmp_path / "no_epoch.json"
    no_epoch.write_text('{"model": {"overall": {"nme": 0.05}}}', encoding="utf-8")
    assert tuner._intermediate_score(args, no_epoch, baseline_metrics=None) is None

    half_written = tmp_path / "partial.json"
    half_written.write_text('{"epoch": 3, "model": {"over', encoding="utf-8")
    assert tuner._intermediate_score(args, half_written, baseline_metrics=None) is None


class _PruningTrial:
    """Trial stub that prunes once a reported epoch reaches a threshold."""

    def __init__(self, prune_at_epoch):
        self.number = 0
        self.reported = []
        self._prune_at = prune_at_epoch
        self._last_epoch = -1

    def report(self, value, step):
        self.reported.append((int(step), float(value)))
        self._last_epoch = int(step)

    def should_prune(self):
        return self._prune_at is not None and self._last_epoch >= self._prune_at


def test_execute_with_pruning_terminates_pruned_trial(tmp_path):
    args = tuner.build_arg_parser().parse_args(
        ["--output-dir", str(tmp_path), "--prune-poll-seconds", "0.5"]
    )
    metrics_path = tmp_path / "metrics.json"
    script = (
        "import json,sys,time\n"
        "p=sys.argv[1]\n"
        "for e in range(20):\n"
        "    json.dump({'epoch': e, 'model': {'overall': {'nme': 0.05}}}, open(p,'w'))\n"
        "    time.sleep(0.3)\n"
    )
    command = [sys.executable, "-c", script, str(metrics_path)]

    trial = _PruningTrial(prune_at_epoch=2)
    status = tuner._execute_with_pruning(
        args, command, metrics_path, trial=trial, baseline_metrics=None
    )

    assert status["pruned"] is True
    assert status["last_epoch"] >= 2
    # The child was killed well before all 20 epochs were written.
    assert status["last_epoch"] < 19


def test_execute_with_pruning_runs_to_completion_when_not_pruned(tmp_path):
    args = tuner.build_arg_parser().parse_args(
        ["--output-dir", str(tmp_path), "--prune-poll-seconds", "0.5"]
    )
    metrics_path = tmp_path / "metrics.json"
    script = (
        "import json,sys\n"
        "p=sys.argv[1]\n"
        "for e in range(3):\n"
        "    json.dump({'epoch': e, 'model': {'overall': {'nme': 0.05}}}, open(p,'w'))\n"
    )
    command = [sys.executable, "-c", script, str(metrics_path)]

    trial = _PruningTrial(prune_at_epoch=None)
    status = tuner._execute_with_pruning(
        args, command, metrics_path, trial=trial, baseline_metrics=None
    )

    assert status["pruned"] is False
    assert status["returncode"] == 0


def test_plain_dry_run_generates_commands_without_metrics(tmp_path, capsys):
    args = tuner.build_arg_parser().parse_args(
        [
            "--output-dir",
            str(tmp_path),
            "--dry-run",
            "--disable-optuna",
            "--star-bracket",
            "0",
            "--optuna-trials",
            "0",
            "--loss-top-k",
            "1",
            "--lr-sweep",
            "0.0001",
            "--lr-top-k",
            "1",
        ]
    )

    recommendation = tuner.run_pipeline(args)
    captured = capsys.readouterr()

    assert recommendation["status"] == "planned_only"
    assert "DRY-RUN" in captured.out
    assert (tmp_path / "runs").exists()
    assert (tmp_path / "best_training_hyperparameters.json").exists()
