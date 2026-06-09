#!/usr/bin/env python3
"""Stage landmark-training hyperparameter tuning runs.

This script implements issue #7's staged plan:

1. Baseline run.
2. Manual STARLoss_v2 bracket.
3. Optuna narrow search over STAR, schema consistency, and auxiliary loss.
4. Multi-seed reruns for top loss-weight finalists.
5. Learning-rate sweep with selected loss weights frozen.
6. Multi-seed reruns for top LR finalists.
7. best_training_hyperparameters.json recommendation.

The script is intentionally usable in two modes:

- dry-run planning, which writes reproducible configs and command lines without
  launching training;
- execute mode, which runs commands and reads metrics JSON emitted by training or
  evaluation.

The loss-weight search uses a persisted Optuna study when Optuna is installed.
The normal repository dependency set includes Optuna. For minimal environments,
--disable-optuna falls back to deterministic sampled trials, while --require-optuna
turns a missing Optuna install into a hard error.

Metrics are expected as JSON files in each run directory. The objective minimizes
heldout 68-point NME plus weighted hard-slice NME terms and regression penalties.
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import os
import random
import shlex
import statistics
import subprocess
import sys
import typing as T
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lib.io_utils import jsonable, read_json, write_json

DEFAULT_STAR_BRACKET = [0.0, 0.005, 0.01, 0.02, 0.05]
DEFAULT_LR_SWEEP = [3e-5, 5e-5, 1e-4, 2e-4, 3e-4]
DEFAULT_LOSS_SEEDS = [17, 29, 43]
DEFAULT_LR_SEEDS = [17, 29, 43]
DEFAULT_OPTUNA_RANGES = {
    "star_loss_weight": (0.0, 0.03),
    "schema_consistency_weight": (0.0, 0.08),
    "auxiliary_loss_weight": (0.0, 0.1),
}
HARD_SLICE_KEYS = (
    "profile_nme",
    "occlusion_nme",
    "profile_occlusion_nme",
    "blur_nme",
    "low_quality_nme",
)
OVERALL_KEYS = ("heldout_68_nme", "overall_68_nme", "nme_68", "NME")
FRONTAL_KEYS = ("frontal_nme", "frontal_68_nme")


class TuningError(RuntimeError):
    pass


def _float(value: T.Any, default: float | None = None) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(out):
        return default
    return out


def _split_csv_floats(raw: str, default: list[float]) -> list[float]:
    raw = str(raw or "").strip()
    if not raw:
        return list(default)
    return [float(item.strip()) for item in raw.split(",") if item.strip()]


def _split_csv_ints(raw: str, default: list[int]) -> list[int]:
    raw = str(raw or "").strip()
    if not raw:
        return list(default)
    return [int(item.strip()) for item in raw.split(",") if item.strip()]


def _load_optuna(*, required: bool = False):
    try:
        return importlib.import_module("optuna")
    except ImportError as exc:
        if required:
            raise TuningError(
                "Optuna is required for --require-optuna. Install dependencies with `pip install -r requirements.txt`."
            ) from exc
        return None


def optuna_storage_url(args: argparse.Namespace) -> str:
    if getattr(args, "optuna_storage", ""):
        return str(args.optuna_storage)
    db_path = Path(args.output_dir) / "optuna_study.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{db_path.resolve()}"


def create_or_load_optuna_study(args: argparse.Namespace):
    if getattr(args, "disable_optuna", False):
        return None
    optuna = _load_optuna(required=bool(getattr(args, "require_optuna", False)))
    if optuna is None:
        return None
    sampler = optuna.samplers.TPESampler(seed=int(args.optuna_seed))
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=int(args.optuna_pruner_startup_trials),
        n_warmup_steps=int(args.optuna_min_pruning_epoch),
    )
    study = optuna.create_study(
        study_name=str(args.optuna_study_name),
        storage=optuna_storage_url(args),
        direction="minimize",
        load_if_exists=True,
        sampler=sampler,
        pruner=pruner,
    )
    try:
        study.set_user_attr("objective", "heldout_68_nme_plus_hard_slices")
        study.set_user_attr("search_space", DEFAULT_OPTUNA_RANGES)
        study.set_user_attr("min_pruning_epoch", int(args.optuna_min_pruning_epoch))
        study.set_user_attr("workers", int(args.optuna_workers))
    except Exception:
        # Older/fake Optuna implementations used in tests may not expose attrs.
        pass
    return study


def config_id(
    stage: str,
    params: dict[str, T.Any],
    *,
    seed: int | None = None,
    index: int | None = None,
) -> str:
    parts = [stage]
    if index is not None:
        parts.append(f"{index:03d}")
    for key in (
        "star_loss_weight",
        "schema_consistency_weight",
        "auxiliary_loss_weight",
        "lr",
    ):
        if key in params:
            text = f"{float(params[key]):.6g}".replace("-", "m").replace(".", "p")
            parts.append(
                f"{key.replace('_loss_weight', '').replace('schema_consistency', 'consistency')}={text}"
            )
    if seed is not None:
        parts.append(f"seed={int(seed)}")
    return "__".join(parts)


def baseline_config(args: argparse.Namespace) -> dict[str, float]:
    return {
        "star_loss_weight": float(args.baseline_star_loss_weight),
        "schema_consistency_weight": float(args.baseline_schema_consistency_weight),
        "auxiliary_loss_weight": float(args.baseline_auxiliary_loss_weight),
        "locw": float(args.locw),
        "hw": float(args.hw),
        "lr": float(args.baseline_lr),
    }


def make_run_config(
    *,
    stage: str,
    params: dict[str, T.Any],
    seed: int,
    index: int | None = None,
    parent_trial: str | None = None,
) -> dict[str, T.Any]:
    params = dict(params)
    return {
        "id": config_id(stage, params, seed=seed, index=index),
        "stage": stage,
        "seed": int(seed),
        "parent_trial": parent_trial,
        "params": params,
    }


def build_train_command(
    args: argparse.Namespace, run: dict[str, T.Any], run_dir: Path
) -> list[str]:
    params = dict(run["params"])
    metrics_path = run_dir / args.metrics_file_name
    cmd = shlex.split(args.train_command)
    cmd += [
        "--star-loss-weight",
        str(params["star_loss_weight"]),
        "--schema-consistency-weight",
        str(params["schema_consistency_weight"]),
        "--auxiliary-loss-weight",
        str(params["auxiliary_loss_weight"]),
        "--lr",
        str(params["lr"]),
        "--locw",
        str(params["locw"]),
        "--hw",
        str(params["hw"]),
        "--seed",
        str(run["seed"]),
        "--ckpt_folder",
        str(run_dir / "checkpoints"),
        "--eval-report-json",
        str(metrics_path),
        "--runtime-metrics-jsonl",
        str(run_dir / "runtime_metrics.jsonl"),
    ]
    if args.extra_train_args:
        cmd += shlex.split(args.extra_train_args)
    return cmd


def select_metric(metrics: dict[str, T.Any], keys: tuple[str, ...]) -> float | None:
    for key in keys:
        value = _float(metrics.get(key))
        if value is not None:
            return value
    return None


def _metric_from_group(
    report: dict[str, T.Any], group_name: str, labels: tuple[str, ...]
) -> float | None:
    group = report.get(group_name)
    if not isinstance(group, dict):
        return None
    for label in labels:
        metrics = group.get(label)
        if isinstance(metrics, dict):
            value = _float(metrics.get("nme"))
            if value is not None:
                return value
    return None


def normalize_metrics(raw_metrics: dict[str, T.Any]) -> dict[str, T.Any]:
    """Flatten trainer/evaluator JSON into objective metric keys.

    The tuner also accepts already-flat metrics for external evaluation jobs.
    When the trainer writes its normal eval report, metrics live under
    ``model.overall`` and slice groups such as ``by_hard_negative_bucket``.
    """

    metrics = dict(raw_metrics)
    report = (
        raw_metrics.get("model")
        if isinstance(raw_metrics.get("model"), dict)
        else raw_metrics
    )
    if not isinstance(report, dict):
        return metrics

    overall = report.get("overall")
    if isinstance(overall, dict):
        nme = _float(overall.get("nme"))
        if nme is not None:
            metrics.setdefault("heldout_68_nme", nme)
            metrics.setdefault("overall_68_nme", nme)

    slice_specs = {
        "profile_nme": (
            ("by_hard_negative_bucket", ("profile",)),
            ("by_pose_bucket", ("profile", "profile_left", "profile_right")),
        ),
        "occlusion_nme": (
            ("by_hard_negative_bucket", ("occlusion",)),
            ("by_occlusion", ("occlusion",)),
        ),
        "profile_occlusion_nme": (
            ("by_hard_negative_bucket", ("profile_occlusion", "profile+occlusion")),
        ),
        "blur_nme": (
            ("by_blur_quality", ("blurred", "blur", "low_quality")),
            ("by_face_size", ("small",)),
        ),
        "low_quality_nme": (
            ("by_landmark_confidence", ("low", "low_quality")),
            ("by_face_size", ("small",)),
        ),
        "frontal_nme": (
            ("by_pose_bucket", ("frontal",)),
            ("by_hard_negative_bucket", ("anchor",)),
        ),
    }
    for metric_key, specs in slice_specs.items():
        if _float(metrics.get(metric_key)) is not None:
            continue
        for group_name, labels in specs:
            value = _metric_from_group(report, group_name, labels)
            if value is not None:
                metrics[metric_key] = value
                break
    return metrics


def objective_score(
    metrics: dict[str, T.Any],
    *,
    baseline_metrics: dict[str, T.Any] | None = None,
    hard_slice_weight: float = 0.25,
    regression_penalty_weight: float = 2.0,
    max_easy_regression: float = 0.0,
    required_slices: tuple[str, ...] = (
        "profile_nme",
        "occlusion_nme",
        "profile_occlusion_nme",
    ),
) -> tuple[float, dict[str, T.Any]]:
    """Return lower-is-better score plus diagnostics."""

    overall = select_metric(metrics, OVERALL_KEYS)
    if overall is None:
        raise TuningError(
            f"metrics missing heldout/overall 68 NME; tried {OVERALL_KEYS}"
        )

    score = overall
    used_slices: dict[str, float] = {}
    missing_slices: list[str] = []
    for key in HARD_SLICE_KEYS:
        value = _float(metrics.get(key))
        if value is None:
            if key in required_slices:
                missing_slices.append(key)
            continue
        used_slices[key] = value
        score += float(hard_slice_weight) * value

    regression_penalty = 0.0
    regressions: dict[str, float] = {}
    if baseline_metrics:
        baseline_overall = select_metric(baseline_metrics, OVERALL_KEYS)
        if baseline_overall is not None:
            delta = overall - baseline_overall
            if delta > max_easy_regression:
                regressions["overall_68_nme"] = delta
                regression_penalty += float(regression_penalty_weight) * delta
        frontal = select_metric(metrics, FRONTAL_KEYS)
        baseline_frontal = select_metric(baseline_metrics, FRONTAL_KEYS)
        if frontal is not None and baseline_frontal is not None:
            delta = frontal - baseline_frontal
            if delta > max_easy_regression:
                regressions["frontal_nme"] = delta
                regression_penalty += float(regression_penalty_weight) * delta

    score += regression_penalty
    diagnostics = {
        "overall_68_nme": overall,
        "used_slices": used_slices,
        "missing_slices": missing_slices,
        "regressions": regressions,
        "regression_penalty": regression_penalty,
        "score": score,
    }
    return score, diagnostics


def summarize_seed_group(results: list[dict[str, T.Any]]) -> dict[str, T.Any]:
    scores = [float(item["score"]) for item in results]
    if not scores:
        raise TuningError("cannot summarize empty seed group")
    return {
        "count": len(scores),
        "mean_score": statistics.mean(scores),
        "std_score": statistics.pstdev(scores) if len(scores) > 1 else 0.0,
        "best_score": min(scores),
        "worst_score": max(scores),
        "run_ids": [item["id"] for item in results],
    }


def rank_results(
    results: list[dict[str, T.Any]], *, top_k: int
) -> list[dict[str, T.Any]]:
    return sorted(results, key=lambda item: (float(item["score"]), item["id"]))[
        : int(top_k)
    ]


def aggregate_by_parent(
    results: list[dict[str, T.Any]], *, top_k: int
) -> list[dict[str, T.Any]]:
    grouped: dict[str, list[dict[str, T.Any]]] = {}
    for result in results:
        parent = str(result.get("parent_trial") or result["id"])
        grouped.setdefault(parent, []).append(result)
    summaries = []
    for parent, items in grouped.items():
        summary = summarize_seed_group(items)
        summary["parent_trial"] = parent
        summary["params"] = dict(items[0]["params"])
        summaries.append(summary)
    return sorted(
        summaries,
        key=lambda item: (
            float(item["mean_score"]),
            float(item["std_score"]),
            item["parent_trial"],
        ),
    )[: int(top_k)]


def generate_star_bracket(
    args: argparse.Namespace, base: dict[str, float]
) -> list[dict[str, T.Any]]:
    out = []
    for index, star_weight in enumerate(
        _split_csv_floats(args.star_bracket, DEFAULT_STAR_BRACKET)
    ):
        params = dict(base)
        params["star_loss_weight"] = float(star_weight)
        out.append(
            make_run_config(
                stage="star_bracket", params=params, seed=args.seed, index=index
            )
        )
    return out


def _sample_range(rng: random.Random, low: float, high: float) -> float:
    return low + (high - low) * rng.random()


def _trial_suggest_params(
    trial, base: dict[str, float], ranges: dict[str, tuple[float, float]]
) -> dict[str, float]:
    params = dict(base)
    for name, (low, high) in ranges.items():
        params[name] = float(trial.suggest_float(name, low, high))
    return params


def _fallback_trial_params(
    rng: random.Random, base: dict[str, float], ranges: dict[str, tuple[float, float]]
) -> dict[str, float]:
    params = dict(base)
    for name, (low, high) in ranges.items():
        params[name] = _sample_range(rng, low, high)
    return params


def _loss_search_plan_path(args: argparse.Namespace) -> Path:
    return Path(args.output_dir) / "optuna_trial_plan.json"


def _load_loss_search_plan(args: argparse.Namespace) -> dict[str, T.Any]:
    path = _loss_search_plan_path(args)
    if path.exists():
        return read_json(path)
    return {
        "study_name": args.optuna_study_name,
        "storage": optuna_storage_url(args),
        "ranges": DEFAULT_OPTUNA_RANGES,
        "trials": [],
        "uses_real_optuna": False,
    }


def generate_loss_search(
    args: argparse.Namespace, base: dict[str, float]
) -> list[dict[str, T.Any]]:
    ranges = dict(DEFAULT_OPTUNA_RANGES)
    plan = _load_loss_search_plan(args)
    plan["study_name"] = args.optuna_study_name
    plan["storage"] = optuna_storage_url(args)
    plan["ranges"] = ranges
    plan.setdefault("trials", [])

    if len(plan["trials"]) < int(args.optuna_trials):
        study = create_or_load_optuna_study(args)
        rng = random.Random(int(args.optuna_seed))
        if study is not None:
            plan["uses_real_optuna"] = True
        elif getattr(args, "require_optuna", False):
            _load_optuna(required=True)

        for index in range(len(plan["trials"]), int(args.optuna_trials)):
            if study is not None:
                trial = study.ask()
                params = _trial_suggest_params(trial, base, ranges)
                trial_number = int(trial.number)
                source = "optuna"
            else:
                params = _fallback_trial_params(rng, base, ranges)
                trial_number = index
                source = "deterministic_fallback"
            run = make_run_config(
                stage="optuna_loss_search",
                params=params,
                seed=args.seed,
                index=trial_number,
            )
            run["optuna_trial_number"] = trial_number
            run["optuna_study_name"] = args.optuna_study_name
            run["optuna_storage"] = optuna_storage_url(args)
            run["optuna_source"] = source
            plan["trials"].append(
                {"number": trial_number, "run": run, "params": params, "source": source}
            )

    write_json(_loss_search_plan_path(args), plan)
    write_json(Path(args.output_dir) / "optuna_study.json", plan)
    return [dict(item["run"]) for item in plan["trials"][: int(args.optuna_trials)]]


def tell_optuna_result(args: argparse.Namespace, result: dict[str, T.Any]) -> None:
    if result.get("stage") != "optuna_loss_search":
        return
    if result.get("optuna_source") != "optuna":
        return
    trial_number = result.get("optuna_trial_number")
    if trial_number is None:
        return
    study = create_or_load_optuna_study(args)
    if study is None:
        return
    try:
        for trial in study.get_trials(deepcopy=False):
            if (
                int(trial.number) == int(trial_number)
                and getattr(trial.state, "is_finished", lambda: False)()
            ):
                return
    except Exception:
        pass
    try:
        study.tell(int(trial_number), float(result["score"]))
    except Exception as exc:
        print(
            f"warning: could not tell Optuna score for trial {trial_number}: {exc}",
            file=sys.stderr,
        )


def generate_loss_finalists(
    args: argparse.Namespace,
    ranked_loss_results: list[dict[str, T.Any]],
) -> list[dict[str, T.Any]]:
    seeds = _split_csv_ints(args.loss_finalist_seeds, DEFAULT_LOSS_SEEDS)
    finalists = ranked_loss_results[: int(args.loss_top_k)]
    out = []
    for finalist_index, finalist in enumerate(finalists):
        for seed in seeds:
            out.append(
                make_run_config(
                    stage="loss_finalist_seed",
                    params=finalist["params"],
                    seed=seed,
                    index=finalist_index,
                    parent_trial=finalist["id"],
                )
            )
    return out


def generate_lr_sweep(
    args: argparse.Namespace, selected_loss_params: dict[str, T.Any]
) -> list[dict[str, T.Any]]:
    lrs = _split_csv_floats(args.lr_sweep, DEFAULT_LR_SWEEP)
    out = []
    for index, lr in enumerate(lrs):
        params = dict(selected_loss_params)
        params["lr"] = float(lr)
        out.append(
            make_run_config(
                stage="lr_sweep", params=params, seed=args.seed, index=index
            )
        )
    return out


def generate_lr_finalists(
    args: argparse.Namespace, ranked_lr_results: list[dict[str, T.Any]]
) -> list[dict[str, T.Any]]:
    seeds = _split_csv_ints(args.lr_finalist_seeds, DEFAULT_LR_SEEDS)
    finalists = ranked_lr_results[: int(args.lr_top_k)]
    out = []
    for finalist_index, finalist in enumerate(finalists):
        for seed in seeds:
            out.append(
                make_run_config(
                    stage="lr_finalist_seed",
                    params=finalist["params"],
                    seed=seed,
                    index=finalist_index,
                    parent_trial=finalist["id"],
                )
            )
    return out


def run_one(
    args: argparse.Namespace,
    run: dict[str, T.Any],
    *,
    baseline_metrics: dict[str, T.Any] | None,
) -> dict[str, T.Any] | None:
    output_dir = Path(args.output_dir)
    run_dir = output_dir / "runs" / run["id"]
    run_dir.mkdir(parents=True, exist_ok=True)
    write_json(run_dir / "config.json", run)
    command = build_train_command(args, run, run_dir)
    (run_dir / "command.txt").write_text(
        " ".join(shlex.quote(item) for item in command) + "\n", encoding="utf-8"
    )

    if args.dry_run and not args.mock_metrics:
        print("DRY-RUN", run["id"], " ".join(shlex.quote(item) for item in command))
        return None

    metrics_path = run_dir / args.metrics_file_name
    if args.mock_metrics:
        metrics = synthetic_metrics_for_config(run["params"], seed=int(run["seed"]))
        write_json(metrics_path, metrics)
    elif args.execute:
        subprocess.run(command, check=True, cwd=args.cwd or None, env=os.environ.copy())
    elif not metrics_path.exists():
        print("SKIP", run["id"], "missing metrics", metrics_path)
        return None

    if not metrics_path.exists():
        print("SKIP", run["id"], "training did not write metrics", metrics_path)
        return None

    raw_metrics = read_json(metrics_path)
    metrics = normalize_metrics(raw_metrics)
    score, diagnostics = objective_score(
        metrics,
        baseline_metrics=baseline_metrics,
        hard_slice_weight=float(args.hard_slice_weight),
        regression_penalty_weight=float(args.regression_penalty_weight),
        max_easy_regression=float(args.max_easy_regression),
    )
    result = {
        **run,
        "run_dir": str(run_dir),
        "command": command,
        "raw_metrics": raw_metrics,
        "metrics": metrics,
        "score": score,
        "objective": diagnostics,
    }
    write_json(run_dir / "result.json", result)
    append_result(output_dir / "results.jsonl", result)
    tell_optuna_result(args, result)
    return result


def append_result(path: Path, result: dict[str, T.Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(jsonable(result), sort_keys=True) + "\n")


def read_results(output_dir: Path) -> list[dict[str, T.Any]]:
    path = output_dir / "results.jsonl"
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            out.append(json.loads(line))
    dedup: dict[str, dict[str, T.Any]] = {}
    for result in out:
        dedup[str(result["id"])] = result
    return list(dedup.values())


def result_by_stage(
    results: list[dict[str, T.Any]], stage: str
) -> list[dict[str, T.Any]]:
    return [item for item in results if item.get("stage") == stage]


def synthetic_metrics_for_config(
    params: dict[str, T.Any], *, seed: int
) -> dict[str, float]:
    """Deterministic mock metrics for dry-run tests and pipeline smoke checks."""
    rng = random.Random(seed + int(float(params.get("lr", 1e-4)) * 1e8))
    star = float(params.get("star_loss_weight", 0.0))
    consistency = float(params.get("schema_consistency_weight", 0.05))
    aux = float(params.get("auxiliary_loss_weight", 0.1))
    lr = float(params.get("lr", 1e-4))
    nme = 0.035
    nme += abs(star - 0.01) * 0.12
    nme += abs(consistency - 0.04) * 0.04
    nme += abs(aux - 0.05) * 0.03
    nme += abs(math.log10(lr) - math.log10(1e-4)) * 0.003
    nme += rng.uniform(-0.0005, 0.0005)
    return {
        "heldout_68_nme": nme,
        "overall_68_nme": nme,
        "profile_nme": nme + 0.010 + abs(star - 0.01) * 0.08,
        "occlusion_nme": nme + 0.008 + abs(aux - 0.05) * 0.05,
        "profile_occlusion_nme": nme + 0.016,
        "blur_nme": nme + 0.006,
        "frontal_nme": nme - 0.006 + max(star - 0.03, 0.0) * 0.2,
    }


def write_recommendation(
    args: argparse.Namespace,
    *,
    baseline_result: dict[str, T.Any] | None,
    selected_loss_summary: dict[str, T.Any],
    selected_lr_summary: dict[str, T.Any],
) -> dict[str, T.Any]:
    params = dict(selected_lr_summary["params"])
    recommendation = {
        "recommended": {
            "star_loss_weight": float(params["star_loss_weight"]),
            "schema_consistency_weight": float(params["schema_consistency_weight"]),
            "auxiliary_loss_weight": float(params["auxiliary_loss_weight"]),
            "locw": float(params["locw"]),
            "hw": float(params["hw"]),
            "lr": float(params["lr"]),
        },
        "rationale": {
            "baseline_score": baseline_result.get("score") if baseline_result else None,
            "selected_loss_parent": selected_loss_summary.get("parent_trial"),
            "selected_loss_mean_score": selected_loss_summary.get("mean_score"),
            "selected_loss_std_score": selected_loss_summary.get("std_score"),
            "selected_lr_parent": selected_lr_summary.get("parent_trial"),
            "selected_lr_mean_score": selected_lr_summary.get("mean_score"),
            "selected_lr_std_score": selected_lr_summary.get("std_score"),
            "lr_sweep_result": selected_lr_summary,
            "baseline_delta": None,
        },
        "training_flags": [
            "--star-loss-weight",
            str(float(params["star_loss_weight"])),
            "--schema-consistency-weight",
            str(float(params["schema_consistency_weight"])),
            "--auxiliary-loss-weight",
            str(float(params["auxiliary_loss_weight"])),
            "--locw",
            str(float(params["locw"])),
            "--hw",
            str(float(params["hw"])),
            "--lr",
            str(float(params["lr"])),
        ],
    }
    if baseline_result is not None:
        recommendation["rationale"]["baseline_delta"] = float(
            selected_lr_summary["mean_score"]
        ) - float(baseline_result["score"])
    write_json(
        Path(args.output_dir) / "best_training_hyperparameters.json", recommendation
    )
    return recommendation


def run_pipeline(args: argparse.Namespace) -> dict[str, T.Any]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "tuning_args.json", vars(args))

    base = baseline_config(args)
    results = read_results(output_dir)

    baseline_runs = result_by_stage(results, "baseline")
    baseline_result = baseline_runs[0] if baseline_runs else None
    if baseline_result is None:
        run = make_run_config(stage="baseline", params=base, seed=args.seed, index=0)
        baseline_result = run_one(args, run, baseline_metrics=None)
        if baseline_result is not None:
            write_json(output_dir / "baseline_config.json", run)
            write_json(output_dir / "baseline_result.json", baseline_result)
    baseline_metrics = baseline_result.get("metrics") if baseline_result else None

    for run in generate_star_bracket(args, base):
        if not already_done(output_dir, run["id"]):
            run_one(args, run, baseline_metrics=baseline_metrics)

    for run in generate_loss_search(args, base):
        if not already_done(output_dir, run["id"]):
            run_one(args, run, baseline_metrics=baseline_metrics)

    results = read_results(output_dir)
    loss_candidates = result_by_stage(results, "star_bracket") + result_by_stage(
        results, "optuna_loss_search"
    )
    ranked_loss = rank_results(loss_candidates, top_k=max(int(args.loss_top_k), 1))
    write_json(output_dir / "ranked_loss_candidates.json", ranked_loss)

    for run in generate_loss_finalists(args, ranked_loss):
        if not already_done(output_dir, run["id"]):
            run_one(args, run, baseline_metrics=baseline_metrics)

    results = read_results(output_dir)
    loss_finalist_summaries = aggregate_by_parent(
        result_by_stage(results, "loss_finalist_seed"),
        top_k=max(int(args.loss_top_k), 1),
    )
    write_json(output_dir / "loss_finalist_summary.json", loss_finalist_summaries)
    if loss_finalist_summaries:
        selected_loss = loss_finalist_summaries[0]
    elif ranked_loss:
        selected_loss = {
            "params": ranked_loss[0]["params"],
            "parent_trial": ranked_loss[0]["id"],
            "mean_score": ranked_loss[0]["score"],
            "std_score": 0.0,
        }
    else:
        selected_loss = {
            "params": base,
            "parent_trial": "baseline",
            "mean_score": baseline_result.get("score") if baseline_result else None,
            "std_score": 0.0,
        }

    for run in generate_lr_sweep(args, selected_loss["params"]):
        if not already_done(output_dir, run["id"]):
            run_one(args, run, baseline_metrics=baseline_metrics)

    results = read_results(output_dir)
    ranked_lr = rank_results(
        result_by_stage(results, "lr_sweep"), top_k=max(int(args.lr_top_k), 1)
    )
    write_json(output_dir / "ranked_lr_candidates.json", ranked_lr)

    for run in generate_lr_finalists(args, ranked_lr):
        if not already_done(output_dir, run["id"]):
            run_one(args, run, baseline_metrics=baseline_metrics)

    results = read_results(output_dir)
    lr_finalist_summaries = aggregate_by_parent(
        result_by_stage(results, "lr_finalist_seed"), top_k=max(int(args.lr_top_k), 1)
    )
    write_json(output_dir / "lr_finalist_summary.json", lr_finalist_summaries)
    if lr_finalist_summaries:
        selected_lr = lr_finalist_summaries[0]
    elif ranked_lr:
        selected_lr = {
            "params": ranked_lr[0]["params"],
            "parent_trial": ranked_lr[0]["id"],
            "mean_score": ranked_lr[0]["score"],
            "std_score": 0.0,
        }
    else:
        selected_lr = selected_loss

    if (
        baseline_result is not None
        and selected_loss.get("mean_score") is not None
        and selected_lr.get("mean_score") is not None
    ):
        recommendation = write_recommendation(
            args,
            baseline_result=baseline_result,
            selected_loss_summary=selected_loss,
            selected_lr_summary=selected_lr,
        )
    else:
        recommendation = {
            "status": "planned_only",
            "message": "No metrics available; rerun with --execute or --mock-metrics.",
        }
        write_json(output_dir / "best_training_hyperparameters.json", recommendation)

    return recommendation


def already_done(output_dir: Path, run_id: str) -> bool:
    return (output_dir / "runs" / run_id / "result.json").exists()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--train-command",
        default=f"{shlex.quote(sys.executable)} TrainHeatmapStageFP16.py",
    )
    parser.add_argument("--extra-train-args", default="")
    parser.add_argument("--cwd", default="")
    parser.add_argument("--metrics-file-name", default="metrics.json")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Write configs and commands without executing training unless --mock-metrics is set.",
    )
    parser.add_argument(
        "--execute", action="store_true", help="Actually launch training commands."
    )
    parser.add_argument(
        "--mock-metrics",
        action="store_true",
        help="Write deterministic synthetic metrics for smoke tests.",
    )
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--baseline-star-loss-weight", type=float, default=0.0)
    parser.add_argument(
        "--baseline-schema-consistency-weight", type=float, default=0.05
    )
    parser.add_argument("--baseline-auxiliary-loss-weight", type=float, default=0.1)
    parser.add_argument("--baseline-lr", type=float, default=1e-4)
    parser.add_argument("--locw", type=float, default=1.0)
    parser.add_argument("--hw", type=float, default=10.0)
    parser.add_argument(
        "--star-bracket", default=",".join(str(x) for x in DEFAULT_STAR_BRACKET)
    )
    parser.add_argument("--optuna-trials", type=int, default=20)
    parser.add_argument("--optuna-seed", type=int, default=2026)
    parser.add_argument("--optuna-study-name", default="landmark_loss_weight_search")
    parser.add_argument(
        "--optuna-storage",
        default="",
        help="Optuna storage URL. Defaults to sqlite:///<output-dir>/optuna_study.db.",
    )
    parser.add_argument(
        "--optuna-workers",
        type=int,
        default=1,
        help="Documented worker count for shared Optuna storage; launch multiple processes with same output dir/storage for parallelism.",
    )
    parser.add_argument("--optuna-pruner-startup-trials", type=int, default=5)
    parser.add_argument("--optuna-min-pruning-epoch", type=int, default=5)
    parser.add_argument(
        "--require-optuna",
        action="store_true",
        help="Fail if Optuna cannot be imported.",
    )
    parser.add_argument(
        "--disable-optuna",
        action="store_true",
        help="Use deterministic sampled fallback instead of a real Optuna study.",
    )
    parser.add_argument("--loss-top-k", type=int, default=3)
    parser.add_argument(
        "--loss-finalist-seeds", default=",".join(str(x) for x in DEFAULT_LOSS_SEEDS)
    )
    parser.add_argument(
        "--lr-sweep", default=",".join(str(x) for x in DEFAULT_LR_SWEEP)
    )
    parser.add_argument("--lr-top-k", type=int, default=2)
    parser.add_argument(
        "--lr-finalist-seeds", default=",".join(str(x) for x in DEFAULT_LR_SEEDS)
    )
    parser.add_argument("--hard-slice-weight", type=float, default=0.25)
    parser.add_argument("--regression-penalty-weight", type=float, default=2.0)
    parser.add_argument("--max-easy-regression", type=float, default=0.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if args.execute and args.dry_run:
        raise SystemExit("--execute and --dry-run cannot both be set")
    if not args.execute and not args.dry_run and not args.mock_metrics:
        print(
            "Neither --execute nor --dry-run was set; defaulting to dry-run planning.",
            file=sys.stderr,
        )
        args.dry_run = True
    recommendation = run_pipeline(args)
    print(json.dumps(recommendation, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
