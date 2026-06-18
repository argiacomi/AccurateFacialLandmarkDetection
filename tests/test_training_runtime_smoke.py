from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pytest
import torch

import TrainHeatmapStageFP16 as train
import lib.training.evaluator as evaluator
from lib.training.cli import build_heatmap_stage_arg_parser
from lib.training.config import (
    DatasetBuildConfig,
    EvalConfig,
    TrainingRuntimeConfig,
    config_dict,
)
from lib.training.loaders import DistributedEvalSampler
from tools import run_cdvit_manifest_training_pipeline as pipeline


def _trainer_args(tmp_path: Path, **overrides):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"samples": []}) + "\n", encoding="utf-8")
    values = {
        "eval_num_workers": 0,
        "eval_batch_size": 8,
        "eval_every": 1,
        "full_eval_every": 0,
        "eval_ema_every": 1,
        "eval_max_samples": 0,
        "eval_slice_reports_every": 1,
        "num_workers": 2,
        "preload": 0,
        "pin_memory": True,
        "persistent_workers": True,
        "prefetch_factor": 2,
        "log_every": 20,
        "runtime_metrics_jsonl": "",
        "ckpt_folder": str(tmp_path / "ckpt"),
        "train_manifest": "",
        "test_manifest": "",
        "manifest": str(manifest),
        "root_folder": "",
        "batch_size": 16,
        "heatmap_size": 32,
        "lmk_num": 68,
        "sched_step": 200,
        "nstack": 8,
        "max_depth": 256,
        "seed": 0,
        "lr": 0.0001,
        "hw": 10.0,
        "locw": 1.0,
        "mul": 1.2,
        "schema_consistency_weight": 0.05,
        "auxiliary_loss_weight": 0.1,
        "schema_aware_training": True,
        "domain_balanced_sampling": False,
        "auxiliary_heads": True,
        "data_name": "FS68Manifest",
        "eval_mode": "random_hash",
        "split_policy": "declared_or_random_hash",
        "bucket_targets": "anchor=0.25,occlusion=0.25,profile=0.25,profile_occlusion=0.25",
        "dataset_targets": "",
        "schema_targets": "",
        "heldout_dataset": [],
        "restore_rng": False,
        "synchronize_runtime_timing": True,
        "roll_quarter_turn_prob": 0.4,
        "roll_diagonal_prob": 0.1,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def _pipeline_args(tmp_path: Path, *extra: str):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"samples": []}) + "\n", encoding="utf-8")
    parser = pipeline._build_arg_parser()
    args = pipeline._normalize_runtime_args(
        parser.parse_args(
            [
                "--manifest",
                str(manifest),
                "--output-root",
                str(tmp_path / "runs"),
                "--run-name",
                "smoke",
                *extra,
            ]
        )
    )
    paths = pipeline.PipelinePaths(
        output_root=args.output_root,
        run_name=args.run_name,
        explicit_manifest=args.manifest.resolve(),
    )
    return args, paths


def test_dataloader_kwargs_disable_prefetch_without_workers(tmp_path):
    args = _trainer_args(
        tmp_path, num_workers=0, persistent_workers=True, prefetch_factor=2
    )
    kwargs = train._dataloader_kwargs(args)
    # Pinned memory only benefits CUDA H2D copies, so it is gated on CUDA
    # availability (off on MPS/CPU) even when --pin-memory is set.
    assert kwargs == {
        "num_workers": 0,
        "pin_memory": torch.cuda.is_available(),
    }


def test_dataloader_kwargs_wires_worker_flags(tmp_path):
    args = _trainer_args(
        tmp_path, num_workers=2, persistent_workers=True, prefetch_factor=4
    )
    kwargs = train._dataloader_kwargs(args)
    assert kwargs["num_workers"] == 2
    assert kwargs["pin_memory"] is torch.cuda.is_available()
    assert kwargs["persistent_workers"] is True
    assert kwargs["prefetch_factor"] == 4
    assert kwargs["worker_init_fn"] is train._seed_worker


def test_restore_rng_forces_non_persistent_workers(tmp_path):
    args = _trainer_args(tmp_path, restore_rng=True, persistent_workers=True)
    train._normalize_runtime_args(args)
    assert args.persistent_workers is False


def test_eval_interval_throttling_runs_intervals_and_final_epoch():
    assert train._should_run_interval(5, 0, 9) is False
    assert train._should_run_interval(5, 4, 9) is True
    assert train._should_run_interval(5, 8, 9) is False
    assert train._should_run_interval(5, 9, 9) is True
    assert train._should_run_interval(0, 9, 9) is False


def test_distributed_eval_sampler_shards_without_padding():
    dataset = list(range(10))

    assert list(DistributedEvalSampler(dataset, rank=0, world_size=3)) == [0, 3, 6, 9]
    assert list(DistributedEvalSampler(dataset, rank=1, world_size=3)) == [1, 4, 7]
    assert list(DistributedEvalSampler(dataset, rank=2, world_size=3)) == [2, 5, 8]
    assert len(DistributedEvalSampler(dataset, rank=0, world_size=3)) == 4
    assert len(DistributedEvalSampler(dataset, rank=2, world_size=3)) == 3


def test_evaluate_landmark_model_reduces_distributed_overall_nmes(monkeypatch):
    points = torch.stack(
        [torch.linspace(0.0, 1.0, 68), torch.linspace(0.1, 0.9, 68)], dim=1
    )

    class PerfectModel(torch.nn.Module):
        def forward(self, data):
            pred = points.unsqueeze(0).repeat(data.shape[0], 1, 1)
            heatmap = torch.zeros(data.shape[0], 68, 8, 8)
            return [(pred, heatmap)]

    def fake_all_reduce(tensor, op=None):
        del op
        tensor[0] += 1.0
        tensor[1] += 0.2
        tensor[2] += 0.04

    def fail_all_gather_object(*args, **kwargs):
        raise AssertionError("overall-only eval should not pickle gather NME lists")

    monkeypatch.setattr(evaluator.dist, "is_available", lambda: True)
    monkeypatch.setattr(evaluator.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(evaluator.dist, "get_world_size", lambda: 2)
    monkeypatch.setattr(evaluator.dist, "get_rank", lambda: 0)
    monkeypatch.setattr(evaluator.dist, "all_reduce", fake_all_reduce)
    monkeypatch.setattr(evaluator.dist, "all_gather_object", fail_all_gather_object)

    batch = (
        torch.zeros(1, 3, 8, 8),
        points.unsqueeze(0),
        torch.ones(1, 68),
        [{"sample_id": "rank0"}],
    )
    report = evaluator.evaluate_landmark_model(
        PerfectModel(),
        [batch],
        torch.device("cpu"),
        build_records=False,
        show_progress=False,
        distributed=True,
    )

    assert report["overall"]["sample_count"] == 2
    assert report["overall"]["nme"] == pytest.approx(0.1)


def test_evaluate_landmark_model_gathers_records_to_rank0_only(monkeypatch):
    points = torch.stack(
        [torch.linspace(0.0, 1.0, 68), torch.linspace(0.1, 0.9, 68)], dim=1
    )

    class PerfectModel(torch.nn.Module):
        def forward(self, data):
            pred = points.unsqueeze(0).repeat(data.shape[0], 1, 1)
            heatmap = torch.zeros(data.shape[0], 68, 8, 8)
            return [(pred, heatmap)]

    def fake_gather_object(local_records, object_gather_list=None, dst=0):
        assert dst == 0
        assert object_gather_list is not None
        object_gather_list[0] = list(local_records)
        object_gather_list[1] = [{"sample_id": "rank1", "nme": 0.2}]

    monkeypatch.setattr(evaluator.dist, "is_available", lambda: True)
    monkeypatch.setattr(evaluator.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(evaluator.dist, "get_world_size", lambda: 2)
    monkeypatch.setattr(evaluator.dist, "get_rank", lambda: 0)
    monkeypatch.setattr(evaluator.dist, "gather_object", fake_gather_object)

    batch = (
        torch.zeros(1, 3, 8, 8),
        points.unsqueeze(0),
        torch.ones(1, 68),
        [{"sample_id": "rank0"}],
    )
    report = evaluator.evaluate_landmark_model(
        PerfectModel(),
        [batch],
        torch.device("cpu"),
        include_records=True,
        show_progress=False,
        distributed=True,
    )

    assert report["overall"]["sample_count"] == 2
    assert {record["sample_id"] for record in report["records"]} == {"rank0", "rank1"}


def test_evaluate_landmark_model_nonzero_rank_returns_empty_report(monkeypatch):
    points = torch.stack(
        [torch.linspace(0.0, 1.0, 68), torch.linspace(0.1, 0.9, 68)], dim=1
    )

    class PerfectModel(torch.nn.Module):
        def forward(self, data):
            pred = points.unsqueeze(0).repeat(data.shape[0], 1, 1)
            heatmap = torch.zeros(data.shape[0], 68, 8, 8)
            return [(pred, heatmap)]

    def fake_gather_object(local_records, object_gather_list=None, dst=0):
        assert dst == 0
        assert local_records
        assert object_gather_list is None

    monkeypatch.setattr(evaluator.dist, "is_available", lambda: True)
    monkeypatch.setattr(evaluator.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(evaluator.dist, "get_world_size", lambda: 2)
    monkeypatch.setattr(evaluator.dist, "get_rank", lambda: 1)
    monkeypatch.setattr(evaluator.dist, "gather_object", fake_gather_object)

    batch = (
        torch.zeros(1, 3, 8, 8),
        points.unsqueeze(0),
        torch.ones(1, 68),
        [{"sample_id": "rank1"}],
    )
    report = evaluator.evaluate_landmark_model(
        PerfectModel(),
        [batch],
        torch.device("cpu"),
        include_records=True,
        show_progress=False,
        distributed=True,
    )

    assert report["overall"]["sample_count"] == 0
    assert report["records"] == []


def test_training_compat_detects_manifest_sha_mismatch(tmp_path):
    args = _trainer_args(tmp_path)
    checkpoint = {
        "format": "cdvit-training-checkpoint-v1",
        "manifest_sha256": "not-the-current-sha",
        "compat_config": train._training_compat_config(args),
    }
    errors = train._checkpoint_compat_errors(checkpoint, args)
    assert "checkpoint manifest SHA differs from the current manifest" in errors


def test_training_compat_accepts_matching_resume_checkpoint(tmp_path):
    args = _trainer_args(tmp_path)
    compat = train._training_compat_config(args)
    checkpoint = {
        "format": "cdvit-training-checkpoint-v1",
        "manifest_sha256": compat["manifest_sha256"],
        "compat_config": compat,
    }
    assert train._checkpoint_compat_errors(checkpoint, args) == []


def test_epoch_timing_payload_keys_and_metrics_append(tmp_path):
    metrics_path = tmp_path / "runtime_metrics.jsonl"
    args = _trainer_args(tmp_path, runtime_metrics_jsonl=str(metrics_path))
    timing = train._empty_epoch_timing()

    started = time.time() - 0.001
    train._accumulate_timing(timing, "data_wait_seconds", started)
    assert timing["data_wait_seconds"] > 0

    final = train._finalize_epoch_timing(timing, epoch_wall_seconds=1.0)
    for key in (
        "data_wait_seconds",
        "device_transfer_seconds",
        "forward_loss_seconds",
        "backward_seconds",
        "optimizer_step_seconds",
        "scaler_update_seconds",
        "eval_seconds",
        "ema_eval_seconds",
        "distributed_eval_wait_seconds",
        "checkpoint_seconds",
        "compute_seconds",
        "forward_backward_update_seconds",
        "epoch_wall_seconds",
        "unattributed_seconds",
    ):
        assert key in final

    train._append_runtime_metrics(
        args,
        {
            "event": "epoch_timing",
            "epoch": 0,
            "timing": {
                key: round(float(value), 6) for key, value in sorted(final.items())
            },
        },
    )
    payload = json.loads(metrics_path.read_text(encoding="utf-8").strip())
    assert payload["event"] == "epoch_timing"
    assert "forward_loss_seconds" in payload["timing"]
    assert "backward_seconds" in payload["timing"]
    assert "forward_backward_update_seconds" in payload["timing"]


def test_typed_training_config_snapshots_from_args(tmp_path):
    args = _trainer_args(tmp_path, eval_ema_scope="full-only", eval_progress=False)

    runtime = TrainingRuntimeConfig.from_args(args)
    eval_config = EvalConfig.from_args(args)
    dataset = DatasetBuildConfig.from_args(args)

    assert runtime.num_workers == 2
    assert eval_config.eval_ema_scope == "full-only"
    assert eval_config.eval_progress is False
    assert dataset.data_name == "FS68Manifest"
    assert dataset.roll_quarter_turn_prob == 0.4
    assert dataset.roll_diagonal_prob == 0.1
    assert config_dict(eval_config)["eval_ema_scope"] == "full-only"


def test_heatmap_stage_cli_builder_preserves_eval_flags():
    parser = build_heatmap_stage_arg_parser()
    args = parser.parse_args(
        [
            "--eval-ema-scope",
            "full-only",
            "--no-eval-progress",
            "--respect-declared-splits",
        ]
    )

    assert args.eval_ema_scope == "full-only"
    assert args.eval_progress is False
    assert args.respect_declared_splits is True


def test_heatmap_stage_cli_builder_exposes_schema_loss_and_resume_flags():
    parser = build_heatmap_stage_arg_parser()
    args = parser.parse_args(
        [
            "--allow-missing-schema-heads",
            "--schema-head-loss-weighting",
            "per_head",
            "--schema-head-loss-weights",
            "landmarks_98=1.5",
            "--star-loss-weight",
            "0.01",
            "--roll-quarter-turn-prob",
            "0.5",
            "--roll-diagonal-prob",
            "0.2",
        ]
    )

    assert args.allow_missing_schema_heads is True
    assert args.schema_head_loss_weighting == "per_head"
    assert args.schema_head_loss_weights == "landmarks_98=1.5"
    assert args.star_loss_weight == 0.01
    assert args.roll_quarter_turn_prob == 0.5
    assert args.roll_diagonal_prob == 0.2


def test_allow_missing_schema_heads_only_accepts_schema_extension_keys():
    net = torch.nn.Module()
    net.output_layers = torch.nn.ModuleList([torch.nn.Linear(1, 1)])
    net.schema_output_layers = torch.nn.ModuleDict(
        {"landmarks_98": torch.nn.ModuleList([torch.nn.Linear(1, 1)])}
    )
    args = argparse.Namespace(allow_missing_schema_heads=True)
    legacy_state = {
        key: value.clone()
        for key, value in net.state_dict().items()
        if not key.startswith("schema_output_layers.")
    }

    train._load_resume_model_state(net, legacy_state, args)

    bad_state = {}
    with pytest.raises(ValueError, match="non_schema_missing"):
        train._load_resume_model_state(net, bad_state, args)


def test_save_best_weights_writes_only_explicit_name(tmp_path):
    state = {"weight": torch.tensor([1.0])}
    ckpt_dir = tmp_path / "ckpt"

    train._save_best_weights(state, ckpt_dir)

    assert (ckpt_dir / "best.weights.pt").is_file()
    assert not (ckpt_dir / "best_model").exists()


def test_save_training_checkpoint_writes_paired_weights_file(tmp_path):
    args = _trainer_args(tmp_path)
    model = torch.nn.Linear(2, 1)
    checkpoint_path = tmp_path / "ckpt" / "last_checkpoint.pt"
    weights_path = tmp_path / "ckpt" / "last_checkpoint.weights.pt"

    train._save_training_checkpoint(
        checkpoint_path,
        model,
        None,
        None,
        None,
        None,
        2,
        0.1,
        [(2, 10.0)],
        args,
        weights_path=weights_path,
    )

    checkpoint = train._torch_load_training_checkpoint(checkpoint_path, "cpu")
    weights = train._torch_load_training_checkpoint(weights_path, "cpu")

    assert checkpoint["format"] == "cdvit-training-checkpoint-v1"
    assert weights.keys() == model.state_dict().keys()
    assert all(
        torch.equal(weights[key], value) for key, value in model.state_dict().items()
    )


def test_allow_incompatible_resume_reapplies_current_optimizer_scheduler_args():
    old_lr = 0.0001
    old_sched_step = 50
    current_lr = 0.0003
    current_sched_step = 25
    checkpoint_model = torch.nn.Linear(2, 1)
    checkpoint_optimizer = torch.optim.AdamW(
        checkpoint_model.parameters(), lr=old_lr, weight_decay=1e-3
    )
    checkpoint_scheduler = torch.optim.lr_scheduler.StepLR(
        checkpoint_optimizer, old_sched_step, gamma=0.5
    )
    checkpoint = {
        "format": "cdvit-training-checkpoint-v1",
        "next_epoch": 3,
        "optimizer": checkpoint_optimizer.state_dict(),
        "scheduler": checkpoint_scheduler.state_dict(),
    }

    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=current_lr, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, current_sched_step, gamma=0.5
    )
    args = argparse.Namespace(
        allow_incompatible_resume=True,
        restore_rng=False,
        lr=current_lr,
        sched_step=current_sched_step,
    )

    start_epoch, _, _ = train._restore_training_checkpoint(
        checkpoint,
        optimizer,
        scheduler,
        None,
        None,
        99999,
        [],
        args,
    )

    assert start_epoch == 3
    assert optimizer.param_groups[0]["lr"] == pytest.approx(current_lr)
    assert optimizer.param_groups[0]["initial_lr"] == pytest.approx(current_lr)
    assert scheduler.step_size == current_sched_step
    assert scheduler.base_lrs == pytest.approx([current_lr])
    assert scheduler.get_last_lr() == pytest.approx([current_lr])


def test_pipeline_auto_resume_accepts_matching_full_checkpoint_metadata(tmp_path):
    args, paths = _pipeline_args(tmp_path, "--epoch", "3")
    ckpt = pipeline._checkpoint_dir(args, paths) / "last_checkpoint.pt"
    ckpt.parent.mkdir(parents=True, exist_ok=True)

    compat = pipeline._pipeline_training_compat_config(args, paths)
    meta = {
        "format": "cdvit-training-checkpoint-v1",
        "next_epoch": 1,
        "manifest_sha256": compat["manifest_sha256"],
        "compat_config": compat,
    }
    Path(str(ckpt) + ".meta.json").write_text(json.dumps(meta) + "\n", encoding="utf-8")

    compatible, reason = pipeline._checkpoint_matches_pipeline_request(
        args, paths, ckpt
    )
    assert compatible, reason


def test_pipeline_auto_resume_rejects_changed_contract(tmp_path):
    args, paths = _pipeline_args(tmp_path, "--epoch", "3")
    ckpt = pipeline._checkpoint_dir(args, paths) / "last_checkpoint.pt"
    ckpt.parent.mkdir(parents=True, exist_ok=True)

    compat = pipeline._pipeline_training_compat_config(args, paths)
    changed = dict(compat)
    changed["batch_size"] = compat["batch_size"] + 1
    meta = {
        "format": "cdvit-training-checkpoint-v1",
        "next_epoch": 1,
        "manifest_sha256": compat["manifest_sha256"],
        "compat_config": changed,
    }
    Path(str(ckpt) + ".meta.json").write_text(json.dumps(meta) + "\n", encoding="utf-8")

    compatible, reason = pipeline._checkpoint_matches_pipeline_request(
        args, paths, ckpt
    )
    assert compatible is False
    assert "training contract" in reason


def test_pipeline_train_command_forwards_runtime_flags(tmp_path):
    args, paths = _pipeline_args(
        tmp_path,
        "--no-pin-memory",
        "--no-persistent-workers",
        "--prefetch-factor",
        "4",
        "--eval-every",
        "5",
        "--full-eval-every",
        "20",
        "--eval-ema-every",
        "10",
        "--eval-max-samples",
        "128",
        "--roll-quarter-turn-prob",
        "0.5",
        "--roll-diagonal-prob",
        "0.2",
        "--no-save-last-checkpoint",
        "--runtime-metrics-jsonl",
        str(tmp_path / "metrics.jsonl"),
        "--synchronize-runtime-timing",
    )

    command = pipeline._train_command(args, paths)
    assert "--no-pin-memory" in command
    assert "--no-persistent-workers" in command
    assert command[command.index("--prefetch-factor") + 1] == "4"
    assert command[command.index("--eval-every") + 1] == "5"
    assert command[command.index("--full-eval-every") + 1] == "20"
    assert command[command.index("--eval-ema-every") + 1] == "10"
    assert command[command.index("--eval-max-samples") + 1] == "128"
    assert command[command.index("--roll-quarter-turn-prob") + 1] == "0.5"
    assert command[command.index("--roll-diagonal-prob") + 1] == "0.2"
    assert "--no-save-last-checkpoint" in command
    assert "--synchronize-runtime-timing" in command
    assert "--no-synchronize-runtime-timing" not in command
    assert command[command.index("--runtime-metrics-jsonl") + 1].endswith(
        "metrics.jsonl"
    )


def test_pipeline_compat_config_honors_train_arg_override(tmp_path):
    args, paths = _pipeline_args(
        tmp_path,
        "--train-arg=--batch-size",
        "--train-arg=32",
    )
    assert pipeline._pipeline_training_compat_config(args, paths)["batch_size"] == 32


def test_pipeline_compat_config_tracks_roll_augmentation(tmp_path):
    args, paths = _pipeline_args(
        tmp_path,
        "--roll-quarter-turn-prob",
        "0.5",
        "--roll-diagonal-prob",
        "0.2",
    )

    compat = pipeline._pipeline_training_compat_config(args, paths)

    assert compat["roll_quarter_turn_prob"] == 0.5
    assert compat["roll_diagonal_prob"] == 0.2
