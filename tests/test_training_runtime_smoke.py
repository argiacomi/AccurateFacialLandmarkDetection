from __future__ import annotations

import argparse
import json
from pathlib import Path

import TrainHeatmapStageFP16 as train
from tools.landmarks import run_cdvit_manifest_training_pipeline as pipeline


def _trainer_args(tmp_path: Path, **overrides):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"samples": []}) + "\n", encoding="utf-8")
    values = {
        "eval_num_workers": 0,
        "num_workers": 2,
        "pin_memory": True,
        "persistent_workers": True,
        "prefetch_factor": 2,
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
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def test_dataloader_kwargs_disable_prefetch_without_workers(tmp_path):
    args = _trainer_args(tmp_path, num_workers=0, persistent_workers=True, prefetch_factor=2)
    kwargs = train._dataloader_kwargs(args)
    assert kwargs == {"num_workers": 0, "pin_memory": True}


def test_restore_rng_forces_non_persistent_workers(tmp_path):
    args = _trainer_args(tmp_path, restore_rng=True, persistent_workers=True)
    train._normalize_runtime_args(args)
    assert args.persistent_workers is False


def test_training_compat_detects_manifest_sha_mismatch(tmp_path):
    args = _trainer_args(tmp_path)
    checkpoint = {
        "format": "cdvit-training-checkpoint-v1",
        "manifest_sha256": "not-the-current-sha",
        "compat_config": train._training_compat_config(args),
    }
    errors = train._checkpoint_compat_errors(checkpoint, args)
    assert "checkpoint manifest SHA differs from the current manifest" in errors


def test_pipeline_train_command_forwards_runtime_flags(tmp_path):
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
                "--no-save-last-checkpoint",
                "--runtime-metrics-jsonl",
                str(tmp_path / "metrics.jsonl"),
            ]
        )
    )
    paths = pipeline.PipelinePaths(
        output_root=args.output_root,
        run_name=args.run_name,
        explicit_manifest=args.manifest.resolve(),
    )

    command = pipeline._train_command(args, paths)
    assert "--no-pin-memory" in command
    assert "--no-persistent-workers" in command
    assert command[command.index("--prefetch-factor") + 1] == "4"
    assert command[command.index("--eval-every") + 1] == "5"
    assert command[command.index("--full-eval-every") + 1] == "20"
    assert command[command.index("--eval-ema-every") + 1] == "10"
    assert command[command.index("--eval-max-samples") + 1] == "128"
    assert "--no-save-last-checkpoint" in command
    assert command[command.index("--runtime-metrics-jsonl") + 1].endswith("metrics.jsonl")
