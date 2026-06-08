from __future__ import annotations

import json
from pathlib import Path

from lib.landmarks.pipeline.config import _extract_config_path, _merge_config_argv
from tools.landmarks import run_cdvit_manifest_training_pipeline as pipeline


def test_config_dry_run_merges_cli_overrides_and_writes_resolved_config(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "pipeline_config.json"
    output_root = tmp_path / "runs"
    config_path.write_text(
        json.dumps(
            {
                "run_name": "from_config",
                "datasets": ["wflw"],
                "dataset_sources": {"wflw": "config/wflw"},
                "training": {
                    "batch_size": 8,
                    "epoch": 3,
                    "train_arg": ["--config-flag 1"],
                },
                "runtime": {"num_workers": 2, "pin_memory": False},
                "eval": {"eval_every": 7},
                "hard_negative": {"write_audit": True, "total_samples": 12},
            }
        ),
        encoding="utf-8",
    )

    argv = [
        "--config",
        str(config_path),
        "--output-root",
        str(output_root),
        "--run-name",
        "cli_run",
        "--dataset",
        "cofw68",
        "--dataset-source",
        "cofw68=cli/cofw68",
        "--batch-size",
        "16",
        "--train-arg",
        "--cli-flag 2",
        "--dry-run",
        "--stop-after",
        "build_dataset_manifests",
    ]

    parser = pipeline._build_arg_parser()
    config_path_from_argv = _extract_config_path(argv)
    merged_argv = _merge_config_argv(parser, config_path_from_argv, argv)
    args = parser.parse_args(merged_argv)

    # Scalar values are overridden by explicit CLI args because config-derived
    # tokens are prepended before user CLI tokens.
    assert args.run_name == "cli_run"
    assert args.dataset == "cofw68"
    assert args.batch_size == 16

    # Append actions intentionally merge config entries with CLI entries.
    assert args.dataset_source == ["wflw=config/wflw", "cofw68=cli/cofw68"]
    assert args.train_arg == ["--config-flag 1", "--cli-flag 2"]
    assert args.hard_negative_arg == ["--write-audit", "--total-samples 12"]

    assert pipeline.main(argv) == 0

    resolved_path = output_root / "cli_run" / "run_config.resolved.json"
    assert resolved_path.is_file()
    resolved = json.loads(resolved_path.read_text(encoding="utf-8"))

    assert resolved["args"]["dry_run"] is True
    assert resolved["args"]["run_name"] == "cli_run"
    assert resolved["args"]["dataset"] == "cofw68"
    assert resolved["args"]["batch_size"] == 16
    assert resolved["args"]["dataset_source"] == [
        "wflw=config/wflw",
        "cofw68=cli/cofw68",
    ]
    assert resolved["args"]["train_arg"] == ["--config-flag 1", "--cli-flag 2"]
    assert resolved["args"]["hard_negative_arg"] == [
        "--write-audit",
        "--total-samples 12",
    ]
    assert resolved["selected_stages"] == ["build_dataset_manifests"]
