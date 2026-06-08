from __future__ import annotations

import json
from pathlib import Path

from lib.landmarks.datasets.manifest import manifest_index_path
from tools.landmarks import run_cdvit_manifest_training_pipeline as pipeline


def _args(tmp_path: Path):
    parser = pipeline._build_arg_parser()
    args = parser.parse_args(
        [
            "--output-root",
            str(tmp_path / "runs"),
            "--run-name",
            "sig",
            "--dataset",
            "wflw",
            "--dataset-source",
            "wflw=fixtures/wflw",
            "--stop-after",
            "build_dataset_manifests",
        ]
    )
    return pipeline._normalize_runtime_args(args)


def _paths(args):
    return pipeline.PipelinePaths(
        output_root=Path(args.output_root),
        run_name=args.run_name,
        explicit_manifest=None,
    )


def test_manifest_stage_signature_controls_dataset_stage_skip(tmp_path: Path) -> None:
    args = _args(tmp_path)
    paths = _paths(args)

    output = Path(pipeline._build_manifest_outputs(args, paths)[0])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text('{"samples": []}\n', encoding="utf-8")

    assert pipeline._stage_complete("build_dataset_manifests", args, paths) is False

    pipeline._write_stage_signature(
        "build_dataset_manifests",
        args,
        paths,
        [str(output)],
        command=["echo", "fake"],
    )
    assert pipeline._stage_complete("build_dataset_manifests", args, paths) is True

    output.write_text('{"samples": [{"id": "changed"}]}\n', encoding="utf-8")
    assert pipeline._stage_complete("build_dataset_manifests", args, paths) is False


def test_manifest_stage_signature_controls_validation_stage_skip(tmp_path: Path) -> None:
    args = _args(tmp_path)
    paths = _paths(args)

    manifest = paths.hard_negative_manifest
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text('{"samples": []}\n', encoding="utf-8")
    paths.validation_report.parent.mkdir(parents=True, exist_ok=True)
    paths.validation_report.write_text('{"ok": true}\n', encoding="utf-8")
    index_path = manifest_index_path(manifest)
    index_path.write_text('{"type": "manifest_index_meta"}\n', encoding="utf-8")

    assert pipeline._stage_complete("validate_cdvit_manifest", args, paths) is False

    pipeline._write_stage_signature(
        "validate_cdvit_manifest",
        args,
        paths,
        [str(paths.validation_report), str(index_path)],
    )
    assert pipeline._stage_complete("validate_cdvit_manifest", args, paths) is True

    manifest.write_text('{"samples": [{"id": "changed"}]}\n', encoding="utf-8")
    assert pipeline._stage_complete("validate_cdvit_manifest", args, paths) is False


def test_train_stage_signature_controls_train_skip(tmp_path: Path) -> None:
    args = _args(tmp_path)
    args.stop_after = "train_cdvit"
    args.epoch = 3
    args.auto_resume = False
    paths = _paths(args)
    ckpt_dir = pipeline._checkpoint_dir(args, paths)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    paths.train_command_json.parent.mkdir(parents=True, exist_ok=True)
    paths.train_command_json.write_text('{"command": []}\n', encoding="utf-8")
    payload = {
        "status": "complete",
        "requested_epochs": 3,
        "pipeline_training_signature_digest": pipeline._pipeline_training_signature_digest(args, paths),
        "pipeline_manifest_sha256": pipeline._safe_sha256_file(Path(pipeline._pipeline_effective_manifest(args, paths))),
    }
    (ckpt_dir / "training_complete.json").write_text(json.dumps(payload) + "\n", encoding="utf-8")
    (ckpt_dir / "last_checkpoint.pt").write_bytes(b"last")
    (ckpt_dir / "best.weights.pt").write_bytes(b"weights")
    (ckpt_dir / "best_checkpoint.pt").write_bytes(b"checkpoint")

    assert pipeline._stage_complete("train_cdvit", args, paths) is False

    pipeline._write_stage_signature(
        "train_cdvit",
        args,
        paths,
        pipeline._train_stage_outputs(args, paths),
        command=pipeline._train_command(args, paths),
    )
    assert pipeline._stage_complete("train_cdvit", args, paths) is True

    (ckpt_dir / "best.weights.pt").write_bytes(b"changed")
    assert pipeline._stage_complete("train_cdvit", args, paths) is False
