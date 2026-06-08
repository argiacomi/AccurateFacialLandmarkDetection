from __future__ import annotations

from pathlib import Path

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

    assert pipeline._stage_complete("validate_cdvit_manifest", args, paths) is False

    pipeline._write_stage_signature(
        "validate_cdvit_manifest",
        args,
        paths,
        [str(paths.validation_report)],
    )
    assert pipeline._stage_complete("validate_cdvit_manifest", args, paths) is True

    manifest.write_text('{"samples": [{"id": "changed"}]}\n', encoding="utf-8")
    assert pipeline._stage_complete("validate_cdvit_manifest", args, paths) is False
