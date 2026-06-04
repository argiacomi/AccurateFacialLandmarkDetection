#!/usr/bin/env python3
"""Build local landmark manifests and train CD-ViT on a hard-negative 68-point mix.

Pipeline stages:

1. build per-dataset manifests with local ``build_quality_dataset.py``
2. merge them with local ``build_hard_negative_manifest.py``
3. validate that the final manifest contains CD-ViT-compatible 68-point samples
4. launch ``TrainHeatmapStageFP16.py --data_name FS68Manifest``
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import subprocess
import sys
import time
import typing as T
from dataclasses import dataclass, field
from pathlib import Path


CDVIT_ROOT = Path(__file__).resolve().parents[2]
TOOLS_ROOT = CDVIT_ROOT / "tools" / "landmarks"
DEFAULT_DATASETS = "wflw,cofw,merl-rav,aflw2000-3d,300w,menpo2d,multipie"
MINED_MANIFEST_NAME = "manifest.json"
PROGRESS_LOG_NAME = "pipeline_progress.jsonl"
TRAIN_COMMAND_NAME = "train_command.json"
VALIDATION_REPORT_NAME = "cdvit_manifest_validation.json"

STAGES: tuple[str, ...] = (
    "build_dataset_manifests",
    "build_hard_negative_manifest",
    "validate_cdvit_manifest",
    "train_cdvit",
)

HARD_NEGATIVE_MANIFEST_FLAGS: dict[str, str] = {
    "wflw": "--wflw-manifest",
    "aflw2000-3d": "--aflw2000-manifest",
    "merl-rav": "--merl-rav-manifest",
    "cofw": "--cofw-manifest",
    "menpo2d": "--menpo2d-manifest",
    "multipie": "--multipie-manifest",
    "300w": "--w300-manifest",
}

DATASET_ALIASES = {
    "aflw2000": "aflw2000-3d",
    "aflw2000-3d": "aflw2000-3d",
    "aflw2000_3d": "aflw2000-3d",
    "merl_rav": "merl-rav",
    "merl-rav": "merl-rav",
    "menpo": "menpo2d",
    "menpo2d": "menpo2d",
    "menpo-2d": "menpo2d",
    "multi-pie": "multipie",
    "multipie": "multipie",
    "w300": "300w",
    "300w": "300w",
    "300-w": "300w",
    "wflw": "wflw",
    "cofw": "cofw",
}


@dataclass(frozen=True)
class PipelinePaths:
    output_root: Path
    run_name: str
    explicit_manifest: Path | None = None
    run_root: Path = field(init=False)
    dataset_root: Path = field(init=False)
    hard_negative_dir: Path = field(init=False)
    hard_negative_manifest: Path = field(init=False)
    validation_report: Path = field(init=False)
    checkpoint_dir: Path = field(init=False)
    progress_log: Path = field(init=False)
    train_command_json: Path = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "run_root", self.output_root / self.run_name)
        object.__setattr__(self, "dataset_root", self.run_root / "datasets")
        object.__setattr__(self, "hard_negative_dir", self.run_root / "hard_negative_mix")
        object.__setattr__(
            self,
            "hard_negative_manifest",
            self.explicit_manifest or self.hard_negative_dir / MINED_MANIFEST_NAME,
        )
        object.__setattr__(self, "validation_report", self.run_root / VALIDATION_REPORT_NAME)
        object.__setattr__(self, "checkpoint_dir", self.run_root / "checkpoints")
        object.__setattr__(self, "progress_log", self.run_root / PROGRESS_LOG_NAME)
        object.__setattr__(self, "train_command_json", self.run_root / TRAIN_COMMAND_NAME)


@dataclass
class StageResult:
    name: str
    status: str
    duration_seconds: float
    command: list[str] = field(default_factory=list)
    outputs: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    error: str = ""

    def to_json(self) -> dict[str, T.Any]:
        return {
            "name": self.name,
            "status": self.status,
            "duration_seconds": round(self.duration_seconds, 3),
            "command": self.command,
            "outputs": self.outputs,
            "notes": self.notes,
            "error": self.error,
        }


def _normalize_dataset_name(value: str) -> str:
    key = str(value or "").strip().lower().replace("_", "-")
    return DATASET_ALIASES.get(key, key)


def _split_csv(value: str | None) -> tuple[str, ...]:
    if not value:
        return ()
    return tuple(item.strip() for item in value.split(",") if item.strip())


def _datasets(args: argparse.Namespace) -> tuple[str, ...]:
    requested = _split_csv(args.dataset) or _split_csv(DEFAULT_DATASETS)
    normalized: list[str] = []
    for item in requested:
        dataset = _normalize_dataset_name(item)
        if dataset not in HARD_NEGATIVE_MANIFEST_FLAGS:
            raise ValueError(f"unsupported dataset for CD-ViT hard-negative mix: {item!r}")
        if dataset not in normalized:
            normalized.append(dataset)
    return tuple(normalized)


def _parse_dataset_mapping(values: T.Sequence[str] | None, option: str) -> dict[str, Path]:
    mapping: dict[str, Path] = {}
    for spec in values or []:
        if "=" not in spec:
            raise ValueError(f"{option} must use dataset=path format, got {spec!r}")
        dataset, raw_path = spec.split("=", 1)
        dataset = _normalize_dataset_name(dataset)
        raw_path = raw_path.strip()
        if dataset not in HARD_NEGATIVE_MANIFEST_FLAGS:
            raise ValueError(f"{option} received unsupported dataset {dataset!r}")
        if not raw_path:
            raise ValueError(f"{option} received empty path for dataset {dataset!r}")
        mapping[dataset] = Path(raw_path)
    return mapping


def _dataset_source_map(args: argparse.Namespace) -> dict[str, Path]:
    mapping = _parse_dataset_mapping(args.dataset_source, "--dataset-source")
    if args.source_dir:
        datasets = _datasets(args)
        if len(datasets) != 1:
            raise ValueError("--source-dir is only valid when exactly one --dataset is selected")
        mapping.setdefault(datasets[0], Path(args.source_dir))
    return mapping


def _dataset_source_zip_map(args: argparse.Namespace) -> dict[str, Path]:
    mapping = _parse_dataset_mapping(args.dataset_source_zip, "--dataset-source-zip")
    if args.source_zip:
        datasets = _datasets(args)
        if len(datasets) != 1:
            raise ValueError("--source-zip is only valid when exactly one --dataset is selected")
        mapping.setdefault(datasets[0], Path(args.source_zip))
    return mapping


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: T.Mapping[str, T.Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _append_progress(paths: PipelinePaths, result: StageResult) -> None:
    paths.progress_log.parent.mkdir(parents=True, exist_ok=True)
    with paths.progress_log.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(result.to_json(), sort_keys=True) + "\n")


def _script(relative_path: str) -> str:
    return str(TOOLS_ROOT / relative_path)


def _append_extra(argv: list[str], extras: T.Sequence[str]) -> list[str]:
    for extra in extras:
        argv.extend(shlex.split(extra))
    return argv


def _dataset_manifest_path(paths: PipelinePaths, dataset: str) -> Path:
    return paths.dataset_root / dataset / "manifest.json"


def _dataset_build_commands(args: argparse.Namespace, paths: PipelinePaths) -> list[list[str]]:
    source_map = _dataset_source_map(args)
    source_zip_map = _dataset_source_zip_map(args)
    overlap = sorted(set(source_map) & set(source_zip_map))
    if overlap:
        raise ValueError("datasets cannot have both source dir and source zip: " + ", ".join(overlap))

    commands: list[list[str]] = []
    for dataset in _datasets(args):
        output_dir = paths.dataset_root / dataset
        argv = [
            args.python_executable,
            _script("build_quality_dataset.py"),
            "--dataset",
            dataset,
            "--output-dir",
            str(output_dir),
            "--manifest-mode",
            "replace",
        ]
        if dataset in source_map:
            argv.extend(["--source-dir", str(source_map[dataset])])
        if dataset in source_zip_map:
            argv.extend(["--source-zip", str(source_zip_map[dataset])])
        extras = list(args.dataset_build_arg or [])
        if dataset in {"menpo2d", "multipie"} and not args.include_39pt_profile:
            extras.insert(0, "--no-39pt-profile")
        commands.append(_append_extra(argv, extras))
    return commands


def _hard_negative_command(args: argparse.Namespace, paths: PipelinePaths) -> list[str]:
    argv = [
        args.python_executable,
        _script("build_hard_negative_manifest.py"),
        "--output-dir",
        str(paths.hard_negative_dir),
    ]
    for dataset in _datasets(args):
        manifest = _dataset_manifest_path(paths, dataset)
        if manifest.is_file() or args.dry_run:
            argv.extend([HARD_NEGATIVE_MANIFEST_FLAGS[dataset], str(manifest)])

    if "--write-audit" not in " ".join(args.hard_negative_arg or []):
        argv.append("--write-audit")
    if args.max_profile_occlusion is not None:
        argv.extend(["--max-profile-occlusion", str(args.max_profile_occlusion)])
    if args.max_profile is not None:
        argv.extend(["--max-profile", str(args.max_profile)])
    if args.max_occlusion is not None:
        argv.extend(["--max-occlusion", str(args.max_occlusion)])
    if args.max_anchors is not None:
        argv.extend(["--max-anchors", str(args.max_anchors)])
    return _append_extra(argv, args.hard_negative_arg or [])


def _train_command(args: argparse.Namespace, paths: PipelinePaths) -> list[str]:
    ckpt_folder = Path(args.ckpt_folder) if args.ckpt_folder else paths.checkpoint_dir
    argv = [
        args.torchrun_executable,
        f"--nproc_per_node={args.nproc_per_node}",
        str(CDVIT_ROOT / "TrainHeatmapStageFP16.py"),
        "--data_name",
        "FS68Manifest",
        "--manifest",
        str(paths.hard_negative_manifest),
        "--ckpt_folder",
        str(ckpt_folder),
        "--batch_size",
        str(args.batch_size),
        "--num_workers",
        str(args.num_workers),
        "--epoch",
        str(args.epoch),
        "--heatmap_size",
        str(args.heatmap_size),
        "--lmk_num",
        str(args.lmk_num),
        "--lr",
        str(args.lr),
    ]
    return _append_extra(argv, args.train_arg or [])


def _stage_slice(start_at: str | None, stop_after: str | None) -> tuple[str, ...]:
    if start_at and start_at not in STAGES:
        raise ValueError(f"unknown --start-at stage {start_at!r}; choose one of {', '.join(STAGES)}")
    if stop_after and stop_after not in STAGES:
        raise ValueError(f"unknown --stop-after stage {stop_after!r}; choose one of {', '.join(STAGES)}")
    start = STAGES.index(start_at) if start_at else 0
    stop = STAGES.index(stop_after) if stop_after else len(STAGES) - 1
    if start > stop:
        raise ValueError(f"--start-at {start_at!r} occurs after --stop-after {stop_after!r}")
    return STAGES[start : stop + 1]


def _require_local_tools(args: argparse.Namespace) -> None:
    if args.manifest:
        return
    required = [TOOLS_ROOT / "build_quality_dataset.py", TOOLS_ROOT / "build_hard_negative_manifest.py"]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing local manifest tool(s): " + ", ".join(missing))


def _run_command(argv: list[str], *, cwd: Path, dry_run: bool) -> None:
    print("+ " + " ".join(shlex.quote(part) for part in argv), flush=True)
    if dry_run:
        return
    subprocess.run(argv, cwd=str(cwd), check=True)


def _load_manifest_samples(manifest_path: Path) -> list[dict[str, T.Any]]:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    samples = payload.get("samples", payload.get("scenarios", []))
    if not isinstance(samples, list):
        raise ValueError(f"manifest samples must be a list: {manifest_path}")
    return [sample for sample in samples if isinstance(sample, dict)]


def _resolve_manifest_path(manifest_path: Path, raw_value: T.Any) -> Path:
    path = Path(str(raw_value or ""))
    if path.is_absolute():
        return path
    return (manifest_path.parent / path).resolve()


def _validate_cdvit_manifest(args: argparse.Namespace, paths: PipelinePaths) -> dict[str, T.Any]:
    try:
        import numpy as np
    except ImportError as err:
        raise RuntimeError("numpy is required to validate CD-ViT landmark manifests") from err

    manifest_path = paths.hard_negative_manifest
    if not manifest_path.is_file():
        raise FileNotFoundError(f"missing manifest: {manifest_path}")

    samples = _load_manifest_samples(manifest_path)
    report: dict[str, T.Any] = {
        "manifest": str(manifest_path),
        "manifest_sha256": _sha256_file(manifest_path),
        "total_samples": len(samples),
        "exact_68_samples": 0,
        "non_68_samples": 0,
        "missing_landmarks": 0,
        "invalid_landmarks": 0,
        "datasets": {},
        "examples": {"non_68": [], "missing": [], "invalid": []},
    }

    for sample in samples:
        dataset = str(sample.get("dataset") or sample.get("source", {}).get("dataset") or "unknown")
        dataset_stats = report["datasets"].setdefault(
            dataset,
            {"total": 0, "exact_68": 0, "non_68": 0, "missing": 0, "invalid": 0},
        )
        dataset_stats["total"] += 1
        landmarks_value = sample.get("landmarks") or sample.get("ground_truth")
        sample_id = str(sample.get("sample_id") or sample.get("id") or "")
        if not landmarks_value:
            report["missing_landmarks"] += 1
            dataset_stats["missing"] += 1
            if len(report["examples"]["missing"]) < 10:
                report["examples"]["missing"].append(sample_id)
            continue
        landmarks_path = _resolve_manifest_path(manifest_path, landmarks_value)
        if not landmarks_path.is_file():
            report["missing_landmarks"] += 1
            dataset_stats["missing"] += 1
            if len(report["examples"]["missing"]) < 10:
                report["examples"]["missing"].append(str(landmarks_path))
            continue
        try:
            landmarks = np.load(landmarks_path)
        except Exception as err:  # noqa: BLE001
            report["invalid_landmarks"] += 1
            dataset_stats["invalid"] += 1
            if len(report["examples"]["invalid"]) < 10:
                report["examples"]["invalid"].append({"path": str(landmarks_path), "error": str(err)})
            continue
        if getattr(landmarks, "ndim", 0) == 2 and landmarks.shape[0] == int(args.lmk_num) and landmarks.shape[1] >= 2:
            report["exact_68_samples"] += 1
            dataset_stats["exact_68"] += 1
        else:
            report["non_68_samples"] += 1
            dataset_stats["non_68"] += 1
            if len(report["examples"]["non_68"]) < 10:
                report["examples"]["non_68"].append(
                    {"sample_id": sample_id, "path": str(landmarks_path), "shape": list(getattr(landmarks, "shape", []))}
                )

    _write_json(paths.validation_report, report)

    if report["exact_68_samples"] <= 0:
        raise ValueError(f"manifest has no {args.lmk_num}-point samples: {manifest_path}")
    if not args.allow_non68 and report["non_68_samples"]:
        raise ValueError(
            f"manifest contains {report['non_68_samples']} non-{args.lmk_num}-point samples. "
            "Pass --allow-non68 to train on the exact-68 subset skipped by DatasetFS68Manifest, "
            "or fix/remap those source manifests."
        )
    if report["missing_landmarks"] or report["invalid_landmarks"]:
        raise ValueError("manifest contains missing or invalid landmark files. See " f"{paths.validation_report}")
    return report


def _stage_complete(stage: str, args: argparse.Namespace, paths: PipelinePaths) -> bool:
    if args.manifest and stage in {"build_dataset_manifests", "build_hard_negative_manifest"}:
        return True
    if stage == "build_dataset_manifests":
        return all(_dataset_manifest_path(paths, dataset).is_file() for dataset in _datasets(args))
    if stage == "build_hard_negative_manifest":
        return paths.hard_negative_manifest.is_file()
    if stage == "validate_cdvit_manifest":
        return paths.validation_report.is_file()
    if stage == "train_cdvit":
        ckpt_folder = Path(args.ckpt_folder) if args.ckpt_folder else paths.checkpoint_dir
        return (ckpt_folder / "best_model").exists()
    raise ValueError(f"unknown stage: {stage}")


def _run_stage(stage: str, args: argparse.Namespace, paths: PipelinePaths) -> StageResult:
    started = time.time()
    command: list[str] = []
    outputs: list[str] = []
    notes: list[str] = []
    try:
        if not args.force and _stage_complete(stage, args, paths):
            return StageResult(stage, "skipped", time.time() - started, notes=["already complete"])

        if stage == "build_dataset_manifests":
            for command in _dataset_build_commands(args, paths):
                _run_command(command, cwd=CDVIT_ROOT, dry_run=args.dry_run)
            outputs = [str(_dataset_manifest_path(paths, dataset)) for dataset in _datasets(args)]

        elif stage == "build_hard_negative_manifest":
            if args.manifest:
                notes.append("using explicit --manifest; hard-negative build skipped")
            else:
                command = _hard_negative_command(args, paths)
                _run_command(command, cwd=CDVIT_ROOT, dry_run=args.dry_run)
            outputs = [str(paths.hard_negative_manifest)]

        elif stage == "validate_cdvit_manifest":
            report = {} if args.dry_run else _validate_cdvit_manifest(args, paths)
            outputs = [str(paths.validation_report)]
            if report:
                notes.append(
                    f"validated {report['exact_68_samples']} exact-{args.lmk_num} sample(s) "
                    f"from {report['total_samples']} manifest entries"
                )

        elif stage == "train_cdvit":
            command = _train_command(args, paths)
            _write_json(
                paths.train_command_json,
                {
                    "command": command,
                    "cwd": str(CDVIT_ROOT),
                    "manifest": str(paths.hard_negative_manifest),
                    "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                },
            )
            _run_command(command, cwd=CDVIT_ROOT, dry_run=args.dry_run)
            ckpt_folder = Path(args.ckpt_folder) if args.ckpt_folder else paths.checkpoint_dir
            outputs = [str(ckpt_folder), str(paths.train_command_json)]

        else:
            raise ValueError(f"unknown stage: {stage}")
        status = "planned" if args.dry_run else "ok"
        return StageResult(stage, status, time.time() - started, command=command, outputs=outputs, notes=notes)
    except Exception as err:  # noqa: BLE001
        return StageResult(stage, "error", time.time() - started, command=command, outputs=outputs, notes=notes, error=str(err))


def _default_run_name() -> str:
    return time.strftime("cdvit_fs68_%Y%m%d_%H%M%S", time.localtime())


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=Path("runs/landmarks"))
    parser.add_argument("--run-name", default=_default_run_name())
    parser.add_argument("--manifest", type=Path, default=None, help="Use an existing hard-negative manifest and skip manifest build stages.")
    parser.add_argument("--dataset", default=DEFAULT_DATASETS, help="Comma-separated dataset list.")
    parser.add_argument("--dataset-source", action="append", default=[], help="dataset=source_dir, repeatable.")
    parser.add_argument("--dataset-source-zip", action="append", default=[], help="dataset=source_zip, repeatable.")
    parser.add_argument("--source-dir", type=Path, default=None, help="Source dir for a single selected dataset.")
    parser.add_argument("--source-zip", type=Path, default=None, help="Source zip for a single selected dataset.")
    parser.add_argument("--dataset-build-arg", action="append", default=[], help="Extra quoted arg(s) passed to build_quality_dataset.py; repeatable.")
    parser.add_argument("--hard-negative-arg", action="append", default=[], help="Extra quoted arg(s) passed to build_hard_negative_manifest.py; repeatable.")
    parser.add_argument("--include-39pt-profile", action="store_true", help="Accepted for compatibility; local builder emits canonical 68 only.")
    parser.add_argument("--allow-non68", action="store_true", help="Allow mixed landmark counts and train on the exact-68 subset.")
    parser.add_argument("--max-profile-occlusion", type=int, default=None)
    parser.add_argument("--max-profile", type=int, default=None)
    parser.add_argument("--max-occlusion", type=int, default=None)
    parser.add_argument("--max-anchors", type=int, default=None)
    parser.add_argument("--python-executable", default=sys.executable)
    parser.add_argument("--torchrun-executable", default="torchrun")
    parser.add_argument("--nproc-per-node", type=int, default=2)
    parser.add_argument("--ckpt-folder", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=12)
    parser.add_argument("--epoch", type=int, default=500)
    parser.add_argument("--heatmap-size", type=int, default=32)
    parser.add_argument("--lmk-num", type=int, default=68)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--train-arg", action="append", default=[], help="Extra quoted arg(s) passed to TrainHeatmapStageFP16.py; repeatable.")
    parser.add_argument("--start-at", choices=STAGES, default=None)
    parser.add_argument("--stop-after", choices=STAGES, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    paths = PipelinePaths(
        output_root=Path(args.output_root),
        run_name=args.run_name,
        explicit_manifest=Path(args.manifest).resolve() if args.manifest else None,
    )
    paths.run_root.mkdir(parents=True, exist_ok=True)
    _require_local_tools(args)

    selected_stages = _stage_slice(args.start_at, args.stop_after)
    print(f"CD-ViT pipeline run root: {paths.run_root}")
    print(f"Stages: {', '.join(selected_stages)}")

    for stage in selected_stages:
        result = _run_stage(stage, args, paths)
        _append_progress(paths, result)
        print(json.dumps(result.to_json(), indent=2), flush=True)
        if result.status == "error":
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
