#!/usr/bin/env python3
"""Build local landmark manifests and train CD-ViT on a schema-aware hard-negative mix.

Pipeline stages:

1. build per-dataset manifests with local ``build_quality_dataset.py``
2. optionally build a ``production_validated`` manifest from ``--prod-dir``
3. merge them with local ``build_hard_negative_manifest.py``
4. validate that the final manifest follows the schema-aware training contract
5. launch ``TrainHeatmapStageFP16.py --data_name FS68Manifest``
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
if str(CDVIT_ROOT) not in sys.path:
    sys.path.insert(0, str(CDVIT_ROOT))

from lib.landmarks.core.manifest_aliases import (
    CANONICAL_MANIFEST_DATA_NAME,
    LEGACY_MANIFEST_DATA_NAME,
    MANIFEST_DATA_NAME_ALIASES,
)

from lib.landmarks.manifest.validator import validate_training_manifest
from lib.landmarks.pipeline.config import (
    _extract_config_path,
    _json_safe_pipeline_value,
    _merge_config_argv,
)
from lib.landmarks.training.checkpoint_compat import (
    build_pipeline_training_compat_config,
    checkpoint_compat_errors_for_config,
    training_compat_digest_from_config,
)


TOOLS_ROOT = CDVIT_ROOT / "tools" / "landmarks"
DEFAULT_DATASETS = "wflw,cofw,merl-rav,aflw2000-3d,300w,menpo2d,multipie"
MINED_MANIFEST_NAME = "manifest.json"
PROGRESS_LOG_NAME = "pipeline_progress.jsonl"
TRAIN_COMMAND_NAME = "train_command.json"
VALIDATION_REPORT_NAME = "training_manifest_validation.json"
PRODUCTION_DATASET = "production_validated"

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
    PRODUCTION_DATASET: "--production-validated-manifest",
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
    "production": PRODUCTION_DATASET,
    "prod": PRODUCTION_DATASET,
    "prod-dir": PRODUCTION_DATASET,
    "production-validated": PRODUCTION_DATASET,
    "production_validated": PRODUCTION_DATASET,
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
        object.__setattr__(
            self, "hard_negative_dir", self.run_root / "hard_negative_mix"
        )
        object.__setattr__(
            self,
            "hard_negative_manifest",
            self.explicit_manifest or self.hard_negative_dir / MINED_MANIFEST_NAME,
        )
        object.__setattr__(
            self, "validation_report", self.run_root / VALIDATION_REPORT_NAME
        )
        object.__setattr__(self, "checkpoint_dir", self.run_root / "checkpoints")
        object.__setattr__(self, "progress_log", self.run_root / PROGRESS_LOG_NAME)
        object.__setattr__(
            self, "train_command_json", self.run_root / TRAIN_COMMAND_NAME
        )


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
        if dataset == PRODUCTION_DATASET:
            raise ValueError(
                "Use --prod-dir for production_validated data instead of adding it to --dataset"
            )
        if dataset not in HARD_NEGATIVE_MANIFEST_FLAGS:
            raise ValueError(
                f"unsupported dataset for CD-ViT hard-negative mix: {item!r}"
            )
        if dataset not in normalized:
            normalized.append(dataset)
    return tuple(normalized)


def _parse_dataset_mapping(
    values: T.Sequence[str] | None, option: str
) -> dict[str, Path]:
    mapping: dict[str, Path] = {}
    for spec in values or []:
        if "=" not in spec:
            raise ValueError(f"{option} must use dataset=path format, got {spec!r}")
        dataset, raw_path = spec.split("=", 1)
        dataset = _normalize_dataset_name(dataset)
        raw_path = raw_path.strip()
        if dataset == PRODUCTION_DATASET:
            raise ValueError(
                f"{option} does not accept production_validated; use --prod-dir"
            )
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
            raise ValueError(
                "--source-dir is only valid when exactly one --dataset is selected"
            )
        mapping.setdefault(datasets[0], Path(args.source_dir))
    return mapping


def _dataset_source_zip_map(args: argparse.Namespace) -> dict[str, Path]:
    mapping = _parse_dataset_mapping(args.dataset_source_zip, "--dataset-source-zip")
    if args.source_zip:
        datasets = _datasets(args)
        if len(datasets) != 1:
            raise ValueError(
                "--source-zip is only valid when exactly one --dataset is selected"
            )
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
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )




def _resolved_pipeline_config(
    args: argparse.Namespace,
    paths: PipelinePaths,
    selected_stages: T.Sequence[str],
) -> dict[str, T.Any]:
    payload = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "selected_stages": list(selected_stages),
        "args": vars(args),
        "paths": {
            "run_root": paths.run_root,
            "dataset_root": paths.dataset_root,
            "hard_negative_dir": paths.hard_negative_dir,
            "hard_negative_manifest": paths.hard_negative_manifest,
            "validation_report": paths.validation_report,
            "checkpoint_dir": _checkpoint_dir(args, paths),
            "progress_log": paths.progress_log,
            "train_command_json": paths.train_command_json,
        },
        "training_signature": _pipeline_training_signature(args, paths),
        "training_signature_digest": _pipeline_training_signature_digest(args, paths),
    }
    return _json_safe_pipeline_value(payload)



def _checkpoint_dir(args: argparse.Namespace, paths: PipelinePaths) -> Path:
    return Path(args.ckpt_folder) if args.ckpt_folder else paths.checkpoint_dir


def _normalize_path_for_signature(value: Path | str | None) -> str:
    if value in (None, ""):
        return ""
    try:
        return str(Path(value).expanduser().resolve())
    except OSError:
        return str(Path(value).expanduser())


def _safe_sha256_file(path: Path) -> str | None:
    try:
        return _sha256_file(path) if path.is_file() else None
    except OSError:
        return None


def _pipeline_train_arg_tokens(args: argparse.Namespace) -> list[str]:
    tokens: list[str] = []
    for extra in args.train_arg or []:
        tokens.extend(shlex.split(extra))
    return tokens


def _pipeline_train_arg_values(args: argparse.Namespace, *names: str) -> list[str]:
    tokens = _pipeline_train_arg_tokens(args)
    values: list[str] = []
    for index, token in enumerate(tokens):
        for name in names:
            if token == name and index + 1 < len(tokens):
                next_value = tokens[index + 1]
                if not next_value.startswith("--"):
                    values.append(next_value)
                break
            prefix = name + "="
            if token.startswith(prefix):
                values.append(token[len(prefix):])
                break
    return values


def _pipeline_train_arg_option(
    args: argparse.Namespace,
    *names: str,
    default: T.Any = None,
) -> T.Any:
    values = _pipeline_train_arg_values(args, *names)
    return values[-1] if values else default


def _pipeline_train_bool_arg(
    args: argparse.Namespace,
    yes_name: str,
    no_name: str | None = None,
    *,
    default: bool = False,
) -> bool:
    tokens = _pipeline_train_arg_tokens(args)
    if no_name and no_name in tokens:
        return False
    if yes_name in tokens:
        return True
    return bool(default)


def _pipeline_effective_runtime_metrics_path(args: argparse.Namespace, paths: PipelinePaths) -> Path:
    return (
        Path(args.runtime_metrics_jsonl)
        if args.runtime_metrics_jsonl is not None
        else paths.run_root / "runtime_metrics.jsonl"
    )


def _pipeline_effective_manifest(args: argparse.Namespace, paths: PipelinePaths) -> str:
    return _normalize_path_for_signature(
        _pipeline_train_arg_option(
            args,
            "--manifest",
            default=paths.hard_negative_manifest,
        )
    )


def _pipeline_effective_training_manifest_for_compat(
    args: argparse.Namespace,
    paths: PipelinePaths,
) -> str:
    """Mirror TrainHeatmapStageFP16._training_manifest_path_for_compat.

    The trainer uses args.train_manifest or args.manifest or args.root_folder
    when computing checkpoint manifest compatibility. The pipeline always
    generates --manifest, but train_arg values are appended after generated
    args, so split-manifest overrides must win here too.
    """
    train_manifest = _pipeline_train_arg_option(
        args,
        "--train_manifest",
        "--train-manifest",
        default="",
    )
    if train_manifest:
        return _normalize_path_for_signature(train_manifest)

    manifest = _pipeline_train_arg_option(
        args,
        "--manifest",
        default=paths.hard_negative_manifest,
    )
    if manifest:
        return _normalize_path_for_signature(manifest)

    root_folder = _pipeline_train_arg_option(
        args,
        "--root_folder",
        "--root-folder",
        default="",
    )
    return _normalize_path_for_signature(root_folder)


def _pipeline_training_compat_config(args: argparse.Namespace, paths: PipelinePaths) -> dict[str, T.Any]:
    """Build the trainer checkpoint contract for a pipeline invocation.

    Contract keys and override semantics live in
    lib.landmarks.training.checkpoint_compat so the trainer and pipeline do not
    drift.
    """
    return build_pipeline_training_compat_config(
        args,
        paths,
        train_arg_option=_pipeline_train_arg_option,
        train_bool_arg=_pipeline_train_bool_arg,
        train_arg_values=_pipeline_train_arg_values,
        effective_training_manifest_for_compat=_pipeline_effective_training_manifest_for_compat,
        safe_sha256_file=_safe_sha256_file,
    )

def _pipeline_training_compat_digest(args: argparse.Namespace, paths: PipelinePaths) -> str:
    return training_compat_digest_from_config(_pipeline_training_compat_config(args, paths))


def _pipeline_training_signature(args: argparse.Namespace, paths: PipelinePaths) -> dict[str, T.Any]:
    ckpt_folder = _checkpoint_dir(args, paths)
    runtime_metrics_jsonl = _pipeline_effective_runtime_metrics_path(args, paths)
    return {
        "version": 2,
        "manifest": _pipeline_effective_manifest(args, paths),
        "manifest_sha256": _safe_sha256_file(Path(_pipeline_effective_manifest(args, paths))),
        "ckpt_folder": _normalize_path_for_signature(ckpt_folder),
        "train_data_name": str(args.train_data_name),
        "nproc_per_node": int(args.nproc_per_node),
        "batch_size": int(args.batch_size),
        "heatmap_size": int(args.heatmap_size),
        "lmk_num": int(args.lmk_num),
        "lr": float(args.lr),
        "train_arg": list(args.train_arg or []),
        "runtime": {
            "num_workers": int(args.num_workers),
            "preload": int(args.preload),
            "pin_memory": bool(args.pin_memory),
            "persistent_workers": bool(args.persistent_workers),
            "prefetch_factor": int(args.prefetch_factor),
            "log_every": int(args.log_every),
            "synchronize_runtime_timing": bool(args.synchronize_runtime_timing),
        },
        "eval": {
            "eval_batch_size": int(args.eval_batch_size),
            "eval_num_workers": int(args.eval_num_workers),
            "eval_every": int(args.eval_every),
            "full_eval_every": int(args.full_eval_every),
            "eval_ema_every": int(args.eval_ema_every),
            "eval_max_samples": int(args.eval_max_samples),
        },
        "checkpoint": {
            "save_last_checkpoint": bool(args.save_last_checkpoint),
            "save_legacy_epoch_state_dict": bool(args.save_legacy_epoch_state_dict),
            "restore_rng": bool(args.restore_rng),
            "allow_incompatible_resume": bool(args.allow_incompatible_resume),
            "auto_resume": bool(args.auto_resume),
            "runtime_metrics_jsonl": _normalize_path_for_signature(runtime_metrics_jsonl),
        },
        "training_compat_config": _pipeline_training_compat_config(args, paths),
        "training_compat_config_digest": _pipeline_training_compat_digest(args, paths),
    }


def _pipeline_training_signature_digest(args: argparse.Namespace, paths: PipelinePaths) -> str:
    payload = json.dumps(
        _pipeline_training_signature(args, paths),
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _write_pipeline_training_signature(
    args: argparse.Namespace,
    paths: PipelinePaths,
    command: list[str],
    ckpt_folder: Path,
) -> None:
    sentinel = ckpt_folder / "training_complete.json"
    payload: dict[str, T.Any] = {}
    if sentinel.is_file():
        try:
            payload = json.loads(sentinel.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            payload = {}
    payload["pipeline_training_signature"] = _pipeline_training_signature(args, paths)
    payload["pipeline_training_signature_digest"] = _pipeline_training_signature_digest(args, paths)
    payload["pipeline_requested_epoch"] = int(args.epoch)
    payload["pipeline_train_command"] = command
    payload["pipeline_manifest_sha256"] = _safe_sha256_file(Path(_pipeline_effective_manifest(args, paths)))
    payload["pipeline_training_compat_config"] = _pipeline_training_compat_config(args, paths)
    payload["pipeline_training_compat_config_digest"] = _pipeline_training_compat_digest(args, paths)
    sentinel.parent.mkdir(parents=True, exist_ok=True)
    sentinel.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _checkpoint_metadata_path(path: Path) -> Path:
    return Path(str(path) + ".meta.json")


def _load_checkpoint_metadata(path: Path) -> dict[str, T.Any] | None:
    meta_path = _checkpoint_metadata_path(path)
    if meta_path.is_file():
        try:
            payload = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as err:
            return {"_load_error": f"could not read checkpoint metadata sidecar: {err}"}
        return payload if isinstance(payload, dict) else None

    try:
        import torch
        try:
            payload = torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            payload = torch.load(path, map_location="cpu")
    except Exception as err:
        return {"_load_error": str(err)}
    return payload if isinstance(payload, dict) else None


def _normalize_runtime_args(args: argparse.Namespace) -> argparse.Namespace:
    if args.restore_rng and args.persistent_workers:
        print(
            "warning: --restore-rng requires epoch-reseeded training workers; "
            "forcing --no-persistent-workers for checkpoint-compatible replay",
            flush=True,
        )
        args.persistent_workers = False
    return args


def _checkpoint_matches_pipeline_request(
    args: argparse.Namespace,
    paths: PipelinePaths,
    checkpoint_path: Path,
) -> tuple[bool, str]:
    payload = _load_checkpoint_metadata(checkpoint_path)
    if not isinstance(payload, dict):
        return False, "checkpoint is not a full PR3 training checkpoint"
    if payload.get("_load_error"):
        return False, f"could not load checkpoint metadata: {payload['_load_error']}"
    if payload.get("format") != "cdvit-training-checkpoint-v1":
        return False, "checkpoint format is not cdvit-training-checkpoint-v1"

    try:
        next_epoch = int(payload.get("next_epoch", int(payload.get("epoch", -1)) + 1))
    except (TypeError, ValueError):
        next_epoch = -1

    ckpt_folder = _checkpoint_dir(args, paths)
    sentinel = ckpt_folder / "training_complete.json"
    current_signature = _pipeline_training_signature_digest(args, paths)
    sentinel_signature = None
    if sentinel.is_file():
        try:
            sentinel_payload = json.loads(sentinel.read_text(encoding="utf-8"))
            sentinel_signature = sentinel_payload.get("pipeline_training_signature_digest")
        except (OSError, json.JSONDecodeError):
            sentinel_signature = None

    if next_epoch >= int(args.epoch) and sentinel_signature != current_signature:
        return (
            False,
            "last_checkpoint.pt has already reached the requested epoch but the "
            "pipeline runtime/eval signature changed; increase --epoch, use --force "
            "with a fresh checkpoint folder, or choose a checkpoint before the final epoch"
        )

    current_manifest_sha = _safe_sha256_file(Path(_pipeline_effective_training_manifest_for_compat(args, paths)))
    checkpoint_manifest_sha = payload.get("manifest_sha256")
    if current_manifest_sha and checkpoint_manifest_sha and current_manifest_sha != checkpoint_manifest_sha:
        return False, "checkpoint manifest SHA differs from the current manifest"

    expected_config = _pipeline_training_compat_config(args, paths)
    compat_errors = checkpoint_compat_errors_for_config(
        payload,
        expected_config,
        current_manifest_sha=current_manifest_sha,
        fallback_expected_args={
            "data_name": str(args.train_data_name),
            "batch_size": int(args.batch_size),
            "heatmap_size": int(args.heatmap_size),
            "lmk_num": int(args.lmk_num),
            "lr": float(args.lr),
            "schema_aware_training": True,
            "domain_balanced_sampling": _pipeline_train_bool_arg(
                args,
                "--domain-balanced-sampling",
                default=False,
            ),
            "auxiliary_heads": _pipeline_train_bool_arg(
                args,
                "--auxiliary-heads",
                "--no-auxiliary-heads",
                default=True,
            ),
        },
    )
    if compat_errors:
        return False, "; ".join(compat_errors)
    return True, "compatible"


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


def _production_manifest_path(paths: PipelinePaths) -> Path:
    return _dataset_manifest_path(paths, PRODUCTION_DATASET)


def _has_prod_dir(args: argparse.Namespace) -> bool:
    return bool(getattr(args, "prod_dir", None))


def _production_build_command(
    args: argparse.Namespace, paths: PipelinePaths
) -> list[str] | None:
    if not _has_prod_dir(args):
        return None
    return _append_extra(
        [
            args.python_executable,
            _script("build_production_validated_manifest.py"),
            "--prod-dir",
            str(args.prod_dir),
            "--output-dir",
            str(paths.dataset_root / PRODUCTION_DATASET),
        ],
        args.production_build_arg or [],
    )


def _dataset_build_commands(
    args: argparse.Namespace, paths: PipelinePaths
) -> list[list[str]]:
    source_map = _dataset_source_map(args)
    source_zip_map = _dataset_source_zip_map(args)
    overlap = sorted(set(source_map) & set(source_zip_map))
    if overlap:
        raise ValueError(
            "datasets cannot have both source dir and source zip: " + ", ".join(overlap)
        )

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

    production_command = _production_build_command(args, paths)
    if production_command is not None:
        commands.append(production_command)
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

    production_manifest = _production_manifest_path(paths)
    if _has_prod_dir(args) and (production_manifest.is_file() or args.dry_run):
        argv.extend(
            [HARD_NEGATIVE_MANIFEST_FLAGS[PRODUCTION_DATASET], str(production_manifest)]
        )

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
    if args.exclude_image_ids_file is not None:
        argv.extend(["--exclude-image-ids-file", str(args.exclude_image_ids_file)])
    return _append_extra(argv, args.hard_negative_arg or [])


def _train_command(args: argparse.Namespace, paths: PipelinePaths) -> list[str]:
    ckpt_folder = Path(args.ckpt_folder) if args.ckpt_folder else paths.checkpoint_dir
    argv = [
        args.torchrun_executable,
        f"--nproc_per_node={args.nproc_per_node}",
        str(CDVIT_ROOT / "TrainHeatmapStageFP16.py"),
        "--data_name",
        args.train_data_name,
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
    argv.extend(["--preload", str(args.preload)])
    argv.append("--pin-memory" if args.pin_memory else "--no-pin-memory")
    argv.append("--persistent-workers" if args.persistent_workers else "--no-persistent-workers")
    argv.extend(["--prefetch-factor", str(args.prefetch_factor)])
    argv.extend(["--eval-batch-size", str(args.eval_batch_size)])
    argv.extend(["--eval-num-workers", str(args.eval_num_workers)])
    argv.extend(["--eval-every", str(args.eval_every)])
    argv.extend(["--full-eval-every", str(args.full_eval_every)])
    argv.extend(["--eval-ema-every", str(args.eval_ema_every)])
    argv.extend(["--eval-max-samples", str(args.eval_max_samples)])
    argv.extend(["--log-every", str(args.log_every)])
    if args.save_last_checkpoint:
        argv.append("--save-last-checkpoint")
    else:
        argv.append("--no-save-last-checkpoint")
    if args.save_legacy_epoch_state_dict:
        argv.append("--save-legacy-epoch-state-dict")
    resume_path = Path(args.resume) if args.resume is not None else None
    if resume_path is None and args.auto_resume and not args.force:
        candidate = ckpt_folder / "last_checkpoint.pt"
        if candidate.is_file():
            compatible, reason = _checkpoint_matches_pipeline_request(args, paths, candidate)
            if compatible:
                resume_path = candidate
            else:
                raise ValueError(
                    f"refusing to auto-resume from {candidate}: {reason}. "
                    "Use --no-auto-resume to start a fresh run in this checkpoint folder, "
                    "or pass --allow-incompatible-resume with an explicit --resume if this is intentional."
                )
    if resume_path is not None:
        argv.extend(["--resume", str(resume_path)])
    if args.restore_rng:
        argv.append("--restore-rng")
    if args.allow_incompatible_resume:
        argv.append("--allow-incompatible-resume")
    runtime_metrics_jsonl = (
        Path(args.runtime_metrics_jsonl)
        if args.runtime_metrics_jsonl is not None
        else paths.run_root / "runtime_metrics.jsonl"
    )
    argv.extend(["--runtime-metrics-jsonl", str(runtime_metrics_jsonl)])
    if args.synchronize_runtime_timing:
        argv.append("--synchronize-runtime-timing")
    else:
        argv.append("--no-synchronize-runtime-timing")
    return _append_extra(argv, args.train_arg or [])


def _stage_slice(start_at: str | None, stop_after: str | None) -> tuple[str, ...]:
    if start_at and start_at not in STAGES:
        raise ValueError(
            f"unknown --start-at stage {start_at!r}; choose one of {', '.join(STAGES)}"
        )
    if stop_after and stop_after not in STAGES:
        raise ValueError(
            f"unknown --stop-after stage {stop_after!r}; choose one of {', '.join(STAGES)}"
        )
    start = STAGES.index(start_at) if start_at else 0
    stop = STAGES.index(stop_after) if stop_after else len(STAGES) - 1
    if start > stop:
        raise ValueError(
            f"--start-at {start_at!r} occurs after --stop-after {stop_after!r}"
        )
    return STAGES[start : stop + 1]


def _require_local_tools(args: argparse.Namespace) -> None:
    if args.manifest:
        return
    required = [
        TOOLS_ROOT / "build_quality_dataset.py",
        TOOLS_ROOT / "build_hard_negative_manifest.py",
    ]
    if _has_prod_dir(args):
        required.append(TOOLS_ROOT / "build_production_validated_manifest.py")
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing local manifest tool(s): " + ", ".join(missing))


def _run_command(argv: list[str], *, cwd: Path, dry_run: bool) -> None:
    print("+ " + " ".join(shlex.quote(part) for part in argv), flush=True)
    if dry_run:
        return
    subprocess.run(argv, cwd=str(cwd), check=True)


def _validate_training_manifest(
    args: argparse.Namespace, paths: PipelinePaths
) -> dict[str, T.Any]:
    manifest_path = paths.hard_negative_manifest
    if not manifest_path.is_file():
        raise FileNotFoundError(f"missing manifest: {manifest_path}")
    return validate_training_manifest(
        manifest_path,
        report_path=paths.validation_report,
        require_images=not args.skip_image_exists_check,
        allow_legacy_68_projection=args.allow_legacy_68_projection,
        allow_missing_projection_audit=args.allow_missing_projection_audit,
        allow_legacy_missing_contract_fields=args.allow_legacy_missing_contract_fields,
        max_examples=args.max_validation_examples,
        raise_on_error=True,
    )


def _validate_cdvit_manifest(
    args: argparse.Namespace, paths: PipelinePaths
) -> dict[str, T.Any]:
    """Compatibility wrapper for the old stage name."""
    return _validate_training_manifest(args, paths)


def _build_manifest_outputs(
    args: argparse.Namespace, paths: PipelinePaths
) -> list[str]:
    outputs = [
        str(_dataset_manifest_path(paths, dataset)) for dataset in _datasets(args)
    ]
    if _has_prod_dir(args):
        outputs.append(str(_production_manifest_path(paths)))
    return outputs


def _stage_complete(stage: str, args: argparse.Namespace, paths: PipelinePaths) -> bool:
    if args.manifest and stage in {
        "build_dataset_manifests",
        "build_hard_negative_manifest",
    }:
        return True
    if stage == "build_dataset_manifests":
        return all(
            Path(path).is_file() for path in _build_manifest_outputs(args, paths)
        )
    if stage == "build_hard_negative_manifest":
        return paths.hard_negative_manifest.is_file()
    if stage == "validate_cdvit_manifest":
        return paths.validation_report.is_file()
    if stage == "train_cdvit":
        ckpt_folder = (
            Path(args.ckpt_folder) if args.ckpt_folder else paths.checkpoint_dir
        )
        if args.resume is not None:
            return False
        sentinel = ckpt_folder / "training_complete.json"
        if not sentinel.is_file():
            return False
        try:
            payload = json.loads(sentinel.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return False
        if payload.get("status") != "complete":
            return False
        try:
            completed_epochs = int(payload.get("requested_epochs", -1))
        except (TypeError, ValueError):
            return False
        if payload.get("pipeline_training_signature_digest") != _pipeline_training_signature_digest(args, paths):
            return False
        if payload.get("pipeline_manifest_sha256") != _safe_sha256_file(Path(_pipeline_effective_manifest(args, paths))):
            return False
        if int(args.eval_every or 0) > 0 and not (ckpt_folder / "best_model").exists():
            return False
        return completed_epochs >= int(args.epoch)
    raise ValueError(f"unknown stage: {stage}")


def _run_stage(
    stage: str, args: argparse.Namespace, paths: PipelinePaths
) -> StageResult:
    started = time.time()
    command: list[str] = []
    outputs: list[str] = []
    notes: list[str] = []
    try:
        if not args.force and _stage_complete(stage, args, paths):
            return StageResult(
                stage, "skipped", time.time() - started, notes=["already complete"]
            )

        if stage == "build_dataset_manifests":
            for command in _dataset_build_commands(args, paths):
                _run_command(command, cwd=CDVIT_ROOT, dry_run=args.dry_run)
            outputs = _build_manifest_outputs(args, paths)

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
                    f"validated {report['valid_samples']} trainable sample(s) "
                    f"from {report['total_samples']} manifest entries "
                    f"across {len(report.get('schemas', {}))} schema(s)"
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
            ckpt_folder = _checkpoint_dir(args, paths)
            if not args.dry_run:
                _write_pipeline_training_signature(args, paths, command, ckpt_folder)
            outputs = [str(ckpt_folder), str(paths.train_command_json)]

        else:
            raise ValueError(f"unknown stage: {stage}")
        status = "planned" if args.dry_run else "ok"
        return StageResult(
            stage,
            status,
            time.time() - started,
            command=command,
            outputs=outputs,
            notes=notes,
        )
    except Exception as err:  # noqa: BLE001
        return StageResult(
            stage,
            "error",
            time.time() - started,
            command=command,
            outputs=outputs,
            notes=notes,
            error=str(err),
        )


def _default_run_name() -> str:
    return time.strftime("cdvit_fs68_%Y%m%d_%H%M%S", time.localtime())


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="JSON/YAML config file. Values are merged before CLI args, so CLI args override scalar config values.",
    )
    parser.add_argument("--output-root", type=Path, default=Path("runs/landmarks"))
    parser.add_argument("--run-name", default=_default_run_name())
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Use an existing hard-negative manifest and skip manifest build stages.",
    )
    parser.add_argument(
        "--dataset",
        default=DEFAULT_DATASETS,
        help="Comma-separated non-production dataset list.",
    )
    parser.add_argument(
        "--dataset-source",
        action="append",
        default=[],
        help="dataset=source_dir, repeatable.",
    )
    parser.add_argument(
        "--dataset-source-zip",
        action="append",
        default=[],
        help="dataset=source_zip, repeatable.",
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=None,
        help="Source dir for a single selected dataset.",
    )
    parser.add_argument(
        "--source-zip",
        type=Path,
        default=None,
        help="Source zip for a single selected dataset.",
    )
    parser.add_argument(
        "--prod-dir",
        "--production-dir",
        dest="prod_dir",
        type=Path,
        default=None,
        help="Directory containing production images and exactly one Faceswap .fsa file.",
    )
    parser.add_argument(
        "--dataset-build-arg",
        action="append",
        default=[],
        help="Extra quoted arg(s) passed to build_quality_dataset.py; repeatable.",
    )
    parser.add_argument(
        "--production-build-arg",
        action="append",
        default=[],
        help="Extra quoted arg(s) passed to build_production_validated_manifest.py; repeatable.",
    )
    parser.add_argument(
        "--hard-negative-arg",
        action="append",
        default=[],
        help="Extra quoted arg(s) passed to build_hard_negative_manifest.py; repeatable.",
    )
    parser.add_argument(
        "--exclude-image-ids-file",
        type=Path,
        default=None,
        help="Drop MERL-RAV samples whose imageNNNNN id appears in this file during hard-negative manifest build.",
    )
    parser.add_argument(
        "--include-39pt-profile",
        action="store_true",
        help="Accepted for compatibility; local builder emits canonical 68 only.",
    )
    parser.add_argument(
        "--allow-non68",
        action="store_true",
        help="Deprecated compatibility flag; mixed-schema manifests are validated by default.",
    )
    parser.add_argument(
        "--allow-legacy-68-projection",
        action="store_true",
        help="Accept old manifests where non-68 source schemas were already projected into 68-point target files.",
    )
    parser.add_argument(
        "--allow-missing-projection-audit",
        action="store_true",
        help="Do not fail samples whose source_schema differs from target_schema but lack mapping/projection audit metadata.",
    )
    parser.add_argument(
        "--allow-legacy-missing-contract-fields",
        action="store_true",
        help="Do not fail legacy manifests missing inferable contract fields: landmark_count, head_name, split_safe_id.",
    )
    parser.add_argument(
        "--skip-image-exists-check",
        action="store_true",
        help="Validate landmark contract without requiring image files to exist on this machine.",
    )
    parser.add_argument("--max-validation-examples", type=int, default=25)
    parser.add_argument("--max-profile-occlusion", type=int, default=None)
    parser.add_argument("--max-profile", type=int, default=None)
    parser.add_argument("--max-occlusion", type=int, default=None)
    parser.add_argument("--max-anchors", type=int, default=None)
    parser.add_argument("--python-executable", default=sys.executable)
    parser.add_argument("--torchrun-executable", default="torchrun")
    parser.add_argument(
        "--train-data-name",
        default=LEGACY_MANIFEST_DATA_NAME,
        choices=MANIFEST_DATA_NAME_ALIASES,
        help="Manifest data_name. FS68Manifest remains the compatibility default; pass MultiSchemaLandmarkManifest to use the canonical alias.",
    )
    parser.add_argument("--nproc-per-node", type=int, default=2)
    parser.add_argument("--ckpt-folder", type=Path, default=None)
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Resume TrainHeatmapStageFP16.py from a model state dict or full training checkpoint.",
    )
    parser.add_argument(
        "--restore-rng",
        action="store_true",
        help="When resuming a full checkpoint, restore RNG state. For exact replay this forces --no-persistent-workers so workers are re-seeded per epoch.",
    )
    parser.add_argument(
        "--allow-incompatible-resume",
        action="store_true",
        help="Forward --allow-incompatible-resume to TrainHeatmapStageFP16.py for intentional checkpoint migration.",
    )
    parser.add_argument(
        "--auto-resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="When train_cdvit is incomplete and no explicit --resume is supplied, resume from <ckpt-folder>/last_checkpoint.pt if present.",
    )
    parser.add_argument(
        "--runtime-metrics-jsonl",
        type=Path,
        default=None,
        help="Runtime metrics JSONL path. Defaults to <run-root>/runtime_metrics.jsonl.",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=12)
    parser.add_argument("--preload", type=int, default=0)
    parser.add_argument(
        "--pin-memory",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Forward pinned-memory DataLoader mode to TrainHeatmapStageFP16.py.",
    )
    parser.add_argument(
        "--persistent-workers",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Forward persistent DataLoader workers for throughput. Worker RNG is seeded once; use --no-persistent-workers for epoch-reseeded workers or --restore-rng replay.",
    )
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--eval-num-workers", type=int, default=0)
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument("--full-eval-every", type=int, default=0)
    parser.add_argument("--eval-ema-every", "--eval-on-ema-every", dest="eval_ema_every", type=int, default=1)
    parser.add_argument("--eval-max-samples", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument(
        "--synchronize-runtime-timing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Forward CUDA event timing around transfer/compute/eval sections. "
            "Pass --no-synchronize-runtime-timing for low-overhead CPU wall-clock timing."
        ),
    )
    parser.add_argument(
        "--save-last-checkpoint",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--save-legacy-epoch-state-dict",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Forward legacy epoch_N state-dict saving to TrainHeatmapStageFP16.py.",
    )
    parser.add_argument("--epoch", type=int, default=500)
    parser.add_argument("--heatmap-size", type=int, default=32)
    parser.add_argument("--lmk-num", type=int, default=68)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument(
        "--train-arg",
        action="append",
        default=[],
        help="Extra quoted arg(s) passed to TrainHeatmapStageFP16.py; repeatable.",
    )
    parser.add_argument("--start-at", choices=STAGES, default=None)
    parser.add_argument("--stop-after", choices=STAGES, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_arg_parser()
    cli_argv = list(sys.argv[1:] if argv is None else argv)
    config_path = _extract_config_path(cli_argv)
    merged_argv = _merge_config_argv(parser, config_path, cli_argv)
    args = _normalize_runtime_args(parser.parse_args(merged_argv))
    paths = PipelinePaths(
        output_root=Path(args.output_root),
        run_name=args.run_name,
        explicit_manifest=Path(args.manifest).resolve() if args.manifest else None,
    )
    paths.run_root.mkdir(parents=True, exist_ok=True)
    _require_local_tools(args)

    selected_stages = _stage_slice(args.start_at, args.stop_after)
    _write_json(
        paths.run_root / "run_config.resolved.json",
        _resolved_pipeline_config(args, paths, selected_stages),
    )
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
