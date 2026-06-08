"""Typed config snapshots for CD-ViT training and pipeline signatures."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import typing as T


@dataclass(frozen=True)
class EvalConfig:
    eval_batch_size: int
    eval_num_workers: int
    eval_every: int
    full_eval_every: int
    eval_ema_every: int
    eval_ema_scope: str
    eval_progress: bool
    eval_max_samples: int
    eval_slice_reports_every: int

    @classmethod
    def from_args(cls, args: T.Any) -> "EvalConfig":
        return cls(
            eval_batch_size=int(args.eval_batch_size),
            eval_num_workers=int(args.eval_num_workers),
            eval_every=int(args.eval_every),
            full_eval_every=int(args.full_eval_every),
            eval_ema_every=int(args.eval_ema_every),
            eval_ema_scope=str(getattr(args, "eval_ema_scope", "same")),
            eval_progress=bool(getattr(args, "eval_progress", True)),
            eval_max_samples=int(args.eval_max_samples),
            eval_slice_reports_every=int(args.eval_slice_reports_every),
        )


@dataclass(frozen=True)
class TrainingRuntimeConfig:
    num_workers: int
    preload: int
    pin_memory: bool
    persistent_workers: bool
    prefetch_factor: int
    log_every: int
    synchronize_runtime_timing: bool

    @classmethod
    def from_args(cls, args: T.Any) -> "TrainingRuntimeConfig":
        return cls(
            num_workers=int(args.num_workers),
            preload=int(args.preload),
            pin_memory=bool(args.pin_memory),
            persistent_workers=bool(args.persistent_workers),
            prefetch_factor=int(args.prefetch_factor),
            log_every=int(args.log_every),
            synchronize_runtime_timing=bool(args.synchronize_runtime_timing),
        )


@dataclass(frozen=True)
class CheckpointConfig:
    save_last_checkpoint: bool
    save_legacy_epoch_state_dict: bool
    restore_rng: bool
    allow_incompatible_resume: bool
    auto_resume: bool
    runtime_metrics_jsonl: str

    @classmethod
    def from_args(cls, args: T.Any, *, runtime_metrics_jsonl: str = "") -> "CheckpointConfig":
        return cls(
            save_last_checkpoint=bool(args.save_last_checkpoint),
            save_legacy_epoch_state_dict=bool(args.save_legacy_epoch_state_dict),
            restore_rng=bool(args.restore_rng),
            allow_incompatible_resume=bool(args.allow_incompatible_resume),
            auto_resume=bool(getattr(args, "auto_resume", False)),
            runtime_metrics_jsonl=str(runtime_metrics_jsonl),
        )


@dataclass(frozen=True)
class DatasetBuildConfig:
    data_name: str
    manifest: str
    train_manifest: str
    test_manifest: str
    split_policy: str
    eval_mode: str
    heldout_dataset: tuple[str, ...]
    schema_aware_training: bool
    domain_balanced_sampling: bool
    bucket_targets: str
    dataset_targets: str
    schema_targets: str
    auto_dataset_balancing: bool
    auto_schema_balancing: bool

    @classmethod
    def from_args(cls, args: T.Any) -> "DatasetBuildConfig":
        return cls(
            data_name=str(args.data_name),
            manifest=str(getattr(args, "manifest", "")),
            train_manifest=str(getattr(args, "train_manifest", "")),
            test_manifest=str(getattr(args, "test_manifest", "")),
            split_policy=str(args.split_policy),
            eval_mode=str(args.eval_mode),
            heldout_dataset=tuple(str(value) for value in getattr(args, "heldout_dataset", ())),
            schema_aware_training=bool(args.schema_aware_training),
            domain_balanced_sampling=bool(args.domain_balanced_sampling),
            bucket_targets=str(args.bucket_targets),
            dataset_targets=str(args.dataset_targets),
            schema_targets=str(args.schema_targets),
            auto_dataset_balancing=bool(getattr(args, "auto_dataset_balancing", True)),
            auto_schema_balancing=bool(getattr(args, "auto_schema_balancing", True)),
        )


@dataclass(frozen=True)
class PipelineConfig:
    train_data_name: str
    nproc_per_node: int
    batch_size: int
    heatmap_size: int
    lmk_num: int
    lr: float
    train_arg: tuple[str, ...]
    runtime: TrainingRuntimeConfig
    eval: EvalConfig
    checkpoint: CheckpointConfig

    @classmethod
    def from_args(
        cls,
        args: T.Any,
        *,
        runtime_metrics_jsonl: str = "",
    ) -> "PipelineConfig":
        return cls(
            train_data_name=str(args.train_data_name),
            nproc_per_node=int(args.nproc_per_node),
            batch_size=int(args.batch_size),
            heatmap_size=int(args.heatmap_size),
            lmk_num=int(args.lmk_num),
            lr=float(args.lr),
            train_arg=tuple(str(value) for value in (args.train_arg or ())),
            runtime=TrainingRuntimeConfig.from_args(args),
            eval=EvalConfig.from_args(args),
            checkpoint=CheckpointConfig.from_args(args, runtime_metrics_jsonl=runtime_metrics_jsonl),
        )


def config_dict(config: T.Any) -> dict[str, T.Any]:
    """Return a JSON-friendly dict for dataclass config snapshots."""

    return asdict(config)


__all__ = [
    "CheckpointConfig",
    "DatasetBuildConfig",
    "EvalConfig",
    "PipelineConfig",
    "TrainingRuntimeConfig",
    "config_dict",
]
