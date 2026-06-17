"""DataLoader construction for CD-ViT heatmap-stage training."""

from __future__ import annotations

import typing as T
from dataclasses import dataclass

import torch

from lib.core.manifest_aliases import is_schema_aware_manifest_dataset
from lib.evaluation.split_safe import validate_no_train_test_leakage
from lib.logging_utils import Verbosity, log_event
from lib.training.data import (
    build_dataset,
    legacy_domain_balanced_collate,
    schema_aware_collate,
)
from lib.training.domain_balanced_sampler import (
    DEFAULT_BUCKET_TARGETS,
    DomainBalancedBatchSampler,
    parse_target_spec,
    parse_target_spec_for_kind,
)
from lib.training.evaluator import eval_collate
from lib.logging_utils import summarize_mapping
from lib.training.runtime import dataloader_kwargs, maybe_limit_eval_dataset


@dataclass(frozen=True)
class TrainerDataLoaders:
    train_dataset: T.Any
    test_dataset: T.Any
    eval_dataset: T.Any
    train_sampler: T.Any
    train_dataloader: torch.utils.data.DataLoader
    test_dataloader: torch.utils.data.DataLoader
    full_test_dataloader: torch.utils.data.DataLoader


class DistributedEvalSampler(torch.utils.data.Sampler[int]):
    """Shard eval datasets by stable index without padding or duplication."""

    def __init__(self, dataset: T.Sized, *, rank: int, world_size: int) -> None:
        self.dataset = dataset
        self.rank = int(rank)
        self.world_size = max(int(world_size), 1)

    def __iter__(self):
        return iter(range(self.rank, len(self.dataset), self.world_size))

    def __len__(self) -> int:
        if self.rank >= len(self.dataset):
            return 0
        return ((len(self.dataset) - 1 - self.rank) // self.world_size) + 1


def _sampler_targets_summary(targets: T.Mapping[str, T.Any]) -> str:
    bucket = targets.get("bucket", {}) if isinstance(targets, dict) else {}
    dataset = targets.get("dataset", {}) if isinstance(targets, dict) else {}
    schema = targets.get("schema", {}) if isinstance(targets, dict) else {}
    return (
        "domain-balanced | "
        f"buckets {len(bucket)} | datasets {len(dataset)} | schemas {len(schema)}"
    )


def _sampler_targets_summary(targets: dict[str, T.Any]) -> str:
    """Compact one-line summary of resolved sampler targets.

    Full resolved target dictionaries are useful for debugging, but too wide for
    normal console startup logs.
    """

    bucket = targets.get("bucket", {}) if isinstance(targets, dict) else {}
    dataset = targets.get("dataset", {}) if isinstance(targets, dict) else {}
    schema = targets.get("schema", {}) if isinstance(targets, dict) else {}
    parts = [
        f"bucket {summarize_mapping(bucket, top_n=4, as_percent=True)}",
        f"dataset {summarize_mapping(dataset, top_n=4)}",
        f"schema {summarize_mapping(schema, top_n=4)}",
    ]
    return " | ".join(parts)


def build_training_loaders(
    args: T.Any,
    *,
    schema_aware_training: bool,
    rank: int,
    world_size: int,
) -> TrainerDataLoaders:
    """Build train/eval datasets, samplers, and dataloaders.

    Keeping this in a dedicated module keeps heatmap_stage.main focused on the
    epoch state machine. It also makes loader behavior testable without bringing
    up distributed training or constructing the model.
    """

    schema_manifest = is_schema_aware_manifest_dataset(args.data_name)
    train_dataset = build_dataset(
        args,
        "train",
        aug=True,
        heatmap_size=args.heatmap_size,
        include_metadata=bool(args.domain_balanced_sampling and schema_manifest),
        schema_aware_training=schema_aware_training,
    )
    test_dataset = build_dataset(
        args,
        "test",
        aug=False,
        heatmap_size=0,
        include_metadata=True,
        schema_aware_training=schema_aware_training,
    )
    if schema_manifest:
        validate_no_train_test_leakage(train_dataset.samples, test_dataset.samples)

    eval_dataset = maybe_limit_eval_dataset(
        test_dataset,
        args.eval_max_samples,
        args.seed,
    )
    eval_sampler = (
        DistributedEvalSampler(eval_dataset, rank=rank, world_size=world_size)
        if world_size > 1
        else None
    )
    test_dataloader = torch.utils.data.DataLoader(
        eval_dataset,
        batch_size=args.eval_batch_size,
        sampler=eval_sampler,
        collate_fn=eval_collate,
        **dataloader_kwargs(args, eval_loader=True),
    )
    full_test_dataloader = test_dataloader
    if int(args.eval_max_samples or 0) > 0 and len(eval_dataset) < len(test_dataset):
        full_eval_sampler = (
            DistributedEvalSampler(test_dataset, rank=rank, world_size=world_size)
            if world_size > 1
            else None
        )
        full_test_dataloader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=args.eval_batch_size,
            sampler=full_eval_sampler,
            collate_fn=eval_collate,
            **dataloader_kwargs(args, eval_loader=True),
        )

    if args.domain_balanced_sampling and schema_manifest:
        train_sampler = DomainBalancedBatchSampler(
            train_dataset.samples,
            bucket_targets=parse_target_spec(
                args.bucket_targets, DEFAULT_BUCKET_TARGETS
            ),
            dataset_targets=parse_target_spec_for_kind(
                args.dataset_targets, kind="dataset"
            ),
            schema_targets=parse_target_spec_for_kind(
                args.schema_targets, kind="schema"
            ),
            batch_size=args.batch_size,
            seed=args.seed,
            rank=rank,
            world_size=world_size,
            auto_balance_datasets=bool(getattr(args, "auto_dataset_balancing", True)),
            auto_balance_schemas=bool(
                getattr(args, "auto_schema_balancing", True) and schema_aware_training
            ),
        )
        if rank == 0:
            resolved_targets = train_sampler.resolved_targets()
            log_event(
                "sampler",
                f"targets | {_sampler_targets_summary(resolved_targets)}",
                level=Verbosity.INFO,
                resolved_targets=resolved_targets,
            )
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_sampler=train_sampler,
            collate_fn=schema_aware_collate
            if schema_aware_training
            else legacy_domain_balanced_collate,
            **dataloader_kwargs(args),
        )
    else:
        # Pass num_replicas/rank explicitly so the sampler is constructed without
        # an initialized process group (single-process MPS/CPU or single-GPU runs).
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
        )
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            sampler=train_sampler,
            collate_fn=schema_aware_collate if schema_aware_training else None,
            **dataloader_kwargs(args),
        )

    return TrainerDataLoaders(
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        eval_dataset=eval_dataset,
        train_sampler=train_sampler,
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
        full_test_dataloader=full_test_dataloader,
    )


__all__ = ["DistributedEvalSampler", "TrainerDataLoaders", "build_training_loaders"]
