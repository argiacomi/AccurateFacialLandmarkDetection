"""DataLoader construction for CD-ViT heatmap-stage training."""

from __future__ import annotations

from dataclasses import dataclass
import typing as T

import torch

from lib.landmarks.core.manifest_aliases import is_schema_aware_manifest_dataset
from lib.landmarks.evaluation.split_safe import validate_no_train_test_leakage
from lib.landmarks.training.data import build_dataset, schema_aware_collate
from lib.landmarks.training.domain_balanced_sampler import (
    DEFAULT_BUCKET_TARGETS,
    DomainBalancedBatchSampler,
    parse_target_spec,
)
from lib.landmarks.training.evaluator import eval_collate
from lib.landmarks.training.runtime import dataloader_kwargs, maybe_limit_eval_dataset


@dataclass(frozen=True)
class TrainerDataLoaders:
    train_dataset: T.Any
    test_dataset: T.Any
    eval_dataset: T.Any
    train_sampler: T.Any
    train_dataloader: torch.utils.data.DataLoader
    test_dataloader: torch.utils.data.DataLoader
    full_test_dataloader: torch.utils.data.DataLoader


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

    train_dataset = build_dataset(
        args,
        "train",
        aug=True,
        heatmap_size=args.heatmap_size,
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
    if is_schema_aware_manifest_dataset(args.data_name):
        validate_no_train_test_leakage(train_dataset.samples, test_dataset.samples)

    eval_dataset = maybe_limit_eval_dataset(
        test_dataset,
        args.eval_max_samples,
        args.seed,
    )
    test_dataloader = torch.utils.data.DataLoader(
        eval_dataset,
        batch_size=args.eval_batch_size,
        collate_fn=eval_collate,
        **dataloader_kwargs(args, eval_loader=True),
    )
    full_test_dataloader = test_dataloader
    if int(args.eval_max_samples or 0) > 0 and len(eval_dataset) < len(test_dataset):
        full_test_dataloader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=args.eval_batch_size,
            collate_fn=eval_collate,
            **dataloader_kwargs(args, eval_loader=True),
        )

    if args.domain_balanced_sampling and is_schema_aware_manifest_dataset(args.data_name):
        train_sampler = DomainBalancedBatchSampler(
            train_dataset.samples,
            bucket_targets=parse_target_spec(args.bucket_targets, DEFAULT_BUCKET_TARGETS),
            dataset_targets=parse_target_spec(args.dataset_targets),
            schema_targets=parse_target_spec(args.schema_targets),
            batch_size=args.batch_size,
            seed=args.seed,
            rank=rank,
            world_size=world_size,
        )
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_sampler=train_sampler,
            collate_fn=schema_aware_collate if schema_aware_training else None,
            **dataloader_kwargs(args),
        )
    else:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
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


__all__ = ["TrainerDataLoaders", "build_training_loaders"]
