"""Distributed setup helpers for CD-ViT training."""

from __future__ import annotations

import os
import typing as T

import torch
import torch.distributed as dist


def setup_distributed_from_env(args: T.Any) -> torch.device:
    """Initialize NCCL DDP from LOCAL_RANK and return the CUDA device."""

    if "LOCAL_RANK" in os.environ and os.environ["LOCAL_RANK"] is not None:
        print(os.environ["LOCAL_RANK"])
        args.local_rank = int(os.environ["LOCAL_RANK"])
    else:
        args.local_rank = 0
    torch.cuda.set_device(args.local_rank)
    dist.init_process_group("nccl", rank=args.local_rank, init_method="env://")
    return torch.device("cuda", args.local_rank)


def distributed_rank() -> int:
    if dist.is_available() and dist.is_initialized():
        return int(dist.get_rank())
    return 0


def distributed_world_size() -> int:
    if dist.is_available() and dist.is_initialized():
        return int(dist.get_world_size())
    return 1


def is_rank_zero() -> bool:
    return distributed_rank() == 0


__all__ = [
    "distributed_rank",
    "distributed_world_size",
    "is_rank_zero",
    "setup_distributed_from_env",
]
