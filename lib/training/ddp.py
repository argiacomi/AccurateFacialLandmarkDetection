"""Distributed setup helpers for CD-ViT training."""

from __future__ import annotations

import os
import typing as T

import torch
import torch.distributed as dist

from lib.training.device import resolve_device


def setup_distributed_from_env(args: T.Any) -> torch.device:
    """Resolve the training device and, on CUDA under torchrun, init NCCL DDP.

    Multi-process distributed training uses the NCCL backend and therefore
    requires CUDA. When CUDA is unavailable (Apple Silicon/MPS or CPU) or the
    process was not launched by torchrun, training runs single-process and no
    process group is initialized -- ``dist.is_initialized()`` stays False so the
    trainer skips every collective operation and DDP wrapper.
    """

    if os.environ.get("LOCAL_RANK") is not None:
        args.local_rank = int(os.environ["LOCAL_RANK"])
    else:
        args.local_rank = 0

    device = resolve_device(getattr(args, "device", "auto"))
    launched_by_torchrun = "LOCAL_RANK" in os.environ or "WORLD_SIZE" in os.environ

    if device.type == "cuda":
        torch.cuda.set_device(args.local_rank)
        if launched_by_torchrun:
            dist.init_process_group("nccl", rank=args.local_rank, init_method="env://")
        return torch.device("cuda", args.local_rank)

    # MPS / CPU: single-process training, no NCCL process group.
    return device


class LocalModelWrapper(torch.nn.Module):
    """DistributedDataParallel-compatible surface for single-process runs.

    Exposes ``.module`` and forwards calls so the trainer's ``net.module`` and
    ``net(data)`` usage works identically whether or not DDP is active. Used on
    MPS/CPU and on single-process CUDA runs where no process group exists.
    """

    def __init__(self, module: torch.nn.Module) -> None:
        super().__init__()
        self.module = module

    def forward(self, *args: T.Any, **kwargs: T.Any) -> T.Any:
        return self.module(*args, **kwargs)


def distributed_is_active() -> bool:
    """Whether a process group is initialized (multi-process DDP training)."""

    return bool(dist.is_available() and dist.is_initialized())


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
    "LocalModelWrapper",
    "distributed_is_active",
    "distributed_rank",
    "distributed_world_size",
    "is_rank_zero",
    "setup_distributed_from_env",
]
