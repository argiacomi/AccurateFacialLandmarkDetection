"""Runtime helpers for CD-ViT landmark training."""

from __future__ import annotations

import random
import typing as T

import numpy as np
import torch
import torch.distributed as dist


def dataloader_kwargs(args: T.Any, *, eval_loader: bool = False) -> dict[str, T.Any]:
    workers = int(args.eval_num_workers if eval_loader else args.num_workers)
    kwargs: dict[str, T.Any] = {
        "num_workers": workers,
        "pin_memory": bool(args.pin_memory),
    }
    if workers > 0:
        kwargs["persistent_workers"] = bool(args.persistent_workers)
        kwargs["worker_init_fn"] = seed_worker
        if (
            getattr(args, "prefetch_factor", None) is not None
            and int(args.prefetch_factor) > 0
        ):
            kwargs["prefetch_factor"] = int(args.prefetch_factor)
    return kwargs


def maybe_limit_eval_dataset(dataset: T.Any, max_samples: int, seed: int = 0) -> T.Any:
    max_samples = int(max_samples or 0)
    if max_samples <= 0 or len(dataset) <= max_samples:
        return dataset
    rng = random.Random(int(seed))
    indices = list(range(len(dataset)))
    rng.shuffle(indices)
    return torch.utils.data.Subset(dataset, sorted(indices[:max_samples]))


def should_run_interval(interval: int, epoch: int, final_epoch: int) -> bool:
    interval = int(interval or 0)
    if interval <= 0:
        return False
    if int(epoch) >= int(final_epoch):
        return True
    return (int(epoch) + 1) % interval == 0


def set_dataset_runtime_epoch(dataset: T.Any, epoch: int, args: T.Any) -> None:
    rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
    targets = [dataset]
    if hasattr(dataset, "dataset"):
        targets.append(dataset.dataset)
    for target in targets:
        try:
            setattr(target, "runtime_epoch", int(epoch))
            setattr(target, "runtime_base_seed", int(args.seed))
            setattr(target, "runtime_rank", int(rank))
        except Exception:
            pass


def seed_worker(worker_id: int) -> None:
    info = torch.utils.data.get_worker_info()
    dataset = info.dataset if info is not None else None
    base_seed = int(getattr(dataset, "runtime_base_seed", 0))
    epoch = int(getattr(dataset, "runtime_epoch", 0))
    rank = int(getattr(dataset, "runtime_rank", 0))
    seed = (base_seed + epoch * 1_000_003 + rank * 10_007 + int(worker_id)) % (2**32)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)


def normalize_runtime_args(args: T.Any) -> T.Any:
    if getattr(args, "restore_rng", False) and getattr(
        args, "persistent_workers", False
    ):
        print(
            "warning: --restore-rng requires epoch-reseeded training workers; "
            "forcing --no-persistent-workers for checkpoint-compatible replay",
            flush=True,
        )
        args.persistent_workers = False
    return args
