"""Checkpoint save/load and RNG helpers for CD-ViT landmark training.

These helpers are split out of ``TrainHeatmapStageFP16.py`` so checkpoint
format, metadata sidecars, and RNG restore behavior can be tested and reused
without growing the training entry point.
"""

from __future__ import annotations

import json
import random
import time
import typing as T
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

from lib.landmarks.training.checkpoint_compat import (
    build_training_compat_config_from_args,
    file_sha256_or_none,
    normalize_path_for_compat,
    training_compat_digest_from_args,
    training_manifest_path_for_compat,
)


_CHECKPOINT_RNG_STATE_BY_RANK: dict[str, T.Any] | None = None


def _current_rank_string() -> str:
    if dist.is_available() and dist.is_initialized():
        return str(dist.get_rank())
    return "0"


def _local_rng_state_for_checkpoint() -> dict[str, T.Any]:
    return {
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "numpy": np.random.get_state(),
        "python": random.getstate(),
    }


def _collect_rng_state_by_rank() -> dict[str, T.Any]:
    local_payload = {
        "rank": _current_rank_string(),
        "rng": _local_rng_state_for_checkpoint(),
    }
    if dist.is_available() and dist.is_initialized():
        gathered: list[T.Any] = [None for _ in range(dist.get_world_size())]
        dist.all_gather_object(gathered, local_payload)
        return {
            str(item["rank"]): item["rng"]
            for item in gathered
            if isinstance(item, dict) and "rank" in item and "rng" in item
        }
    return {local_payload["rank"]: local_payload["rng"]}


def _set_checkpoint_rng_state_by_rank(rng_state_by_rank: dict[str, T.Any] | None) -> None:
    global _CHECKPOINT_RNG_STATE_BY_RANK
    _CHECKPOINT_RNG_STATE_BY_RANK = rng_state_by_rank


def _checkpoint_rng_state_for_payload() -> dict[str, T.Any]:
    if _CHECKPOINT_RNG_STATE_BY_RANK:
        return _CHECKPOINT_RNG_STATE_BY_RANK
    return {_current_rank_string(): _local_rng_state_for_checkpoint()}


def _rng_state_for_current_rank(checkpoint: dict[str, T.Any]) -> dict[str, T.Any] | None:
    rng_by_rank = checkpoint.get("rng_by_rank")
    if isinstance(rng_by_rank, dict):
        rank_key = _current_rank_string()
        if rank_key in rng_by_rank:
            return rng_by_rank[rank_key]
        print(
            f"warning: checkpoint has rank-keyed RNG but no RNG state for rank {rank_key}; "
            "skipping RNG restore on this rank",
            flush=True,
        )
        return None

    legacy_rng = checkpoint.get("rng")
    if legacy_rng:
        if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
            print(
                "warning: checkpoint has legacy rank-0 RNG only; skipping RNG restore "
                "for distributed resume. Use a rank-keyed checkpoint for exact DDP replay.",
                flush=True,
            )
            return None
        return legacy_rng
    return None


def _torch_load_training_checkpoint(path: str | Path, device: torch.device | str | int) -> T.Any:
    """Load trusted local CD-ViT training checkpoints.

    Full training checkpoints include optimizer, scheduler, scaler, EMA, RNG,
    and argparse state, so they are intentionally not weights-only payloads.
    """
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        # Older PyTorch releases do not accept weights_only.
        return torch.load(path, map_location=device)


def _checkpoint_metadata_path(path: str | Path) -> Path:
    return Path(str(Path(path)) + ".meta.json")


def _json_safe_checkpoint_value(value: T.Any) -> T.Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [_json_safe_checkpoint_value(item) for item in value]
    if isinstance(value, dict):
        return {
            str(key): _json_safe_checkpoint_value(val)
            for key, val in sorted(value.items(), key=lambda item: str(item[0]))
        }
    return str(value)


def _checkpoint_metadata_payload(payload: dict[str, T.Any]) -> dict[str, T.Any]:
    heavy_keys = {
        "model",
        "optimizer",
        "scheduler",
        "scaler",
        "ema",
        "rng",
        "rng_by_rank",
    }
    metadata = {
        key: value
        for key, value in payload.items()
        if key not in heavy_keys
    }
    metadata["has_model"] = "model" in payload
    metadata["has_optimizer"] = payload.get("optimizer") is not None
    metadata["has_scheduler"] = payload.get("scheduler") is not None
    metadata["has_scaler"] = payload.get("scaler") is not None
    metadata["has_ema"] = payload.get("ema") is not None
    metadata["has_rng_by_rank"] = payload.get("rng_by_rank") is not None
    return _json_safe_checkpoint_value(metadata)


def _write_checkpoint_metadata(path: str | Path, payload: dict[str, T.Any]) -> None:
    meta_path = _checkpoint_metadata_path(path)
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(
        json.dumps(_checkpoint_metadata_payload(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _model_state(net: T.Any) -> dict[str, T.Any]:
    model = net.module if hasattr(net, "module") else net
    return model.state_dict()


def _save_training_checkpoint(
    path: str | Path,
    net: T.Any,
    optimizer: T.Any,
    scheduler: T.Any,
    scaler: T.Any,
    ema: T.Any,
    epoch: int,
    best_nme: T.Any,
    best_record: T.Any,
    args: T.Any,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path = training_manifest_path_for_compat(args)
    payload: dict[str, T.Any] = {
        "format": "cdvit-training-checkpoint-v1",
        "epoch": int(epoch),
        "next_epoch": int(epoch) + 1,
        "model": _model_state(net),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "scaler": scaler.state_dict() if scaler is not None else None,
        "best_nme": best_nme,
        "best_record": list(best_record),
        "args": vars(args),
        "manifest": normalize_path_for_compat(manifest_path),
        "manifest_sha256": file_sha256_or_none(manifest_path),
        "compat_config": build_training_compat_config_from_args(args),
        "compat_config_digest": training_compat_digest_from_args(args),
        "rng_schema": "rank-keyed-v1",
        "rng_by_rank": _checkpoint_rng_state_for_payload(),
    }
    if ema is not None:
        payload["ema"] = ema.model.state_dict()
        payload["ema_n_iter"] = int(getattr(ema, "n_iter", 0))
    torch.save(payload, path)
    _write_checkpoint_metadata(path, payload)


def _restore_training_checkpoint(
    checkpoint: T.Any,
    optimizer: T.Any,
    scheduler: T.Any,
    scaler: T.Any,
    ema: T.Any,
    best_nme: T.Any,
    best_record: T.Any,
    args: T.Any,
) -> tuple[int, T.Any, T.Any]:
    if not isinstance(checkpoint, dict) or checkpoint.get("format") != "cdvit-training-checkpoint-v1":
        return 0, best_nme, best_record

    if checkpoint.get("optimizer") is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])
    if checkpoint.get("scheduler") is not None:
        scheduler.load_state_dict(checkpoint["scheduler"])
    if checkpoint.get("scaler") is not None:
        scaler.load_state_dict(checkpoint["scaler"])
    if ema is not None and checkpoint.get("ema") is not None:
        ema.model.load_state_dict(checkpoint["ema"])
        ema.n_iter = int(checkpoint.get("ema_n_iter", getattr(ema, "n_iter", 0)))

    if getattr(args, "restore_rng", False):
        rng = _rng_state_for_current_rank(checkpoint)
        if rng is not None:
            try:
                if rng.get("torch") is not None:
                    torch.set_rng_state(rng["torch"].cpu())
                if torch.cuda.is_available() and rng.get("cuda") is not None:
                    torch.cuda.set_rng_state_all(rng["cuda"])
                if rng.get("numpy") is not None:
                    np.random.set_state(rng["numpy"])
                if rng.get("python") is not None:
                    random.setstate(rng["python"])
            except Exception as err:
                print(f"warning: failed to restore RNG state from checkpoint: {err}", flush=True)

    start_epoch = int(checkpoint.get("next_epoch", int(checkpoint.get("epoch", -1)) + 1))
    best_nme = checkpoint.get("best_nme", best_nme)
    best_record = checkpoint.get("best_record", best_record)
    return start_epoch, best_nme, best_record


def _write_training_complete_sentinel(
    args: T.Any,
    epoch: int,
    best_nme: T.Any,
    best_record: T.Any,
    global_train_samples: int,
) -> None:
    path = Path(args.ckpt_folder) / "training_complete.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path = training_manifest_path_for_compat(args)
    payload = {
        "status": "complete",
        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "epoch": int(epoch),
        "requested_epochs": int(args.epoch),
        "best_nme": best_nme,
        "best_record": list(best_record),
        "global_train_samples_last_epoch": int(global_train_samples),
        "manifest": normalize_path_for_compat(manifest_path),
        "manifest_sha256": file_sha256_or_none(manifest_path),
        "compat_config": build_training_compat_config_from_args(args),
        "compat_config_digest": training_compat_digest_from_args(args),
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
