import argparse
import hashlib
import json
import os
from pathlib import Path
import torch
import torch.utils.data.distributed
import torch.distributed as dist
from DatasetAll import GetDataset
from torch.optim.lr_scheduler import StepLR
from Net import VitAttnStage, HeadingNet
from torch.utils.data._utils.collate import default_collate

# from Vit import Vit
from Attention import SA2SA1_2

# from Attention import  SA2SA1_twins
# from UNet2 import UNet
import torch.nn.functional as F
import time
from tqdm import tqdm
import numpy as np
from loss import AWingLoss
from EMA import EMA
import math
from torch.nn.attention import sdpa_kernel, SDPBackend
import random
from lib.landmarks.core.schema import (
    DEFAULT_SCHEMA_HEADS,
    MAP_98_TO_68,
    head_name_for_schema,
)
from lib.landmarks.core.manifest_aliases import (
    CANONICAL_MANIFEST_DATA_NAME,
    LEGACY_MANIFEST_DATA_NAME,
    is_schema_aware_manifest_dataset,
)
from lib.landmarks.evaluation.split_safe import (
    EVAL_MODES,
    SPLIT_POLICIES,
    build_slice_report,
    record_for_sample,
    validate_no_train_test_leakage,
    write_eval_csv,
    write_eval_json,
    write_eval_records_csv,
    write_eval_records_jsonl,
)
from lib.landmarks.training.domain_balanced_sampler import (
    DEFAULT_BUCKET_TARGETS,
    DomainBalancedBatchSampler,
    parse_target_spec,
)


# from torch.cuda.amp import autocast as autocast


LEGACY_FS68_DATASET_NAME = LEGACY_MANIFEST_DATA_NAME
MULTI_SCHEMA_MANIFEST_DATASET_NAME = CANONICAL_MANIFEST_DATA_NAME
FS68_DATASET_NAME = LEGACY_FS68_DATASET_NAME


def _is_schema_aware_manifest_dataset(data_name):
    return is_schema_aware_manifest_dataset(data_name)


# PR3 training runtime helpers.
def _dataloader_kwargs(args, *, eval_loader=False):
    workers = int(args.eval_num_workers if eval_loader else args.num_workers)
    kwargs = {
        "num_workers": workers,
        "pin_memory": bool(args.pin_memory),
    }
    if workers > 0:
        kwargs["persistent_workers"] = bool(args.persistent_workers)
        kwargs["worker_init_fn"] = _seed_worker
        if args.prefetch_factor is not None and int(args.prefetch_factor) > 0:
            kwargs["prefetch_factor"] = int(args.prefetch_factor)
    return kwargs


def _maybe_limit_eval_dataset(dataset, max_samples, seed=0):
    max_samples = int(max_samples or 0)
    if max_samples <= 0 or len(dataset) <= max_samples:
        return dataset
    rng = random.Random(int(seed))
    indices = list(range(len(dataset)))
    rng.shuffle(indices)
    return torch.utils.data.Subset(dataset, sorted(indices[:max_samples]))


def _should_run_interval(interval, epoch, final_epoch):
    interval = int(interval or 0)
    if interval <= 0:
        return False
    if int(epoch) >= int(final_epoch):
        return True
    return (int(epoch) + 1) % interval == 0


def _cuda_peak_memory_mb(device):
    if not torch.cuda.is_available():
        return None
    try:
        return round(torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0), 3)
    except Exception:
        return None


def _runtime_metrics_path(args):
    if args.runtime_metrics_jsonl:
        return Path(args.runtime_metrics_jsonl)
    return Path(args.ckpt_folder) / "runtime_metrics.jsonl"


def _append_runtime_metrics(args, payload):
    path = _runtime_metrics_path(args)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def _empty_epoch_timing():
    return {
        "data_wait_seconds": 0.0,
        "device_transfer_seconds": 0.0,
        "forward_backward_update_seconds": 0.0,
        "eval_seconds": 0.0,
        "ema_eval_seconds": 0.0,
        "checkpoint_seconds": 0.0,
    }


def _accumulate_timing(timing, key, started_at):
    timing[key] = float(timing.get(key, 0.0)) + (time.time() - started_at)



def _normalize_path_for_compat(value):
    if value in (None, ""):
        return ""
    try:
        return str(Path(value).expanduser().resolve())
    except OSError:
        return str(Path(value).expanduser())


def _file_sha256_or_none(value):
    if value in (None, ""):
        return None
    path = Path(value)
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _training_manifest_path_for_compat(args):
    return args.train_manifest or args.manifest or args.root_folder


def _coerce_config_scalar(value):
    if isinstance(value, Path):
        return _normalize_path_for_compat(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [_coerce_config_scalar(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _coerce_config_scalar(val) for key, val in sorted(value.items())}
    return str(value)


def _training_compat_config(args):
    int_keys = {
        "batch_size",
        "heatmap_size",
        "lmk_num",
        "sched_step",
        "nstack",
        "max_depth",
        "seed",
    }
    float_keys = {
        "lr",
        "hw",
        "locw",
        "mul",
        "schema_consistency_weight",
        "auxiliary_loss_weight",
    }
    bool_keys = {
        "schema_aware_training",
        "domain_balanced_sampling",
        "auxiliary_heads",
    }
    string_keys = {
        "data_name",
        "eval_mode",
        "split_policy",
        "bucket_targets",
        "dataset_targets",
        "schema_targets",
    }

    config = {
        "manifest_sha256": _file_sha256_or_none(_training_manifest_path_for_compat(args)),
        "train_manifest_sha256": _file_sha256_or_none(getattr(args, "train_manifest", "")),
        "test_manifest_sha256": _file_sha256_or_none(getattr(args, "test_manifest", "")),
    }

    for key in sorted(int_keys):
        try:
            config[key] = int(getattr(args, key))
        except (TypeError, ValueError):
            config[key] = getattr(args, key, None)
    for key in sorted(float_keys):
        try:
            config[key] = float(getattr(args, key))
        except (TypeError, ValueError):
            config[key] = getattr(args, key, None)
    for key in sorted(bool_keys):
        config[key] = bool(getattr(args, key, False))
    for key in sorted(string_keys):
        config[key] = str(getattr(args, key, ""))

    config["heldout_dataset"] = [
        str(item) for item in list(getattr(args, "heldout_dataset", []) or [])
    ]
    return config


def _training_compat_digest(args):
    payload = json.dumps(
        _training_compat_config(args),
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _checkpoint_compat_errors(checkpoint, args):
    if not isinstance(checkpoint, dict) or checkpoint.get("format") != "cdvit-training-checkpoint-v1":
        return []

    errors = []
    expected_config = _training_compat_config(args)
    actual_config = checkpoint.get("compat_config")
    if isinstance(actual_config, dict):
        comparable_actual = {key: actual_config.get(key) for key in expected_config}
        if comparable_actual != expected_config:
            errors.append("checkpoint data/model training contract differs from the current invocation")
    else:
        actual_digest = checkpoint.get("compat_config_digest")
        if actual_digest:
            if actual_digest != _training_compat_digest(args):
                errors.append("checkpoint data/model training contract digest differs from the current invocation")
        else:
            # Fallback for PR3 checkpoints produced before compat_config existed.
            saved_args = checkpoint.get("args") if isinstance(checkpoint.get("args"), dict) else {}
            critical_keys = {
                "data_name": str(args.data_name),
                "batch_size": int(args.batch_size),
                "heatmap_size": int(args.heatmap_size),
                "lmk_num": int(args.lmk_num),
                "lr": float(args.lr),
                "schema_aware_training": bool(args.schema_aware_training),
                "domain_balanced_sampling": bool(args.domain_balanced_sampling),
                "auxiliary_heads": bool(args.auxiliary_heads),
            }
            for key, expected in critical_keys.items():
                if key not in saved_args:
                    continue
                actual = saved_args[key]
                try:
                    if isinstance(expected, bool):
                        matches = bool(actual) == expected
                    elif isinstance(expected, int):
                        matches = int(actual) == expected
                    elif isinstance(expected, float):
                        matches = float(actual) == expected
                    else:
                        matches = str(actual) == str(expected)
                except (TypeError, ValueError):
                    matches = False
                if not matches:
                    errors.append(
                        f"checkpoint arg {key!r} differs: checkpoint={actual!r}, current={expected!r}"
                    )

    current_manifest_sha = _file_sha256_or_none(_training_manifest_path_for_compat(args))
    checkpoint_manifest_sha = checkpoint.get("manifest_sha256")
    if current_manifest_sha and checkpoint_manifest_sha and current_manifest_sha != checkpoint_manifest_sha:
        errors.append("checkpoint manifest SHA differs from the current manifest")

    return errors


def _set_dataset_runtime_epoch(dataset, epoch, args):
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


def _seed_worker(worker_id):
    info = torch.utils.data.get_worker_info()
    dataset = info.dataset if info is not None else None
    base_seed = int(getattr(dataset, "runtime_base_seed", 0))
    epoch = int(getattr(dataset, "runtime_epoch", 0))
    rank = int(getattr(dataset, "runtime_rank", 0))
    seed = (base_seed + epoch * 1_000_003 + rank * 10_007 + int(worker_id)) % (2**32)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)



def _normalize_runtime_args(args):
    if args.restore_rng and args.persistent_workers:
        print(
            "warning: --restore-rng requires epoch-reseeded training workers; "
            "forcing --no-persistent-workers for checkpoint-compatible replay",
            flush=True,
        )
        args.persistent_workers = False
    return args


_CHECKPOINT_RNG_STATE_BY_RANK = None


def _current_rank_string():
    if dist.is_available() and dist.is_initialized():
        return str(dist.get_rank())
    return "0"


def _local_rng_state_for_checkpoint():
    return {
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "numpy": np.random.get_state(),
        "python": random.getstate(),
    }


def _collect_rng_state_by_rank():
    local_payload = {
        "rank": _current_rank_string(),
        "rng": _local_rng_state_for_checkpoint(),
    }
    if dist.is_available() and dist.is_initialized():
        gathered = [None for _ in range(dist.get_world_size())]
        dist.all_gather_object(gathered, local_payload)
        return {
            str(item["rank"]): item["rng"]
            for item in gathered
            if isinstance(item, dict) and "rank" in item and "rng" in item
        }
    return {local_payload["rank"]: local_payload["rng"]}


def _set_checkpoint_rng_state_by_rank(rng_state_by_rank):
    global _CHECKPOINT_RNG_STATE_BY_RANK
    _CHECKPOINT_RNG_STATE_BY_RANK = rng_state_by_rank


def _checkpoint_rng_state_for_payload():
    if _CHECKPOINT_RNG_STATE_BY_RANK:
        return _CHECKPOINT_RNG_STATE_BY_RANK
    return {_current_rank_string(): _local_rng_state_for_checkpoint()}


def _rng_state_for_current_rank(checkpoint):
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
                "for distributed resume. Use a rank-keyed PR3 checkpoint for exact DDP replay.",
                flush=True,
            )
            return None
        return legacy_rng
    return None


def _write_training_complete_sentinel(args, epoch, best_nme, best_record, global_train_samples):
    path = Path(args.ckpt_folder) / "training_complete.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "status": "complete",
        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "epoch": int(epoch),
        "requested_epochs": int(args.epoch),
        "best_nme": best_nme,
        "best_record": list(best_record),
        "global_train_samples_last_epoch": int(global_train_samples),
        "manifest": _normalize_path_for_compat(_training_manifest_path_for_compat(args)),
        "manifest_sha256": _file_sha256_or_none(_training_manifest_path_for_compat(args)),
        "compat_config": _training_compat_config(args),
        "compat_config_digest": _training_compat_digest(args),
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")



def _torch_load_training_checkpoint(path, device):
    """Load trusted local CD-ViT training checkpoints.

    Full training checkpoints include optimizer, scheduler, scaler, EMA, RNG,
    and argparse state, so they are intentionally not weights-only payloads.
    """
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        # Older PyTorch releases do not accept weights_only.
        return torch.load(path, map_location=device)


def _checkpoint_metadata_path(path):
    return Path(str(Path(path)) + ".meta.json")


def _json_safe_checkpoint_value(value):
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


def _checkpoint_metadata_payload(payload):
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


def _write_checkpoint_metadata(path, payload):
    meta_path = _checkpoint_metadata_path(path)
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(
        json.dumps(_checkpoint_metadata_payload(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _model_state(net):
    model = net.module if hasattr(net, "module") else net
    return model.state_dict()


def _save_training_checkpoint(path, net, optimizer, scheduler, scaler, ema, epoch, best_nme, best_record, args):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
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
        "manifest": _normalize_path_for_compat(_training_manifest_path_for_compat(args)),
        "manifest_sha256": _file_sha256_or_none(_training_manifest_path_for_compat(args)),
        "compat_config": _training_compat_config(args),
        "compat_config_digest": _training_compat_digest(args),
        "rng_schema": "rank-keyed-v1",
        "rng_by_rank": _checkpoint_rng_state_for_payload(),
    }
    if ema is not None:
        payload["ema"] = ema.model.state_dict()
        payload["ema_n_iter"] = int(getattr(ema, "n_iter", 0))
    torch.save(payload, path)
    _write_checkpoint_metadata(path, payload)


def _restore_training_checkpoint(checkpoint, optimizer, scheduler, scaler, ema, best_nme, best_record, args):
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

    if args.restore_rng:
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


AUXILIARY_CLASS_NAMES = {
    "pose_bucket": ("frontal", "profile", "profile_left", "profile_right"),
    "occlusion": ("no_occlusion", "occlusion"),
    "visibility": ("all_visible", "partially_visible"),
    "blur_quality": ("clear", "blurred"),
    "illumination_quality": ("normal", "challenging"),
    "profile_side": ("not_profile", "left", "right"),
    "landmark_confidence": ("normal", "low"),
}
AUXILIARY_CLASS_INDEX = {
    name: {label: index for index, label in enumerate(labels)}
    for name, labels in AUXILIARY_CLASS_NAMES.items()
}


def setup_seed(seed=0, deterministic=False):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = bool(deterministic)
    torch.backends.cudnn.benchmark = not bool(deterministic)


def _landmark_count_for_dataset(args):
    if args.data_name == "WFLW":
        return 98
    if args.data_name == "COFW":
        return 29
    if args.data_name == "300W":
        return 68
    if _is_schema_aware_manifest_dataset(args.data_name):
        return int(args.lmk_num)
    raise ValueError(f"unknown data_name: {args.data_name}")


def _manifest_for_split(args, split):
    if split == "train":
        return args.train_manifest or args.manifest or args.root_folder
    if split == "test":
        return args.test_manifest or args.manifest or args.root_folder
    return args.manifest or args.root_folder


def _build_dataset(
    args,
    split,
    aug,
    heatmap_size=0,
    include_metadata=False,
    schema_aware_training=False,
):
    manifest_path = (
        _manifest_for_split(args, split)
        if _is_schema_aware_manifest_dataset(args.data_name)
        else ""
    )
    return GetDataset(
        args.data_name,
        args.root_folder,
        split,
        preload=args.preload != 0,
        aug=aug,
        heatmap_size=heatmap_size,
        manifest_path=manifest_path,
        eval_mode=args.eval_mode
        if _is_schema_aware_manifest_dataset(args.data_name)
        else "random_hash",
        heldout_datasets=args.heldout_dataset
        if _is_schema_aware_manifest_dataset(args.data_name)
        else None,
        include_metadata=include_metadata,
        schema_aware_training=schema_aware_training,
        split_policy=args.split_policy
        if _is_schema_aware_manifest_dataset(args.data_name)
        else "declared_or_random_hash",
    )


def _unpack_train_batch(batch, device, non_blocking=False):
    if isinstance(batch, dict):
        data = batch["image"].to(device, non_blocking=non_blocking)
        heads = {}
        for head_name, payload in batch["heads"].items():
            heads[head_name] = {
                "indices": payload["indices"].to(device, non_blocking=non_blocking),
                "target": payload["target"].to(device, non_blocking=non_blocking).float(),
                "heatmap": payload["heatmap"].to(device, non_blocking=non_blocking).float(),
                "landmark_mask": payload["landmark_mask"].to(device, non_blocking=non_blocking).float(),
                "sample_weight": payload["sample_weight"].to(device, non_blocking=non_blocking).float(),
            }
            heads[head_name]["sample_weight"] = heads[head_name][
                "sample_weight"
            ] / heads[head_name]["sample_weight"].mean().clamp_min(1e-6)
        aux_labels = {
            task: labels.to(device, non_blocking=non_blocking)
            for task, labels in batch.get("aux_labels", {}).items()
        }
        return data, heads, aux_labels

    if len(batch) == 5:
        data, target, heatmap, sample_weight, landmark_mask = batch
    elif len(batch) == 4:
        data, target, heatmap, sample_weight = batch
        landmark_mask = None
    elif len(batch) == 3:
        data, target, heatmap = batch
        sample_weight = None
        landmark_mask = None
    else:
        raise ValueError(
            f"expected train batch with 3, 4, or 5 items, got {len(batch)}"
        )

    data = data.to(device, non_blocking=non_blocking)
    target = target.to(device, non_blocking=non_blocking).float()
    heatmap = heatmap.to(device, non_blocking=non_blocking)
    if sample_weight is not None:
        sample_weight = sample_weight.to(device, non_blocking=non_blocking).float()
        sample_weight = sample_weight / sample_weight.mean().clamp_min(1e-6)
    if landmark_mask is None:
        landmark_mask = torch.ones(target.shape[:2], device=device, dtype=torch.float32)
    else:
        landmark_mask = landmark_mask.to(device, non_blocking=non_blocking).float()
    return data, target, heatmap, sample_weight, landmark_mask


def _schema_aware_collate(batch):
    images = default_collate([item["image"] for item in batch])
    grouped = {}
    for index, item in enumerate(batch):
        head_name = item["head_name"]
        grouped.setdefault(
            head_name,
            {
                "indices": [],
                "target": [],
                "heatmap": [],
                "landmark_mask": [],
                "sample_weight": [],
                "metadata": [],
            },
        )
        grouped[head_name]["indices"].append(index)
        grouped[head_name]["target"].append(item["target"])
        grouped[head_name]["heatmap"].append(item["heatmap"])
        grouped[head_name]["landmark_mask"].append(item["landmark_mask"])
        grouped[head_name]["sample_weight"].append(item["sample_weight"])
        grouped[head_name]["metadata"].append(item.get("metadata", {}))

    heads = {}
    for head_name, payload in grouped.items():
        heads[head_name] = {
            "indices": torch.as_tensor(payload["indices"], dtype=torch.long),
            "target": default_collate(payload["target"]),
            "heatmap": default_collate(payload["heatmap"]),
            "landmark_mask": default_collate(payload["landmark_mask"]),
            "sample_weight": default_collate(payload["sample_weight"]),
            "metadata": payload["metadata"],
        }
    mix = {"bucket": {}, "dataset": {}, "schema": {}}
    aux_labels = {name: [] for name in AUXILIARY_CLASS_NAMES}
    for item in batch:
        metadata = item.get("metadata", {})
        bucket = str(
            metadata.get("hard_negative_bucket")
            or metadata.get("condition")
            or "unknown"
        )
        dataset = str(metadata.get("dataset") or "unknown")
        schema = str(item.get("schema") or metadata.get("source_schema") or "unknown")
        mix["bucket"][bucket] = mix["bucket"].get(bucket, 0) + 1
        mix["dataset"][dataset] = mix["dataset"].get(dataset, 0) + 1
        mix["schema"][schema] = mix["schema"].get(schema, 0) + 1
        for task in aux_labels:
            aux_labels[task].append(_auxiliary_label(task, metadata, item))
    return {
        "image": images,
        "heads": heads,
        "mix": mix,
        "aux_labels": {
            task: torch.as_tensor(values, dtype=torch.long)
            for task, values in aux_labels.items()
        },
    }


def _auxiliary_label(task, metadata, item):
    attributes = (
        metadata.get("attributes")
        if isinstance(metadata.get("attributes"), dict)
        else {}
    )
    conditions = metadata.get("conditions") or ()
    if isinstance(conditions, str):
        conditions = (conditions,)
    condition_labels = {
        str(value).strip().lower().replace("-", "_") for value in conditions
    }
    condition = str(metadata.get("condition") or "").strip().lower().replace("-", "_")
    if condition:
        condition_labels.add(condition)

    label = None
    if task == "pose_bucket":
        raw = (
            str(metadata.get("pose_bucket") or metadata.get("pose") or "")
            .strip()
            .lower()
            .replace("-", "_")
        )
        if raw in {"profile_left", "large_yaw_left", "left_profile"}:
            label = "profile_left"
        elif raw in {"profile_right", "large_yaw_right", "right_profile"}:
            label = "profile_right"
        elif raw in {"profile", "large_yaw", "1"} or attributes.get("pose"):
            label = "profile"
        elif (
            raw in {"frontal", "normal", "0"}
            or "frontal" in condition_labels
            or "anchor" in condition_labels
        ):
            label = "frontal"
    elif task == "occlusion":
        raw = metadata.get("occlusion", attributes.get("occlusion"))
        if raw is not None:
            label = (
                "occlusion"
                if bool(raw) and str(raw).lower() not in {"0", "false", "none"}
                else "no_occlusion"
            )
        elif any(
            "occlusion" in value or "occlud" in value for value in condition_labels
        ):
            label = "occlusion"
        else:
            label = "no_occlusion"
    elif task == "visibility":
        mask = item.get("landmark_mask")
        if mask is not None:
            label = (
                "all_visible"
                if bool(torch.as_tensor(mask).float().min().item() > 0.5)
                else "partially_visible"
            )
    elif task == "blur_quality":
        raw = metadata.get("blur", attributes.get("blur"))
        if raw is not None:
            label = (
                "blurred"
                if bool(raw) and str(raw).lower() not in {"0", "false", "none"}
                else "clear"
            )
    elif task == "illumination_quality":
        raw = metadata.get("illumination", attributes.get("illumination"))
        if raw is not None:
            label = (
                "challenging"
                if bool(raw) and str(raw).lower() not in {"0", "false", "none"}
                else "normal"
            )
    elif task == "profile_side":
        raw = (
            str(metadata.get("profile_side") or metadata.get("side") or "")
            .strip()
            .lower()
        )
        if raw in {"left", "right"}:
            label = raw
        elif any("left" in value for value in condition_labels):
            label = "left"
        elif any("right" in value for value in condition_labels):
            label = "right"
        elif not any(
            value in condition_labels
            for value in ("profile", "large_yaw", "profile_pose")
        ):
            label = "not_profile"
    elif task == "landmark_confidence":
        weight = float(item.get("sample_weight", torch.tensor(1.0)).item())
        label = "low" if weight > 2.0 else "normal"

    if label is None:
        return -1
    return AUXILIARY_CLASS_INDEX[task].get(label, -1)


def _weighted_smooth_l1(pred_loc, target, sample_weight, landmark_mask, beta=0.001):
    per_point = F.smooth_l1_loss(pred_loc, target, beta=beta, reduction="none").mean(
        dim=2
    )
    landmark_mask = landmark_mask.to(per_point.device).float()
    per_sample = (per_point * landmark_mask).sum(dim=1) / landmark_mask.sum(
        dim=1
    ).clamp_min(1.0)
    if sample_weight is not None:
        return (per_sample * sample_weight).mean()
    return per_sample.mean()


def _schema_head_loss(stage_pred, heads, aux_labels, heatmap_loss_func, args):
    loss = torch.tensor(0.0, device=next(iter(heads.values()))["target"].device)
    loss_loc = torch.tensor(0.0, device=loss.device)
    loss_heatmap = torch.tensor(0.0, device=loss.device)
    loss_aux = torch.tensor(0.0, device=loss.device)
    for head_name, payload in heads.items():
        pred_loc, pred_heatmap = stage_pred[head_name]
        indices = payload["indices"]
        pred_loc = pred_loc.index_select(0, indices)
        pred_heatmap = pred_heatmap.index_select(0, indices)
        target = payload["target"]
        heatmap = payload["heatmap"]
        sample_weight = payload["sample_weight"]
        landmark_mask = payload["landmark_mask"]
        B, C, H, W = pred_heatmap.shape
        head_loc = (
            _weighted_smooth_l1(
                pred_loc, target, sample_weight, landmark_mask, beta=0.001
            )
            * args.locw
        )
        pred_prob = F.softmax(pred_heatmap.reshape((B, C, -1)), dim=2).reshape(
            (B, C, H, W)
        )
        head_heatmap = (
            heatmap_loss_func(
                pred_prob,
                heatmap,
                batch_weights=_heatmap_batch_weight(
                    sample_weight, pred_heatmap, landmark_mask
                ),
            )
            * args.hw
        )
        loss = loss + head_loc + head_heatmap
        loss_loc = loss_loc + head_loc.detach()
        loss_heatmap = loss_heatmap + head_heatmap.detach()

    if (
        args.schema_consistency_weight > 0
        and "landmarks_98" in heads
        and "landmarks_68" in stage_pred
    ):
        payload = heads["landmarks_98"]
        indices = payload["indices"]
        pred_98 = stage_pred["landmarks_98"][0].index_select(0, indices)
        pred_68 = stage_pred["landmarks_68"][0].index_select(0, indices)
        projected = pred_98[:, torch.as_tensor(MAP_98_TO_68, device=pred_98.device), :]
        loss = loss + float(args.schema_consistency_weight) * F.smooth_l1_loss(
            pred_68, projected.detach(), beta=0.001
        )

    aux_outputs = stage_pred.get("_aux", {}) if isinstance(stage_pred, dict) else {}
    if args.auxiliary_loss_weight > 0 and aux_outputs:
        for task, logits in aux_outputs.items():
            labels = aux_labels.get(task)
            if labels is None:
                continue
            valid = labels >= 0
            if bool(valid.any()):
                loss_aux = loss_aux + F.cross_entropy(
                    logits[valid], labels[valid]
                ) * float(args.auxiliary_loss_weight)
        loss = loss + loss_aux

    return loss, loss_loc, loss_heatmap, loss_aux


def _heatmap_batch_weight(sample_weight, pred_heatmap, landmark_mask=None):
    if sample_weight is None and landmark_mask is None:
        return None
    weights = torch.ones(
        (pred_heatmap.shape[0], pred_heatmap.shape[1]),
        device=pred_heatmap.device,
        dtype=pred_heatmap.dtype,
    )
    if landmark_mask is not None:
        weights = weights * landmark_mask.to(pred_heatmap.device).to(pred_heatmap.dtype)
    if sample_weight is not None:
        weights = weights * sample_weight.to(pred_heatmap.device).to(
            pred_heatmap.dtype
        ).reshape(-1, 1)
    return weights.reshape(pred_heatmap.shape[0], pred_heatmap.shape[1], 1, 1)


def _eval_collate(batch):
    if batch and len(batch[0]) >= 4 and isinstance(batch[0][3], dict):
        data = default_collate([item[0] for item in batch])
        metadata = [item[3] for item in batch]
        head_names = []
        for item in batch:
            meta = item[3]
            head_name = str(meta.get("head_name") or "")
            if not head_name:
                schema = meta.get("source_schema") or meta.get("schema") or ""
                if schema:
                    try:
                        head_name = head_name_for_schema(schema)
                    except ValueError:
                        head_name = ""
            head_names.append(head_name or "landmarks_68")
        target_shapes = {tuple(item[1].shape) for item in batch}
        if len(target_shapes) == 1 and set(head_names) == {"landmarks_68"}:
            target = default_collate([item[1] for item in batch])
            landmark_mask = default_collate([item[2] for item in batch])
            return data, target, landmark_mask, metadata

        grouped = {}
        for index, (item, head_name) in enumerate(zip(batch, head_names)):
            grouped.setdefault(
                head_name,
                {"indices": [], "target": [], "landmark_mask": [], "metadata": []},
            )
            grouped[head_name]["indices"].append(index)
            grouped[head_name]["target"].append(item[1])
            grouped[head_name]["landmark_mask"].append(item[2])
            meta = dict(item[3])
            meta["head_name"] = head_name
            grouped[head_name]["metadata"].append(meta)
        return {
            "image": data,
            "heads": {
                head_name: {
                    "indices": torch.as_tensor(payload["indices"], dtype=torch.long),
                    "target": default_collate(payload["target"]),
                    "landmark_mask": default_collate(payload["landmark_mask"]),
                    "metadata": payload["metadata"],
                }
                for head_name, payload in grouped.items()
            },
        }
    return default_collate(batch)


def _unpack_eval_batch(batch):
    data = batch[0]
    target = batch[1]
    if len(batch) >= 3:
        landmark_mask = batch[2]
    else:
        landmark_mask = torch.ones(target.shape[:2], dtype=torch.float32)
    if len(batch) >= 4:
        metadata = batch[3]
    else:
        metadata = [{} for _ in range(int(target.shape[0]))]
    return data, target, landmark_mask, metadata


def _masked_nme_values(pred_keypoints, keypoints, landmark_mask):
    pred = pred_keypoints.detach().float().cpu().numpy()
    target = keypoints.detach().float().cpu().numpy()
    mask = landmark_mask.detach().float().cpu().numpy() > 0.5

    values = []
    for pred_i, target_i, mask_i in zip(pred, target, mask):
        if mask_i.sum() <= 0:
            values.append(float("nan"))
            continue

        valid = target_i[mask_i]
        if valid.shape[0] <= 1:
            values.append(float("nan"))
            continue

        span = np.max(valid, axis=0) - np.min(valid, axis=0)
        span_norm = float(max(span[0], span[1]))

        eye_norm = None
        if mask_i.shape[0] > 45 and mask_i[36] and mask_i[45]:
            eye_norm = float(np.linalg.norm(target_i[36] - target_i[45]))

        # Prefer canonical outer-eye interocular only when it is plausible.
        #
        # MERL-RAV has some frontal samples where landmarks 36/45 are valid but
        # nearly collapsed, e.g. eye_norm ~= 0.04 in normalized coordinates.
        # That explodes NME even when the rest of the face is reasonable.
        #
        # Since targets are normalized to roughly [0, 1], 0.05 is ~12.75px on
        # a 256 crop. Also require eye_norm to be at least 15% of the visible
        # landmark span, otherwise fall back to span normalization.
        if (
            eye_norm is not None
            and np.isfinite(eye_norm)
            and eye_norm > 0.05
            and np.isfinite(span_norm)
            and span_norm > 1e-6
            and eye_norm >= 0.15 * span_norm
        ):
            normalizer = eye_norm
        else:
            normalizer = span_norm

        if not np.isfinite(normalizer) or normalizer <= 1e-6:
            values.append(float("nan"))
            continue

        dist = np.linalg.norm(target_i[mask_i] - pred_i[mask_i], axis=1)
        values.append(float(dist.mean() / normalizer))

    return np.asarray(values, dtype=np.float32)


def _normalizer_for_masked_target(target_i, mask_i):
    valid = target_i[mask_i]
    if valid.shape[0] <= 1:
        return None
    span = np.max(valid, axis=0) - np.min(valid, axis=0)
    span_norm = float(max(span[0], span[1]))
    eye_norm = None
    if mask_i.shape[0] > 45 and mask_i[36] and mask_i[45]:
        eye_norm = float(np.linalg.norm(target_i[36] - target_i[45]))
    if (
        eye_norm is not None
        and np.isfinite(eye_norm)
        and eye_norm > 0.05
        and np.isfinite(span_norm)
        and span_norm > 1e-6
        and eye_norm >= 0.15 * span_norm
    ):
        return eye_norm
    if np.isfinite(span_norm) and span_norm > 1e-6:
        return span_norm
    return None


def _visibility_target_from_meta(meta, expected_count):
    raw = meta.get("visibility_target") if isinstance(meta, dict) else None
    if raw is None:
        return np.full((expected_count,), -1, dtype=np.int64)
    arr = np.asarray(raw, dtype=np.int64).reshape(-1)
    if arr.size != expected_count:
        return np.full((expected_count,), -1, dtype=np.int64)
    return arr


def _visibility_logits_from_stage(stage_pred, head_name="landmarks_68"):
    if not isinstance(stage_pred, dict):
        return None
    point_suffix = head_name.removeprefix("landmarks_").removeprefix("profile")
    for key in (
        f"{head_name}_visibility_logits",
        f"visibility_{point_suffix}_logits",
        "visibility_logits",
    ):
        value = stage_pred.get(key)
        if torch.is_tensor(value):
            return value
    head_payload = stage_pred.get(head_name)
    if (
        isinstance(head_payload, (list, tuple))
        and len(head_payload) >= 3
        and torch.is_tensor(head_payload[2])
    ):
        return head_payload[2]
    return None


def _visibility_aware_records(
    pred_keypoints, keypoints, landmark_mask, metadata, visibility_logits=None
):
    pred = pred_keypoints.detach().float().cpu().numpy()
    target = keypoints.detach().float().cpu().numpy()
    mask = landmark_mask.detach().float().cpu().numpy() > 0.5
    logits = None
    if visibility_logits is not None:
        logits = visibility_logits.detach().float().cpu().numpy()

    records = []
    for index, (pred_i, target_i, mask_i, meta) in enumerate(
        zip(pred, target, mask, metadata)
    ):
        meta = meta if isinstance(meta, dict) else {}
        normalizer = _normalizer_for_masked_target(target_i, mask_i)
        if normalizer is None:
            continue
        point_errors = np.full((target_i.shape[0],), np.nan, dtype=np.float32)
        point_errors[mask_i] = np.linalg.norm(
            target_i[mask_i] - pred_i[mask_i], axis=1
        ) / float(normalizer)
        nme = float(np.nanmean(point_errors[mask_i]))

        visibility_target = _visibility_target_from_meta(meta, target_i.shape[0])
        visible_mask = mask_i & (visibility_target == 1)
        occluded_mask = mask_i & (visibility_target == 0)
        unknown_mask = mask_i & ~np.isin(visibility_target, (0, 1))

        record = record_for_sample(meta, nme)
        record["evaluation_head"] = str(meta.get("head_name") or "")
        record["visibility_target_source"] = str(
            meta.get("visibility_target_source") or ""
        )
        record["visible_landmark_count"] = int(np.sum(visible_mask))
        record["occluded_landmark_count"] = int(np.sum(occluded_mask))
        record["visibility_label_skipped_count"] = int(np.sum(unknown_mask))
        record["nme_visible"] = (
            float(np.nanmean(point_errors[visible_mask]))
            if np.any(visible_mask)
            else None
        )
        record["nme_occluded"] = (
            float(np.nanmean(point_errors[occluded_mask]))
            if np.any(occluded_mask)
            else None
        )
        record["visibility_targets"] = [
            int(value) if mask_i[pos] else -1
            for pos, value in enumerate(visibility_target.tolist())
        ]
        if (
            logits is not None
            and index < logits.shape[0]
            and logits.shape[1] == target_i.shape[0]
        ):
            record["visibility_scores"] = [
                float(value) if mask_i[pos] else float("nan")
                for pos, value in enumerate(logits[index].tolist())
            ]
        records.append(record)
    return records


def _masked_nme_list(pred_keypoints, keypoints, landmark_mask):
    values = _masked_nme_values(pred_keypoints, keypoints, landmark_mask)
    return values[np.isfinite(values)]


def _evaluate_landmark_model(model, test_dataloader, device, *, include_records=False, non_blocking=False):
    model.eval()
    records = []
    for batch_idx, batch in enumerate(tqdm(test_dataloader)):
        if isinstance(batch, dict) and "heads" in batch:
            data = batch["image"].to(device, non_blocking=non_blocking)
            stage_pred = model(data)[-1]
            for head_name, payload in batch["heads"].items():
                indices = payload["indices"].to(device, non_blocking=non_blocking)
                pred_keypoints, heatmap = _landmark_prediction_for_head(
                    stage_pred, head_name
                )
                pred_keypoints = pred_keypoints.index_select(0, indices)
                keypoints = payload["target"].to(device, non_blocking=non_blocking)
                landmark_mask = payload["landmark_mask"].to(device, non_blocking=non_blocking)
                visibility_logits = _visibility_logits_from_stage(stage_pred, head_name)
                if visibility_logits is not None:
                    visibility_logits = visibility_logits.index_select(0, indices)
                records.extend(
                    _visibility_aware_records(
                        pred_keypoints,
                        keypoints,
                        landmark_mask,
                        payload["metadata"],
                        visibility_logits=visibility_logits,
                    )
                )
            continue

        data, target, landmark_mask, metadata = _unpack_eval_batch(batch)
        data = data.to(device, non_blocking=non_blocking)
        keypoints = target.to(device, non_blocking=non_blocking)
        landmark_mask = landmark_mask.to(device, non_blocking=non_blocking)
        stage_pred = model(data)[-1]
        pred_keypoints, heatmap = _landmark_prediction_for_head(
            stage_pred, "landmarks_68"
        )
        records.extend(
            _visibility_aware_records(
                pred_keypoints,
                keypoints,
                landmark_mask,
                metadata,
                visibility_logits=_visibility_logits_from_stage(
                    stage_pred, "landmarks_68"
                ),
            )
        )
    report = build_slice_report(records)
    if include_records:
        report["records"] = records
    return report


def _records_from_report(report):
    records = report.get("records", [])
    return records if isinstance(records, list) else []


def _landmarks_68_prediction(stage_pred):
    return _landmark_prediction_for_head(stage_pred, "landmarks_68")


def _landmark_prediction_for_head(stage_pred, head_name):
    if isinstance(stage_pred, dict):
        if head_name not in stage_pred:
            available = ", ".join(
                sorted(key for key in stage_pred if not key.startswith("_"))
            )
            raise ValueError(
                f"model output does not include evaluation head '{head_name}' (available: {available})"
            )
        return stage_pred[head_name]
    if head_name != "landmarks_68":
        raise ValueError(
            f"legacy model output can only evaluate landmarks_68, not '{head_name}'"
        )
    return stage_pred


def _print_eval_summary(title, report):
    metrics = report["overall"]
    print(f"\n------------ {title} ------------")
    if metrics["sample_count"] == 0:
        print("NME %: nan")
        print("FR_{}% : nan".format(0.10))
        print("AUC_{}: nan".format(0.10))
        return
    print("NME %: {}".format(metrics["nme_percent"]))
    print("FR_{}% : {}".format(0.10, metrics["fr_percent"]))
    print("AUC_{}: {}".format(0.10, metrics["auc"]))


def _eval_report_json_path(args):
    if args.eval_report_json:
        return args.eval_report_json
    return os.path.join(args.ckpt_folder, "eval_report.json")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_folder", type=str, default="WFLW")
    parser.add_argument("--ckpt_folder", type=str, default="checkpoint")
    parser.add_argument("--batch_size", type=int, default="16")
    parser.add_argument("--num_workers", type=int, default="12")
    parser.add_argument("--epoch", type=int, default="500")
    parser.add_argument("--lr", type=float, default="0.0001")
    parser.add_argument("--local_rank", type=int, help="local rank, will passed by ddp")
    parser.add_argument("--local-rank", type=int, help="local rank, will passed by ddp")
    parser.add_argument("--sched_step", type=int, default="200")
    parser.add_argument("--save_n_epoch", type=int, default="100")
    parser.add_argument(
        "--preload",
        type=int,
        default="0",
        help="0 streams samples through DataLoader workers; 1 preloads the dataset in memory.",
    )

    parser.add_argument(
        "--pin-memory",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Pin DataLoader host memory so CUDA transfers can use non_blocking=True.",
    )
    parser.add_argument(
        "--persistent-workers",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep DataLoader workers alive for throughput. Worker RNG is seeded once; use --no-persistent-workers for epoch-reseeded workers or --restore-rng replay.",
    )
    parser.add_argument(
        "--prefetch-factor",
        type=int,
        default=2,
        help="DataLoader batches prefetched per worker. Ignored when num_workers == 0.",
    )
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--eval-num-workers", type=int, default=0)
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument("--full-eval-every", type=int, default=0)
    parser.add_argument("--eval-ema-every", "--eval-on-ema-every", dest="eval_ema_every", type=int, default=1)
    parser.add_argument("--eval-max-samples", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument(
        "--save-last-checkpoint",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write <ckpt_folder>/last_checkpoint.pt after each epoch.",
    )
    parser.add_argument(
        "--save-legacy-epoch-state-dict",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Also write legacy epoch_N model state-dict files at --save_n_epoch intervals.",
    )
    parser.add_argument(
        "--runtime-metrics-jsonl",
        type=str,
        default="",
        help="Optional runtime metrics JSONL path. Defaults to <ckpt_folder>/runtime_metrics.jsonl.",
    )
    parser.add_argument(
        "--restore-rng",
        action="store_true",
        help="Restore RNG state from full checkpoints. For exact replay this forces --no-persistent-workers so workers are re-seeded per epoch.",
    )
    parser.add_argument(
        "--allow-incompatible-resume",
        action="store_true",
        help="Allow loading a full checkpoint even when manifest/config compatibility metadata differs.",
    )
    parser.add_argument("--hw", type=float, default="10")
    parser.add_argument("--locw", type=float, default="1")
    parser.add_argument("--nstack", type=int, default="8")
    parser.add_argument("--heatmap_size", type=int, default="32")
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument("--max_depth", type=int, default="256")
    parser.add_argument("--mul", type=float, default="1.2")
    parser.add_argument(
        "--lmk_num",
        type=int,
        default="68",
        help="fallback landmark count for schema-aware manifest aliases",
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default="",
        help="schema-aware landmark manifest for train/test",
    )
    parser.add_argument(
        "--train_manifest",
        type=str,
        default="",
        help="schema-aware train manifest",
    )
    parser.add_argument(
        "--test_manifest",
        type=str,
        default="",
        help="schema-aware test manifest",
    )
    parser.add_argument("--eval-mode", choices=EVAL_MODES, default="random_hash")
    parser.add_argument(
        "--split-policy", choices=SPLIT_POLICIES, default="declared_or_random_hash"
    )
    parser.add_argument(
        "--respect-declared-splits",
        action="store_true",
        help="Alias for --split-policy declared.",
    )
    parser.add_argument(
        "--ignore-declared-splits",
        action="store_true",
        help="Alias for --split-policy random_hash.",
    )
    parser.add_argument(
        "--heldout-dataset",
        action="append",
        default=[],
        help="Dataset label to hold out. by_dataset accepts one or more; leave_one_dataset_out requires exactly one.",
    )
    parser.add_argument(
        "--eval-report-json",
        type=str,
        default="",
        help="Evaluation JSON path. Defaults to <ckpt_folder>/eval_report.json",
    )
    parser.add_argument(
        "--eval-report-csv", type=str, default="", help="Optional evaluation CSV path"
    )
    parser.add_argument(
        "--eval-records-jsonl",
        type=str,
        default="",
        help="Optional per-sample evaluation records JSONL path",
    )
    parser.add_argument(
        "--eval-records-csv",
        type=str,
        default="",
        help="Optional per-sample evaluation records CSV path",
    )
    parser.add_argument(
        "--schema-aware-training",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="For schema-aware manifest aliases, train schema-specific heads from mixed-schema manifests.",
    )
    parser.add_argument("--schema-consistency-weight", type=float, default=0.05)
    parser.add_argument("--domain-balanced-sampling", action="store_true")
    parser.add_argument(
        "--bucket-targets",
        default="anchor=0.25,occlusion=0.25,profile=0.25,profile_occlusion=0.25",
        help="Comma-separated hard bucket target weights for domain-balanced sampling.",
    )
    parser.add_argument(
        "--dataset-targets", default="", help="Comma-separated dataset target weights."
    )
    parser.add_argument(
        "--schema-targets", default="", help="Comma-separated schema target weights."
    )
    parser.add_argument(
        "--auxiliary-heads",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable optional pose/quality/visibility auxiliary heads for schema-aware manifest training.",
    )
    parser.add_argument("--auxiliary-loss-weight", type=float, default=0.1)
    parser.add_argument("--data_name", type=str, default="WFLW")
    parser.add_argument("--seed", type=int, default="0")
    parser.add_argument(
        "--deterministic",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable deterministic cuDNN behavior. Default favors training throughput.",
    )
    parser.add_argument(
        "--find_unused_parameters",
        action="store_true",
        help="Enable only if the model forward pass can skip trainable parameters",
    )
    args = parser.parse_args()
    if args.respect_declared_splits and args.ignore_declared_splits:
        parser.error(
            "pass only one of --respect-declared-splits or --ignore-declared-splits"
        )
    if args.respect_declared_splits:
        args.split_policy = "declared"
    if args.ignore_declared_splits:
        args.split_policy = "random_hash"
    args = _normalize_runtime_args(args)
    setup_seed(args.seed, deterministic=args.deterministic)
    lmk_num = _landmark_count_for_dataset(args)
    if "LOCAL_RANK" in os.environ and os.environ["LOCAL_RANK"] is not None:
        print(os.environ["LOCAL_RANK"])
        args.local_rank = int(os.environ["LOCAL_RANK"])
    else:
        args.local_rank = 0
    torch.cuda.set_device(args.local_rank)
    dist.init_process_group("nccl", rank=args.local_rank, init_method="env://")
    device = torch.device("cuda", args.local_rank)

    with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        # if True:

        schema_aware_training = (
            _is_schema_aware_manifest_dataset(args.data_name)
            and args.schema_aware_training
        )
        train_dataset = _build_dataset(
            args,
            "train",
            aug=True,
            heatmap_size=args.heatmap_size,
            schema_aware_training=schema_aware_training,
        )
        print("----------------------len(train_dataset)", len(train_dataset))
        test_dataset = _build_dataset(
            args,
            "test",
            aug=False,
            heatmap_size=0,
            include_metadata=True,
            schema_aware_training=schema_aware_training,
        )
        if _is_schema_aware_manifest_dataset(args.data_name):
            validate_no_train_test_leakage(train_dataset.samples, test_dataset.samples)
        eval_dataset = _maybe_limit_eval_dataset(test_dataset, args.eval_max_samples, args.seed)
        test_dataloader = torch.utils.data.DataLoader(
            eval_dataset,
            batch_size=args.eval_batch_size,
            collate_fn=_eval_collate,
            **_dataloader_kwargs(args, eval_loader=True),
        )
        full_test_dataloader = test_dataloader
        if int(args.eval_max_samples or 0) > 0 and len(eval_dataset) < len(test_dataset):
            full_test_dataloader = torch.utils.data.DataLoader(
                test_dataset,
                batch_size=args.eval_batch_size,
                collate_fn=_eval_collate,
                **_dataloader_kwargs(args, eval_loader=True),
            )
        if args.domain_balanced_sampling and _is_schema_aware_manifest_dataset(
            args.data_name
        ):
            train_sampler = DomainBalancedBatchSampler(
                train_dataset.samples,
                bucket_targets=parse_target_spec(
                    args.bucket_targets, DEFAULT_BUCKET_TARGETS
                ),
                dataset_targets=parse_target_spec(args.dataset_targets),
                schema_targets=parse_target_spec(args.schema_targets),
                batch_size=args.batch_size,
                seed=args.seed,
                rank=dist.get_rank(),
                world_size=dist.get_world_size(),
            )
            train_dataloader = torch.utils.data.DataLoader(
                train_dataset,
                batch_sampler=train_sampler,
                collate_fn=_schema_aware_collate if schema_aware_training else None,
                **_dataloader_kwargs(args),
            )
        else:
            train_sampler = torch.utils.data.distributed.DistributedSampler(
                train_dataset
            )
            train_dataloader = torch.utils.data.DataLoader(
                train_dataset,
                batch_size=args.batch_size,
                sampler=train_sampler,
                collate_fn=_schema_aware_collate if schema_aware_training else None,
                **_dataloader_kwargs(args),
            )
        # net = NetAttnStage(
        #     args.lmk_num, Attn=lambda:SA2SA1_2(args.heatmap_size, args.max_depth), nstack=args.nstack, heatmap_size=args.heatmap_size, max_depth=args.max_depth
        # ).cuda()
        assert (
            args.heatmap_size == 8
            or args.heatmap_size == 16
            or args.heatmap_size == 32
            or args.heatmap_size == 64
        )
        win_size = 2
        if args.heatmap_size == 8:
            def backbone_net(max_depth):
                return HeadingNet([32, 64, 128, 128, max_depth])
            win_size = 1
        elif args.heatmap_size == 16:
            def backbone_net(max_depth):
                return HeadingNet([32, 64, 128, max_depth])
            win_size = 1
        if args.heatmap_size == 32:
            def backbone_net(max_depth):
                return HeadingNet([32, 64, max_depth])
            win_size = 2
        if args.heatmap_size == 64:
            def backbone_net(max_depth):
                return HeadingNet([32, max_depth])
            win_size = 2

        net = VitAttnStage(
            lmk_num=lmk_num,
            nstack=args.nstack,
            Attn=lambda: SA2SA1_2(args.heatmap_size, args.max_depth, win_size=win_size),
            # Attn=lambda: Hourglass(3, args.max_depth),
            # Attn = lambda :SelfAttention_block2(args.max_depth),
            # Attn = lambda :SelfAttention2_block(args.heatmap_size, args.max_depth,args.max_depth),
            # Attn = lambda :UNet([256, 256, 256]),
            # Attn=lambda: nn.Sequential(RCCAModule(256, 256, 256), RCCAModule(256, 256, 256)),
            heatmap_size=args.heatmap_size,
            max_depth=args.max_depth,
            backbone_net=backbone_net,
            schema_heads=DEFAULT_SCHEMA_HEADS if schema_aware_training else None,
            auxiliary_heads={
                name: len(labels) for name, labels in AUXILIARY_CLASS_NAMES.items()
            }
            if schema_aware_training and args.auxiliary_heads
            else None,
        ).cuda()
        # net = VitAttnStage(
        #     nstack=args.nstack,
        #     Attn=lambda: SA2SA1_2(args.heatmap_size, args.max_depth),
        #     heatmap_size=args.heatmap_size,
        #     max_depth=args.max_depth,
        # ).cuda()
        # net = UNetStage(
        #     lmk_num=lmk_num,
        #     nstack=args.nstack,
        #     heatmap_size=args.heatmap_size,
        #     max_depth=args.max_depth,
        #     feature_extractor=Vit
        # ).cuda()
        resume_checkpoint = None
        start_epoch = 0
        if args.resume != "":
            resume_checkpoint = _torch_load_training_checkpoint(args.resume, device)
            if isinstance(resume_checkpoint, dict) and "model" in resume_checkpoint:
                compat_errors = _checkpoint_compat_errors(resume_checkpoint, args)
                if compat_errors and not args.allow_incompatible_resume:
                    raise ValueError(
                        "refusing to resume incompatible checkpoint: "
                        + "; ".join(compat_errors)
                        + ". Pass --allow-incompatible-resume only if this is intentional."
                    )
                net.load_state_dict(resume_checkpoint["model"])
                start_epoch = int(
                    resume_checkpoint.get(
                        "next_epoch",
                        int(resume_checkpoint.get("epoch", -1)) + 1,
                    )
                )
            else:
                net.load_state_dict(resume_checkpoint)
        net = torch.nn.parallel.DistributedDataParallel(
            net,
            device_ids=[args.local_rank],
            find_unused_parameters=args.find_unused_parameters or schema_aware_training,
        )

        optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
        scheduler = StepLR(optimizer, args.sched_step, gamma=0.5)

        best_nme = 99999
        weights = [1 / math.pow(args.mul, i) for i in range(args.nstack)]
        weights.reverse()
        best_record = []

        # heatmap_loss_func = HeatMapLoss2
        heatmap_loss_func = AWingLoss()
        # vertex_loss_func = STARLoss_v2()
        ema = EMA(net.module, 0.99, 100, 10) if dist.get_rank() == 0 else None
        scaler = torch.amp.GradScaler("cuda")
        if isinstance(resume_checkpoint, dict) and "model" in resume_checkpoint:
            start_epoch, best_nme, best_record = _restore_training_checkpoint(
                resume_checkpoint,
                optimizer,
                scheduler,
                scaler,
                ema,
                best_nme,
                best_record,
                args,
            )
            if dist.get_rank() == 0:
                print(f"resumed training checkpoint from epoch {start_epoch}")
        if start_epoch >= args.epoch:
            if dist.get_rank() == 0:
                print(
                    f"resume checkpoint next_epoch={start_epoch} is >= requested epoch={args.epoch}; "
                    "training is already complete for this epoch target",
                    flush=True,
                )
            if dist.is_initialized():
                dist.barrier()
            return
        for epoch in range(start_epoch, args.epoch):
            n = 0
            net.train()
            if dist.get_rank() == 0:
                ema.train()
            if dist.get_rank() == 0:
                epoch_start_time = time.time()
                epoch_timing = _empty_epoch_timing()
                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats(device)
            train_sampler.set_epoch(epoch)
            _set_dataset_runtime_epoch(train_dataset, epoch, args)
            if args.persistent_workers and epoch == start_epoch and dist.get_rank() == 0:
                print(
                    "note: --persistent-workers seeds DataLoader workers once for throughput; "
                    "use --no-persistent-workers for epoch-reseeded worker RNG",
                    flush=True,
                )
            batch_fetch_start_time = time.time()
            for batch_idx, batch in enumerate(train_dataloader):
                if dist.get_rank() == 0:
                    _accumulate_timing(epoch_timing, "data_wait_seconds", batch_fetch_start_time)
                optimizer.zero_grad(set_to_none=True)
                schema_batch = isinstance(batch, dict)
                transfer_start_time = time.time()
                if schema_batch:
                    data, schema_heads, aux_labels = _unpack_train_batch(batch, device, non_blocking=args.pin_memory)
                else:
                    data, target, heatmap, sample_weight, landmark_mask = (
                        _unpack_train_batch(batch, device, non_blocking=args.pin_memory)
                    )
                if dist.get_rank() == 0:
                    _accumulate_timing(epoch_timing, "device_transfer_seconds", transfer_start_time)
                compute_start_time = time.time()
                loss = 0
                loss_loc = torch.tensor(0.0, device=device)
                loss_heatmap = torch.tensor(0.0, device=device)
                loss_aux = torch.tensor(0.0, device=device)
                # if True:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    pred_info = net(data)
                    for i in range(len(pred_info)):
                        if schema_batch:
                            stage_loss, stage_loc, stage_heatmap, stage_aux = (
                                _schema_head_loss(
                                    pred_info[i],
                                    schema_heads,
                                    aux_labels,
                                    heatmap_loss_func,
                                    args,
                                )
                            )
                            loss_loc = stage_loc
                            loss_heatmap = stage_heatmap
                            loss_aux = stage_aux
                            loss = loss + stage_loss * weights[i]
                        else:
                            pred_loc, pred_heatmap = pred_info[i]
                            B, C, H, W = pred_heatmap.shape
                            # loss_loc = vertex_loss_func(pred_heatmap, target)
                            loss_loc = (
                                _weighted_smooth_l1(
                                    pred_loc,
                                    target,
                                    sample_weight,
                                    landmark_mask,
                                    beta=0.001,
                                )
                                * args.locw
                            )
                            pred_prob = F.softmax(
                                pred_heatmap.reshape((B, C, -1)), dim=2
                            ).reshape((B, C, H, W))
                            loss_heatmap = (
                                heatmap_loss_func(
                                    pred_prob,
                                    heatmap,
                                    batch_weights=_heatmap_batch_weight(
                                        sample_weight, pred_heatmap, landmark_mask
                                    ),
                                )
                                * args.hw
                            )  # for awing loss
                            # loss_heatmap = heatmap_loss_func(pred_heatmap, heatmap) * args.hw
                            loss = loss + (loss_loc + loss_heatmap) * weights[i]
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                if dist.get_rank() == 0:
                    _accumulate_timing(epoch_timing, "forward_backward_update_seconds", compute_start_time)

                #                 loss.backward()
                #                 optimizer.step()

                if dist.get_rank() == 0:
                    ema.update_parameters(net.module)
                n += data.shape[0]
                if args.log_every > 0 and batch_idx % args.log_every == 0 and dist.get_rank() == 0:
                    mix_text = (
                        f" mix: {batch.get('mix')}"
                        if isinstance(batch, dict) and "mix" in batch
                        else ""
                    )
                    print(
                        f"train epoch {epoch} batch_idx {batch_idx} rank {dist.get_rank()}  {n}/{len(train_dataset)} loss: {loss.item()} loss_loc: {loss_loc.item()} loss_heatmap: {loss_heatmap.item()} loss_aux: {loss_aux.item()}{mix_text}"
                    )
                batch_fetch_start_time = time.time()

            if (
                args.save_legacy_epoch_state_dict
                and dist.get_rank() == 0
                and args.save_n_epoch > 0
                and (epoch + 1) % args.save_n_epoch == 0
            ):
                if not os.path.exists(args.ckpt_folder):
                    os.mkdir(args.ckpt_folder)
                torch.save(
                    net.module.state_dict(),
                    os.path.join(args.ckpt_folder, ("epoch_%d") % (epoch,)),
                )

            scheduler.step()

            global_train_samples = int(n)
            if dist.is_initialized():
                sample_count = torch.tensor(
                    [global_train_samples], device=device, dtype=torch.long
                )
                dist.all_reduce(sample_count, op=dist.ReduceOp.SUM)
                global_train_samples = int(sample_count.item())

            # All ranks must participate so full checkpoints can resume each rank's
            # own RNG state instead of replaying rank 0 RNG everywhere.
            _set_checkpoint_rng_state_by_rank(_collect_rng_state_by_rank())

            if dist.get_rank() == 0:
                if args.save_n_epoch > 0 and (epoch + 1) % args.save_n_epoch == 0:
                    _save_training_checkpoint(
                        Path(args.ckpt_folder) / f"checkpoint_epoch_{epoch:04d}.pt",
                        net,
                        optimizer,
                        scheduler,
                        scaler,
                        ema,
                        epoch,
                        best_nme,
                        best_record,
                        args,
                    )
                duration = time.time() - epoch_start_time
                samples_per_second = float(global_train_samples) / max(duration, 1e-9)
                peak_memory_mb = _cuda_peak_memory_mb(device)
                print(
                    f"#epoch runtime epoch={epoch} duration={duration:.3f}s "
                    f"samples_per_second={samples_per_second:.3f} "
                    f"peak_cuda_memory_mb={peak_memory_mb}"
                )
                _append_runtime_metrics(
                    args,
                    {
                        "epoch": int(epoch),
                        "duration_seconds": round(duration, 6),
                        "train_samples": int(global_train_samples),
                        "rank0_train_samples": int(n),
                        "samples_per_second": samples_per_second,
                        "peak_cuda_memory_mb": peak_memory_mb,
                        "lr": float(scheduler.get_last_lr()[0]),
                    },
                )

                final_epoch = int(args.epoch) - 1
                should_eval_model = _should_run_interval(args.eval_every, epoch, final_epoch)
                run_full_eval = _should_run_interval(args.full_eval_every, epoch, final_epoch)
                limited_eval = eval_dataset is not test_dataset
                if limited_eval and should_eval_model and epoch >= final_epoch and not run_full_eval:
                    print(
                        "running full final eval so best_model and best_checkpoint are selected "
                        "from the full validation set"
                    )
                    run_full_eval = True
                eval_loader = full_test_dataloader if run_full_eval else test_dataloader
                eval_scope = "full" if (run_full_eval or not limited_eval) else "sampled"
                is_full_eval = eval_scope == "full"
                model_report = None
                ema_report = None

                if should_eval_model:
                    eval_start_time = time.time()
                    with torch.no_grad():
                        model_report = _evaluate_landmark_model(
                            net.module,
                            eval_loader,
                            device,
                            include_records=bool(
                                args.eval_records_jsonl or args.eval_records_csv
                            ),
                            non_blocking=args.pin_memory,
                        )
                    eval_seconds = time.time() - eval_start_time
                    epoch_timing["eval_seconds"] = float(epoch_timing.get("eval_seconds", 0.0)) + eval_seconds
                    nme = model_report["overall"]["nme"]
                    if is_full_eval and nme is not None and best_nme > nme:
                        best_nme = nme
                        best_record.append((epoch, best_nme * 100))
                        if not os.path.exists(args.ckpt_folder):
                            os.mkdir(args.ckpt_folder)
                        torch.save(
                            net.module.state_dict(),
                            os.path.join(args.ckpt_folder, "best_model"),
                        )
                        _save_training_checkpoint(
                            Path(args.ckpt_folder) / "best_checkpoint.pt",
                            net,
                            optimizer,
                            scheduler,
                            scaler,
                            ema,
                            epoch,
                            best_nme,
                            best_record,
                            args,
                        )
                    _print_eval_summary(f"test {eval_scope}", model_report)
                    print("BEST NME %: {}".format(best_nme * 100))
                    _append_runtime_metrics(
                        args,
                        {
                            "epoch": int(epoch),
                            "eval_scope": eval_scope,
                            "eval_seconds": round(eval_seconds, 6),
                            "eval_samples": int(model_report["overall"].get("sample_count") or 0),
                        },
                    )
                else:
                    print(f"skipping model eval at epoch {epoch}; --eval-every={args.eval_every}")

                should_eval_ema = (
                    ema is not None
                    and should_eval_model
                    and _should_run_interval(args.eval_ema_every, epoch, final_epoch)
                )
                if should_eval_ema:
                    ema_eval_start_time = time.time()
                    with torch.no_grad():
                        ema_report = _evaluate_landmark_model(
                            ema,
                            eval_loader,
                            device,
                            non_blocking=args.pin_memory,
                        )
                    epoch_timing["ema_eval_seconds"] = (
                        float(epoch_timing.get("ema_eval_seconds", 0.0))
                        + time.time()
                        - ema_eval_start_time
                    )
                    nme = ema_report["overall"]["nme"]
                    if is_full_eval and nme is not None and best_nme > nme:
                        best_nme = nme
                        best_record.append((epoch, "ema", best_nme * 100))
                        if not os.path.exists(args.ckpt_folder):
                            os.mkdir(args.ckpt_folder)
                        torch.save(
                            ema.model.state_dict(),
                            os.path.join(args.ckpt_folder, "best_model"),
                        )
                        _save_training_checkpoint(
                            Path(args.ckpt_folder) / "best_checkpoint.pt",
                            net,
                            optimizer,
                            scheduler,
                            scaler,
                            ema,
                            epoch,
                            best_nme,
                            best_record,
                            args,
                        )
                    _print_eval_summary(f"test ema {eval_scope}", ema_report)
                    print(best_record)
                elif ema is not None and should_eval_model:
                    print(f"skipping EMA eval at epoch {epoch}; --eval-ema-every={args.eval_ema_every}")

                if model_report is not None:
                    records = _records_from_report(model_report)
                    compact_model_report = {
                        key: value
                        for key, value in model_report.items()
                        if key != "records"
                    }
                    eval_payload = {
                        "epoch": epoch,
                        "eval_mode": args.eval_mode,
                        "eval_scope": eval_scope,
                        "heldout_datasets": list(args.heldout_dataset),
                        "model": compact_model_report,
                    }
                    if ema_report is not None:
                        eval_payload["ema"] = ema_report
                    write_eval_json(_eval_report_json_path(args), eval_payload)
                    if args.eval_report_csv:
                        write_eval_csv(args.eval_report_csv, eval_payload)
                    if args.eval_records_jsonl:
                        write_eval_records_jsonl(args.eval_records_jsonl, records)
                    if args.eval_records_csv:
                        write_eval_records_csv(args.eval_records_csv, records)
                # Save last checkpoint after eval so best_nme and best_record are current.
                if args.save_last_checkpoint:
                    checkpoint_start_time = time.time()
                    _save_training_checkpoint(
                        Path(args.ckpt_folder) / "last_checkpoint.pt",
                        net,
                        optimizer,
                        scheduler,
                        scaler,
                        ema,
                        epoch,
                        best_nme,
                        best_record,
                        args,
                    )
                    epoch_timing["checkpoint_seconds"] = (
                        float(epoch_timing.get("checkpoint_seconds", 0.0))
                        + time.time()
                        - checkpoint_start_time
                    )

                final_epoch_timing = dict(epoch_timing)
                final_epoch_timing["epoch_wall_seconds"] = time.time() - epoch_start_time
                final_epoch_timing["unattributed_seconds"] = max(
                    0.0,
                    final_epoch_timing["epoch_wall_seconds"]
                    - sum(
                        float(final_epoch_timing.get(key, 0.0))
                        for key in (
                            "data_wait_seconds",
                            "device_transfer_seconds",
                            "forward_backward_update_seconds",
                            "eval_seconds",
                            "ema_eval_seconds",
                            "checkpoint_seconds",
                        )
                    ),
                )
                _append_runtime_metrics(
                    args,
                    {
                        "event": "epoch_timing",
                        "epoch": int(epoch),
                        "timing": {
                            key: round(float(value), 6)
                            for key, value in sorted(final_epoch_timing.items())
                        },
                    },
                )

                if epoch + 1 >= args.epoch:
                    _write_training_complete_sentinel(
                        args,
                        epoch,
                        best_nme,
                        best_record,
                        global_train_samples,
                    )

            if dist.is_initialized():
                dist.barrier()


if __name__ == "__main__":
    main()
