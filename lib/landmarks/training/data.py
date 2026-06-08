
from __future__ import annotations

import torch
from torch.utils.data._utils.collate import default_collate

from lib.landmarks.core.manifest_aliases import is_schema_aware_manifest_dataset
from lib.landmarks.training.domain_balanced_sampler import sample_bucket, sample_dataset, sample_schema
from lib.landmarks.training.auxiliary import (
    AUXILIARY_CLASS_INDEX,
    AUXILIARY_CLASS_NAMES,
    resolve_auxiliary_label,
)

class LegacyBatchWithMix(tuple):
    """Tuple-compatible legacy batch with domain-mix diagnostics attached."""

    def __new__(cls, values, mix):
        obj = super().__new__(cls, values)
        obj.mix = mix
        return obj


def is_schema_aware_manifest_dataset_name(data_name):
    return is_schema_aware_manifest_dataset(data_name)

def landmark_count_for_dataset(args):
    if args.data_name == "WFLW":
        return 98
    if args.data_name == "COFW":
        return 29
    if args.data_name == "300W":
        return 68
    if is_schema_aware_manifest_dataset_name(args.data_name):
        return int(args.lmk_num)
    raise ValueError(f"unknown data_name: {args.data_name}")

def manifest_for_split(args, split):
    if split == "train":
        return args.train_manifest or args.manifest or args.root_folder
    if split == "test":
        return args.test_manifest or args.manifest or args.root_folder
    return args.manifest or args.root_folder

def build_dataset(
    args,
    split,
    aug,
    heatmap_size=0,
    include_metadata=False,
    schema_aware_training=False,
):
    manifest_path = (
        manifest_for_split(args, split)
        if is_schema_aware_manifest_dataset_name(args.data_name)
        else ""
    )
    from lib.landmarks.datasets.registry import GetDataset

    return GetDataset(
        args.data_name,
        args.root_folder,
        split,
        preload=args.preload != 0,
        aug=aug,
        heatmap_size=heatmap_size,
        manifest_path=manifest_path,
        eval_mode=args.eval_mode
        if is_schema_aware_manifest_dataset_name(args.data_name)
        else "random_hash",
        heldout_datasets=args.heldout_dataset
        if is_schema_aware_manifest_dataset_name(args.data_name)
        else None,
        include_metadata=include_metadata,
        schema_aware_training=schema_aware_training,
        split_policy=args.split_policy
        if is_schema_aware_manifest_dataset_name(args.data_name)
        else "declared_or_random_hash",
    )

def unpack_train_batch(batch, device, non_blocking=False):
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
            if "visibility_target" in payload:
                heads[head_name]["visibility_target"] = payload["visibility_target"].to(
                    device,
                    non_blocking=non_blocking,
                ).float()
            if "visibility_target_weight" in payload:
                heads[head_name]["visibility_target_weight"] = payload[
                    "visibility_target_weight"
                ].to(device, non_blocking=non_blocking).float()
            if "visibility_target_provenance" in payload:
                heads[head_name]["visibility_target_provenance"] = list(
                    payload["visibility_target_provenance"]
                )
            heads[head_name]["sample_weight"] = heads[head_name][
                "sample_weight"
            ] / heads[head_name]["sample_weight"].mean().clamp_min(1e-6)
        aux_labels = {
            task: labels.to(device, non_blocking=non_blocking)
            for task, labels in batch.get("aux_labels", {}).items()
        }
        return data, heads, aux_labels

    if len(batch) == 6:
        data, target, heatmap, sample_weight, landmark_mask, _metadata = batch
    elif len(batch) == 5:
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

def mix_for_samples(samples):
    mix = {"bucket": {}, "dataset": {}, "schema": {}}
    for sample in samples:
        if not isinstance(sample, dict):
            sample = {}
        merged = dict(sample)
        merged["metadata"] = sample
        for key, label in (
            ("bucket", sample_bucket(merged)),
            ("dataset", sample_dataset(merged)),
            ("schema", sample_schema(merged)),
        ):
            mix[key][label] = mix[key].get(label, 0) + 1
    return mix

def batch_mix(batch):
    if isinstance(batch, dict):
        return batch.get("mix")
    return getattr(batch, "mix", None)

def legacy_domain_balanced_collate(batch):
    if batch and isinstance(batch[0], tuple) and len(batch[0]) == 6:
        metadata = [item[5] if isinstance(item[5], dict) else {} for item in batch]
        values = [item[:5] for item in batch]
    else:
        metadata = []
        values = batch
    return LegacyBatchWithMix(default_collate(values), mix_for_samples(metadata))

def schema_aware_collate(batch):
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
                "visibility_target": [],
                "visibility_target_weight": [],
                "visibility_target_provenance": [],
                "metadata": [],
            },
        )
        grouped[head_name]["indices"].append(index)
        grouped[head_name]["target"].append(item["target"])
        grouped[head_name]["heatmap"].append(item["heatmap"])
        grouped[head_name]["landmark_mask"].append(item["landmark_mask"])
        grouped[head_name]["sample_weight"].append(item["sample_weight"])
        visibility_target = item.get(
            "visibility_target",
            torch.full((item["target"].shape[0],), -1.0, dtype=torch.float32),
        )
        grouped[head_name]["visibility_target"].append(visibility_target)
        grouped[head_name]["visibility_target_weight"].append(
            item.get("visibility_target_weight", torch.ones_like(visibility_target).float())
        )
        grouped[head_name]["visibility_target_provenance"].append(
            item.get("visibility_target_provenance", "")
        )
        grouped[head_name]["metadata"].append(item.get("metadata", {}))

    heads = {}
    for head_name, payload in grouped.items():
        heads[head_name] = {
            "indices": torch.as_tensor(payload["indices"], dtype=torch.long),
            "target": default_collate(payload["target"]),
            "heatmap": default_collate(payload["heatmap"]),
            "landmark_mask": default_collate(payload["landmark_mask"]),
            "sample_weight": default_collate(payload["sample_weight"]),
            "visibility_target": default_collate(payload["visibility_target"]),
            "visibility_target_weight": default_collate(payload["visibility_target_weight"]),
            "visibility_target_provenance": list(payload["visibility_target_provenance"]),
            "metadata": payload["metadata"],
        }

    mix = {"bucket": {}, "dataset": {}, "schema": {}}
    aux_labels = {name: [] for name in AUXILIARY_CLASS_NAMES}
    aux_provenance = {name: [] for name in AUXILIARY_CLASS_NAMES}
    for item in batch:
        metadata = item.get("metadata", {})
        sample = dict(metadata)
        sample["metadata"] = metadata
        sample["source_schema"] = item.get("schema") or metadata.get("source_schema")
        bucket = sample_bucket(sample)
        dataset = sample_dataset(sample)
        schema = sample_schema(sample)
        mix["bucket"][bucket] = mix["bucket"].get(bucket, 0) + 1
        mix["dataset"][dataset] = mix["dataset"].get(dataset, 0) + 1
        mix["schema"][schema] = mix["schema"].get(schema, 0) + 1
        for task in aux_labels:
            resolved = resolve_auxiliary_label(task, metadata, item)
            aux_labels[task].append(resolved.label)
            aux_provenance[task].append(resolved.provenance)

    return {
        "image": images,
        "heads": heads,
        "mix": mix,
        "aux_labels": {
            task: torch.as_tensor(values, dtype=torch.long)
            for task, values in aux_labels.items()
        },
        "aux_provenance": aux_provenance,
    }


def auxiliary_label(task, metadata, item):
    """Compatibility wrapper returning only the integer label.

    New training code should use resolve_auxiliary_label() so provenance is kept.
    """

    return resolve_auxiliary_label(task, metadata, item).label


# Public trainer data API.
__all__ = [
    "AUXILIARY_CLASS_NAMES",
    "AUXILIARY_CLASS_INDEX",
    "is_schema_aware_manifest_dataset_name",
    "landmark_count_for_dataset",
    "manifest_for_split",
    "build_dataset",
    "unpack_train_batch",
    "batch_mix",
    "legacy_domain_balanced_collate",
    "mix_for_samples",
    "schema_aware_collate",
    "auxiliary_label",
]

# Legacy private aliases kept for TrainHeatmapStageFP16.py and older tests/tools.
_is_schema_aware_manifest_dataset = is_schema_aware_manifest_dataset_name
_landmark_count_for_dataset = landmark_count_for_dataset
_manifest_for_split = manifest_for_split
_build_dataset = build_dataset
_unpack_train_batch = unpack_train_batch
_schema_aware_collate = schema_aware_collate
_auxiliary_label = auxiliary_label
