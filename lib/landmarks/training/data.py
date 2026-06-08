
from __future__ import annotations

import torch
from torch.utils.data._utils.collate import default_collate

from DatasetAll import GetDataset
from lib.landmarks.core.manifest_aliases import is_schema_aware_manifest_dataset


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

def _is_schema_aware_manifest_dataset(data_name):
    return is_schema_aware_manifest_dataset(data_name)

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
