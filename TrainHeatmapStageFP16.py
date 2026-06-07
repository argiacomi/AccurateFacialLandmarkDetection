import argparse
import os
import torch
import torch.utils.data.distributed
import torch.distributed as dist
from DatasetAll import GetDataset
from torch.optim.lr_scheduler import StepLR
from Net import VitAttnStage, HeadingNet
import torch.nn as nn
from torch.utils.data._utils.collate import default_collate
from Hourglass import Hourglass
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
from lib.landmarks.core.schema import MAP_98_TO_68
from lib.landmarks.evaluation.split_safe import (
    EVAL_MODES,
    build_slice_report,
    slice_labels,
    validate_no_train_test_leakage,
    write_eval_csv,
    write_eval_json,
)
from lib.landmarks.training.domain_balanced_sampler import (
    DEFAULT_BUCKET_TARGETS,
    DomainBalancedBatchSampler,
    parse_target_spec,
)


# from torch.cuda.amp import autocast as autocast


FS68_DATASET_NAME = "FS68Manifest"
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


def setup_seed(seed=0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def _landmark_count_for_dataset(args):
    if args.data_name == "WFLW":
        return 98
    if args.data_name == "COFW":
        return 29
    if args.data_name == "300W":
        return 68
    if args.data_name == FS68_DATASET_NAME:
        return int(args.lmk_num)
    raise ValueError(f"unknown data_name: {args.data_name}")


def _manifest_for_split(args, split):
    if split == "train":
        return args.train_manifest or args.manifest or args.root_folder
    if split == "test":
        return args.test_manifest or args.manifest or args.root_folder
    return args.manifest or args.root_folder


def _build_dataset(args, split, aug, heatmap_size=0, include_metadata=False, schema_aware_training=False):
    manifest_path = _manifest_for_split(args, split) if args.data_name == FS68_DATASET_NAME else ""
    return GetDataset(
        args.data_name,
        args.root_folder,
        split,
        preload=args.preload != 0,
        aug=aug,
        heatmap_size=heatmap_size,
        manifest_path=manifest_path,
        eval_mode=args.eval_mode if args.data_name == FS68_DATASET_NAME else "random_hash",
        heldout_datasets=args.heldout_dataset if args.data_name == FS68_DATASET_NAME else None,
        include_metadata=include_metadata,
        schema_aware_training=schema_aware_training,
    )


def _unpack_train_batch(batch, device):
    if isinstance(batch, dict):
        data = batch["image"].to(device)
        heads = {}
        for head_name, payload in batch["heads"].items():
            heads[head_name] = {
                "indices": payload["indices"].to(device),
                "target": payload["target"].to(device).float(),
                "heatmap": payload["heatmap"].to(device).float(),
                "landmark_mask": payload["landmark_mask"].to(device).float(),
                "sample_weight": payload["sample_weight"].to(device).float(),
            }
            heads[head_name]["sample_weight"] = heads[head_name]["sample_weight"] / heads[head_name][
                "sample_weight"
            ].mean().clamp_min(1e-6)
        aux_labels = {
            task: labels.to(device)
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
        raise ValueError(f"expected train batch with 3, 4, or 5 items, got {len(batch)}")

    data = data.to(device)
    target = target.to(device).float()
    heatmap = heatmap.to(device)
    if sample_weight is not None:
        sample_weight = sample_weight.to(device).float()
        sample_weight = sample_weight / sample_weight.mean().clamp_min(1e-6)
    if landmark_mask is None:
        landmark_mask = torch.ones(target.shape[:2], device=device, dtype=torch.float32)
    else:
        landmark_mask = landmark_mask.to(device).float()
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
        bucket = str(metadata.get("hard_negative_bucket") or metadata.get("condition") or "unknown")
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
        "aux_labels": {task: torch.as_tensor(values, dtype=torch.long) for task, values in aux_labels.items()},
    }


def _auxiliary_label(task, metadata, item):
    attributes = metadata.get("attributes") if isinstance(metadata.get("attributes"), dict) else {}
    conditions = metadata.get("conditions") or ()
    if isinstance(conditions, str):
        conditions = (conditions,)
    condition_labels = {str(value).strip().lower().replace("-", "_") for value in conditions}
    condition = str(metadata.get("condition") or "").strip().lower().replace("-", "_")
    if condition:
        condition_labels.add(condition)

    label = None
    if task == "pose_bucket":
        raw = str(metadata.get("pose_bucket") or metadata.get("pose") or "").strip().lower().replace("-", "_")
        if raw in {"profile_left", "large_yaw_left", "left_profile"}:
            label = "profile_left"
        elif raw in {"profile_right", "large_yaw_right", "right_profile"}:
            label = "profile_right"
        elif raw in {"profile", "large_yaw", "1"} or attributes.get("pose"):
            label = "profile"
        elif raw in {"frontal", "normal", "0"} or "frontal" in condition_labels or "anchor" in condition_labels:
            label = "frontal"
    elif task == "occlusion":
        raw = metadata.get("occlusion", attributes.get("occlusion"))
        if raw is not None:
            label = "occlusion" if bool(raw) and str(raw).lower() not in {"0", "false", "none"} else "no_occlusion"
        elif any("occlusion" in value or "occlud" in value for value in condition_labels):
            label = "occlusion"
        else:
            label = "no_occlusion"
    elif task == "visibility":
        mask = item.get("landmark_mask")
        if mask is not None:
            label = "all_visible" if bool(torch.as_tensor(mask).float().min().item() > 0.5) else "partially_visible"
    elif task == "blur_quality":
        raw = metadata.get("blur", attributes.get("blur"))
        if raw is not None:
            label = "blurred" if bool(raw) and str(raw).lower() not in {"0", "false", "none"} else "clear"
    elif task == "illumination_quality":
        raw = metadata.get("illumination", attributes.get("illumination"))
        if raw is not None:
            label = "challenging" if bool(raw) and str(raw).lower() not in {"0", "false", "none"} else "normal"
    elif task == "profile_side":
        raw = str(metadata.get("profile_side") or metadata.get("side") or "").strip().lower()
        if raw in {"left", "right"}:
            label = raw
        elif any("left" in value for value in condition_labels):
            label = "left"
        elif any("right" in value for value in condition_labels):
            label = "right"
        elif not any(value in condition_labels for value in ("profile", "large_yaw", "profile_pose")):
            label = "not_profile"
    elif task == "landmark_confidence":
        weight = float(item.get("sample_weight", torch.tensor(1.0)).item())
        label = "low" if weight > 2.0 else "normal"

    if label is None:
        return -1
    return AUXILIARY_CLASS_INDEX[task].get(label, -1)


def _weighted_smooth_l1(pred_loc, target, sample_weight, landmark_mask, beta=0.001):
    per_point = F.smooth_l1_loss(pred_loc, target, beta=beta, reduction="none").mean(dim=2)
    landmark_mask = landmark_mask.to(per_point.device).float()
    per_sample = (per_point * landmark_mask).sum(dim=1) / landmark_mask.sum(dim=1).clamp_min(1.0)
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
        head_loc = _weighted_smooth_l1(pred_loc, target, sample_weight, landmark_mask, beta=0.001) * args.locw
        pred_prob = F.softmax(pred_heatmap.reshape((B, C, -1)), dim=2).reshape((B, C, H, W))
        head_heatmap = heatmap_loss_func(
            pred_prob,
            heatmap,
            batch_weights=_heatmap_batch_weight(sample_weight, pred_heatmap, landmark_mask),
        ) * args.hw
        loss = loss + head_loc + head_heatmap
        loss_loc = loss_loc + head_loc.detach()
        loss_heatmap = loss_heatmap + head_heatmap.detach()

    if args.schema_consistency_weight > 0 and "landmarks_98" in heads and "landmarks_68" in stage_pred:
        payload = heads["landmarks_98"]
        indices = payload["indices"]
        pred_98 = stage_pred["landmarks_98"][0].index_select(0, indices)
        pred_68 = stage_pred["landmarks_68"][0].index_select(0, indices)
        projected = pred_98[:, torch.as_tensor(MAP_98_TO_68, device=pred_98.device), :]
        loss = loss + float(args.schema_consistency_weight) * F.smooth_l1_loss(pred_68, projected.detach(), beta=0.001)

    aux_outputs = stage_pred.get("_aux", {}) if isinstance(stage_pred, dict) else {}
    if args.auxiliary_loss_weight > 0 and aux_outputs:
        for task, logits in aux_outputs.items():
            labels = aux_labels.get(task)
            if labels is None:
                continue
            valid = labels >= 0
            if bool(valid.any()):
                loss_aux = loss_aux + F.cross_entropy(logits[valid], labels[valid]) * float(args.auxiliary_loss_weight)
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
        weights = weights * sample_weight.to(pred_heatmap.device).to(pred_heatmap.dtype).reshape(-1, 1)
    return weights.reshape(pred_heatmap.shape[0], pred_heatmap.shape[1], 1, 1)


def _eval_collate(batch):
    if batch and len(batch[0]) >= 4 and isinstance(batch[0][3], dict):
        data = default_collate([item[0] for item in batch])
        target = default_collate([item[1] for item in batch])
        landmark_mask = default_collate([item[2] for item in batch])
        metadata = [item[3] for item in batch]
        return data, target, landmark_mask, metadata
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


def _masked_nme_list(pred_keypoints, keypoints, landmark_mask):
    values = _masked_nme_values(pred_keypoints, keypoints, landmark_mask)
    return values[np.isfinite(values)]


def _evaluate_landmark_model(model, test_dataloader, device):
    model.eval()
    records = []
    for batch_idx, batch in enumerate(tqdm(test_dataloader)):
        data, target, landmark_mask, metadata = _unpack_eval_batch(batch)
        data = data.to(device)
        keypoints = target.to(device)
        landmark_mask = landmark_mask.to(device)
        pred_keypoints, heatmap = _landmarks_68_prediction(model(data)[-1])
        nme_values = _masked_nme_values(pred_keypoints, keypoints, landmark_mask)
        for nme, meta in zip(nme_values, metadata):
            if not np.isfinite(float(nme)):
                continue
            meta = meta if isinstance(meta, dict) else {}
            record = {"nme": float(nme), **slice_labels(meta)}
            if meta.get("sample_id"):
                record["sample_id"] = str(meta["sample_id"])
            records.append(record)
    return build_slice_report(records)


def _landmarks_68_prediction(stage_pred):
    if isinstance(stage_pred, dict):
        return stage_pred["landmarks_68"]
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
    parser.add_argument("--preload", type=int, default="1")
    parser.add_argument("--hw", type=float, default="10")
    parser.add_argument("--locw", type=float, default="1")
    parser.add_argument("--nstack", type=int, default="8")
    parser.add_argument("--heatmap_size", type=int, default="32")
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument("--max_depth", type=int, default="256")
    parser.add_argument("--mul", type=float, default="1.2")
    parser.add_argument("--lmk_num", type=int, default="68", help="landmark count for FS68Manifest")
    parser.add_argument("--manifest", type=str, default="", help="faceswap-compatible manifest for FS68Manifest train/test")
    parser.add_argument("--train_manifest", type=str, default="", help="faceswap-compatible train manifest for FS68Manifest")
    parser.add_argument("--test_manifest", type=str, default="", help="faceswap-compatible test manifest for FS68Manifest")
    parser.add_argument("--eval-mode", choices=EVAL_MODES, default="random_hash")
    parser.add_argument(
        "--heldout-dataset",
        action="append",
        default=[],
        help="Dataset label to hold out for by_dataset or leave_one_dataset_out evaluation. May be repeated.",
    )
    parser.add_argument("--eval-report-json", type=str, default="", help="Evaluation JSON path. Defaults to <ckpt_folder>/eval_report.json")
    parser.add_argument("--eval-report-csv", type=str, default="", help="Optional evaluation CSV path")
    parser.add_argument(
        "--schema-aware-training",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="For FS68Manifest, train schema-specific 68/98/profile39 heads from mixed-schema manifests.",
    )
    parser.add_argument("--schema-consistency-weight", type=float, default=0.05)
    parser.add_argument("--domain-balanced-sampling", action="store_true")
    parser.add_argument(
        "--bucket-targets",
        default="anchor=0.25,occlusion=0.25,profile=0.25,profile_occlusion=0.25",
        help="Comma-separated hard bucket target weights for domain-balanced sampling.",
    )
    parser.add_argument("--dataset-targets", default="", help="Comma-separated dataset target weights.")
    parser.add_argument("--schema-targets", default="", help="Comma-separated schema target weights.")
    parser.add_argument(
        "--auxiliary-heads",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable optional pose/quality/visibility auxiliary heads for schema-aware FS68 training.",
    )
    parser.add_argument("--auxiliary-loss-weight", type=float, default=0.1)
    parser.add_argument("--data_name", type=str, default="WFLW")
    parser.add_argument("--seed", type=int, default="0")
    parser.add_argument("--find_unused_parameters", action="store_true", help="Enable only if the model forward pass can skip trainable parameters")
    args = parser.parse_args()
    setup_seed(args.seed)
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

        schema_aware_training = args.data_name == FS68_DATASET_NAME and args.schema_aware_training
        train_dataset = _build_dataset(
            args,
            "train",
            aug=True,
            heatmap_size=args.heatmap_size,
            schema_aware_training=schema_aware_training,
        )
        print('----------------------len(train_dataset)', len(train_dataset))
        test_dataset = _build_dataset(args, "test", aug=False, heatmap_size=0, include_metadata=True)
        if args.data_name == FS68_DATASET_NAME:
            validate_no_train_test_leakage(train_dataset.samples, test_dataset.samples)
        test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=8, collate_fn=_eval_collate)
        if args.domain_balanced_sampling and args.data_name == FS68_DATASET_NAME:
            train_sampler = DomainBalancedBatchSampler(
                train_dataset.samples,
                bucket_targets=parse_target_spec(args.bucket_targets, DEFAULT_BUCKET_TARGETS),
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
                num_workers=args.num_workers,
                collate_fn=_schema_aware_collate if schema_aware_training else None,
            )
        else:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
            train_dataloader = torch.utils.data.DataLoader(
                train_dataset,
                batch_size=args.batch_size,
                sampler=train_sampler,
                num_workers=args.num_workers,
                collate_fn=_schema_aware_collate if schema_aware_training else None,
            )
        # net = NetAttnStage(
        #     args.lmk_num, Attn=lambda:SA2SA1_2(args.heatmap_size, args.max_depth), nstack=args.nstack, heatmap_size=args.heatmap_size, max_depth=args.max_depth
        # ).cuda()
        assert args.heatmap_size==8 or args.heatmap_size==16 or args.heatmap_size==32 or args.heatmap_size==64
        win_size=2
        if args.heatmap_size==8:
            backbone_net=lambda max_depth: HeadingNet([32, 64,128,128, max_depth])
            win_size=1
        elif args.heatmap_size==16:
            backbone_net=lambda max_depth: HeadingNet([32, 64,128, max_depth])
            win_size=1
        if args.heatmap_size==32:
            backbone_net=lambda max_depth: HeadingNet([32, 64, max_depth])
            win_size=2
        if args.heatmap_size==64:
            backbone_net=lambda max_depth: HeadingNet([32,  max_depth])
            win_size=2
            
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
            schema_heads={"landmarks_68": 68, "landmarks_98": 98, "profile39": 39}
            if schema_aware_training
            else None,
            auxiliary_heads={name: len(labels) for name, labels in AUXILIARY_CLASS_NAMES.items()}
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
        if args.resume != "":
            ckpt = torch.load(args.resume)
            net.load_state_dict(ckpt)
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
        if dist.get_rank() == 0:
            ema = EMA(net.module, 0.99, 100, 10)
        scaler = torch.amp.GradScaler("cuda")
        for epoch in range(args.epoch):
            n = 0
            net.train()
            if dist.get_rank() == 0:
                ema.train()
            if dist.get_rank() == 0:
                epoch_start_time = time.time()
            train_sampler.set_epoch(epoch)
            for batch_idx, batch in enumerate(train_dataloader):
                optimizer.zero_grad()
                schema_batch = isinstance(batch, dict)
                if schema_batch:
                    data, schema_heads, aux_labels = _unpack_train_batch(batch, device)
                else:
                    data, target, heatmap, sample_weight, landmark_mask = _unpack_train_batch(batch, device)
                loss = 0
                loss_loc = torch.tensor(0.0, device=device)
                loss_heatmap = torch.tensor(0.0, device=device)
                loss_aux = torch.tensor(0.0, device=device)
                # if True:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    pred_info = net(data)
                    for i in range(len(pred_info)):
                        if schema_batch:
                            stage_loss, stage_loc, stage_heatmap, stage_aux = _schema_head_loss(
                                pred_info[i],
                                schema_heads,
                                aux_labels,
                                heatmap_loss_func,
                                args,
                            )
                            loss_loc = stage_loc
                            loss_heatmap = stage_heatmap
                            loss_aux = stage_aux
                            loss = loss + stage_loss * weights[i]
                        else:
                            pred_loc, pred_heatmap = pred_info[i]
                            B, C, H, W = pred_heatmap.shape
                            # loss_loc = vertex_loss_func(pred_heatmap, target)
                            loss_loc = _weighted_smooth_l1(pred_loc, target, sample_weight, landmark_mask, beta=0.001) * args.locw
                            pred_prob = F.softmax(pred_heatmap.reshape((B, C, -1)), dim=2).reshape((B, C, H, W))
                            loss_heatmap = heatmap_loss_func(
                                pred_prob,
                                heatmap,
                                batch_weights=_heatmap_batch_weight(sample_weight, pred_heatmap, landmark_mask),
                            ) * args.hw  # for awing loss
                            # loss_heatmap = heatmap_loss_func(pred_heatmap, heatmap) * args.hw
                            loss = loss + (loss_loc + loss_heatmap) * weights[i]
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

#                 loss.backward()
#                 optimizer.step()

                if dist.get_rank() == 0:
                    ema.update_parameters(net.module)
                n += data.shape[0]
                if batch_idx % 20 == 0 and dist.get_rank() == 0:
                    mix_text = f" mix: {batch.get('mix')}" if isinstance(batch, dict) and "mix" in batch else ""
                    print(
                        f"train epoch {epoch} batch_idx {batch_idx} rank {dist.get_rank()}  {n}/{len(train_dataset)} loss: {loss.item()} loss_loc: {loss_loc.item()} loss_heatmap: {loss_heatmap.item()} loss_aux: {loss_aux.item()}{mix_text}"
                    )

            if dist.get_rank() == 0 and (epoch + 1) % args.save_n_epoch == 0:
                if not os.path.exists(args.ckpt_folder):
                    os.mkdir(args.ckpt_folder)
                torch.save(net.module.state_dict(), os.path.join(args.ckpt_folder, ("epoch_%d") % (epoch,)))

            scheduler.step()

            if dist.get_rank() == 0:
                duration = time.time() - epoch_start_time
                print("#epoch duration", duration)
                with torch.no_grad():
                    model_report = _evaluate_landmark_model(net, test_dataloader, device)
                    nme = model_report["overall"]["nme"]
                    if nme is not None and best_nme > nme:
                        best_nme = nme
                        if not os.path.exists(args.ckpt_folder):
                            os.mkdir(args.ckpt_folder)
                        torch.save(net.module.state_dict(), os.path.join(args.ckpt_folder, "best_model"))
                        best_record.append((epoch, best_nme * 100))
                    _print_eval_summary("test", model_report)
                    print("BEST NME %: {}".format(best_nme * 100))

                with torch.no_grad():
                    ema_report = _evaluate_landmark_model(ema, test_dataloader, device)
                    nme = ema_report["overall"]["nme"]
                    if nme is not None and best_nme > nme:
                        best_nme = nme
                        if not os.path.exists(args.ckpt_folder):
                            os.mkdir(args.ckpt_folder)
                        torch.save(ema.model.state_dict(), os.path.join(args.ckpt_folder, "best_model"))
                        best_record.append((epoch, "ema", best_nme * 100))
                    _print_eval_summary("test ema", ema_report)
                    # print("BEST NME %: {}".format(best_nme * 100))
                    print(best_record)
                    eval_payload = {
                        "epoch": epoch,
                        "eval_mode": args.eval_mode,
                        "heldout_datasets": list(args.heldout_dataset),
                        "model": model_report,
                        "ema": ema_report,
                    }
                    write_eval_json(_eval_report_json_path(args), eval_payload)
                    if args.eval_report_csv:
                        write_eval_csv(args.eval_report_csv, eval_payload)


if __name__ == "__main__":
    main()
