import argparse
import os
import torch
import torch.utils.data.distributed
import torch.distributed as dist
from DatasetAll import GetDataset
from torch.optim.lr_scheduler import StepLR
from Net import VitAttnStage, HeadingNet
import torch.nn as nn
from Hourglass import Hourglass
# from Vit import Vit
from Attention import SA2SA1_2
# from Attention import  SA2SA1_twins
# from UNet2 import UNet
import torch.nn.functional as F
import time
from tqdm import tqdm
from loss_function import calc_nme, compute_fr_and_auc
import numpy as np
from loss import AWingLoss
from EMA import EMA
import math
from torch.nn.attention import sdpa_kernel, SDPBackend
import random


# from torch.cuda.amp import autocast as autocast


FS68_DATASET_NAME = "FS68Manifest"


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


def _build_dataset(args, split, aug, heatmap_size=0):
    manifest_path = _manifest_for_split(args, split) if args.data_name == FS68_DATASET_NAME else ""
    return GetDataset(
        args.data_name,
        args.root_folder,
        split,
        preload=args.preload != 0,
        aug=aug,
        heatmap_size=heatmap_size,
        manifest_path=manifest_path,
    )


def _unpack_train_batch(batch, device):
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


def _weighted_smooth_l1(pred_loc, target, sample_weight, landmark_mask, beta=0.001):
    per_point = F.smooth_l1_loss(pred_loc, target, beta=beta, reduction="none").mean(dim=2)
    landmark_mask = landmark_mask.to(per_point.device).float()
    per_sample = (per_point * landmark_mask).sum(dim=1) / landmark_mask.sum(dim=1).clamp_min(1.0)
    if sample_weight is not None:
        return (per_sample * sample_weight).mean()
    return per_sample.mean()


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


def _unpack_eval_batch(batch):
    data = batch[0]
    target = batch[1]
    if len(batch) >= 3:
        landmark_mask = batch[2]
    else:
        landmark_mask = torch.ones(target.shape[:2], dtype=torch.float32)
    return data, target, landmark_mask


def _masked_nme_list(pred_keypoints, keypoints, landmark_mask):
    pred = pred_keypoints.detach().float().cpu().numpy()
    target = keypoints.detach().float().cpu().numpy()
    mask = landmark_mask.detach().float().cpu().numpy() > 0.5

    values = []
    for pred_i, target_i, mask_i in zip(pred, target, mask):
        if mask_i.sum() <= 0:
            continue

        valid = target_i[mask_i]
        if valid.shape[0] <= 1:
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
            continue

        dist = np.linalg.norm(target_i[mask_i] - pred_i[mask_i], axis=1)
        values.append(float(dist.mean() / normalizer))

    return np.asarray(values, dtype=np.float32)


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

        train_dataset = _build_dataset(args, "train", aug=True, heatmap_size=args.heatmap_size)
        print('----------------------len(train_dataset)', len(train_dataset))
        test_dataset = _build_dataset(args, "test", aug=False, heatmap_size=0)
        test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=8)
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, sampler=train_sampler, num_workers=args.num_workers
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
            backbone_net=backbone_net
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
        net = torch.nn.parallel.DistributedDataParallel(net, device_ids=[args.local_rank], find_unused_parameters=args.find_unused_parameters)

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
                data, target, heatmap, sample_weight, landmark_mask = _unpack_train_batch(batch, device)
                loss = 0
                loss_loc = torch.tensor(0.0, device=device)
                loss_heatmap = torch.tensor(0.0, device=device)
                # if True:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    pred_info = net(data)
                    for i in range(len(pred_info)):
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
                    print(
                        f"train epoch {epoch} batch_idx {batch_idx} rank {dist.get_rank()}  {n}/{len(train_dataset)} loss: {loss.item()} loss_loc: {loss_loc.item()} loss_heatmap: {loss_heatmap.item()}"
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
                    net.eval()
                    SME = 0.0
                    IONs = None
                    for batch_idx, batch in enumerate(tqdm(test_dataloader)):
                        data, target, landmark_mask = _unpack_eval_batch(batch)
                        data = data.to(device)
                        keypoints = target.to(device)
                        landmark_mask = landmark_mask.to(device)
                        pred_keypoints, heatmap = net(data)[-1]
                        ion_list = _masked_nme_list(pred_keypoints, keypoints, landmark_mask)
                        IONs = np.concatenate((IONs, ion_list), 0) if IONs is not None else ion_list

                    nme, fr, auc = compute_fr_and_auc(IONs, thres=0.10, step=0.0001)
                    if best_nme > nme:
                        best_nme = nme
                        if not os.path.exists(args.ckpt_folder):
                            os.mkdir(args.ckpt_folder)
                        torch.save(net.module.state_dict(), os.path.join(args.ckpt_folder, "best_model"))
                        best_record.append((epoch, best_nme * 100))
                    print(f"\n------------ test ------------")
                    print("NME %: {}".format(nme * 100))
                    print("FR_{}% : {}".format(0.10, fr * 100))
                    print("AUC_{}: {}".format(0.10, auc))
                    print("BEST NME %: {}".format(best_nme * 100))

                with torch.no_grad():
                    ema.eval()
                    SME = 0.0
                    IONs = None
                    for batch_idx, batch in enumerate(tqdm(test_dataloader)):
                        data, target, landmark_mask = _unpack_eval_batch(batch)
                        data = data.to(device)
                        keypoints = target.to(device)
                        landmark_mask = landmark_mask.to(device)
                        pred_keypoints, heatmap = ema(data)[-1]
                        ion_list = _masked_nme_list(pred_keypoints, keypoints, landmark_mask)
                        IONs = np.concatenate((IONs, ion_list), 0) if IONs is not None else ion_list

                    nme, fr, auc = compute_fr_and_auc(IONs, thres=0.10, step=0.0001)
                    if best_nme > nme:
                        best_nme = nme
                        if not os.path.exists(args.ckpt_folder):
                            os.mkdir(args.ckpt_folder)
                        torch.save(ema.model.state_dict(), os.path.join(args.ckpt_folder, "best_model"))
                        best_record.append((epoch, "ema", best_nme * 100))
                    print(f"\n------------ test ema------------")
                    print("NME %: {}".format(nme * 100))
                    print("FR_{}% : {}".format(0.10, fr * 100))
                    print("AUC_{}: {}".format(0.10, auc))
                    # print("BEST NME %: {}".format(best_nme * 100))
                    print(best_record)


if __name__ == "__main__":
    main()
