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
from torch.backends.cuda import sdp_kernel, SDPBackend
import random


# from torch.cuda.amp import autocast as autocast



def setup_seed(seed=0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True





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
    # parser.add_argument("--lmk_num", type=int, default="98", help="WFLW-98,COFW-29")
    parser.add_argument("--data_name", type=str, default="WFLW")
    parser.add_argument("--seed", type=int, default="0")
    args = parser.parse_args()
    setup_seed(args.seed)
    if args.data_name == "WFLW":
        lmk_num = 98
    elif args.data_name == "COFW":
        lmk_num = 29
    elif args.data_name == "300W":
        lmk_num = 68
    else:
        assert 0 == 1
    if "LOCAL_RANK" in os.environ and os.environ["LOCAL_RANK"] is not None:
        print(os.environ["LOCAL_RANK"])
        args.local_rank = int(os.environ["LOCAL_RANK"])
    else:
        args.local_rank = 0
    torch.cuda.set_device(args.local_rank)
    dist.init_process_group("nccl", rank=args.local_rank, init_method="env://")

    backend_map = {
        SDPBackend.MATH: {"enable_math": True, "enable_flash": False, "enable_mem_efficient": False},
        SDPBackend.FLASH_ATTENTION: {"enable_math": False, "enable_flash": True, "enable_mem_efficient": False},
        SDPBackend.EFFICIENT_ATTENTION: {"enable_math": False, "enable_flash": False, "enable_mem_efficient": True},
    }
    with sdp_kernel(**backend_map[SDPBackend.FLASH_ATTENTION]):
        # if True:

        train_dataset = GetDataset(
            args.data_name, args.root_folder, "train", preload=args.preload != 0, heatmap_size=args.heatmap_size
        )
        print('----------------------len(train_dataset)', len(train_dataset))
        test_dataset = GetDataset(args.data_name, args.root_folder, "test", aug=False)
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
        net = torch.nn.parallel.DistributedDataParallel(net, device_ids=[args.local_rank], find_unused_parameters=True)

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
        scaler = torch.cuda.amp.GradScaler()
        for epoch in range(args.epoch):
            n = 0
            net.train()
            if dist.get_rank() == 0:
                ema.train()
            if dist.get_rank() == 0:
                epoch_start_time = time.time()
            train_sampler.set_epoch(epoch)
            for batch_idx, (data, target, heatmap) in enumerate(train_dataloader):
                optimizer.zero_grad()
                data = data.to(args.local_rank)
                target = target.to(args.local_rank).float()
                heatmap = heatmap.to(args.local_rank)
                loss = 0
                # if True:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    pred_info = net(data)
                    for i in range(len(pred_info)):
                        pred_loc, pred_heatmap = pred_info[i]
                        B, C, H, W = pred_heatmap.shape
                        # loss_loc = vertex_loss_func(pred_heatmap, target)
                        loss_loc = F.smooth_l1_loss(pred_loc, target, 0.001) * args.locw
                        loss_heatmap = heatmap_loss_func( F.softmax(pred_heatmap.reshape((B, C, -1)), dim=2).reshape((B, C, H, W)), heatmap) * args.hw  # for awing loss
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
                    for batch_idx, (data, target) in enumerate(tqdm(test_dataloader)):
                        data = data.to(args.local_rank)
                        keypoints = target.to(args.local_rank)
                        pred_keypoints, heatmap = net(data)[-1]
                        sum_ion, ion_list = calc_nme(pred_keypoints, keypoints, data_name=args.data_name)
                        SME += sum_ion
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
                    for batch_idx, (data, target) in enumerate(tqdm(test_dataloader)):
                        data = data.to(args.local_rank)
                        keypoints = target.to(args.local_rank)
                        pred_keypoints, heatmap = ema(data)[-1]
                        sum_ion, ion_list = calc_nme(pred_keypoints, keypoints, data_name=args.data_name)
                        SME += sum_ion
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
