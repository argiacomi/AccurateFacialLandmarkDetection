import argparse
import os
import torch
from Dataset import LandmarkDataset
from Net import VitAttnStage, VitAttnStageResSkip
from Attention import SA2SA1_2, SelfAttention_block2, SelfAttention2_block
from tqdm import tqdm
from loss_function import calc_nme, compute_fr_and_auc
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=str)
    parser.add_argument("--root_folder", type=str, default="WFLW")
    parser.add_argument("--data_name", type=str, default="WFLW")
    parser.add_argument("--lmk_num", type=int, default="98")
    parser.add_argument("--max_depth", type=int, default="256")

    args = parser.parse_args()
    test_dataset = LandmarkDataset(args.root_folder, "test", aug=False, preload=False)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=8)


    
    net = (
        VitAttnStage(
            lmk_num=args.lmk_num,
            nstack=8,
            Attn=lambda: SA2SA1_2(32, args.max_depth),
            # Attn=lambda: SelfAttention2_block(32, 256, 256),
            heatmap_size=32,
            max_depth=args.max_depth,
        )
        .cuda()
        .eval()
        .requires_grad_(False)
    )
    ckpt = torch.load(args.checkpoint)
    net.load_state_dict(ckpt)

    with torch.no_grad():
        net.eval()
        SME = 0.0
        IONs = None

        for batch_idx, (data, target) in enumerate(tqdm(test_dataloader)):
            data = data.cuda()
            keypoints = target.cuda()

            pred_keypoints = net(data)[-1][0]

            sum_ion, ion_list = calc_nme(pred_keypoints, keypoints, args.data_name)
            SME += sum_ion
            IONs = np.concatenate((IONs, ion_list), 0) if IONs is not None else ion_list

        nme, fr, auc = compute_fr_and_auc(IONs, thres=0.10, step=0.0001)
        print(f"\n------------ test ------------")
        print("NME %: {}".format(nme * 100))
        print("FR_{}% : {}".format(0.10, fr * 100))
        print("AUC_{}: {}".format(0.10, auc))


if __name__ == "__main__":
    main()
