import argparse
import os

import numpy as np
import torch
from tqdm import tqdm

from lib.landmarks.datasets.registry import GetDataset
from lib.landmarks.models.attention import SA2SA1_2
from lib.landmarks.models.cdvit import HeadingNet, VitAttnStage
from lib.landmarks.training.loss_function import calc_nme, compute_fr_and_auc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=str)
    parser.add_argument("--root_folder", type=str, default="WFLW")
    parser.add_argument("--data_name", type=str, default="WFLW", help="WFLW, COFW, 300W")
    parser.add_argument("--max_depth", type=int, default="256")
    args = parser.parse_args()
    if args.data_name == "WFLW":
        lmk_num = 98
    elif args.data_name == "COFW":
        lmk_num = 29
    elif args.data_name == "300W":
        lmk_num = 68
    else:
        assert 0 == 1

    test_sets = []
    for sub_name in os.listdir(args.root_folder):
        sub_folder = os.path.join(args.root_folder, sub_name)
        if os.path.isdir(sub_folder) and sub_name.startswith("test"):
            test_sets.append(sub_name)
    print(test_sets)
    net = (
        VitAttnStage(
            lmk_num=lmk_num,
            nstack=8,
            Attn=lambda: SA2SA1_2(32, args.max_depth),
            # Attn=lambda: SelfAttention_block2(256),
            # Attn=lambda: SelfAttention2_block(32, 256, 256),
            heatmap_size=32,
            max_depth=args.max_depth,
            backbone_net=lambda max_depth: HeadingNet([32, 64, max_depth]),
        )
        .cuda()
        .eval()
        .requires_grad_(False)
    )
    ckpt = torch.load(args.checkpoint)
    net.load_state_dict(ckpt)
    for test_set in test_sets:
        test_dataset = GetDataset(args.data_name, args.root_folder, test_set, aug=False)
        test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=8)
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
            print(f"\n------------ {test_set} ------------")
            print("NME %: {:.3f}".format(nme * 100))
            print("FR_{}% : {:.3f}".format(0.10, fr * 100))
            print("AUC_{}: {:.3f}".format(0.10, auc))
            print("{:.3f},{:.3f},{:.3f}".format(nme * 100, fr * 100, auc))


if __name__ == "__main__":
    main()