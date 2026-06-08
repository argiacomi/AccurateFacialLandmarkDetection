import argparse
import os.path

import cv2
import numpy as np
from ledq_helpers import *
from torchvision.transforms import transforms
from tqdm import tqdm

from lib.landmarks.models.attention import SA2SA1_2
from lib.landmarks.models.cdvit import HeadingNet, VitAttnStageDenseConn
from lib.landmarks.training.loss_function import video_NME_NMJ


def GetNet(ckpt="saved_ckpt/heading_net/bk_heading_sa2sa1_2_nstack8"):
    net = (
        VitAttnStageDenseConn(
            lmk_num=98,
            nstack=8,
            Attn=lambda: SA2SA1_2(32, 256),
            # Attn=lambda: SelfAttention_block2(256),
            # Attn=lambda: SelfAttention2_block(32, 256, 256),
            heatmap_size=32,
            max_depth=256,
            backbone_net=lambda max_depth: HeadingNet([32, 64, max_depth]),
        )
        .cuda()
        .eval()
        .requires_grad_(False)
    )
    ckpt = torch.load(ckpt)
    net.load_state_dict(ckpt)

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    return net, transform


def GetVideList(filename: str):
    res = []
    with open(filename, "r") as fh:
        for line in fh.readlines():
            name = line.strip()
            if name != "":
                res.append(name)
    return res


def main():
    parser = argparse.ArgumentParser("")
    parser.add_argument("data_folder", type=str)
    parser.add_argument("ckpt", type=str)
    parser.add_argument("--mode", type=str, default="easy")

    args = parser.parse_args()

    video_dir = os.path.join(args.data_folder, "videos")
    bbox_dir = os.path.join(args.data_folder, "bboxes")
    lmk_dir = os.path.join(args.data_folder, "landmarks")
    # video_names = [name.split("/")[-1] for name in glob(os.path.join(video_dir, "*.mp4"))]

    easy_name_file = os.path.join(args.data_folder, "easy_video_IDs.txt")
    hard_name_file = os.path.join(args.data_folder, "hard_video_IDs.txt")
    if args.mode == "easy":
        video_names = [name + ".mp4" for name in GetVideList(easy_name_file)]
    elif args.mode == "hard":
        video_names = [name + ".mp4" for name in GetVideList(hard_name_file)]
    else:
        assert 0

    net, transform = GetNet(args.ckpt)

    nmes = []
    nmjs = []
    for video_name in tqdm(video_names, total=len(video_names)):

        # video_name = "0QY9cT3sR_I.mp4"
        video_capture = cv2.VideoCapture(os.path.join(video_dir, video_name))
        bboxes = np.load(os.path.join(bbox_dir, video_name[:-4] + ".npy"))
        oracle_kpts = np.load(os.path.join(lmk_dir, video_name[:-4] + ".npy"))
        video_FourCC = cv2.VideoWriter_fourcc(*"mp4v")
        video_out = None

        n_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
        video_frames = [video_capture.read()[1] for _ in range(n_frames)]
        video_capture.release()

        pred_lmks = []
        for frame_idx in range(n_frames):
            frame = video_frames[frame_idx]
            with torch.no_grad():
                transform_matrix = get_transform_from_bbox(bboxes[frame_idx], extra_scale=1.2, target_im_size=256)
                face_np = cv2.warpAffine(video_frames[frame_idx], transform_matrix, (256, 256), flags=cv2.INTER_LINEAR)

                face_torch = transform(face_np[:, :, [2, 1, 0]])[None].cuda()
                pred_info = net(face_torch)
                pred_lmk = pred_info[-1][0]
                pred_lmk = apply_affine_transform_to_kpts(
                    pred_lmk.cpu().numpy().squeeze() * 256, transform_matrix, inverse=True
                )
                pred_lmks.append(pred_lmk)
                # draw_landmark(pred_lmk, frame)
                # if video_out is None:
                #     video_out = cv2.VideoWriter(
                #         f"video_out/video_out.avi", video_FourCC, 24.0, (frame.shape[1], frame.shape[0])
                #     )
                # video_out.write(frame)
        nme, nmj = video_NME_NMJ(pred_lmks, oracle_kpts)
        nmes.append(nme)
        nmjs.append(nmj)
        # video_out.release()
    print(np.mean(nmes), np.mean(nmjs))


if __name__ == "__main__":
    main()
