import os.path
import torch
from torch.utils.data import Dataset
from torchvision.transforms import transforms
import numpy as np
import cv2
from glob import glob
from PIL import Image
from ImageAugmentation import GetAugTransform
from DrawHeatmap import GenerateHeatmap
from RandomFlip import *
from sixdrepnet import SixDRepNet
import math


class LandmarkDataset(Dataset):
    def __init__(self, data_root, split, preload=True, aug=True, heatmap_size=0, perturbation=0):
        super(LandmarkDataset, self).__init__()
        self.split = split

        img_files, lmk_files = self.loaddata(data_root, split)
        model = SixDRepNet()
        self.head_angles = []
        for img_file in img_files:
            img = cv2.imread(img_file)
            pitch, yaw, roll = model.predict(img)
            self.head_angles.append(math.fabs(pitch) + math.fabs(yaw) + math.fabs(roll))

        self.annotation_files = lmk_files
        self.image_files = img_files

        self.data_list = None
        if preload:
            self.data_list = self.loaditem_list(self.image_files, self.annotation_files)
        self.transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]
        )
        self.aug_transform = None
        if aug:
            self.aug_transform = GetAugTransform()
        self.encoder = None
        self.heatmap_size = heatmap_size
        if heatmap_size > 0:
            self.generateHM = GenerateHeatmap(heatmap_size)

    def loaddata(self, root_dir, split):
        img_files = []
        npy_files = []
        if split == "train":
            dir = os.path.join(root_dir, "train_data")
            D300W_tmp_train = ["afw_processed", "hellen_trainset", "lfpw_trainset"]
            D300W_tmp2_train = ["afw", "helen/trainset", "lfpw/trainset"]
            for sub_name in D300W_tmp2_train:
                sub_folder = os.path.join(dir, sub_name)
                sub_img_files = glob(os.path.join(sub_folder, "*.jpg")) + glob(os.path.join(sub_folder, "*.png"))
                for img_file in sub_img_files:
                    npy_file = img_file[:-4] + ".npy"
                    img_files.append(img_file)
                    npy_files.append(npy_file)
        elif split == "test":
            dir = os.path.join(root_dir, "test_data")
            D300W_tmp_test = ["hellen_testset", "ibug_processed", "lfpw_testset"]
            D300W_tmp2_test = ["helen/testset", "ibug", "lfpw/testset"]
            for sub_name in D300W_tmp2_test:
                sub_folder = os.path.join(dir, sub_name)
                sub_img_files = glob(os.path.join(sub_folder, "*.jpg")) + glob(os.path.join(sub_folder, "*.png"))
                for img_file in sub_img_files:
                    npy_file = img_file[:-4] + ".npy"
                    img_files.append(img_file)
                    npy_files.append(npy_file)
        return img_files, npy_files

    def loaditem(self, image_file, annotation_file):
        img = cv2.imread(image_file)[:, :, [2, 1, 0]]
        assert img.shape[0] == 256 and img.shape[1] == 256
        lmk = np.load(annotation_file).astype(np.float32)
        return img, lmk

    def loaditem_list(self, image_files, annotation_files):
        data_list = []
        for i in range(len(image_files)):
            image_file = image_files[i]
            (img, lmk) = self.loaditem(image_file, annotation_files[i])
            data_list.append((img, lmk))
        return data_list

    def __len__(self):
        return len(self.image_files)

    def MakeLMKInsideImage(self, img, lmk):
        lt = np.min(lmk, axis=0)
        rb = np.max(lmk, axis=0)
        padding = 0
        margin = 5
        if lt[0] < margin:
            padding = margin - lt[0]
        if lt[1] < margin:
            padding = max(margin - lt[1], padding)
        if rb[0] > img.shape[1] - margin:
            padding = max(padding, rb[0] - img.shape[1] + margin)
        if rb[1] > img.shape[0] - margin:
            padding = max(padding, rb[1] - img.shape[0] + margin)
        if padding > 0:
            padding = int(round(padding))
            new_img = cv2.copyMakeBorder(img, padding, padding, padding, padding, cv2.BORDER_CONSTANT)
            lmk = lmk + padding
            lmk = lmk * img.shape[0] / new_img.shape[0]
            new_img = cv2.resize(new_img, (img.shape[0], img.shape[1]))

            return new_img, lmk
        return img, lmk

    def __getitem__(self, item):
        if self.data_list is None:
            img, lmk = self.loaditem(self.image_files[item], self.annotation_files[item])
            print(self.image_files[item])
        else:
            img, lmk = self.data_list[item]

        if self.aug_transform is not None:
            transformed = self.aug_transform(image=img, keypoints=lmk)
            transformed_image = transformed["image"]
            transformed_keypoints = transformed["keypoints"]
            transformed_keypoints = np.array(transformed_keypoints)
            img = transformed_image
            lmk = transformed_keypoints
            img, lmk = random_flip(img, lmk, flip_points("300W"), p=0.5)
        img, lmk = self.MakeLMKInsideImage(img, lmk)

        angle_w = 1.0
        if self.head_angles[item] > 35:
            angle_w = 0.75

        if self.heatmap_size > 0:
            img = self.transform(img)
            lmk = torch.from_numpy(lmk / 255)

            heatmap = self.generateHM.Generate(lmk * (self.heatmap_size - 1))
            heatmap = torch.from_numpy(heatmap)

            return img, lmk, heatmap / torch.sum(heatmap, dim=(1, 2), keepdim=True), angle_w

        img = self.transform(img)
        lmk = torch.from_numpy(lmk / 255)

        return img, lmk, angle_w


if __name__ == "__main__":

    heatmap_size = 32
    dataset = LandmarkDataset(
        "/media/mm/b14440f0-8231-4c05-aaa9-8a5dcfeca275/HumanFace/FaceLandmarkDataset/300W/300w_tmp2",
        "train",
        False,
        aug=False,
        heatmap_size=heatmap_size,
    )
    print(len(dataset))

    d = dataset[100]

    img, lmk0, heatmap = d
    # from loss import STARLoss
    #
    # loss_func = STARLoss()
    # loss = loss_func(heatmap[None], lmk0[None])

    img = torch.permute((img + 1) / 2.0, (1, 2, 0))
    img = (img * 255).numpy().astype(np.uint8)

    img = np.ascontiguousarray(img)
    lmk = lmk0 * 255

    for i in range(lmk.shape[0]):
        p = lmk[i]
        x, y = round(float(p[0])), round(float(p[1]))
        cv2.circle(img, (x, y), 1, (0, 225, 255), -1)
        cv2.putText(img, str(i), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.2, (255, 0, 0), 1, cv2.LINE_AA)
    img = Image.fromarray(img)
    img.show()
    # print(lmk)

    row, col = torch.meshgrid(torch.arange(heatmap_size), torch.arange(heatmap_size))
    c = heatmap_size - 1
    row = row / c
    col = col / c
    yy_loc, xx_loc = row.reshape((1, heatmap_size, heatmap_size)), col.reshape((1, heatmap_size, heatmap_size))

    xx = (xx_loc * heatmap).sum(dim=[1, 2])
    yy = (yy_loc * heatmap).sum(dim=[1, 2])
    loc = torch.stack([xx, yy], dim=1)
    error = torch.nn.functional.l1_loss(loc, lmk / 255)
    print("#error", error)
    # heatmap = encoder.generate_heatmap(lmk)
    print(torch.min((heatmap)), torch.max((heatmap)))

    # print(torch.sum(heatmap, dim=[1, 2]))
    heatmap = torch.sum(heatmap, dim=0).unsqueeze(-1).repeat((1, 1, 3))
    heatmap = (torch.clip(heatmap, 0, 1) * 500).numpy().astype(np.uint8)
    heatmap = Image.fromarray(heatmap)
    heatmap.show()
