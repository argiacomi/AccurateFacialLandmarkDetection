import os.path
from glob import glob

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import transforms

from lib.landmarks.training.augmentation import GetAugTransform
from lib.landmarks.training.heatmap_targets import GenerateLineHeatmap


class LandmarkDataset(Dataset):
    def __init__(self, data_root, split, preload=True, aug=True, heatmap_size=0):
        super(LandmarkDataset, self).__init__()
        self.split = split

        self.annotation = self.loadannotation(os.path.join(data_root, split + ".txt"))
        self.image_files = glob(os.path.join(os.path.join(data_root, split, "*.jpg")))

        self.data_list = None
        if preload:
            self.data_list = self.loaditem_list(self.image_files, self.annotation)
        self.transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]
        )
        self.aug_transform = None
        if aug:
            self.aug_transform = GetAugTransform()

        self.heatmap_size = heatmap_size
        if self.heatmap_size > 0:
            self.generateHM = GenerateLineHeatmap(heatmap_size)

    def loadannotation(self, filename):
        annotation = {}
        with open(filename, "r") as fh:
            for line in fh.readlines():
                a = line.strip()
                if a != "":
                    items = a.split(" ")
                    key = items[0]
                    lmk = np.array([float(x) for x in items[1:]]).reshape((-1, 2)).astype(np.float32)
                    annotation[key] = lmk * 255
        return annotation

    def loaditem(self, image_file, annotation):
        img = cv2.imread(image_file)[:, :, [2, 1, 0]]
        assert img.shape[0] == 256 and img.shape[1] == 256
        image_name = image_file.split("/")[-1]
        if self.split == "train":
            image_name = image_name.replace("wflw_train_with_box", "wflw_train")
        return img, annotation[image_name]

    def loaditem_list(self, image_files, annotation):
        data_list = []
        for image_file in image_files:
            (img, lmk) = self.loaditem(image_file, annotation)
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
            img, lmk = self.loaditem(self.image_files[item], self.annotation)
        else:
            img, lmk = self.data_list[item]

        if self.aug_transform is not None:
            transformed = self.aug_transform(image=img, keypoints=lmk)
            transformed_image = transformed["image"]
            transformed_keypoints = transformed["keypoints"]
            transformed_keypoints = np.array(transformed_keypoints)
            img = transformed_image
            lmk = transformed_keypoints
        img, lmk = self.MakeLMKInsideImage(img, lmk)

        img = self.transform(img)
        lmk = torch.from_numpy(lmk / 255)
        # heatmap = self.encoder.generate_heatmap(lmk * 255)

        if self.heatmap_size > 0:
            heatmap = self.generateHM.Generate(lmk * (self.heatmap_size - 1))
            heatmap = torch.from_numpy(heatmap)
            return img, lmk, heatmap / torch.sum(heatmap, dim=-1, keepdim=True)
        else:
            return img, lmk


if __name__ == "__main__":

    heatmap_size = 32
    dataset = LandmarkDataset("WFLW", "test", False, aug=False, heatmap_size=heatmap_size)
    d = dataset[1]

    img, lmk0, heatmap = d

    img = torch.permute((img + 1) / 2.0, (1, 2, 0))
    img = (img * 255).numpy().astype(np.uint8)

    img = np.ascontiguousarray(img)
    lmk = lmk0 * 255

    for i in range(lmk.shape[0]):
        p = lmk[i]
        x, y = round(float(p[0])), round(float(p[1]))

        cv2.circle(img, (x, y), 2, (0, 225, 255), 2)
    img = Image.fromarray(img)
    img.show()

    row = torch.arange(heatmap_size)
    c = heatmap_size - 1
    row = row / c
    row = row.reshape((1, 1, heatmap_size))

    loc = (row * heatmap).sum(dim=-1)

    error = torch.nn.functional.l1_loss(loc, lmk / 255)
    print("#error", error)
