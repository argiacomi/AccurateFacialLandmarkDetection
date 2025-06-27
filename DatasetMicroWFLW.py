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


class LandmarkDataset(Dataset):
    def __init__(self, data_root, split, mst_root, num, aug=True, heatmap_size=0):
        super(LandmarkDataset, self).__init__()
        self.split = split

        self.annotation = self.loadannotation(os.path.join(data_root, split + ".txt"))
        self.image_files = glob(os.path.join(os.path.join(data_root, split, "*.jpg")))

        mst_images = []
        for i in range(num):
            mst_images.append(os.path.join(mst_root, "%06d.png" % (i,)))
        self.data_type = [0] * len(self.image_files) + [1] * len(mst_images)
        self.image_files = self.image_files + mst_images

        self.transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]
        )
        self.aug_transform = None
        if aug:
            self.aug_transform = GetAugTransform()

        self.heatmap_size = heatmap_size
        if self.heatmap_size > 0:
            self.generateHM = GenerateHeatmap(self.heatmap_size)

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

    def loaditem(self, image_file, annotation, data_type=1):
        if data_type == 0:
            img = cv2.imread(image_file)[:, :, [2, 1, 0]]
            image_name = image_file.split("/")[-1]
            if self.split == "train":
                image_name = image_name.replace("wflw_train_with_box", "wflw_train")
            return img, annotation[image_name]
        else:
            img = cv2.imread(image_file)[:, :, [2, 1, 0]]
            img = cv2.resize(img, (256, 256))
            lmk_file = image_file.replace(".png", "_ldmks.txt")
            lmk = np.loadtxt(lmk_file) / 2
            return img, lmk.astype(np.float32)

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
        img, lmk = self.loaditem(self.image_files[item], self.annotation, self.data_type[item])

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

        if len(lmk) < 98:
            lmk = torch.cat([lmk, torch.zeros((28, 2)) + 0.5], dim=0)

        if self.heatmap_size <= 0:
            return img, lmk, self.data_type[item]
        else:
            hm = self.generateHM.Generate(lmk * (self.heatmap_size - 1))
            hm = torch.from_numpy(hm)
            hm = hm / torch.sum(hm, dim=[1, 2], keepdim=True)
            return img, lmk, hm, self.data_type[item]


if __name__ == "__main__":

    heatmap_size = 64
    dataset = LandmarkDataset("WFLW", "test", "/home/mm/Downloads/dataset_100", 100, True, heatmap_size)
    d = dataset[-2]

    img, lmk, hm, data_type = d
    print(data_type)
    print("#lmk", lmk.shape, hm.shape)

    img = torch.permute((img + 1) / 2.0, (1, 2, 0))
    img = (img * 255).numpy().astype(np.uint8)

    img = np.ascontiguousarray(img)
    lmk = lmk * 255

    for i in range(lmk.shape[0]):
        p = lmk[i]
        x, y = round(float(p[0])), round(float(p[1]))

        cv2.circle(img, (x, y), 1, (0, 225, 255), 2)

    img = Image.fromarray(img)
    img.show()

    hm = torch.sum(hm[:70], dim=0)
    hm = torch.stack([hm, hm, hm], dim=2)
    hm = torch.clip(hm * 200, 0, 255).numpy().astype(np.uint8)
    hm = Image.fromarray(hm)
    hm.show()
