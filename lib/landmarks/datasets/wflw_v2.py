import os.path
from glob import glob

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import transforms

from lib.landmarks.training.augmentation import GetAugTransform


class LandmarkDataset(Dataset):
    def __init__(self, data_root, split, preload=True, aug=True, perturbation=False):
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

        self.perturbation = perturbation

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

        edges = cv2.Canny(img, 50, 50)
        edges = torch.from_numpy(edges / 255).float()
        img = self.transform(img)
        lmk = torch.from_numpy(lmk / 255)

        if self.perturbation:
            sampled_num = 8
            perturbed_imgs = []
            perturbed_lmks = []
            img = torch.permute(img, (1, 2, 0)).numpy()
            ldmks = (lmk * 255).numpy()
            for i in range(sampled_num):
                angle = 20 * i / (sampled_num - 1) - 10
                rot = cv2.getRotationMatrix2D((128, 128), angle, 1)
                warped_img = cv2.warpAffine(img, rot, (256, 256))
                warped_lmks = np.transpose(rot[:2, :2] @ np.transpose(ldmks) + rot[:2, [2]])
                perturbed_imgs.append(torch.from_numpy(warped_img).permute((2, 0, 1)))
                perturbed_lmks.append((torch.from_numpy(warped_lmks) / 255))
            perturbed_imgs = torch.stack(perturbed_imgs, dim=0)
            perturbed_lmks = torch.stack(perturbed_lmks, dim=0)
            return (perturbed_imgs, perturbed_lmks)
        return torch.cat([img, edges.unsqueeze(0)], dim=0), lmk


if __name__ == "__main__":

    dataset = LandmarkDataset("WFLW", "test", False, True, False)
    d = dataset[1]

    img_edge, lmk = d
    img = img_edge[:3]
    edges = img_edge[3]
    img = torch.permute((img + 1) / 2.0, (1, 2, 0))
    img = (img * 255).numpy().astype(np.uint8)

    img = np.ascontiguousarray(img)
    lmk = lmk * 255

    for i in range(lmk.shape[0]):
        p = lmk[i]
        x, y = round(float(p[0])), round(float(p[1]))

        cv2.circle(img, (x, y), 2, (0, 225, 255), 2)

    edges = (edges * 255).unsqueeze(-1)

    edges = np.ascontiguousarray(edges.repeat((1, 1, 3)).numpy().astype(np.uint8))

    img = np.concatenate([img, edges], axis=1)
    img = Image.fromarray(img)
    img.show()
