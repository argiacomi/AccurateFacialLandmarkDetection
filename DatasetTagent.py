import os.path
import torch
from torch.utils.data import Dataset
from torchvision.transforms import transforms
import numpy as np
import cv2
from glob import glob
from PIL import Image
from ImageAugmentation import GetAugTransform


class LandmarkDataset(Dataset):
    def __init__(self, data_root, split, preload=True, aug=True):
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

    def MakeTangent(self, lmk: torch.Tensor):
        def NormalizeVector(v):
            return v / torch.sqrt(torch.sum(v * v))

        tangent = torch.zeros_like(lmk)
        tangent[0] = NormalizeVector(lmk[1] - lmk[0])
        for i in range(1, 32):
            tangent[i] = NormalizeVector(lmk[i + 1] - lmk[i - 1])
        tangent[32] = NormalizeVector(lmk[32] - lmk[31])

        tangent[33] = NormalizeVector(lmk[34] - lmk[33])
        for i in range(34, 37):
            tangent[i] = NormalizeVector(lmk[i + 1] - lmk[i - 1])
        tangent[37] = NormalizeVector(lmk[37] - lmk[36])
        tangent[38] = NormalizeVector(lmk[39] - lmk[38])
        for i in range(39, 41):
            tangent[i] = NormalizeVector(lmk[i + 1] - lmk[i - 1])
        tangent[41] = NormalizeVector(lmk[41] - lmk[40])

        tangent[42] = NormalizeVector(lmk[43] - lmk[42])
        for i in range(43, 46):
            tangent[i] = NormalizeVector(lmk[i + 1] - lmk[i - 1])
        tangent[46] = NormalizeVector(lmk[46] - lmk[45])
        for i in range(47, 50):
            tangent[i] = NormalizeVector(lmk[i + 1] - lmk[i - 1])
        tangent[50] = NormalizeVector(lmk[50] - lmk[49])

        tangent[51] = NormalizeVector(lmk[52] - lmk[51])
        for i in range(52, 54):
            tangent[i] = NormalizeVector(lmk[i + 1] - lmk[i - 1])
        tangent[54] = NormalizeVector(lmk[54] - lmk[53])

        tangent[55] = NormalizeVector(lmk[56] - lmk[55])
        for i in range(56, 59):
            tangent[i] = NormalizeVector(lmk[i + 1] - lmk[i - 1])
        tangent[59] = NormalizeVector(lmk[59] - lmk[58])

        tangent[60] = NormalizeVector(lmk[64] - lmk[60])
        tangent[64] = NormalizeVector(lmk[64] - lmk[60])
        for i in range(61, 64):
            tangent[i] = NormalizeVector(lmk[i + 1] - lmk[i - 1])
        for i in range(65, 67):
            tangent[i] = NormalizeVector(lmk[i + 1] - lmk[i - 1])
        tangent[67] = NormalizeVector(lmk[66] - lmk[60])

        tangent[68] = NormalizeVector(lmk[72] - lmk[68])
        tangent[72] = NormalizeVector(lmk[72] - lmk[68])
        for i in range(69, 72):
            tangent[i] = NormalizeVector(lmk[i + 1] - lmk[i - 1])
        for i in range(73, 75):
            tangent[i] = NormalizeVector(lmk[i + 1] - lmk[i - 1])

        tangent[75] = NormalizeVector(lmk[74] - lmk[68])

        tangent[76] = NormalizeVector(lmk[82] - lmk[76])
        tangent[82] = NormalizeVector(lmk[82] - lmk[76])
        for i in range(77, 82):
            tangent[i] = NormalizeVector(lmk[i + 1] - lmk[i - 1])
        for i in range(83, 87):
            tangent[i] = NormalizeVector(lmk[i + 1] - lmk[i - 1])
        tangent[87] = NormalizeVector(lmk[86] - lmk[76])

        tangent[88] = NormalizeVector(lmk[92] - lmk[88])
        tangent[89] = NormalizeVector(lmk[90] - lmk[89])
        tangent[90] = NormalizeVector(lmk[91] - lmk[89])
        tangent[91] = NormalizeVector(lmk[91] - lmk[90])
        tangent[92] = NormalizeVector(lmk[92] - lmk[88])

        tangent[93] = NormalizeVector(lmk[94] - lmk[93])
        tangent[94] = NormalizeVector(lmk[95] - lmk[93])
        tangent[95] = NormalizeVector(lmk[95] - lmk[94])

        tangent[96][0] = 1
        tangent[96][1] = 0
        tangent[97][0] = 1
        tangent[97][1] = 0

        flag = torch.sum(torch.isnan(tangent), dim=1) > 0
        tangent[flag, 0] = 1
        tangent[flag, 1] = 0

        return tangent

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

        img = self.transform(img)
        lmk = torch.from_numpy(lmk / 255)

        return img, lmk, self.MakeTangent(lmk)


if __name__ == "__main__":

    dataset = LandmarkDataset("WFLW", "test", False, False)
    d = dataset[0]

    img, lmk = d
    print(img.shape)
    img = torch.permute((img + 1) / 2.0, (1, 2, 0))
    img = (img * 255).numpy().astype(np.uint8)

    img = np.ascontiguousarray(img)
    lmk = lmk * 255

    for i in range(lmk.shape[0]):
        p = lmk[i]
        x, y = round(float(p[0])), round(float(p[1]))

        cv2.circle(img, (x, y), 2, (0, 225, 255), 2)
    img = Image.fromarray(img)
    img.show()
