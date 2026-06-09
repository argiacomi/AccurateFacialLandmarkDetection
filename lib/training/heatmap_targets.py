import copy
import numpy as np

import torch
import torch.nn.functional as F
import math


def DistPoint2Edge(points: np.ndarray, edge: np.ndarray):
    v = edge[1] - edge[0]
    length = np.linalg.norm(v)
    if length < 1e-4:
        dist = np.linalg.norm(points - (edge[0] + edge[1]) / 2, axis=1)
        return dist
    else:
        dist = np.ones(points.shape[0])
        left_dist = np.linalg.norm(points - edge[0], axis=1)
        flag_left = np.sum((points - edge[0]) * v, axis=1) < 0
        dist = np.where(flag_left, left_dist, dist)
        right_dist = np.linalg.norm(points - edge[1], axis=1)
        flag_right = np.sum((points - edge[1]) * v, axis=1) > 0
        dist = np.where(flag_right, right_dist, dist)

        unit_v = v / length
        dist_v = points - edge[0]
        dist_v = dist_v - np.sum(dist_v * unit_v, axis=1, keepdims=True) * unit_v
        dist_v = np.linalg.norm(dist_v, axis=1)

        dist = np.where(np.logical_or(flag_right, flag_left), dist, dist_v)
        return dist


class encoder_default:
    def __init__(self, image_height, image_width, scale=0.25, sigma=1.5):
        self.image_height = image_height
        self.image_width = image_width
        self.scale = scale
        self.sigma = sigma

    def generate_heatmap(self, points):
        # points = (num_pts, 2)
        h, w = self.image_height, self.image_width
        pointmaps = []
        for i in range(len(points)):
            pointmap = np.zeros([h, w], dtype=np.float32)
            # align_corners: False.
            point = copy.deepcopy(points[i])
            point[0] = max(0, min(w - 1, point[0]))
            point[1] = max(0, min(h - 1, point[1]))
            pointmap = self._circle(pointmap, point, sigma=self.sigma)

            pointmaps.append(pointmap)
        pointmaps = np.stack(pointmaps, axis=0) / 255.0
        pointmaps = torch.from_numpy(pointmaps).float().unsqueeze(0)
        pointmaps = F.interpolate(
            pointmaps,
            size=(int(w * self.scale), int(h * self.scale)),
            mode="bilinear",
            align_corners=True,
        ).squeeze()
        return pointmaps

    def _circle(self, img, ptt, sigma=1.0, label_type="Gaussian"):
        # Check that any part of the gaussian is in-bounds
        tmp_size = sigma * 3
        pt = round(float(ptt[0]), 4), round(float(ptt[1]), 4)
        ul = [int(pt[0] - tmp_size), int(pt[1] - tmp_size)]
        br = [int(pt[0] + tmp_size + 1), int(pt[1] + tmp_size + 1)]
        if (
            ul[0] > img.shape[1] - 1
            or ul[1] > img.shape[0] - 1
            or br[0] - 1 < 0
            or br[1] - 1 < 0
        ):
            # If not, just return the image as is
            return img

        # Generate gaussian
        size = 2 * tmp_size + 1
        x = np.arange(0, size, 1, np.float32)
        y = x[:, np.newaxis]
        x0 = y0 = size // 2
        # The gaussian is not normalized, we want the center value to equal 1
        if label_type == "Gaussian":
            g = np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma**2))
        else:
            g = sigma / (((x - x0) ** 2 + (y - y0) ** 2 + sigma**2) ** 1.5)

        # Usable gaussian range
        g_x = max(0, -ul[0]), min(br[0], img.shape[1]) - ul[0]
        g_y = max(0, -ul[1]), min(br[1], img.shape[0]) - ul[1]

        # Image range
        img_x = max(0, ul[0]), min(br[0], img.shape[1])
        img_y = max(0, ul[1]), min(br[1], img.shape[0])

        # if img_x[1] - img_x[0] != g_x[1] - g_x[0] or img_y[1] - img_y[0] != g_y[1] - g_y[0]:
        #     print("buging.....", pt, sigma)
        # if g.shape[0] != 10 or g.shape[1] != 10:
        #     print("buging.....2", pt, sigma)
        # shape1 = img[img_y[0] : img_y[1], img_x[0] : img_x[1]].shape
        # shape2 = g[g_y[0] : g_y[1], g_x[0] : g_x[1]].shape
        # if shape1[0] != shape2[0] or shape1[1] != shape2[1]:
        #     print("buging....3", pt, sigma, shape1, shape2, img_x, img_y)
        img[img_y[0] : img_y[1], img_x[0] : img_x[1]] = (
            255 * g[g_y[0] : g_y[1], g_x[0] : g_x[1]]
        )
        return img


class GenerateHeatmap:
    def __init__(self, img_size):
        super(GenerateHeatmap, self).__init__()
        self.img_size = img_size
        row = np.arange(img_size).reshape((img_size, 1)).repeat(img_size, axis=1)
        col = np.arange(img_size).reshape((1, img_size)).repeat(img_size, axis=0)

        self.grid_index = np.stack([row, col], axis=2).astype(np.float32)

    def Generate(self, lmk):

        hm = np.zeros((len(lmk), self.img_size, self.img_size), dtype=np.float32)
        for i in range(len(lmk)):
            x, y = lmk[i]
            loc_idx = np.array([[[y, x]]], dtype=np.float32)
            dist = self.grid_index - loc_idx
            dist = np.sqrt(np.sum(dist * dist, axis=2))
            # hm_i = np.exp(-dist / 0.6)  # 64-0.6
            # hm_i = np.exp(-dist  / 0.75)  # 32-0.75
            hm_i = np.exp(-dist * dist / 1.0)  # 32-0.75
            hm[i] = hm_i
        return hm

    def GenerateDebug(self, lmk):
        hm = np.zeros((len(lmk), self.img_size, self.img_size), dtype=np.float32)
        for i in range(len(lmk)):
            x, y = lmk[i]
            x_b = math.floor(x)
            a = x - x_b
            x_t = math.ceil(x)
            if x_t == x_b:
                if x_b + 1 < self.img_size:
                    x_t = x_b + 1
                else:
                    x_b = x_b - 1
                    a = 1
            y_b = math.floor(y)
            b = y - y_b
            y_t = math.ceil(y)
            if y_t == y_b:
                if y_b + 1 < self.img_size:
                    y_t = y_b + 1
                else:
                    y_b = y_b - 1
                    b = 1

            hm[i, int(y_b), int(x_b)] = (1 - a) * (1 - b)
            hm[i, int(y_b), int(x_t)] = a * (1 - b)
            hm[i, int(y_t), int(x_b)] = (1 - a) * b
            hm[i, int(y_t), int(x_t)] = a * b

        return hm

    def GenerateEdgeHeatmap(self, lmk, edge_info):
        hm = np.zeros((len(edge_info), self.img_size, self.img_size), dtype=np.float32)
        for i in range(len(edge_info)):
            edge = lmk[list(edge_info[i][1])]
            closed = edge_info[i][0]
            hm_e = self.GenerateEdge(edge, closed)
            hm[i] = hm_e
        return hm

    def GenerateEdge(self, edge, closed=False):
        points = self.grid_index.reshape((-1, 2))[:, [1, 0]]
        dist = np.ones(points.shape[0]) * 99999
        for i in range(len(edge) - 1):
            dist_e = DistPoint2Edge(points, edge[i : i + 2])
            dist = np.where(dist_e < dist, dist_e, dist)

        if closed:
            dist_e = DistPoint2Edge(points, (edge[0], edge[-1]))
            dist = np.where(dist_e < dist, dist_e, dist)

        dist = dist.reshape((self.img_size, self.img_size))

        hm = np.exp(-dist * dist / 1.0)
        return hm


class GenerateLineHeatmap:
    def __init__(self, img_size):
        super(GenerateLineHeatmap, self).__init__()
        self.img_size = img_size
        self.grid_index = np.arange(img_size).astype(np.float32)

    def Generate(self, lmk):
        hm = np.zeros((len(lmk), 2, self.img_size), dtype=np.float32)
        for i in range(len(lmk)):
            x, y = lmk[i]

            dist_x = self.grid_index - float(x)
            dist_x = np.abs(dist_x)
            hm_x = np.exp(-dist_x * dist_x)

            dist_y = self.grid_index - float(y)
            dist_y = np.abs(dist_y)
            hm_y = np.exp(-dist_y * dist_y)
            hm_i = np.stack([hm_x, hm_y])

            hm[i] = hm_i
        return hm


if __name__ == "__main__":
    encoder = encoder_default(256, 256)

    points = torch.FloatTensor([[43.0994, 124.4999999]])
    hp = encoder.generate_heatmap(points)
    print(hp.shape)
