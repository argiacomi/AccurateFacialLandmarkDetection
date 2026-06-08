from scipy import integrate
from scipy.interpolate import BSpline
import torch
import numpy as np


def MakeGrid(grid_num=30, k=3):
    t = [0] * k + np.linspace(0, 1, grid_num).tolist() + [1] * k

    dim = grid_num + k - 1
    A = np.zeros((dim, dim))
    B = np.zeros((dim, dim))
    for i in range(grid_num + k - 1):
        bi = BSpline.basis_element(t[i : i + k + 2], False)
        # bi = CubicBasis(t[i : i + k + 2])
        su = 0
        for m in range(i, i + k + 1):
            if t[m] < t[m + 1]:
                su += integrate.fixed_quad(bi, t[m], t[m + 1])[0]

        suu = 0
        for m in range(i, i + k + 1):
            if t[m] < t[m + 1]:
                suu += integrate.fixed_quad(lambda x: bi(x) * x, t[m], t[m + 1])[0]
        for j in range(grid_num + k - 1):
            bj = BSpline.basis_element(t[j : j + k + 2], False)
            # bj = CubicBasis(t[j : j + k + 2])\
            sv = 0
            for m in range(j, j + k + 1):
                if t[m] < t[m + 1]:
                    sv += integrate.fixed_quad(bj, t[m], t[m + 1])[0]
            B[j, i] = su * sv
            A[j, i] = suu * sv
    return A, B


class Heatmap(torch.nn.Module):
    def __init__(self, grid_num=30):
        super(Heatmap, self).__init__()

        A, B = MakeGrid(grid_num)
        self.register_buffer("A", torch.from_numpy(A).float().unsqueeze(0).unsqueeze(0), False)
        self.register_buffer("B", torch.from_numpy(B).float().unsqueeze(0).unsqueeze(0), False)

    def forward(self, heatmap):
        heatmap = heatmap - torch.amax(heatmap, dim=(2, 3), keepdim=True)
        heatmap = torch.exp(heatmap)
        heatmap = heatmap / (1e-6 +  torch.sum(heatmap, dim=(2, 3), keepdim=True))
        Pij = heatmap / self.B

        xx = (Pij * self.A).sum([2, 3])
        yy = (Pij * self.A.permute((0, 1, 3, 2))).sum([2, 3])
        return torch.stack([xx, yy], dim=2)
