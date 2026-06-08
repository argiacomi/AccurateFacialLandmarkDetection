import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from scipy.io import loadmat
from vit_pytorch import SimpleViT

from lib.landmarks.models.blocks import WSConv2d, bottleneck_IR_SE
from lib.landmarks.models.coord_conv import CoordConvTh


class Res50(nn.Module):
    def __init__(self, lmk_num=98):
        super(Res50, self).__init__()
        # self.res50 = torchvision.models.resnet101(pretrained=True)
        self.res50 = torchvision.models.resnet50(pretrained=True)
        self.lmk_num = lmk_num

        self.linear = nn.Linear(2048, self.lmk_num * 2)
        self.res50.fc = self.linear

    def forward(self, img):

        r = self.res50(img)
        return torch.sigmoid(r.reshape((-1, self.lmk_num, 2)))


class SimpleRegressor(nn.Module):
    def __init__(
        self,
        basis_mat,
        lmk_num=98,
        regressing_transformation=False,
        basis_num=80,
        res_backone=True,
    ):
        super(SimpleRegressor, self).__init__()
        if basis_num is None:
            basis_num = lmk_num * 2
        else:
            basis_num = min(lmk_num * 2, basis_num)
        self.basis_num = basis_num

        if res_backone:
            self.backbone = torchvision.models.resnet50(
                torchvision.models.ResNet50_Weights.IMAGENET1K_V1
            )
            if not regressing_transformation:
                self.head = nn.Linear(2048, basis_num)
            else:
                self.head = nn.Linear(2048, basis_num + 1 + 1 + 2)
            self.backbone.fc = self.head
        else:
            num_coeff = basis_num
            if regressing_transformation:
                num_coeff = basis_num + 1 + 1 + 2
            self.backbone = SimpleViT(
                image_size=256,
                patch_size=16,
                num_classes=num_coeff,
                dim=1024,
                depth=6,
                heads=16,
                mlp_dim=2048,
                channels=3,
            )

        self.lmk_num = lmk_num
        self.regress_transformation = regressing_transformation

        mat = loadmat(basis_mat)

        mu = torch.from_numpy(mat["mu"]).reshape((1, lmk_num * 2)).float()
        basis = torch.from_numpy(mat["basis"]).float()[:basis_num]

        self.register_buffer("mu", mu, persistent=False)
        self.register_buffer("basis", basis, persistent=False)

    def GetR(self, angle):
        B, _ = angle.shape
        theta = angle / 180.0 * torch.pi

        a00 = torch.cos(theta)
        a01 = -torch.sin(theta)
        a10 = torch.sin(theta)
        a11 = torch.cos(theta)

        R = torch.cat([a00, a01, a10, a11], dim=1).reshape((B, 2, 2))

        return R

    def forward(self, img):
        B, _, _, _ = img.shape
        feat = self.backbone(img)

        loc = (
            torch.einsum("bi, ij->bj", feat[:, : self.basis_num], self.basis) + self.mu
        )
        loc = loc.reshape((-1, self.lmk_num, 2))
        if self.regress_transformation:
            angle = feat[:, [self.basis_num]]
            scale = feat[:, [self.basis_num + 1]].reshape((B, 1, 1))
            T = feat[:, self.basis_num + 1 + 1 :].reshape((B, 1, 2))
            R = self.GetR(angle)
            loc = torch.einsum("bli, bij->blj", loc, R) * scale + T

        return loc, feat[:, : self.basis_num]


class MLP(nn.Module):
    def __init__(self, in_channel, out_channel, num_units=1):
        super(MLP, self).__init__()
        units = [bottleneck_IR_SE(in_channel, out_channel, 1)]
        for b in range(num_units - 1):
            units.append(bottleneck_IR_SE(out_channel, out_channel, 1))
        self.body = nn.Sequential(*units)

    def forward(self, x):
        return self.body(x)


class DoubleConv(nn.Module):
    def __init__(self, in_channel, out_channel, mid_channel=None):
        super(DoubleConv, self).__init__()
        if not mid_channel:
            mid_channel = out_channel
        self.double_conv = nn.Sequential(
            WSConv2d(
                in_channel, mid_channel, kernel_size=3, stride=1, padding=1, bias=False
            ),
            nn.BatchNorm2d(mid_channel),
            nn.ReLU(inplace=True),
            WSConv2d(
                mid_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=False
            ),
            nn.BatchNorm2d(out_channel),
        )

        self.short_cut = (
            nn.Identity()
            if in_channel == out_channel
            else WSConv2d(
                in_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=False
            )
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, input):
        x = self.double_conv(input)
        x = self.relu(x + self.short_cut(input))
        return x


class LBottleNeck(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=17):
        super().__init__()
        self.conv1 = WSConv2d(in_channels, out_channels, 1)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = WSConv2d(
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.conv3 = WSConv2d(out_channels, out_channels, 1)
        self.bn3 = nn.BatchNorm2d(out_channels)

        self.relu = nn.LeakyReLU(0.1)

        if in_channels == out_channels:
            self.short_cut = nn.Identity()
        else:
            self.short_cut = WSConv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        short_cut = self.short_cut(x)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))

        return x + short_cut


class LKernelBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num=3, kernel_size=17):
        super(LKernelBlock, self).__init__()
        self.lk = nn.Sequential(
            LBottleNeck(in_channels, out_channels, kernel_size=kernel_size),
            MLP(out_channels, out_channels, num),
        )

    def forward(self, x):
        return self.lk(x)


class LKernelNet(nn.Module):
    def __init__(self, lmk_num=98):
        super(LKernelNet, self).__init__()
        self.lmk_num = lmk_num
        self.pre = nn.Sequential(
            self.Conv(3, 32, 3),
            self.Conv(32, 32, 17),
            self.Conv(32, 32, 15),
            self.Conv(32, 32, 13),
            nn.MaxPool2d(2),
            self.Conv(32, 64, 17),
            self.Conv(64, 64, 15),
            self.Conv(64, 64, 13),
            nn.MaxPool2d(2),
            LKernelBlock(64, 64, 3, kernel_size=17),
            LKernelBlock(64, 64, 3, kernel_size=17),
            nn.MaxPool2d(2),
            LKernelBlock(64, 128, 3, kernel_size=7),
            nn.MaxPool2d(2),
            LKernelBlock(128, 256, 3, kernel_size=5),
            nn.MaxPool2d(2),
            LKernelBlock(256, 512, 3, kernel_size=5),
            nn.MaxPool2d(2),
            LKernelBlock(512, 1024, 8, kernel_size=5),
            nn.AvgPool2d(4),
            nn.Flatten(),
        )
        self.head = nn.Linear(1024, self.lmk_num * 2)

    def Conv(self, in_channels, out_channels, kernel_size):
        return LBottleNeck(in_channels, out_channels, kernel_size)

    def forward(self, x):
        B, C, H, W = x.shape
        feat = self.pre(x)
        lmk = self.head(feat)
        return lmk.reshape((B, self.lmk_num, 2))


class Res50PredictBasisCoefficients(nn.Module):
    def __init__(self, lmk_num=98):
        super(Res50PredictBasisCoefficients, self).__init__()
        self.res50 = torchvision.models.resnet101(pretrained=True)
        # self.res50 = torchvision.models.resnet50(pretrained=True)
        self.lmk_num = lmk_num

        self.linear = nn.Linear(2 * 8 * 8, self.lmk_num)

        row_loc, col_loc = self.make_grid("cpu", size=32)
        self.register_buffer("xx_loc", col_loc, False)
        self.register_buffer("yy_loc", row_loc, False)
        hm_basis = self.GenerateBasis()
        self.register_buffer("hm_basis", hm_basis, False)

    def make_grid(self, device="cpu", size=14):
        row, col = torch.meshgrid(
            torch.arange(size, device=device),
            torch.arange(size, device=device),
            indexing="ij",
        )
        c = size - 1.0
        row = row / c
        col = col / c
        return row.reshape((1, 1, size, size)), col.reshape((1, 1, size, size))

    def GetCoord(self, heatmap):
        B, C, H, W = heatmap.shape
        heatmap = F.softmax(heatmap.reshape((B, C, -1)), dim=-1).reshape((B, C, H, W))
        xx = (heatmap * self.xx_loc).sum([2, 3])
        yy = (heatmap * self.yy_loc).sum([2, 3])
        return torch.stack([xx, yy], dim=2)

    def GetHeatmapFromBasis(self, basis_coeff):
        B, C, N = basis_coeff.shape
        # hm = basis_coeff.reshape((B, C, N, 1, 1)) * self.hm_basis
        # hm = torch.sum(hm, dim=2)
        hm = torch.einsum("bcn,njk->bcjk", basis_coeff, self.hm_basis)
        return hm

    def GenerateBasis(self):
        # hm = np.zeros((len(lmk), self.img_size, self.img_size), dtype=np.float32)
        heatmap_size = 32
        row = (
            torch.arange(heatmap_size)
            .reshape((heatmap_size, 1))
            .repeat((1, heatmap_size))
        )
        col = (
            torch.arange(heatmap_size)
            .reshape((1, heatmap_size))
            .repeat((heatmap_size, 1))
        )

        self.grid_index = torch.stack([row, col], dim=2)

        lmk = torch.stack([col, row], dim=2).reshape((heatmap_size * heatmap_size, 2))
        hm = torch.zeros((len(lmk), 32, 32))
        for i in range(len(lmk)):
            x, y = lmk[i]
            loc_idx = torch.FloatTensor([[[y, x]]])
            dist = self.grid_index - loc_idx
            dist = torch.sqrt(torch.sum(dist * dist, dim=2))
            # hm_i = np.exp(-dist / 0.6)  # 64-0.6
            # hm_i = np.exp(-dist  / 0.75)  # 32-0.75
            hm_i = torch.exp(-dist * dist / 1.0)  # 32-0.75

            # hm_i = hm_i / torch.sum(hm_i, dim=[0, 1], keepdim=True)
            hm[i] = hm_i
        return hm

    def forward(self, img):
        B, C, H, W = img.shape
        x = img
        r = self.res50
        x = r.conv1(x)
        x = r.bn1(x)
        x = r.relu(x)
        x = r.maxpool(x)
        x = r.layer1(x)  # output shape (256,64,64)
        x = r.layer2(x)  # output shape (256,32,32)
        x = r.layer3(x)  # output shape (256,16,16)
        x = r.layer4(x)

        hm_basis = x.reshape((B, 1024, 2 * 8 * 8))

        hm = self.linear(hm_basis).permute((0, 2, 1))
        # hm = hm.reshape((B, self.lmk_num, 32, 32))
        basis_coeff = hm.reshape((B, self.lmk_num, -1))
        hm = self.GetHeatmapFromBasis(basis_coeff)
        coord = self.GetCoord(hm)
        return coord, hm


class Res50PredictBasisCoefficients2(nn.Module):
    def __init__(self, lmk_num=98):
        super(Res50PredictBasisCoefficients2, self).__init__()
        self.res50 = torchvision.models.resnet101(pretrained=True)
        # self.res50 = torchvision.models.resnet50(pretrained=True)
        self.lmk_num = lmk_num

        self.linear = nn.Linear(2 * 8 * 8, self.lmk_num)

        row_loc, col_loc = self.make_grid("cpu", size=32)
        self.register_buffer("xx_loc", col_loc, False)
        self.register_buffer("yy_loc", row_loc, False)

        self.head = nn.Conv2d(2048, self.lmk_num * 16, 1)
        self.coord_conv1 = CoordConvTh(
            64, 64, True, False, 256, 256, kernel_size=3, padding=1
        )
        self.coord_conv2 = CoordConvTh(
            32, 32, True, False, 512, 512, kernel_size=3, padding=1
        )
        self.coord_conv3 = CoordConvTh(
            16, 16, True, False, 1024, 1024, kernel_size=3, padding=1
        )

    def make_grid(self, device="cpu", size=14):
        row, col = torch.meshgrid(
            torch.arange(size, device=device),
            torch.arange(size, device=device),
            indexing="ij",
        )
        c = size - 1.0
        row = row / c
        col = col / c
        return row.reshape((1, 1, size, size)), col.reshape((1, 1, size, size))

    def GetCoord(self, heatmap):
        B, C, H, W = heatmap.shape
        heatmap = F.softmax(heatmap.reshape((B, C, -1)), dim=-1).reshape((B, C, H, W))
        xx = (heatmap * self.xx_loc).sum([2, 3])
        yy = (heatmap * self.yy_loc).sum([2, 3])
        return torch.stack([xx, yy], dim=2)

    def GetHeatmapFromBasis(self, basis_coeff):
        B, C, N = basis_coeff.shape
        # hm = basis_coeff.reshape((B, C, N, 1, 1)) * self.hm_basis
        # hm = torch.sum(hm, dim=2)
        hm = torch.einsum("bcn,njk->bcjk", basis_coeff, self.hm_basis)
        return hm

    def GenerateBasis(self):
        # hm = np.zeros((len(lmk), self.img_size, self.img_size), dtype=np.float32)
        heatmap_size = 32
        row = (
            torch.arange(heatmap_size)
            .reshape((heatmap_size, 1))
            .repeat((1, heatmap_size))
        )
        col = (
            torch.arange(heatmap_size)
            .reshape((1, heatmap_size))
            .repeat((heatmap_size, 1))
        )

        self.grid_index = torch.stack([row, col], dim=2)

        lmk = torch.stack([col, row], dim=2).reshape((heatmap_size * heatmap_size, 2))
        hm = torch.zeros((len(lmk), 32, 32))
        for i in range(len(lmk)):
            x, y = lmk[i]
            loc_idx = torch.FloatTensor([[[y, x]]])
            dist = self.grid_index - loc_idx
            dist = torch.sqrt(torch.sum(dist * dist, dim=2))
            # hm_i = np.exp(-dist / 0.6)  # 64-0.6
            # hm_i = np.exp(-dist  / 0.75)  # 32-0.75
            hm_i = torch.exp(-dist * dist / 1.0)  # 32-0.75

            hm_i = hm_i / torch.sum(hm_i, dim=[0, 1], keepdim=True)
            hm[i] = hm_i
        return hm

    def forward(self, img):
        B, C, H, W = img.shape
        x = img
        r = self.res50
        x = r.conv1(x)
        x = r.bn1(x)
        x = r.relu(x)
        x = r.maxpool(x)
        x = r.layer1(x)  # output shape (256,64,64)
        x = self.coord_conv1(x)
        x = r.layer2(x)  # output shape (512,32,32)
        x = self.coord_conv2(x)
        x = r.layer3(x)  # output shape (1024,16,16)
        x = self.coord_conv3(x)
        x = r.layer4(x)  # output shape (2048,8,8)
        x = self.head(x)

        hm = x.reshape((B, self.lmk_num, 4, 4, 8, 8))
        hm = torch.cat(
            [hm[:, :, :, 0], hm[:, :, :, 1], hm[:, :, :, 2], hm[:, :, :, 3]], dim=-1
        )
        hm = torch.cat([hm[:, :, 0], hm[:, :, 1], hm[:, :, 2], hm[:, :, 3]], dim=-2)

        coord = self.GetCoord(hm)
        return coord, hm


class Res50MultiResHM(nn.Module):
    def __init__(self, lmk_num=98):
        super(Res50MultiResHM, self).__init__()
        # self.res50 = torchvision.models.resnet101(pretrained=True)
        self.res50 = torchvision.models.resnet._resnet(
            torchvision.models.resnet.Bottleneck,
            [8, 8, 8, 8],
            weights=None,
            progress=False,
        )
        ckpt = torch.load("resnet101_8_8_8_8.pt")
        self.res50.load_state_dict(ckpt)

        self.lmk_num = lmk_num

        row_loc, col_loc = self.make_grid("cpu", size=32)
        self.register_buffer("xx_loc32", col_loc, False)
        self.register_buffer("yy_loc32", row_loc, False)

        row_loc, col_loc = self.make_grid("cpu", size=16)
        self.register_buffer("xx_loc16", col_loc, False)
        self.register_buffer("yy_loc16", row_loc, False)

        row_loc, col_loc = self.make_grid("cpu", size=8)
        self.register_buffer("xx_loc8", col_loc, False)
        self.register_buffer("yy_loc8", row_loc, False)

        self.head32 = nn.Conv2d(512, self.lmk_num, 1)
        self.head16 = nn.Conv2d(1024, self.lmk_num, 1)
        self.head8 = nn.Conv2d(2048, self.lmk_num, 1)

        self.short_cut1 = nn.Conv2d(256, 512, 3, stride=2, padding=1)
        self.short_cut2 = nn.Conv2d(512, 1024, 3, stride=2, padding=1)

    def make_grid(self, device="cpu", size=14):
        row, col = torch.meshgrid(
            torch.arange(size, device=device),
            torch.arange(size, device=device),
            indexing="ij",
        )
        c = size - 1.0
        row = row / c
        col = col / c
        return row.reshape((1, 1, size, size)), col.reshape((1, 1, size, size))

    def GetCoord32(self, heatmap):
        B, C, H, W = heatmap.shape
        heatmap = F.softmax(heatmap.reshape((B, C, -1)), dim=-1).reshape((B, C, H, W))
        xx = (heatmap * self.xx_loc32).sum([2, 3])
        yy = (heatmap * self.yy_loc32).sum([2, 3])
        return torch.stack([xx, yy], dim=2)

    def GetCoord16(self, heatmap):
        B, C, H, W = heatmap.shape
        heatmap = F.softmax(heatmap.reshape((B, C, -1)), dim=-1).reshape((B, C, H, W))
        xx = (heatmap * self.xx_loc16).sum([2, 3])
        yy = (heatmap * self.yy_loc16).sum([2, 3])
        return torch.stack([xx, yy], dim=2)

    def GetCoord8(self, heatmap):
        B, C, H, W = heatmap.shape
        heatmap = F.softmax(heatmap.reshape((B, C, -1)), dim=-1).reshape((B, C, H, W))
        xx = (heatmap * self.xx_loc8).sum([2, 3])
        yy = (heatmap * self.yy_loc8).sum([2, 3])
        return torch.stack([xx, yy], dim=2)

    def forward(self, img):
        B, C, H, W = img.shape
        x = img
        r = self.res50
        x = r.conv1(x)
        x = r.bn1(x)
        x = r.relu(x)
        x = r.maxpool(x)
        x = r.layer1(x)  # output shape (256,64,64)
        x32 = r.layer2(x)  # output shape (512,32,32)
        hm32 = self.head32(x32)
        coord32 = self.GetCoord32(hm32)
        x16 = r.layer3(x32 + self.short_cut1(x))  # output shape (1024,16,16)
        hm16 = self.head16(x16)
        coord16 = self.GetCoord16(hm16)

        x8 = r.layer4(x16 + self.short_cut2(x32))  # output shape (2048,8,8)
        hm8 = self.head8(x8)
        coord8 = self.GetCoord8(hm8)

        return [(coord32, hm32), (coord16, hm16), (coord8, hm8)]


class Res50MultiTuckedUpHM(nn.Module):
    def __init__(self, lmk_num=98):
        super(Res50MultiTuckedUpHM, self).__init__()
        # self.res50 = torchvision.models.resnet101(pretrained=True)
        self.res50 = torchvision.models.resnet._resnet(
            torchvision.models.resnet.Bottleneck, [8, 8, 8, 8], weights=None, progress=False
        )
        self.lmk_num = lmk_num

        self.linear = nn.Linear(2 * 8 * 8, self.lmk_num)

        row_loc, col_loc = self.make_grid("cpu", size=32)
        self.register_buffer("xx_loc", col_loc, False)
        self.register_buffer("yy_loc", row_loc, False)

        self.output_layer32 = nn.Conv2d(512, self.lmk_num, 1)

        self.head16 = nn.Conv2d(1024, self.lmk_num * 4, 1)
        self.head8 = nn.Conv2d(2048, self.lmk_num * 16, 1)

    def make_grid(self, device="cpu", size=14):
        row, col = torch.meshgrid(torch.arange(size, device=device), torch.arange(size, device=device), indexing="ij")
        c = size - 1.0
        row = row / c
        col = col / c
        return row.reshape((1, 1, size, size)), col.reshape((1, 1, size, size))

    def GetCoord(self, heatmap):
        B, C, H, W = heatmap.shape
        heatmap = F.softmax(heatmap.reshape((B, C, -1)), dim=-1).reshape((B, C, H, W))
        xx = (heatmap * self.xx_loc).sum([2, 3])
        yy = (heatmap * self.yy_loc).sum([2, 3])
        return torch.stack([xx, yy], dim=2)

    def GetHeatmapFromBasis(self, basis_coeff):
        B, C, N = basis_coeff.shape
        # hm = basis_coeff.reshape((B, C, N, 1, 1)) * self.hm_basis
        # hm = torch.sum(hm, dim=2)
        hm = torch.einsum("bcn,njk->bcjk", basis_coeff, self.hm_basis)
        return hm

    def GenerateBasis(self):
        # hm = np.zeros((len(lmk), self.img_size, self.img_size), dtype=np.float32)
        heatmap_size = 32
        row = torch.arange(heatmap_size).reshape((heatmap_size, 1)).repeat((1, heatmap_size))
        col = torch.arange(heatmap_size).reshape((1, heatmap_size)).repeat((heatmap_size, 1))

        self.grid_index = torch.stack([row, col], dim=2)

        lmk = torch.stack([col, row], dim=2).reshape((heatmap_size * heatmap_size, 2))
        hm = torch.zeros((len(lmk), 32, 32))
        for i in range(len(lmk)):
            x, y = lmk[i]
            loc_idx = torch.FloatTensor([[[y, x]]])
            dist = self.grid_index - loc_idx
            dist = torch.sqrt(torch.sum(dist * dist, dim=2))
            # hm_i = np.exp(-dist / 0.6)  # 64-0.6
            # hm_i = np.exp(-dist  / 0.75)  # 32-0.75
            hm_i = torch.exp(-dist * dist / 1.0)  # 32-0.75

            hm_i = hm_i / torch.sum(hm_i, dim=[0, 1], keepdim=True)
            hm[i] = hm_i
        return hm

    def forward(self, img):
        B, C, H, W = img.shape
        x = img
        r = self.res50
        x = r.conv1(x)
        x = r.bn1(x)
        x = r.relu(x)
        x = r.maxpool(x)
        x = r.layer1(x)  # output shape (256,64,64)
        x32 = r.layer2(x)  # output shape (512,32,32)

        hm32 = self.output_layer32(x32)

        x16 = r.layer3(x32)  # output shape (1024,16,16)
        hm16 = self.head16(x16)
        hm16 = hm16.reshape((B, self.lmk_num, 2, 2, 16, 16))
        hm16 = torch.cat([hm16[:, :, :, 0], hm16[:, :, :, 1]], dim=-1)
        hm16 = torch.cat([hm16[:, :, 0], hm16[:, :, 1]], dim=-2)

        x8 = r.layer4(x16)  # output shape (2048,8,8)
        hm8 = self.head8(x8)
        hm8 = hm8.reshape((B, self.lmk_num, 4, 4, 8, 8))
        hm8 = torch.cat([hm8[:, :, :, 0], hm8[:, :, :, 1], hm8[:, :, :, 2], hm8[:, :, :, 3]], dim=-1)
        hm8 = torch.cat([hm8[:, :, 0], hm8[:, :, 1], hm8[:, :, 2], hm8[:, :, 3]], dim=-2)

        coord32 = self.GetCoord(hm32)
        coord16 = self.GetCoord(hm16)
        coord8 = self.GetCoord(hm8)
        return [(coord32, hm32), (coord16, hm16), (coord8, hm8)]


if __name__ == "__main__":
    x = torch.rand((2, 3, 256, 256))

    net = Res50PredictBasisCoefficients()
    net.GenerateBasis()
    loc, hm = net(x)
    print(loc.shape, hm.shape)
