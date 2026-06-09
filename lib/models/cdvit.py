import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from lib.models.attention import (
    SA2SA1,
    SA2SA1_2,
    DoubleConv,
    SelfAttention,
    SelfAttention2,
)
from lib.models.blocks import bottleneck_IR_SE, get_block
from lib.models.coord_conv import CoordConvTh
from lib.models.heatmap import Heatmap
from lib.models.unet import UNet
from lib.models.vit import Vit as Vit


class Res50(nn.Module):
    def __init__(self, out_depth=256):
        super(Res50, self).__init__()
        self.res50 = torchvision.models.resnet50(pretrained=True)
        self.output_layer = nn.Sequential(
            nn.ConvTranspose2d(2048, 512, kernel_size=2, stride=2),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(512, out_depth, kernel_size=2, stride=2),
            nn.BatchNorm2d(out_depth),
            nn.ReLU(inplace=True),
        )

    def forward(self, img):
        x = img
        r = self.res50
        x = r.conv1(x)
        x = r.bn1(x)
        x = r.relu(x)
        x = r.maxpool(x)
        x = r.layer1(x)
        x = r.layer2(x)
        x = r.layer3(x)
        x = r.layer4(x)
        x = self.output_layer(x)
        return x


class Res50PredictBasisCoefficients(nn.Module):
    def __init__(self, lmk_num=98):
        super(Res50PredictBasisCoefficients, self).__init__()
        self.res50 = torchvision.models.resnet101(pretrained=True)
        self.lmk_num = lmk_num

        self.linear = nn.Linear(2 * 8 * 8, self.lmk_num)

        row_loc, col_loc = self.make_grid("cpu", size=32)
        self.register_buffer("xx_loc", col_loc, False)
        self.register_buffer("yy_loc", row_loc, False)

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

    def forward(self, img):
        B, C, H, W = img.shape
        x = img
        r = self.res50
        x = r.conv1(x)
        x = r.bn1(x)
        x = r.relu(x)
        x = r.maxpool(x)
        x = r.layer1(x)
        x = r.layer2(x)
        x = r.layer3(x)
        x = r.layer4(x)

        hm_basis = x.reshape((B, 1024, 2 * 8 * 8))

        hm = self.linear(hm_basis).permute((0, 2, 1))
        hm = hm.reshape((B, self.lmk_num, 32, 32))
        coord = self.GetCoord(hm)
        return coord, hm


class HeadingNet(torch.nn.Module):
    def __init__(self, channels=(128, 256), in_channel=3):
        super(HeadingNet, self).__init__()

        # body
        blocks = [
            get_block(in_channel=in_channel, depth=channels[0], num_units=3),
            # get_block(in_channel=channels[0], depth=channels[1], num_units=3),
        ]
        for i in range(len(channels) - 1):
            blocks.append(
                get_block(in_channel=channels[i], depth=channels[i + 1], num_units=3)
            )
        units = []
        for bottlenecks in blocks:
            for b in bottlenecks:
                units.append(bottleneck_IR_SE(b.in_channel, b.depth, b.stride))
        self.body = nn.Sequential(*units)

    def forward(self, x):
        x = self.body(x)
        return x


class Net(nn.Module):
    def __init__(self, lmk_num=98):
        super(Net, self).__init__()
        self.pre = HeadingNet()

        self.unet = UNet([256, 320, 384, 448, 512])

        self.output_layer = nn.Conv2d(256, lmk_num, 1)

        row_loc, col_loc = self.make_grid("cpu", size=64)

        self.register_buffer("xx_loc", col_loc, False)
        self.register_buffer("yy_loc", row_loc, False)

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

    def forward(self, img):
        x = self.pre(img)
        heatmap = self.unet(x)
        heatmap = self.output_layer(heatmap)

        heatmap = heatmap - torch.amax(heatmap, dim=(2, 3), keepdim=True)
        heatmap = torch.exp(heatmap)
        heatmap = heatmap / (1e-4 + torch.sum(heatmap, dim=(2, 3), keepdim=True))

        xx = (heatmap * self.xx_loc).sum([2, 3])
        yy = (heatmap * self.yy_loc).sum([2, 3])
        return torch.stack([xx, yy], dim=2), heatmap


class NetAttn(nn.Module):
    def __init__(self, lmk_num=98):
        super(NetAttn, self).__init__()
        self.pre = HeadingNet([64, 128, 256])

        # self.unet = UNet([128,152,176,200])
        self.unet = UNet([256, 320, 384, 448, 512])

        # self.output_layer= nn.Conv2d(256, lmk_num,1)
        self.output_layer = nn.Sequential(
            CoordConvTh(32, 32, True, False, 256, 256, kernel_size=3, padding=1),
            SelfAttention(256),
            CoordConvTh(32, 32, True, False, 256, 256, kernel_size=3, padding=1),
            SelfAttention(256),
            nn.Conv2d(256, lmk_num, 1),
        )

        row_loc, col_loc = self.make_grid("cpu", size=32)

        self.register_buffer("xx_loc", col_loc, False)
        self.register_buffer("yy_loc", row_loc, False)

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

    def forward(self, img):
        x = self.pre(img)
        heatmap = self.unet(x)
        heatmap = self.output_layer(heatmap)

        heatmap = heatmap - torch.amax(heatmap, dim=(2, 3), keepdim=True)
        heatmap = torch.exp(heatmap)
        heatmap = heatmap / (1e-4 + torch.sum(heatmap, dim=(2, 3), keepdim=True))

        xx = (heatmap * self.xx_loc).sum([2, 3])
        yy = (heatmap * self.yy_loc).sum([2, 3])
        return torch.stack([xx, yy], dim=2), heatmap


class NetAttn2(nn.Module):
    def __init__(self, lmk_num=98):
        super(NetAttn2, self).__init__()
        self.pre = HeadingNet([32, 64, 256])

        self.unet = UNet([256, 256, 256, 256])

        # self.output_layer= nn.Conv2d(256, lmk_num,1)
        self.output_layer = nn.Sequential(
            CoordConvTh(32, 32, True, False, 256, 256, kernel_size=3, padding=1),
            SelfAttention(256),
            CoordConvTh(32, 32, True, False, 256, 256, kernel_size=3, padding=1),
            SelfAttention(256),
            CoordConvTh(32, 32, True, False, 256, 256, kernel_size=3, padding=1),
            SelfAttention(256),
            CoordConvTh(32, 32, True, False, 256, 256, kernel_size=3, padding=1),
            SelfAttention(256),
            nn.Conv2d(256, lmk_num, 1),
        )

        row_loc, col_loc = self.make_grid("cpu", size=32)

        self.register_buffer("xx_loc", col_loc, False)
        self.register_buffer("yy_loc", row_loc, False)

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

    def forward(self, img):
        x = self.pre(img)
        heatmap = self.unet(x)
        heatmap = self.output_layer(heatmap)

        heatmap = heatmap - torch.amax(heatmap, dim=(2, 3), keepdim=True)
        heatmap = torch.exp(heatmap)
        heatmap = heatmap / (1e-4 + torch.sum(heatmap, dim=(2, 3), keepdim=True))

        xx = (heatmap * self.xx_loc).sum([2, 3])
        yy = (heatmap * self.yy_loc).sum([2, 3])
        return torch.stack([xx, yy], dim=2), heatmap


class NetAttnSA2(nn.Module):
    def __init__(self, lmk_num=98, in_channel=3):
        super(NetAttnSA2, self).__init__()
        self.pre = HeadingNet([32, 64, 256], in_channel)

        self.unet = UNet([256, 256, 256, 256])

        # self.output_layer= nn.Conv2d(256, lmk_num,1)
        self.output_layer = nn.Sequential(
            CoordConvTh(32, 32, True, False, 256, 256, kernel_size=3, padding=1),
            SelfAttention2(32),
            CoordConvTh(32, 32, True, False, 256, 256, kernel_size=3, padding=1),
            SelfAttention2(32),
            CoordConvTh(32, 32, True, False, 256, 256, kernel_size=3, padding=1),
            SelfAttention2(32),
            CoordConvTh(32, 32, True, False, 256, 256, kernel_size=3, padding=1),
            SelfAttention2(32),
            nn.Conv2d(256, lmk_num, 1),
        )

        row_loc, col_loc = self.make_grid("cpu", size=32)

        self.register_buffer("xx_loc", col_loc, False)
        self.register_buffer("yy_loc", row_loc, False)

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

    def forward(self, img):
        x = self.pre(img)
        heatmap = self.unet(x)
        heatmap = self.output_layer(heatmap)
        heatmap = heatmap - torch.amax(heatmap, dim=(2, 3), keepdim=True)
        heatmap = torch.exp(heatmap)
        heatmap = heatmap / (1e-6 + torch.sum(heatmap, dim=(2, 3), keepdim=True))

        xx = (heatmap * self.xx_loc).sum([2, 3])
        yy = (heatmap * self.yy_loc).sum([2, 3])
        return torch.stack([xx, yy], dim=2), heatmap


class NetAttnSA2Condition(nn.Module):
    def __init__(self, lmk_num=98, in_channel=3):
        super(NetAttnSA2Condition, self).__init__()
        self.embed = nn.Embedding(2, 256 * 32 * 32)
        self.pre = HeadingNet([32, 64, 256], in_channel)

        self.unet = UNet([256, 256, 256, 256])

        # self.output_layer= nn.Conv2d(256, lmk_num,1)
        self.block1 = nn.Sequential(
            CoordConvTh(32, 32, True, False, 256, 256, kernel_size=3, padding=1),
            SelfAttention2(32),
        )
        self.block2 = nn.Sequential(
            CoordConvTh(32, 32, True, False, 256, 256, kernel_size=3, padding=1),
            SelfAttention2(32),
        )
        self.block3 = nn.Sequential(
            CoordConvTh(32, 32, True, False, 256, 256, kernel_size=3, padding=1),
            SelfAttention2(32),
        )
        self.block4 = nn.Sequential(
            CoordConvTh(32, 32, True, False, 256, 256, kernel_size=3, padding=1),
            SelfAttention2(32),
        )

        self.output_layer = nn.Sequential(
            nn.Conv2d(256, lmk_num, 1),
        )

        row_loc, col_loc = self.make_grid("cpu", size=32)

        self.register_buffer("xx_loc", col_loc, False)
        self.register_buffer("yy_loc", row_loc, False)

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

    def forward(self, img, data_type):
        B, C, H, W = img.shape
        x = self.pre(img)
        heatmap = self.unet(x)
        embeding = self.embed(data_type).reshape((B, 256, 32, 32))
        heatmap = self.block1(heatmap + embeding)
        heatmap = self.block2(heatmap + embeding)
        heatmap = self.block3(heatmap + embeding)
        heatmap = self.block4(heatmap + embeding)
        heatmap = self.output_layer(heatmap)
        heatmap = heatmap - torch.amax(heatmap, dim=(2, 3), keepdim=True)
        heatmap = torch.exp(heatmap)
        heatmap = heatmap / (1e-6 + torch.sum(heatmap, dim=(2, 3), keepdim=True))

        xx = (heatmap * self.xx_loc).sum([2, 3])
        yy = (heatmap * self.yy_loc).sum([2, 3])
        return torch.stack([xx, yy], dim=2), heatmap


class NetAttnSA2Branch(nn.Module):
    def __init__(self, lmk_num=98, in_channel=3):
        super(NetAttnSA2Branch, self).__init__()
        self.pre = HeadingNet([32, 64, 256], in_channel)

        self.unet = UNet([256, 256, 256, 256])

        # self.output_layer= nn.Conv2d(256, lmk_num,1)

        self.branch1 = nn.Sequential(
            CoordConvTh(32, 32, True, False, 256, 256, kernel_size=3, padding=1),
            SelfAttention2(32),
            CoordConvTh(32, 32, True, False, 256, 256, kernel_size=3, padding=1),
            SelfAttention2(32),
            CoordConvTh(32, 32, True, False, 256, 256, kernel_size=3, padding=1),
            SelfAttention2(32),
            CoordConvTh(32, 32, True, False, 256, 256, kernel_size=3, padding=1),
            SelfAttention2(32),
        )
        self.branch2 = nn.Sequential(
            CoordConvTh(32, 32, True, False, 256, 256, kernel_size=3, padding=1),
            SelfAttention2(32),
            CoordConvTh(32, 32, True, False, 256, 256, kernel_size=3, padding=1),
            SelfAttention2(32),
            CoordConvTh(32, 32, True, False, 256, 256, kernel_size=3, padding=1),
            SelfAttention2(32),
            CoordConvTh(32, 32, True, False, 256, 256, kernel_size=3, padding=1),
            SelfAttention2(32),
        )
        self.branch3 = nn.Sequential(
            CoordConvTh(32, 32, True, False, 256, 256, kernel_size=3, padding=1),
            SelfAttention2(32),
            CoordConvTh(32, 32, True, False, 256, 256, kernel_size=3, padding=1),
            SelfAttention2(32),
            CoordConvTh(32, 32, True, False, 256, 256, kernel_size=3, padding=1),
            SelfAttention2(32),
            CoordConvTh(32, 32, True, False, 256, 256, kernel_size=3, padding=1),
            SelfAttention2(32),
        )

        self.branch4 = nn.Sequential(
            CoordConvTh(32, 32, True, False, 256, 256, kernel_size=3, padding=1),
            SelfAttention2(32),
            CoordConvTh(32, 32, True, False, 256, 256, kernel_size=3, padding=1),
            SelfAttention2(32),
            CoordConvTh(32, 32, True, False, 256, 256, kernel_size=3, padding=1),
            SelfAttention2(32),
            CoordConvTh(32, 32, True, False, 256, 256, kernel_size=3, padding=1),
            SelfAttention2(32),
        )

        self.output_layer = nn.Conv2d(256, lmk_num, 1)

        row_loc, col_loc = self.make_grid("cpu", size=32)

        self.register_buffer("xx_loc", col_loc, False)
        self.register_buffer("yy_loc", row_loc, False)

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
        heatmap = heatmap - torch.amax(heatmap, dim=(2, 3), keepdim=True)
        heatmap = torch.exp(heatmap)
        heatmap = heatmap / (1e-4 + torch.sum(heatmap, dim=(2, 3), keepdim=True))

        xx = (heatmap * self.xx_loc).sum([2, 3])
        yy = (heatmap * self.yy_loc).sum([2, 3])
        return torch.stack([xx, yy], dim=2)

    def forward(self, img):
        x = self.pre(img)
        feat = self.unet(x)
        b1 = self.branch1(feat)
        b2 = self.branch2(feat)
        b3 = self.branch3(feat)
        b4 = self.branch4(feat)

        heatmap = self.output_layer(b1 + b2 + b3 + b4)
        coord = self.GetCoord(heatmap)
        return coord, heatmap


class NetAttnSA2BranchPatch(nn.Module):
    def __init__(self, lmk_num=98, in_channel=3):
        super(NetAttnSA2BranchPatch, self).__init__()
        self.pre = HeadingNet([32, 64, 256], in_channel)

        self.unet = UNet([256, 256, 256, 256])

        # self.output_layer= nn.Conv2d(256, lmk_num,1)

        self.branch1 = nn.Sequential(
            CoordConvTh(32, 32, True, False, 256, 256, kernel_size=3, padding=1),
            SelfAttention2(32),
            CoordConvTh(32, 32, True, False, 256, 256, kernel_size=3, padding=1),
            SelfAttention2(32),
            CoordConvTh(32, 32, True, False, 256, 256, kernel_size=3, padding=1),
            SelfAttention2(32),
            CoordConvTh(32, 32, True, False, 256, 256, kernel_size=3, padding=1),
            SelfAttention2(32),
        )
        self.branch2 = nn.Sequential(
            CoordConvTh(32, 32, True, False, 256, 256, kernel_size=3, padding=1),
            SelfAttention2(32),
            CoordConvTh(32, 32, True, False, 256, 256, kernel_size=3, padding=1),
            SelfAttention2(32),
            CoordConvTh(32, 32, True, False, 256, 256, kernel_size=3, padding=1),
            SelfAttention2(32),
            CoordConvTh(32, 32, True, False, 256, 256, kernel_size=3, padding=1),
            SelfAttention2(32),
        )
        self.branch3 = nn.Sequential(
            CoordConvTh(32, 32, True, False, 256, 256, kernel_size=3, padding=1),
            SelfAttention2(32),
            CoordConvTh(32, 32, True, False, 256, 256, kernel_size=3, padding=1),
            SelfAttention2(32),
            CoordConvTh(32, 32, True, False, 256, 256, kernel_size=3, padding=1),
            SelfAttention2(32),
            CoordConvTh(32, 32, True, False, 256, 256, kernel_size=3, padding=1),
            SelfAttention2(32),
        )

        self.branch4 = nn.Sequential(
            CoordConvTh(32, 32, True, False, 256, 256, kernel_size=3, padding=1),
            SelfAttention2(32),
            CoordConvTh(32, 32, True, False, 256, 256, kernel_size=3, padding=1),
            SelfAttention2(32),
            CoordConvTh(32, 32, True, False, 256, 256, kernel_size=3, padding=1),
            SelfAttention2(32),
            CoordConvTh(32, 32, True, False, 256, 256, kernel_size=3, padding=1),
            SelfAttention2(32),
        )

        self.output_layer = nn.Conv2d(256, lmk_num, 1)

        row_loc, col_loc = self.make_grid("cpu", size=32)

        self.register_buffer("xx_loc", col_loc, False)
        self.register_buffer("yy_loc", row_loc, False)

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
        heatmap = heatmap - torch.amax(heatmap, dim=(2, 3), keepdim=True)
        heatmap = torch.exp(heatmap)
        heatmap = heatmap / (1e-4 + torch.sum(heatmap, dim=(2, 3), keepdim=True))

        xx = (heatmap * self.xx_loc).sum([2, 3])
        yy = (heatmap * self.yy_loc).sum([2, 3])
        return torch.stack([xx, yy], dim=2)

    def forward(self, img, training=True):
        x = self.pre(img)
        B, C, H, W = x.shape
        feat = self.unet(x)

        if training:
            n = B // 4
            b1 = self.branch1(feat[:n])
            b2 = self.branch2(feat[n : n + n])
            b3 = self.branch3(feat[n + n : n * 3])
            b4 = self.branch4(feat[n * 3 :])
            b = torch.cat([b1, b2, b3, b4], dim=0)
        else:
            b1 = self.branch1(feat)
            b2 = self.branch2(feat)
            b3 = self.branch3(feat)
            b4 = self.branch4(feat)
            b = (b1 + b2 + b3 + b4) / 4
        heatmap = self.output_layer(b)
        coord = self.GetCoord(heatmap)
        return coord, heatmap


class NetAttnSA2Heatmap(nn.Module):
    def __init__(self, lmk_num=98):
        super(NetAttnSA2Heatmap, self).__init__()
        self.pre = HeadingNet([32, 64, 256])

        self.unet = UNet([256, 256, 256, 256])

        # self.output_layer= nn.Conv2d(256, lmk_num,1)
        self.output_layer = nn.Sequential(
            CoordConvTh(32, 32, True, False, 256, 256, kernel_size=3, padding=1),
            SelfAttention2(32),
            CoordConvTh(32, 32, True, False, 256, 256, kernel_size=3, padding=1),
            SelfAttention2(32),
            CoordConvTh(32, 32, True, False, 256, 256, kernel_size=3, padding=1),
            SelfAttention2(32),
            CoordConvTh(32, 32, True, False, 256, 256, kernel_size=3, padding=1),
            SelfAttention2(32),
            nn.Conv2d(256, lmk_num, 1),
        )
        self.heatmap = Heatmap(30)

    def forward(self, img):
        x = self.pre(img)
        heatmap = self.unet(x)
        heatmap = self.output_layer(heatmap)
        return self.heatmap(heatmap)


class NetAttnSA2GobalLocal(nn.Module):
    """
    how is the performance to use a global key?

    """

    def __init__(self, lmk_num=98):
        super(NetAttnSA2GobalLocal, self).__init__()
        self.pre = HeadingNet([32, 64, 256])

        self.unet = UNet([256, 256, 256, 256])

        self.sa_g = SelfAttention2(32)

        self.coordconv1 = CoordConvTh(
            32, 32, True, False, 256, 256, kernel_size=3, padding=1
        )
        self.sa_l1 = SelfAttention2(32)
        self.coordconv2 = CoordConvTh(
            32, 32, True, False, 256, 256, kernel_size=3, padding=1
        )
        self.sa_l2 = SelfAttention2(32)
        self.coordconv3 = CoordConvTh(
            32, 32, True, False, 256, 256, kernel_size=3, padding=1
        )
        self.sa_l3 = SelfAttention2(32)
        self.coordconv4 = CoordConvTh(
            32, 32, True, False, 256, 256, kernel_size=3, padding=1
        )
        self.sa_l4 = SelfAttention2(32)
        self.output_layer = nn.Sequential(
            nn.Conv2d(256, lmk_num, 1),
        )

        row_loc, col_loc = self.make_grid("cpu", size=32)

        self.register_buffer("xx_loc", col_loc, False)
        self.register_buffer("yy_loc", row_loc, False)

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

    def forward(self, img):
        x = self.pre(img)
        feature = self.unet(x)

        heatmap = self.coordconv1(feature)
        heatmap = (self.sa_l1(heatmap) + self.sa_g(heatmap)) / 2.0

        heatmap = self.coordconv2(heatmap)
        heatmap = (self.sa_l2(heatmap) + self.sa_g(heatmap)) / 2.0
        heatmap = self.coordconv3(heatmap)
        heatmap = (self.sa_l3(heatmap) + self.sa_g(heatmap)) / 2.0
        heatmap = self.coordconv4(heatmap)
        heatmap = (self.sa_l4(heatmap) + self.sa_g(heatmap)) / 2.0

        heatmap = self.output_layer(heatmap)
        heatmap = heatmap - torch.amax(heatmap, dim=(2, 3), keepdim=True)
        heatmap = torch.exp(heatmap)
        heatmap = heatmap / (1e-4 + torch.sum(heatmap, dim=(2, 3), keepdim=True))

        xx = (heatmap * self.xx_loc).sum([2, 3])
        yy = (heatmap * self.yy_loc).sum([2, 3])
        return torch.stack([xx, yy], dim=2), heatmap


class NetAttnStage(nn.Module):
    def __init__(
        self,
        lmk_num=98,
        Attn=lambda: SA2SA1(32, 256),
        nstack=4,
        heatmap_size=32,
        max_depth=256,
        increase=False,
    ):
        super(NetAttnStage, self).__init__()
        if heatmap_size == 16:
            self.pre = HeadingNet([32, 64, 128, max_depth])
            self.unet = UNet([max_depth, max_depth, max_depth, max_depth])
        elif heatmap_size == 32:
            self.pre = HeadingNet([32, 64, max_depth])
            self.unet = UNet([max_depth, max_depth, max_depth, max_depth])
        elif heatmap_size == 64:
            self.pre = HeadingNet([64, max_depth])
            self.unet = UNet([max_depth, max_depth, max_depth, max_depth, max_depth])
        else:
            assert 0

        stages = []
        output_layers = []
        merge = []
        for i in range(nstack):
            block = nn.Sequential(
                CoordConvTh(
                    heatmap_size,
                    heatmap_size,
                    True,
                    False,
                    max_depth,
                    max_depth,
                    kernel_size=3,
                    padding=1,
                ),
                Attn(),
                CoordConvTh(
                    heatmap_size,
                    heatmap_size,
                    True,
                    False,
                    max_depth,
                    max_depth,
                    kernel_size=3,
                    padding=1,
                ),
                Attn(),
            )
            if increase:
                for j in range(i):
                    block.append(
                        CoordConvTh(
                            heatmap_size,
                            heatmap_size,
                            True,
                            False,
                            max_depth,
                            max_depth,
                            kernel_size=3,
                            padding=1,
                        )
                    )
                    block.append(Attn())
            stages.append(block)
            output_layers.append(nn.Conv2d(max_depth, lmk_num, 1))
            if i > 0:
                merge.append(DoubleConv(max_depth * 2, max_depth, max_depth))
        self.stages = nn.ModuleList(stages)
        self.output_layers = nn.ModuleList(output_layers)
        self.merge = nn.ModuleList(merge)

        row_loc, col_loc = self.make_grid("cpu", size=heatmap_size)
        self.register_buffer("xx_loc", col_loc, False)
        self.register_buffer("yy_loc", row_loc, False)

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

    def forward(self, img):
        x = self.pre(img)
        feat = self.unet(x)
        pre_hm = 0
        res = []
        for i in range(len(self.stages)):
            if i > 0:
                merge = self.merge[i - 1](torch.cat([feat, pre_hm], dim=1))
            else:
                merge = feat
            hm_0 = self.stages[i](merge)
            pre_hm = hm_0
            hm = self.output_layers[i](hm_0)
            coord = self.GetCoord(hm)
            res.append((coord, hm))

        return res


class NetAttnStageResSkip(nn.Module):
    def __init__(
        self,
        lmk_num=98,
        Attn=lambda: SA2SA1(32, 256),
        nstack=4,
        heatmap_size=32,
        max_depth=256,
        increase=False,
    ):
        super(NetAttnStageResSkip, self).__init__()
        if heatmap_size == 16:
            self.pre = HeadingNet([32, 64, 128, max_depth])
            self.unet = UNet([max_depth, max_depth, max_depth, max_depth])
        elif heatmap_size == 32:
            self.pre = HeadingNet([32, 64, max_depth])
            self.unet = UNet([max_depth, max_depth, max_depth, max_depth])
        elif heatmap_size == 64:
            self.pre = HeadingNet([64, max_depth])
            self.unet = UNet([max_depth, max_depth, max_depth, max_depth, max_depth])
        else:
            assert 0

        stages = []
        output_layers = []
        merge = []
        for i in range(nstack):
            block = nn.Sequential(
                CoordConvTh(
                    heatmap_size,
                    heatmap_size,
                    True,
                    False,
                    max_depth,
                    max_depth,
                    kernel_size=3,
                    padding=1,
                ),
                Attn(),
                CoordConvTh(
                    heatmap_size,
                    heatmap_size,
                    True,
                    False,
                    max_depth,
                    max_depth,
                    kernel_size=3,
                    padding=1,
                ),
                Attn(),
            )
            if increase:
                for j in range(i):
                    block.append(
                        CoordConvTh(
                            heatmap_size,
                            heatmap_size,
                            True,
                            False,
                            max_depth,
                            max_depth,
                            kernel_size=3,
                            padding=1,
                        )
                    )
                    block.append(Attn())
            stages.append(block)
            output_layers.append(nn.Conv2d(max_depth, lmk_num, 1))
            if i > 0:
                merge.append(DoubleConv(max_depth * 2, max_depth, max_depth))
        self.stages = nn.ModuleList(stages)
        self.output_layers = nn.ModuleList(output_layers)
        self.merge = nn.ModuleList(merge)

        row_loc, col_loc = self.make_grid("cpu", size=heatmap_size)
        self.register_buffer("xx_loc", col_loc, False)
        self.register_buffer("yy_loc", row_loc, False)

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

    def forward(self, img):
        feat = self.pre(img)
        pre_input = 0
        pre_output = feat
        res = []
        for i in range(len(self.stages)):
            if i > 0:
                merge = self.merge[i - 1](torch.cat([pre_input, pre_output], dim=1))
            else:
                merge = feat
            hm_0 = self.stages[i](merge)
            hm = self.output_layers[i](hm_0)
            coord = self.GetCoord(hm)
            res.append((coord, hm))

            pre_input = pre_output
            pre_output = hm_0

        return res


class Net50AttnStage(nn.Module):
    def __init__(
        self,
        lmk_num=98,
        Attn=lambda: SA2SA1(32, 256),
        nstack=4,
        heatmap_size=32,
        max_depth=256,
    ):
        super(Net50AttnStage, self).__init__()
        assert heatmap_size == 32
        self.pre = Res50(max_depth)

        stages = []
        output_layers = []
        merge = []
        for i in range(nstack):
            block = nn.Sequential(
                CoordConvTh(
                    heatmap_size,
                    heatmap_size,
                    True,
                    False,
                    max_depth,
                    max_depth,
                    kernel_size=3,
                    padding=1,
                ),
                Attn(),
                CoordConvTh(
                    heatmap_size,
                    heatmap_size,
                    True,
                    False,
                    max_depth,
                    max_depth,
                    kernel_size=3,
                    padding=1,
                ),
                Attn(),
            )
            stages.append(block)
            output_layers.append(nn.Conv2d(max_depth, lmk_num, 1))
            if i > 0:
                merge.append(DoubleConv(max_depth * 2, max_depth, max_depth))
        self.stages = nn.ModuleList(stages)
        self.output_layers = nn.ModuleList(output_layers)
        self.merge = nn.ModuleList(merge)

        row_loc, col_loc = self.make_grid("cpu", size=heatmap_size)
        self.register_buffer("xx_loc", col_loc, False)
        self.register_buffer("yy_loc", row_loc, False)

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

    def forward(self, img):
        feat = self.pre(img)
        pre_hm = 0
        res = []
        for i in range(len(self.stages)):
            if i > 0:
                merge = self.merge[i - 1](torch.cat([feat, pre_hm], dim=1))
            else:
                merge = feat
            hm_0 = self.stages[i](merge)
            pre_hm = hm_0
            hm = self.output_layers[i](hm_0)
            coord = self.GetCoord(hm)
            res.append((coord, hm))

        return res


class VitAttnStageMultiResHM(nn.Module):
    def __init__(
        self,
        lmk_num=98,
        Attn=lambda: SA2SA1(32, 256),
        nstack=4,
        heatmap_size=32,
        max_depth=256,
        backbone_net=lambda max_depth: HeadingNet([32, 64, max_depth]),
        num_dvit_per_pred_blk=2,
    ):
        super(VitAttnStageMultiResHM, self).__init__()
        # assert heatmap_size == 32
        assert (
            max_depth == 256 or max_depth == 192 or max_depth == 128 or max_depth == 64
        )

        self.pre = backbone_net(max_depth)

        stages = []
        output_layers = []
        output_layers16 = []
        output_layers8 = []
        merge = []
        for i in range(nstack):
            vit_list = []
            for num in range(num_dvit_per_pred_blk):
                vit_list.append(
                    CoordConvTh(
                        heatmap_size,
                        heatmap_size,
                        True,
                        False,
                        max_depth,
                        max_depth,
                        kernel_size=3,
                        padding=1,
                    )
                )
                vit_list.append(Attn())
            block = nn.Sequential(*vit_list)
            stages.append(block)
            output_layers.append(nn.Conv2d(max_depth, lmk_num, 1))
            output_layers16.append(
                nn.Conv2d(max_depth, lmk_num, 3, stride=2, padding=1)
            )
            output_layers8.append(
                nn.Sequential(
                    nn.AvgPool2d(2),
                    nn.Conv2d(max_depth, lmk_num, 3, stride=2, padding=1),
                )
            )
            if i > 0:
                merge.append(DoubleConv(max_depth * 2, max_depth, max_depth))
        self.stages = nn.ModuleList(stages)
        self.output_layers = nn.ModuleList(output_layers)
        self.output_layers16 = nn.ModuleList(output_layers16)
        self.output_layers8 = nn.ModuleList(output_layers8)
        self.merge = nn.ModuleList(merge)

        row_loc, col_loc = self.make_grid("cpu", size=heatmap_size)
        self.register_buffer("xx_loc", col_loc, False)
        self.register_buffer("yy_loc", row_loc, False)

        row_loc, col_loc = self.make_grid("cpu", size=heatmap_size // 2)
        self.register_buffer("xx_loc16", col_loc, False)
        self.register_buffer("yy_loc16", row_loc, False)

        row_loc, col_loc = self.make_grid("cpu", size=heatmap_size // 4)
        self.register_buffer("xx_loc8", col_loc, False)
        self.register_buffer("yy_loc8", row_loc, False)

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

    def GetCoord16(self, heatmap):
        # print(heatmap.shape, self.xx_loc16.shape, self.yy_loc16.shape)
        B, C, H, W = heatmap.shape
        heatmap = F.softmax(heatmap.reshape((B, C, -1)), dim=-1).reshape((B, C, H, W))
        xx = (heatmap * self.xx_loc16).sum([2, 3])
        yy = (heatmap * self.yy_loc16).sum([2, 3])
        return torch.stack([xx, yy], dim=2)

    def GetCoord8(self, heatmap):
        # print(heatmap.shape, self.xx_loc16.shape, self.yy_loc16.shape)
        B, C, H, W = heatmap.shape
        heatmap = F.softmax(heatmap.reshape((B, C, -1)), dim=-1).reshape((B, C, H, W))
        xx = (heatmap * self.xx_loc8).sum([2, 3])
        yy = (heatmap * self.yy_loc8).sum([2, 3])
        return torch.stack([xx, yy], dim=2)

    def forward_NoConnection(self, img):
        feat = self.pre(img)
        pre_hm = feat
        res = []
        for i in range(len(self.stages)):
            hm_0 = self.stages[i](pre_hm)
            pre_hm = hm_0
            hm = self.output_layers[i](hm_0)
            coord = self.GetCoord(hm)
            res.append((coord, hm))

        return res

    def forward_res(self, img):
        feat = self.pre(img)
        pre_input = 0
        pre_output = feat
        res = []
        for i in range(len(self.stages)):
            if i > 0:
                merge = self.merge[i - 1](torch.cat([pre_input, pre_output], dim=1))
            else:
                merge = feat
            hm_0 = self.stages[i](merge)
            hm = self.output_layers[i](hm_0)
            coord = self.GetCoord(hm)
            res.append((coord, hm))

            pre_input = pre_output
            pre_output = hm_0

        return res

    def forward_res3(self, img):
        feat = self.pre(img)
        pre_merged = 0
        pre_output = feat
        res = []
        for i in range(len(self.stages)):
            if i > 0:
                merge = self.merge[i - 1](torch.cat([pre_merged, pre_output], dim=1))
            else:
                merge = feat
            hm_0 = self.stages[i](merge)
            hm = self.output_layers[i](hm_0)
            coord = self.GetCoord(hm)
            res.append((coord, hm))

            pre_merged = merge
            pre_output = hm_0

        return res

    def forward(self, img, connect_type=1):
        if connect_type == 1:
            feat = self.pre(img)
            pre_hm = 0
            res = []
            for i in range(len(self.stages)):
                if i > 0:
                    merge = self.merge[i - 1](torch.cat([feat, pre_hm], dim=1))
                else:
                    merge = feat
                hm_0 = self.stages[i](merge)
                pre_hm = hm_0
                hm = self.output_layers[i](hm_0)
                hm16 = self.output_layers16[i](hm_0)
                hm8 = self.output_layers8[i](hm_0)
                coord = self.GetCoord(hm)
                coord16 = self.GetCoord16(hm16)
                coord8 = self.GetCoord8(hm8)
                res.append([coord, hm, coord16, hm16, coord8, hm8])

            return res
        elif connect_type == 0:
            return self.forward_NoConnection(img)
        elif connect_type == 2:
            return self.forward_res(img)
        elif connect_type == 3:
            return self.forward_res3(img)


class NetAttnStageDW(nn.Module):
    """
    DW: dynamic weighted
    """

    def __init__(
        self, lmk_num=98, Attn=SA2SA1, nstack=4, heatmap_size=32, max_depth=256
    ):
        super(NetAttnStageDW, self).__init__()
        if heatmap_size == 16:
            self.pre = HeadingNet([32, 64, 128, max_depth])
            self.unet = UNet([max_depth, max_depth, max_depth, max_depth])
        elif heatmap_size == 32:
            self.pre = HeadingNet([32, 64, max_depth])
            self.unet = UNet([max_depth, max_depth, max_depth, max_depth])
        elif heatmap_size == 64:
            self.pre = HeadingNet([64, max_depth])
            self.unet = UNet([max_depth, max_depth, max_depth, max_depth, max_depth])
        else:
            assert 0

        stages = []
        output_layers = []
        merge = []
        weighed = []
        for i in range(nstack):
            block = nn.Sequential(
                CoordConvTh(
                    heatmap_size,
                    heatmap_size,
                    True,
                    False,
                    max_depth,
                    max_depth,
                    kernel_size=3,
                    padding=1,
                ),
                Attn(heatmap_size, max_depth),
                CoordConvTh(
                    heatmap_size,
                    heatmap_size,
                    True,
                    False,
                    max_depth,
                    max_depth,
                    kernel_size=3,
                    padding=1,
                ),
                Attn(heatmap_size, max_depth),
            )
            stages.append(block)
            output_layers.append(nn.Conv2d(max_depth, lmk_num, 1))
            weighed.append(
                nn.Sequential(
                    nn.BatchNorm2d(max_depth),
                    nn.Conv2d(max_depth, lmk_num, 3, stride=2, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(lmk_num, lmk_num, 3, stride=2, padding=1),
                )
            )
            if i > 0:
                merge.append(DoubleConv(max_depth * 2, max_depth, max_depth))
        self.stages = nn.ModuleList(stages)
        self.output_layers = nn.ModuleList(output_layers)
        self.merge = nn.ModuleList(merge)
        self.weighted = nn.ModuleList(weighed)

        row_loc, col_loc = self.make_grid("cpu", size=heatmap_size)
        self.register_buffer("xx_loc", col_loc, False)
        self.register_buffer("yy_loc", row_loc, False)

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

    def GetWeight(self, w):
        B, C, _, _ = w.shape
        return torch.softmax(torch.mean(w, dim=[2, 3]), dim=1) + 1

    def forward(self, img):
        x = self.pre(img)
        feat = self.unet(x)
        pre_hm = 0
        res = []
        for i in range(len(self.stages)):
            if i > 0:
                merge = self.merge[i - 1](torch.cat([feat, pre_hm], dim=1))
            else:
                merge = feat
            hm_0 = self.stages[i](merge)
            pre_hm = hm_0
            hm = self.output_layers[i](hm_0)
            coord = self.GetCoord(hm)

            weight = self.weighted[i](hm_0)
            weight = self.GetWeight(weight)

            res.append((coord, hm, weight))

        return res


class NetAttn64SA2(nn.Module):
    def __init__(self, lmk_num=98):
        super(NetAttn64SA2, self).__init__()
        self.pre = HeadingNet([32, 128])

        self.unet = UNet([128, 128, 128, 128])

        # self.output_layer= nn.Conv2d(256, lmk_num,1)
        self.output_layer = nn.Sequential(
            CoordConvTh(64, 64, True, False, 128, 128, kernel_size=3, padding=1),
            SelfAttention2(64),
            CoordConvTh(64, 64, True, False, 128, 128, kernel_size=3, padding=1),
            SelfAttention2(64),
            CoordConvTh(64, 64, True, False, 128, 128, kernel_size=3, padding=1),
            SelfAttention2(64),
            CoordConvTh(64, 64, True, False, 128, 128, kernel_size=3, padding=1),
            SelfAttention2(64),
            nn.Conv2d(128, lmk_num, 1),
        )

        row_loc, col_loc = self.make_grid("cpu", size=64)

        self.register_buffer("xx_loc", col_loc, False)
        self.register_buffer("yy_loc", row_loc, False)

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

    def forward(self, img):
        x = self.pre(img)
        heatmap = self.unet(x)
        heatmap = self.output_layer(heatmap)

        heatmap = heatmap - torch.amax(heatmap, dim=(2, 3), keepdim=True)
        heatmap = torch.exp(heatmap)
        heatmap = heatmap / (1e-8 + torch.sum(heatmap, dim=(2, 3), keepdim=True))

        xx = (heatmap * self.xx_loc).sum([2, 3])
        yy = (heatmap * self.yy_loc).sum([2, 3])
        return torch.stack([xx, yy], dim=2), heatmap


class UNetStage(nn.Module):
    def __init__(
        self,
        lmk_num=98,
        nstack=4,
        heatmap_size=32,
        max_depth=256,
        feature_extractor=None,
    ):
        super(UNetStage, self).__init__()
        if feature_extractor is None:
            if heatmap_size == 16:
                self.pre = HeadingNet([32, 64, 128, max_depth])
            elif heatmap_size == 32:
                self.pre = HeadingNet([32, 64, max_depth])
            elif heatmap_size == 64:
                self.pre = HeadingNet([64, max_depth])
            else:
                assert 0
        else:
            self.pre = feature_extractor()

        stages = []
        output_layers = []
        merge = []
        for i in range(nstack):
            if heatmap_size == 32 or heatmap_size == 16:
                block = UNet([max_depth, max_depth, max_depth, max_depth])
            elif heatmap_size == 64:
                block = UNet([max_depth, max_depth, max_depth, max_depth, max_depth])
            stages.append(block)
            output_layers.append(nn.Conv2d(max_depth, lmk_num, 1))
            if i > 0:
                merge.append(DoubleConv(max_depth * 2, max_depth, max_depth))
        self.stages = nn.ModuleList(stages)
        self.output_layers = nn.ModuleList(output_layers)
        self.merge = nn.ModuleList(merge)

        row_loc, col_loc = self.make_grid("cpu", size=heatmap_size)
        self.register_buffer("xx_loc", col_loc, False)
        self.register_buffer("yy_loc", row_loc, False)

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

    def forward_res(self, img):
        feat = self.pre(img)
        pre_input = 0
        pre_output = feat
        res = []
        for i in range(len(self.stages)):
            if i > 0:
                merge = self.merge[i - 1](torch.cat([pre_input, pre_output], dim=1))
            else:
                merge = feat
            hm_0 = self.stages[i](merge)
            hm = self.output_layers[i](hm_0)
            coord = self.GetCoord(hm)
            res.append((coord, hm))

            pre_input = pre_output
            pre_output = hm_0

        return res

    def forward_LSC(self, img):
        feat = self.pre(img)
        pre_hm = 0
        res = []
        for i in range(len(self.stages)):
            if i > 0:
                merge = self.merge[i - 1](torch.cat([feat, pre_hm], dim=1))
            else:
                merge = feat
            hm_0 = self.stages[i](merge)
            pre_hm = hm_0
            hm = self.output_layers[i](hm_0)
            coord = self.GetCoord(hm)
            res.append((coord, hm))

        return res

    def forward(self, img, lsc=True):
        if lsc:
            return self.forward_LSC(img)
        else:
            return self.forward_res(img)


class LocalCrossLandmarkVisibilityHead(nn.Module):
    """OccFace-inspired per-point visibility reasoning head.

    This replaces the older simple visibility path:

        Conv2d(max_depth, point_count, 1).mean(dim=(2, 3))

    with:

    1. Local landmark-conditioned tokens:
       heatmap-weighted pooling of image features around each predicted landmark.

    2. Cross-landmark context:
       self-attention over all landmark tokens for the schema/head.

    3. Gated fusion:
       learnable blend of local occluder evidence and cross-landmark context.

    Output:
        logits with shape [B, point_count]

    Target convention remains unchanged:
        1 = visible
        0 = occluded
       -1 = unknown / masked out by loss
    """

    def __init__(
        self,
        in_channels: int,
        point_count: int,
        hidden_channels: int = 128,
        num_attention_heads: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.point_count = int(point_count)

        if hidden_channels % num_attention_heads != 0:
            raise ValueError(
                "hidden_channels must be divisible by num_attention_heads: "
                f"{hidden_channels=} {num_attention_heads=}"
            )

        self.feature_proj = nn.Conv2d(
            int(in_channels),
            int(hidden_channels),
            kernel_size=1,
        )

        self.local_norm = nn.LayerNorm(hidden_channels)
        self.local_mlp = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, hidden_channels),
        )

        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_channels,
            num_heads=num_attention_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.cross_norm = nn.LayerNorm(hidden_channels)
        self.cross_mlp = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, hidden_channels),
        )

        self.fusion_gate = nn.Sequential(
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.GELU(),
            nn.Linear(hidden_channels, hidden_channels),
            nn.Sigmoid(),
        )

        self.output = nn.Linear(hidden_channels, 1)

    def forward(self, feature, landmark_heatmap):
        """Return per-point visibility logits.

        Args:
            feature:
                Stage feature map, shape [B, C, H, W].
            landmark_heatmap:
                Landmark heatmap logits for the same schema/head,
                shape [B, N, Hh, Wh].

        Returns:
            Tensor with shape [B, N].
        """
        if landmark_heatmap.ndim != 4:
            raise ValueError(
                "landmark_heatmap must have shape [B, N, H, W], "
                f"got {tuple(landmark_heatmap.shape)}"
            )

        local_feature = self.feature_proj(feature)
        batch_size, hidden_channels, height, width = local_feature.shape

        if landmark_heatmap.shape[0] != batch_size:
            raise ValueError(
                "feature and landmark_heatmap batch sizes differ: "
                f"{batch_size=} heatmap_batch={landmark_heatmap.shape[0]}"
            )

        point_count = int(landmark_heatmap.shape[1])
        if point_count != self.point_count:
            raise ValueError(
                "visibility head point count mismatch: "
                f"expected {self.point_count}, got {point_count}"
            )

        if landmark_heatmap.shape[-2:] != (height, width):
            landmark_heatmap = F.interpolate(
                landmark_heatmap,
                size=(height, width),
                mode="bilinear",
                align_corners=False,
            )

        # Landmark-conditioned local evidence.
        #
        # heatmap_prob:  [B, N, H*W]
        # feature_flat:  [B, C, H*W]
        # local_tokens:  [B, N, C]
        heatmap_prob = F.softmax(
            landmark_heatmap.reshape(batch_size, point_count, -1),
            dim=-1,
        )
        feature_flat = local_feature.reshape(batch_size, hidden_channels, -1)
        local_tokens = torch.einsum("bnh,bch->bnc", heatmap_prob, feature_flat)

        local_tokens = self.local_norm(local_tokens)
        local_context = self.local_mlp(local_tokens)

        # Cross-landmark context for structured occlusion / self-occlusion.
        cross_tokens, _ = self.cross_attention(
            local_tokens,
            local_tokens,
            local_tokens,
            need_weights=False,
        )
        cross_tokens = self.cross_norm(local_tokens + cross_tokens)
        cross_context = self.cross_mlp(cross_tokens)

        # Gated local/cross fusion.
        gate = self.fusion_gate(torch.cat([local_context, cross_context], dim=-1))
        fused = gate * local_context + (1.0 - gate) * cross_context

        return self.output(fused).squeeze(-1)


class VitAttnStage(nn.Module):
    def __init__(
        self,
        lmk_num=98,
        Attn=lambda: SA2SA1(32, 256),
        nstack=4,
        heatmap_size=32,
        max_depth=256,
        backbone_net=lambda max_depth: HeadingNet([32, 64, max_depth]),
        # backbone_net=lambda max_depth: HeadingNet([32, 64, 128, max_depth]),
        # backbone_net=lambda max_depth: HeadingNet([32, 64, 128, 256, max_depth]),
        # backbone_net=Vit
        num_dvit_per_pred_blk=2,
        schema_heads=None,
        auxiliary_heads=None,
        visibility_heads=False,
        visibility_all_stages=False,
    ):
        super(VitAttnStage, self).__init__()
        # assert heatmap_size == 32
        # assert max_depth == 256 or max_depth == 192 or max_depth == 128 or max_depth == 64

        self.pre = backbone_net(max_depth)
        self.schema_heads = dict(schema_heads or {})
        self.multi_schema = bool(self.schema_heads)
        if self.multi_schema:
            self.schema_heads.setdefault("landmarks_68", lmk_num)
        self.auxiliary_heads = dict(auxiliary_heads or {})
        self.visibility_heads_enabled = bool(visibility_heads and self.multi_schema)
        # When False (default), the landmark-conditioned visibility head only
        # runs on the final stage, since the loss and evaluator consume only
        # the final stage's visibility output unless all-stage supervision is
        # requested. Set True to compute it on every stage (e.g. when
        # auxiliary_loss_stage == 'all'). Modules are always instantiated for
        # every stage so checkpoints stay compatible regardless of this flag.
        self.visibility_all_stages = bool(visibility_all_stages)

        stages = []
        output_layers = []
        schema_output_layers = {
            name: [] for name in self.schema_heads if name != "landmarks_68"
        }
        visibility_output_layers = (
            {name: [] for name in self.schema_heads}
            if self.visibility_heads_enabled
            else {}
        )
        merge = []
        for i in range(nstack):
            vit_list = []
            for num in range(num_dvit_per_pred_blk):
                vit_list.append(
                    CoordConvTh(
                        heatmap_size,
                        heatmap_size,
                        True,
                        False,
                        max_depth,
                        max_depth,
                        kernel_size=3,
                        padding=1,
                    )
                )
                vit_list.append(Attn())
            block = nn.Sequential(*vit_list)
            # block = nn.Sequential(
            #     CoordConvTh(heatmap_size, heatmap_size, True, False, max_depth, max_depth, kernel_size=3, padding=1),
            #     Attn(),
            #     CoordConvTh(heatmap_size, heatmap_size, True, False, max_depth, max_depth, kernel_size=3, padding=1),
            #     Attn(),
            # )
            stages.append(block)
            output_layers.append(nn.Conv2d(max_depth, lmk_num, 1))
            for name, point_count in schema_output_layers.items():
                point_count = int(self.schema_heads[name])
                schema_output_layers[name].append(nn.Conv2d(max_depth, point_count, 1))
            for name, point_count in visibility_output_layers.items():
                point_count = int(self.schema_heads[name])
                visibility_output_layers[name].append(
                    LocalCrossLandmarkVisibilityHead(
                        in_channels=max_depth,
                        point_count=point_count,
                    )
                )
            if i > 0:
                merge.append(DoubleConv(max_depth * 2, max_depth, max_depth))
        self.stages = nn.ModuleList(stages)
        self.output_layers = nn.ModuleList(output_layers)
        self.schema_output_layers = nn.ModuleDict(
            {
                name: nn.ModuleList(layers)
                for name, layers in schema_output_layers.items()
            }
        )
        self.visibility_output_layers = nn.ModuleDict(
            {
                name: nn.ModuleList(layers)
                for name, layers in visibility_output_layers.items()
            }
        )
        self.auxiliary_output_layers = nn.ModuleDict(
            {
                name: nn.Linear(max_depth, int(classes))
                for name, classes in self.auxiliary_heads.items()
            }
        )
        self.merge = nn.ModuleList(merge)

        row_loc, col_loc = self.make_grid("cpu", size=heatmap_size)
        self.register_buffer("xx_loc", col_loc, False)
        self.register_buffer("yy_loc", row_loc, False)

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

    def _prediction_for_stage(self, stage_index, feature, is_final_stage=True):
        hm = self.output_layers[stage_index](feature)
        coord = self.GetCoord(hm)
        if not self.multi_schema:
            return coord, hm
        out = {"landmarks_68": (coord, hm)}
        for name, layers in self.schema_output_layers.items():
            head_hm = layers[stage_index](feature)
            out[name] = (self.GetCoord(head_hm), head_hm)
        # The visibility head is the most expensive per-stage head (dominated by
        # its 1x1 feature projection). Skip it on non-final stages unless all
        # stages are supervised, matching where its output is actually consumed.
        if is_final_stage or self.visibility_all_stages:
            self._append_visibility_outputs(stage_index, feature, out)
        if self.auxiliary_output_layers:
            pooled = F.adaptive_avg_pool2d(feature, 1).flatten(1)
            out["_aux"] = {
                name: layer(pooled)
                for name, layer in self.auxiliary_output_layers.items()
            }
        return out

    def _append_visibility_outputs(self, stage_index, feature, out):
        for name, layers in self.visibility_output_layers.items():
            if name not in out:
                continue
            visibility_key = (
                "visibility_profile39"
                if name == "profile39"
                else "visibility_" + name.split("_", 1)[1]
            )
            landmark_heatmap = out[name][1]
            out[visibility_key] = layers[stage_index](feature, landmark_heatmap)

    def forward_res(self, img):
        feat = self.pre(img)
        pre_input = 0
        pre_output = feat
        res = []
        for i in range(len(self.stages)):
            if i > 0:
                merge = self.merge[i - 1](torch.cat([pre_input, pre_output], dim=1))
            else:
                merge = feat
            hm_0 = self.stages[i](merge)
            res.append(
                self._prediction_for_stage(
                    i, hm_0, is_final_stage=(i == len(self.stages) - 1)
                )
            )

            pre_input = pre_output
            pre_output = hm_0

        return res

    def forward_res3(self, img):
        feat = self.pre(img)
        pre_merged = 0
        pre_output = feat
        res = []
        for i in range(len(self.stages)):
            if i > 0:
                merge = self.merge[i - 1](torch.cat([pre_merged, pre_output], dim=1))
            else:
                merge = feat
            hm_0 = self.stages[i](merge)
            res.append(
                self._prediction_for_stage(
                    i, hm_0, is_final_stage=(i == len(self.stages) - 1)
                )
            )

            pre_merged = merge
            pre_output = hm_0

        return res

    def forward_NoConnection(self, img):
        feat = self.pre(img)
        pre_hm = feat
        res = []
        for i in range(len(self.stages)):
            hm_0 = self.stages[i](pre_hm)
            pre_hm = hm_0
            res.append(
                self._prediction_for_stage(
                    i, hm_0, is_final_stage=(i == len(self.stages) - 1)
                )
            )

        return res

    def forward(self, img, connect_type=1):
        if connect_type == 1:
            feat = self.pre(img)
            pre_hm = 0
            res = []
            for i in range(len(self.stages)):
                if i > 0:
                    merge = self.merge[i - 1](torch.cat([feat, pre_hm], dim=1))
                else:
                    merge = feat
                hm_0 = self.stages[i](merge)
                pre_hm = hm_0
                res.append(
                    self._prediction_for_stage(
                        i, hm_0, is_final_stage=(i == len(self.stages) - 1)
                    )
                )

            return res
        elif connect_type == 0:
            return self.forward_NoConnection(img)
        elif connect_type == 2:
            return self.forward_res(img)
        elif connect_type == 3:
            return self.forward_res3(img)


class VitAttnStageDenseConn(nn.Module):
    def __init__(
        self,
        lmk_num=98,
        Attn=lambda: SA2SA1(32, 256),
        nstack=4,
        heatmap_size=32,
        max_depth=256,
        backbone_net=lambda max_depth: HeadingNet([32, 64, max_depth]),
    ):
        super(VitAttnStageDenseConn, self).__init__()
        assert heatmap_size == 32
        assert (
            max_depth == 256 or max_depth == 192 or max_depth == 128 or max_depth == 64
        )

        self.pre = backbone_net(max_depth)

        stages = []
        output_layers = []
        merge = []
        for i in range(nstack):
            block = nn.Sequential(
                CoordConvTh(
                    heatmap_size,
                    heatmap_size,
                    True,
                    False,
                    max_depth,
                    max_depth,
                    kernel_size=3,
                    padding=1,
                ),
                Attn(),
                CoordConvTh(
                    heatmap_size,
                    heatmap_size,
                    True,
                    False,
                    max_depth,
                    max_depth,
                    kernel_size=3,
                    padding=1,
                ),
                Attn(),
            )
            stages.append(block)
            output_layers.append(nn.Conv2d(max_depth, lmk_num, 1))
            if i > 0:
                merge.append(DoubleConv(max_depth * (i + 1), max_depth, max_depth))
        self.stages = nn.ModuleList(stages)
        self.output_layers = nn.ModuleList(output_layers)
        self.merge = nn.ModuleList(merge)

        row_loc, col_loc = self.make_grid("cpu", size=heatmap_size)
        self.register_buffer("xx_loc", col_loc, False)
        self.register_buffer("yy_loc", row_loc, False)

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

    def forward_NoConnection(self, img):
        feat = self.pre(img)
        pre_hm = feat
        res = []
        for i in range(len(self.stages)):
            hm_0 = self.stages[i](pre_hm)
            pre_hm = hm_0
            hm = self.output_layers[i](hm_0)
            coord = self.GetCoord(hm)
            res.append((coord, hm))

        return res

    def forward_res(self, img):
        feat = self.pre(img)
        pre_input = 0
        pre_output = feat
        res = []
        for i in range(len(self.stages)):
            if i > 0:
                merge = self.merge[i - 1](torch.cat([pre_input, pre_output], dim=1))
            else:
                merge = feat
            hm_0 = self.stages[i](merge)
            hm = self.output_layers[i](hm_0)
            coord = self.GetCoord(hm)
            res.append((coord, hm))

            pre_input = pre_output
            pre_output = hm_0

        return res

    def forward_res3(self, img):
        feat = self.pre(img)
        pre_merged = 0
        pre_output = feat
        res = []
        for i in range(len(self.stages)):
            if i > 0:
                merge = self.merge[i - 1](torch.cat([pre_merged, pre_output], dim=1))
            else:
                merge = feat
            hm_0 = self.stages[i](merge)
            hm = self.output_layers[i](hm_0)
            coord = self.GetCoord(hm)
            res.append((coord, hm))

            pre_merged = merge
            pre_output = hm_0

        return res

    def forward(self, img, connect_type=1):
        if connect_type == 1:
            feat = self.pre(img)
            pre_merged = []
            pre_hm = 0
            res = []
            for i in range(len(self.stages)):
                if i > 0:
                    merge = self.merge[i - 1](torch.cat([pre_hm] + pre_merged, dim=1))
                else:
                    merge = feat
                pre_merged.append(merge)
                hm_0 = self.stages[i](merge)
                pre_hm = hm_0
                hm = self.output_layers[i](hm_0)
                coord = self.GetCoord(hm)
                res.append((coord, hm))

            return res


class VitAttnStageResSkip(nn.Module):
    def __init__(
        self,
        lmk_num=98,
        Attn=lambda: SA2SA1(32, 256),
        nstack=4,
        heatmap_size=32,
        max_depth=256,
    ):
        super(VitAttnStageResSkip, self).__init__()
        assert heatmap_size == 32
        assert max_depth == 256 or max_depth == 192

        self.pre = Vit(max_depth=max_depth)

        stages = []
        output_layers = []
        merge = []
        for i in range(nstack):
            block = nn.Sequential(
                CoordConvTh(
                    heatmap_size,
                    heatmap_size,
                    True,
                    False,
                    max_depth,
                    max_depth,
                    kernel_size=3,
                    padding=1,
                ),
                Attn(),
                CoordConvTh(
                    heatmap_size,
                    heatmap_size,
                    True,
                    False,
                    max_depth,
                    max_depth,
                    kernel_size=3,
                    padding=1,
                ),
                Attn(),
            )
            stages.append(block)
            output_layers.append(nn.Conv2d(max_depth, lmk_num, 1))
            if i > 0:
                merge.append(DoubleConv(max_depth * 2, max_depth, max_depth))
        self.stages = nn.ModuleList(stages)
        self.output_layers = nn.ModuleList(output_layers)
        self.merge = nn.ModuleList(merge)

        row_loc, col_loc = self.make_grid("cpu", size=heatmap_size)
        self.register_buffer("xx_loc", col_loc, False)
        self.register_buffer("yy_loc", row_loc, False)

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

    def forward(self, img):
        feat = self.pre(img)
        pre_input = 0
        pre_output = feat
        res = []
        for i in range(len(self.stages)):
            if i > 0:
                merge = self.merge[i - 1](torch.cat([pre_input, pre_output], dim=1))
            else:
                merge = feat
            hm_0 = self.stages[i](merge)
            hm = self.output_layers[i](hm_0)
            coord = self.GetCoord(hm)
            res.append((coord, hm))

            pre_input = pre_output
            pre_output = hm_0

        return res


class NetVitAttnStage(nn.Module):
    def __init__(
        self,
        lmk_num=98,
        Attn=lambda: SA2SA1(32, 256),
        nstack=4,
        heatmap_size=32,
        max_depth=256,
    ):
        super(NetVitAttnStage, self).__init__()
        if heatmap_size == 16:
            self.pre = HeadingNet([32, 64, 128, max_depth])
            self.unet = UNet([max_depth, max_depth, max_depth, max_depth])
        elif heatmap_size == 32:
            self.pre = HeadingNet([32, 64, max_depth])
            self.unet = UNet([max_depth, max_depth, max_depth, max_depth])
        elif heatmap_size == 64:
            self.pre = HeadingNet([64, max_depth])
            self.unet = UNet([max_depth, max_depth, max_depth, max_depth, max_depth])
        else:
            assert 0

        stages = []
        output_layers = []
        merge = []
        for i in range(nstack):
            if i % 2 == 0:
                block = nn.Sequential(
                    CoordConvTh(
                        heatmap_size,
                        heatmap_size,
                        True,
                        False,
                        max_depth,
                        max_depth,
                        kernel_size=3,
                        padding=1,
                    ),
                    Attn(),
                    CoordConvTh(
                        heatmap_size,
                        heatmap_size,
                        True,
                        False,
                        max_depth,
                        max_depth,
                        kernel_size=3,
                        padding=1,
                    ),
                    Attn(),
                )
            else:
                block = UNet([max_depth, max_depth, max_depth, max_depth])
            stages.append(block)
            output_layers.append(nn.Conv2d(max_depth, lmk_num, 1))
            if i > 0:
                merge.append(DoubleConv(max_depth * 2, max_depth, max_depth))
        self.stages = nn.ModuleList(stages)
        self.output_layers = nn.ModuleList(output_layers)
        self.merge = nn.ModuleList(merge)

        row_loc, col_loc = self.make_grid("cpu", size=heatmap_size)
        self.register_buffer("xx_loc", col_loc, False)
        self.register_buffer("yy_loc", row_loc, False)

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

    def forward(self, img):
        x = self.pre(img)
        feat = self.unet(x)
        pre_hm = 0
        res = []
        for i in range(len(self.stages)):
            if i > 0:
                merge = self.merge[i - 1](torch.cat([feat, pre_hm], dim=1))
            else:
                merge = feat
            hm_0 = self.stages[i](merge)
            pre_hm = hm_0
            hm = self.output_layers[i](hm_0)
            coord = self.GetCoord(hm)
            res.append((coord, hm))

        return res


"""
stacked hg结构融合SA2SA1效果如何

动态权重：每个关键点的权重是不一样的，动态权重效果如何呢？
Attention_block的block中PatchEmbed怎么眼样？
"""


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # import torchsummary

    x = torch.randn((2, 3, 256, 256)).cuda()

    max_depth = 160
    net = VitAttnStage(
        lmk_num=98,
        nstack=4,
        Attn=lambda: SA2SA1_2(32, max_depth, win_size=2),
        # Attn=lambda: Hourglass(3, args.max_depth),
        # Attn = lambda :SelfAttention_block2(args.max_depth),
        # Attn = lambda :SelfAttention2_block(args.heatmap_size, args.max_depth,args.max_depth),
        # Attn = lambda :UNet([256, 256, 256]),
        # Attn=lambda: nn.Sequential(RCCAModule(256, 256, 256), RCCAModule(256, 256, 256)),
        heatmap_size=32,
        max_depth=max_depth,
        backbone_net=lambda max_depth: HeadingNet([32, 64, max_depth]),
    ).cuda()

    print(count_parameters(net) / 1024 / 1024)
