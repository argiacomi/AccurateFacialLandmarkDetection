import torch
import torch.nn as nn

from lib.models.attention import *
from lib.models.coord_conv import CoordConvTh


class PatchEmbed(nn.Module):
    """Image to Patch Embedding"""

    # 第1步：通过patch_size=4，设置块的大小，对原始图像进行分块
    def __init__(self, img_size=224, in_chans=3, embed_dim=768, patch_size=2):
        super().__init__()
        img_size = (img_size, img_size)
        patch_size = (patch_size, patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        # self.project = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.project = CoordConvTh(
            img_size[0],
            img_size[0],
            True,
            False,
            in_chans,
            embed_dim,
            kernel_size=3,
            stride=2,
            padding=1,
        )

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], (
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        )

        x = self.project(x)  # 第2步：通过2d卷积进行线性变换
        # x = x.flatten(2)  # 第3步：拉平生成线性变量
        # x = x.transpose(1, 2)  # 第4步：块的个数 与 每块的向量维度交换位置
        return x


class PatchEmbed2(nn.Module):
    """Image to Patch Embedding"""

    # 第1步：通过patch_size=4，设置块的大小，对原始图像进行分块
    def __init__(self, img_size=224, in_chans=3, embed_dim=768, patch_size=2, stride=2):
        super().__init__()
        img_size = (img_size, img_size)
        patch_size = (patch_size, patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.project = nn.Sequential(
            CoordConvTh(
                img_size[0],
                img_size[0],
                True,
                False,
                in_chans,
                embed_dim,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.Conv2d(
                embed_dim,
                embed_dim,
                kernel_size=patch_size,
                stride=stride,
                padding=(patch_size[0] - 1) // 2,
            ),
        )

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], (
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        )

        x = self.project(x)  # 第2步：通过2d卷积进行线性变换
        # x = x.flatten(2)  # 第3步：拉平生成线性变量
        # x = x.transpose(1, 2)  # 第4步：块的个数 与 每块的向量维度交换位置
        return x


class Vit(nn.Module):
    def __init__(self, max_depth=256):
        super(Vit, self).__init__()
        self.body = nn.Sequential(
            PatchEmbed(256, 3, 64),
            SelfAttention_block3(64, win_size=4),
            SelfAttention_block3(64, win_size=4),
            PatchEmbed(128, 64, 128),
            SelfAttention_block3(128, win_size=4),
            SelfAttention_block3(128, win_size=4),
            PatchEmbed(64, 128, max_depth),
            # SelfAttention_block(256),
            SelfAttention_block3(max_depth),
        )

    def forward(self, x):
        return self.body(x)


class Vit_cofw68_1(nn.Module):
    def __init__(self, max_depth=256):
        super(Vit_cofw68_1, self).__init__()
        self.body = nn.Sequential(
            PatchEmbed(256, 3, 16),
            SelfAttention_block3(16, win_size=4),
            SelfAttention_block3(16, win_size=4),
            PatchEmbed(128, 16, 32),
            SelfAttention_block3(32, win_size=4),
            SelfAttention_block3(32, win_size=4),
            PatchEmbed(64, 32, max_depth),
            # SelfAttention_block(256),
            SelfAttention_block3(max_depth),
        )

    def forward(self, x):
        return self.body(x)


class Vit_cofw68_2(nn.Module):
    def __init__(self, max_depth=256):
        super(Vit_cofw68_2, self).__init__()
        self.body = nn.Sequential(
            PatchEmbed(256, 3, 32),
            SelfAttention_block3(32, win_size=4),
            SelfAttention_block3(32, win_size=4),
            PatchEmbed(128, 32, 64),
            SelfAttention_block3(64, win_size=4),
            SelfAttention_block3(64, win_size=4),
            PatchEmbed(64, 64, max_depth),
            # SelfAttention_block(256),
            SelfAttention_block3(max_depth),
        )

    def forward(self, x):
        return self.body(x)


class Vit_2(nn.Module):
    def __init__(self, max_depth=256):
        super(Vit_2, self).__init__()
        self.body = nn.Sequential(
            PatchEmbed(256, 3, 64),
            SelfAttention_block3(64, win_size=4),
            SelfAttention_block3(64, win_size=4),
            SelfAttention_block3(64, win_size=4),
            SelfAttention_block3(64, win_size=4),
            PatchEmbed(128, 64, 128),
            SelfAttention_block3(128, win_size=4),
            SelfAttention_block3(128, win_size=4),
            SelfAttention_block3(128, win_size=4),
            SelfAttention_block3(128, win_size=4),
            PatchEmbed(64, 128, max_depth),
            # SelfAttention_block(256),
            SelfAttention_block3(max_depth),
            SelfAttention_block3(max_depth),
        )

    def forward(self, x):
        return self.body(x)


class Vit_3(nn.Module):
    def __init__(self, max_depth=256):
        super(Vit_3, self).__init__()
        self.body = nn.Sequential(
            PatchEmbed2(256, 3, 64, patch_size=5),
            SelfAttention_block3(64, win_size=4),
            SelfAttention_block3(64, win_size=4),
            PatchEmbed2(128, 64, 128, patch_size=5),
            SelfAttention_block3(128, win_size=4),
            SelfAttention_block3(128, win_size=4),
            PatchEmbed2(64, 128, max_depth, patch_size=3),
            # SelfAttention_block(256),
            SelfAttention_block3(max_depth),
        )

    def forward(self, x):
        return self.body(x)


class MixBlk(nn.Module):
    def __init__(self, in_channel, win_size=4):
        super(MixBlk, self).__init__()
        self.sa = SelfAttention_block3(in_channel // 2, win_size=win_size)
        self.conv = DoubleConv(in_channel // 2, in_channel // 2)
        self.merge = DoubleConv(in_channel, in_channel)

    def forward(self, img):
        B, C, H, W = img.shape

        img_1 = img[:, : C // 2]
        img_2 = img[:, C // 2 :]

        f1 = self.sa(img_1)
        f2 = self.conv(img_2)

        f = torch.concat([f1, f2], dim=1)
        f = self.merge(f)
        return f


class Vit2(nn.Module):
    def __init__(self, max_depth=256):
        super(Vit2, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            PatchEmbed(256, 32, 64),
            MixBlk(64, win_size=4),
            MixBlk(64, win_size=4),
            PatchEmbed(128, 64, 128),
            MixBlk(128, win_size=4),
            MixBlk(128, win_size=2),
            PatchEmbed(64, 128, max_depth),
            # SelfAttention_block(256),
            MixBlk(max_depth, win_size=2),
        )

    def forward(self, x):
        return self.body(x)


class MixBlkC(nn.Module):
    def __init__(self, in_channel, win_size=4):
        super(MixBlkC, self).__init__()
        self.sa = SelfAttention_block3(in_channel, win_size=win_size)
        self.conv = DoubleConv(in_channel, in_channel)
        self.merge = DoubleConv(in_channel * 2, in_channel, in_channel)

    def forward(self, img):
        f1 = self.sa(img)
        f2 = self.conv(f1)

        f = torch.concat([f1, f2], dim=1)
        f = self.merge(f)
        return f


class Vit3(nn.Module):
    def __init__(self, max_depth=256):
        super(Vit3, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            PatchEmbed(256, 32, 64),
            MixBlkC(64, win_size=4),
            MixBlkC(64, win_size=4),
            PatchEmbed(128, 64, 128),
            MixBlkC(128, win_size=4),
            MixBlkC(128, win_size=4),
            PatchEmbed(64, 128, max_depth),
            # SelfAttention_block(256),
            MixBlkC(max_depth, win_size=4),
        )

    def forward(self, x):
        return self.body(x)


class VitRegressor(nn.Module):
    def __init__(self, lmd_num=98):
        super(VitRegressor, self).__init__()
        self.lmd_num = lmd_num
        self.body = nn.Sequential(
            PatchEmbed(256, 3, 32),
            SelfAttention_block(32, win_size=4),
            SelfAttention_block(32, win_size=4),
            PatchEmbed(128, 32, 64),
            SelfAttention_block(64, win_size=4),
            SelfAttention_block(64, win_size=4),
            PatchEmbed(64, 64, 128),
            SelfAttention_block(128, win_size=2),
            SelfAttention_block(128, win_size=2),
            PatchEmbed(32, 128, 128),
            SelfAttention_block(128, win_size=2),
            SelfAttention_block(128, win_size=2),
            PatchEmbed(16, 128, 128),
            SelfAttention_block(128, win_size=2),
            SelfAttention_block(128, win_size=2),
            PatchEmbed(8, 128, 128),
            SelfAttention_block(128, win_size=2),
            SelfAttention_block(128, win_size=2),
            PatchEmbed(4, 128, 256),
            SelfAttention_block(256, win_size=2),
            SelfAttention_block(256, win_size=2),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, lmd_num * 2, 2),
        )

    def forward(self, x):
        B, C, H, W = x.shape
        coord = self.body(x).reshape((B, self.lmd_num, 2))
        return coord


if __name__ == "__main__":
    x = torch.rand([2, 3, 256, 256])

    model = VitRegressor()
    y = model(x)

    print(y.shape)
