import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.landmarks.models.blocks import WSConv2d


class DoubleConvRes(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, res_oper="concat"):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            WSConv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            WSConv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.short_cut = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Sequential(
                WSConv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        )
        self.res_oper = res_oper
        if res_oper == "concat":
            self.merge = nn.Sequential(
                WSConv2d(out_channels * 2, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x_short = self.short_cut(x)
        y = self.double_conv(x)
        if self.res_oper == "concat":
            r = torch.cat([x_short, y], dim=1)
            r = self.relu(self.merge(r))
        elif self.res_oper == "add":
            r = self.relu(x_short + y)
        else:
            assert 0
        return r


class DoubleConv_backup(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            WSConv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            WSConv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        self.double_conv = DoubleConv_backup(in_channels, out_channels, mid_channels)

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)), DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=(1, 2), mode="bilinear", align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, channel_list, bilinear=True):
        super(UNet, self).__init__()

        down_list = []
        for i in range(1, len(channel_list)):
            down = Down(channel_list[i - 1], channel_list[i])
            down_list.append(down)

        up_list = []
        for i in range(len(channel_list) - 1, 0, -1):
            up = Up(channel_list[i] + channel_list[i - 1], channel_list[i - 1], bilinear=bilinear)
            up_list.append(up)

        self.down = nn.ModuleList(down_list)
        self.up = nn.ModuleList(up_list)

    def forward(self, x):
        nodes = [x]
        for i in range(len(self.down)):
            n = self.down[i](x)
            nodes.append(n)
            x = n

        for i in range(len(self.up)):
            y = self.up[i](x, nodes[-i - 2])
            x = y
        return x


if __name__ == "__main__":

    net = UNet(
        [
            256,
            256,
            256,
            256,
            256,
        ]
    )

    x = torch.randn((3, 256, 2, 32))
    y = net(x)
    print(y.shape)
