import torch.nn as nn
import torch


class DoubleConv(nn.Module):
    def __init__(self, in_channel, out_channel, mid_channel=None):
        super(DoubleConv, self).__init__()
        if not mid_channel:
            mid_channel = out_channel
        self.double_conv = nn.Sequential(
            nn.Conv2d(
                in_channel, mid_channel, kernel_size=3, stride=1, padding=1, bias=False
            ),
            nn.BatchNorm2d(mid_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                mid_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=False
            ),
            nn.BatchNorm2d(out_channel),
        )

        self.short_cut = (
            nn.Identity()
            if in_channel == out_channel
            else nn.Conv2d(
                in_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=False
            )
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, input):
        x = self.double_conv(input)
        x = self.relu(x + self.short_cut(input))
        return x


class Down(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Down, self).__init__()
        self.down_sample = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            DoubleConv(in_channel, out_channel),
        )

    def forward(self, x):
        return self.down_sample(x)


class Up(nn.Module):
    def __init__(self, in_channel, dual_channel):
        super(Up, self).__init__()
        self.up_sample = nn.ConvTranspose2d(
            in_channel, dual_channel, kernel_size=2, stride=2
        )
        self.conv = DoubleConv(dual_channel * 2, dual_channel)

    def forward(self, pre_level, dual_level):
        x1 = self.up_sample(pre_level)
        x = torch.cat([x1, dual_level], dim=1)
        x = self.conv(x)
        return x


class UNet(nn.Module):
    def __init__(self, channel_list=[64, 128, 256, 512, 512]):
        super(UNet, self).__init__()
        assert len(channel_list) >= 2

        down_list = []
        for i in range(1, len(channel_list)):
            e = Down(channel_list[i - 1], channel_list[i])
            down_list.append(e)

        self.down_list = nn.ModuleList(down_list)

        up_list = []
        for i in range(len(channel_list) - 1, 0, -1):
            e = Up(channel_list[i], channel_list[i - 1])
            up_list.append(e)
        self.up_list = nn.ModuleList(up_list)

    def forward(self, x):
        down_values = [x]
        for e in self.down_list:
            x = e(x)
            down_values.append(x)

        down_values.reverse()

        for i in range(len(self.up_list)):
            e = self.up_list[i]
            x = e(x, down_values[i + 1])
        return x


if __name__ == "__main__":
    import torchsummary

    # net = UNet([32, 32, 32, 32, 256, 1024])
    # x = torch.randn((1, 32, 512, 512))

    net = UNet(
        [
            256,
            256,
            256,
            256,
            256,
        ]
    )

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    torchsummary.summary(net.to(device), (256, 64, 64))
