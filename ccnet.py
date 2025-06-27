import torch.nn as nn
from torch.nn import functional as F
import math
import torch.utils.model_zoo as model_zoo
import torch
import numpy as np
from torch.autograd import Variable

affine_par = True
import functools

import sys, os

# from cc_attention import CrissCrossAttention
from CC import CC_module as CrissCrossAttention

# from .utils.pyt_utils import load_model

# from Synchronized.sync_batchnorm import SynchronizedBatchNorm2d as SyncBN
BatchNorm2d = nn.BatchNorm2d  # SyncBN#functools.partial(InPlaceABNSync, activation='identity')


def outS(i):
    i = int(i)
    i = (i + 1) / 2
    i = int(np.ceil((i + 1) / 2.0))
    i = (i + 1) / 2
    return i


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None, fist_dilation=1, multi_grid=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=dilation * multi_grid, dilation=dilation * multi_grid, bias=False)
        self.bn2 = BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=False)
        self.relu_inplace = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.dilation = dilation
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out + residual
        out = self.relu_inplace(out)

        return out


class RCCAModule(nn.Module):
    def __init__(self, in_channels, out_channels, num_classes):
        super(RCCAModule, self).__init__()
        # inter_channels = in_channels // 4
        inter_channels = in_channels // 2
        self.conva = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                                   BatchNorm2d(inter_channels), nn.ReLU(inplace=False))
        self.cca = CrissCrossAttention(inter_channels)
        self.convb = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
                                   BatchNorm2d(inter_channels), nn.ReLU(inplace=False))

        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels + inter_channels, out_channels, kernel_size=3, padding=1, dilation=1, bias=False),
            BatchNorm2d(out_channels), nn.ReLU(inplace=False),
            nn.Dropout2d(0.1),
            nn.Conv2d(out_channels, num_classes, kernel_size=1, stride=1, padding=0, bias=True)
        )

    def forward(self, x, recurrence=2):
        output = self.conva(x)
        for i in range(recurrence):
            output = self.cca(output)
        output = self.convb(output)

        output = self.bottleneck(torch.cat([x, output], 1))
        return output


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes, recurrence):
        self.inplanes = 128
        super(ResNet, self).__init__()
        self.conv1 = conv3x3(3, 64, stride=2)
        self.bn1 = BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=False)
        self.conv2 = conv3x3(64, 64)
        self.bn2 = BatchNorm2d(64)
        self.relu2 = nn.ReLU(inplace=False)
        self.conv3 = conv3x3(64, 128)
        self.bn3 = BatchNorm2d(128)
        self.relu3 = nn.ReLU(inplace=False)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.relu = nn.ReLU(inplace=False)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=True)  # change
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=1, dilation=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilation=4, multi_grid=(1, 1, 1))
        # self.layer5 = PSPModule(2048, 512)
        self.head = RCCAModule(2048, 512, num_classes)

        self.dsn = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(512), nn.ReLU(inplace=False),
            nn.Dropout2d(0.1),
            nn.Conv2d(512, num_classes, kernel_size=1, stride=1, padding=0, bias=True)
        )
        self.conv4 = nn.Conv2d(num_classes * 2, num_classes, kernel_size=1, stride=1, bias=False)
        # self.criterion = criterion
        self.recurrence = recurrence

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1, multi_grid=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                BatchNorm2d(planes * block.expansion, affine=affine_par))

        layers = []
        generate_multi_grid = lambda index, grids: grids[index % len(grids)] if isinstance(grids, tuple) else 1
        layers.append(block(self.inplanes, planes, stride, dilation=dilation, downsample=downsample,
                            multi_grid=generate_multi_grid(0, multi_grid)))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(
                block(self.inplanes, planes, dilation=dilation, multi_grid=generate_multi_grid(i, multi_grid)))

        return nn.Sequential(*layers)

    def forward(self, x, labels=None):
        size = (x.shape[2], x.shape[3])
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x_dsn = self.dsn(x)
        x = self.layer4(x)
        x = self.head(x, self.recurrence)
        outs = torch.cat([x, x_dsn], 1)
        outs = self.conv4(outs)
        # outs = nn.Upsample(size, mode='bilinear', align_corners=True)(outs)
        return outs


def resnet152(num_classes=4, pretrained_model=None, recurrence=2, **kwargs):
    model = ResNet(Bottleneck, [3, 8, 36, 3], num_classes, recurrence)
    return model


def resnet101(num_classes=4, pretrained_model=None, recurrence=2, **kwargs):
    model = ResNet(Bottleneck, [3, 4, 23, 3], num_classes, recurrence)
    return model


def resnet50(num_classes=4, pretrained_model=None, recurrence=2, **kwargs):
    model = ResNet(Bottleneck, [3, 4, 6, 3], num_classes, recurrence)
    return model


class CCNet(nn.Module):
    def __init__(self, lmk_num=10, heatmap_size=33):
        super(CCNet, self).__init__()
        assert heatmap_size == 33
        self.heatmap_size = heatmap_size

        self.net = resnet101(lmk_num)
        row_loc, col_loc = self.make_grid("cpu", size=heatmap_size)
        self.register_buffer("xx_loc", col_loc, False)
        self.register_buffer("yy_loc", row_loc, False)

        self.lmk_num = lmk_num

    def make_grid(self, device="cpu", size=14):
        row, col = torch.meshgrid(torch.arange(size, device=device), torch.arange(size, device=device))
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
        hm = self.net(img)
        coord = self.GetCoord(hm)
        return [(coord, hm)]


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # model = resnet152(num_classes=256)
    # model = model.to(device)
    # x = torch.rand((4, 3, 256, 256))
    # x = x.to(device)
    # print('====================')
    # output = model(x)
    # print('====================')
    # print(output.shape)
    # # torch.save(model.state_dict(), 'ccnet_{}.pth'.format(0))
    # # torch.save(model, 'model_{}.pt'.format(0))

    net = CCNet().cuda()
    x = torch.randn((2, 3, 256, 256)).cuda()
    r = net(x)[0]
    print(r[0].shape, r[1].shape)

    print(count_parameters(net) / 1024 / 1024)
