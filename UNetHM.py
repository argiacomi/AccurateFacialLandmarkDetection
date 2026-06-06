import torch

from helpers import bottleneck_IR_SE, get_block
from UNet2 import UNet
from coord_conv import CoordConvTh
from Heatmap import Heatmap
import torch.nn.functional as F
from Vit import Vit
from Attention import *
import torchvision
from Hourglass import Hourglass


class HeadingNet(torch.nn.Module):
    def __init__(self, channels=(128, 256), in_channel=3):
        super(HeadingNet, self).__init__()

        # body
        blocks = [
            get_block(in_channel=in_channel, depth=channels[0], num_units=3),
            # get_block(in_channel=channels[0], depth=channels[1], num_units=3),
        ]
        for i in range(len(channels) - 1):
            blocks.append(get_block(in_channel=channels[i], depth=channels[i + 1], num_units=3))
        units = []
        for bottlenecks in blocks:
            for b in bottlenecks:
                units.append(bottleneck_IR_SE(b.in_channel, b.depth, b.stride))
        self.body = nn.Sequential(*units)

    def forward(self, x):
        x = self.body(x)
        return x


class UNetStage(nn.Module):
    def __init__(
        self,
        lmk_num=98,
        Attn=lambda: UNet([256, 256, 256, 256, 256]),
        nstack=4,
        heatmap_size=32,
        max_depth=256,
        backbone_net=lambda max_depth: HeadingNet([32, 64, max_depth]),
        num_dvit_per_pred_blk=1,
    ):
        super(UNetStage, self).__init__()
        # assert heatmap_size == 32
        assert max_depth == 256 or max_depth == 192 or max_depth == 128 or max_depth == 64

        self.pre = backbone_net(max_depth)

        stages = []
        output_layers = []
        merge = []
        for i in range(nstack):
            vit_list = []
            for num in range(num_dvit_per_pred_blk):
                vit_list.append(
                    CoordConvTh(heatmap_size, heatmap_size, True, False, max_depth, max_depth, kernel_size=3, padding=1)
                )
                vit_list.append(Attn())
            block = nn.Sequential(*vit_list)

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
                coord = self.GetCoord(hm)
                res.append((coord, hm))

            return res
        elif connect_type == 0:
            return self.forward_NoConnection(img)
        elif connect_type == 2:
            return self.forward_res(img)
        elif connect_type == 3:
            return self.forward_res3(img)


class StackedHG(nn.Module):
    def __init__(
        self,
        lmk_num=98,
        # Attn=lambda: UNet([256, 256, 256, 256, 256]),
        Attn=lambda: Hourglass(3, 256),
        nstack=4,
        heatmap_size=32,
        max_depth=256,
        backbone_net=lambda max_depth: HeadingNet([32, 64, max_depth]),
        num_dvit_per_pred_blk=1,
        merge_oper="cat",
    ):
        super(StackedHG, self).__init__()
        # assert heatmap_size == 32
        # assert max_depth == 256 or max_depth == 192 or max_depth == 128 or max_depth == 64

        self.pre = backbone_net(max_depth)

        stages = []
        output_layers = []
        for i in range(nstack):
            vit_list = []
            for num in range(num_dvit_per_pred_blk):
                vit_list.append(
                    CoordConvTh(heatmap_size, heatmap_size, True, False, max_depth, max_depth, kernel_size=3, padding=1)
                )
                vit_list.append(Attn())
            block = nn.Sequential(*vit_list)
            stages.append(block)
            output_layers.append(nn.Conv2d(max_depth, lmk_num, 1))

        self.stages = nn.ModuleList(stages)
        self.output_layers = nn.ModuleList(output_layers)

        if merge_oper == "cat":
            merge = []
            for i in range(nstack):
                merge.append(DoubleConv(max_depth * 2, max_depth, max_depth))
            self.merge = nn.ModuleList(merge)
        self.merge_oper = merge_oper
        row_loc, col_loc = self.make_grid("cpu", size=heatmap_size)
        self.register_buffer("xx_loc", col_loc, False)
        self.register_buffer("yy_loc", row_loc, False)

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

    def forward_classic_hg(self, img):
        low_feat = self.pre(img)
        pre_feat = low_feat
        res = []
        for i in range(len(self.stages)):
            cur_feat = self.stages[i](pre_feat)
            if self.merge_oper == "cat":
                merged_feat = self.merge[i](torch.cat([pre_feat, cur_feat], dim=1))
            else:
                merged_feat = pre_feat + cur_feat
            hm = self.output_layers[i](merged_feat)
            coord = self.GetCoord(hm)
            res.append((coord, hm))

            pre_feat = merged_feat
        return res

    def forward_long_skip(self, img):
        low_feat = self.pre(img)
        pre_output = 0
        res = []
        for i in range(len(self.stages)):
            if i == 0:
                merged_feat = low_feat
            else:
                if self.merge_oper == "cat":
                    merged_feat = self.merge[i](torch.cat([low_feat, pre_output], dim=1))
                else:
                    merged_feat = low_feat + pre_output
            hm_basis = self.stages[i](merged_feat)
            hm = self.output_layers[i](hm_basis)
            coord = self.GetCoord(hm)
            res.append((coord, hm))
            pre_output = hm_basis
        return res

    def forward_ResCBSP_type1(self, img):
        # this is the residual connection between two prediction blocks in the paper dual vit
        low_feat = self.pre(img)
        pre_input = low_feat
        pre_output = 0
        res = []
        for i in range(len(self.stages)):
            if i == 0:
                merged_feat = low_feat
            else:
                if self.merge_oper == "cat":
                    merged_feat = self.merge[i](torch.cat([pre_input, pre_output], dim=1))
                else:
                    merged_feat = pre_input + pre_output
            hm_basis = self.stages[i](merged_feat)
            hm = self.output_layers[i](hm_basis)
            coord = self.GetCoord(hm)
            res.append((coord, hm))
            pre_input = merged_feat
            pre_output = hm_basis
        return res

    def forward(self, img, type=2):
        if 0 == type:
            return self.forward_classic_hg(img)
        elif 1 == type:
            return self.forward_ResCBSP_type1(img)
        elif 2 == type:
            return self.forward_long_skip(img)
        else:
            assert 0
            
class StackedHGCoraseAndFine(nn.Module):
    def __init__(
        self,
        lmk_num=98,
        # Attn=lambda: UNet([256, 256, 256, 256, 256]),
        Attn=lambda: Hourglass(3, 256),
        nstack=4,
        heatmap_size=32,
        max_depth=256,
        backbone_net=lambda max_depth: HeadingNet([32, 64, max_depth]),
        num_dvit_per_pred_blk=1,
        merge_oper="cat",
    ):
        super(StackedHGCoraseAndFine, self).__init__()
        # assert heatmap_size == 32
        # assert max_depth == 256 or max_depth == 192 or max_depth == 128 or max_depth == 64

        self.pre = backbone_net(max_depth)

        coarse_stages = []
        fine_stages = []
        coarse_output_layers = []
        fine_output_layers = []
        for i in range(nstack):
            coarse_list = []
            fine_list = []

            for num in range(num_dvit_per_pred_blk):
                coarse_list.append(
                    CoordConvTh(heatmap_size, heatmap_size, True, False, max_depth, max_depth, kernel_size=3, padding=1)
                )
                coarse_list.append(Attn())
                fine_list.append(
                    CoordConvTh(heatmap_size, heatmap_size, True, False, max_depth, max_depth, kernel_size=3, padding=1)
                )
                fine_list.append(Attn())
            corase_block = nn.Sequential(*coarse_list)
            fine_block = nn.Sequential(*fine_list)
            coarse_stages.append(corase_block)
            fine_stages.append(fine_block)
            coarse_output_layers.append(nn.Conv2d(max_depth, lmk_num, 1))
            fine_output_layers.append(nn.Conv2d(max_depth, lmk_num, 1))

        self.coarse_stages = nn.ModuleList(coarse_stages)
        self.fine_stages = nn.ModuleList(fine_stages)
        self.coarse_output_layers = nn.ModuleList(coarse_output_layers)
        self.fine_output_layers = nn.ModuleList(fine_output_layers)

        if merge_oper == "cat":
            merge = []
            for i in range(nstack):
                merge.append(DoubleConv(max_depth * 2, max_depth, max_depth))
            self.merge = nn.ModuleList(merge)
        self.merge_oper = merge_oper
        row_loc, col_loc = self.make_grid("cpu", size=heatmap_size)
        self.register_buffer("xx_loc", col_loc, False)
        self.register_buffer("yy_loc", row_loc, False)

        row_loc, col_loc = self.make_fine_grid("cpu", local_size=heatmap_size, global_size=heatmap_size * 2)
        self.register_buffer("f_xx_loc", col_loc, False)
        self.register_buffer("f_yy_loc", row_loc, False)

    def make_grid(self, device="cpu", size=14):
        row, col = torch.meshgrid(torch.arange(size, device=device), torch.arange(size, device=device), indexing="ij")
        c = size - 1.0
        row = row / c
        col = col / c
        return row.reshape((1, 1, size, size)), col.reshape((1, 1, size, size))

    def make_fine_grid(self, device="cpu", local_size=8, global_size=32):
        row, col = torch.meshgrid(torch.arange(local_size, device=device), torch.arange(local_size, device=device), indexing="ij")
        c = global_size - 1.0
        row = row / c
        col = col / c
        return row.reshape((1, 1, local_size, local_size)), col.reshape((1, 1, local_size, local_size))

    def GetFineCoord(self, fine_hm, coarse_coord):
        B, C, H, W = fine_hm.shape
        heatmap = F.softmax(fine_hm.reshape((B, C, -1)), dim=-1).reshape((B, C, H, W))
        xx = (heatmap * self.f_xx_loc).sum([2, 3])
        yy = (heatmap * self.f_yy_loc).sum([2, 3])
        return torch.stack([xx, yy], dim=2) + coarse_coord

    def GetCoord(self, heatmap):
        B, C, H, W = heatmap.shape
        heatmap = F.softmax(heatmap.reshape((B, C, -1)), dim=-1).reshape((B, C, H, W))
        xx = (heatmap * self.xx_loc).sum([2, 3])
        yy = (heatmap * self.yy_loc).sum([2, 3])
        return torch.stack([xx, yy], dim=2)

    def forward_classic_hg(self, img):
        low_feat = self.pre(img)
        pre_feat = low_feat
        res = []
        for i in range(len(self.stages)):
            cur_feat = self.stages[i](pre_feat)
            if self.merge_oper == "cat":
                merged_feat = self.merge[i](torch.cat([pre_feat, cur_feat], dim=1))
            else:
                merged_feat = pre_feat + cur_feat
            hm = self.output_layers[i](merged_feat)
            coord = self.GetCoord(hm)
            res.append((coord, hm))

            pre_feat = merged_feat
        return res

    def forward_ResCBSP_type1(self, img):
        # this is the residual connection between two prediction blocks in the paper dual vit
        low_feat = self.pre(img)
        pre_input = low_feat
        pre_output = 0
        res = []
        for i in range(len(self.stages)):
            if i == 0:
                merged_feat = low_feat
            else:
                if self.merge_oper == "cat":
                    merged_feat = self.merge[i](torch.cat([pre_input, pre_output], dim=1))
                else:
                    merged_feat = pre_input + pre_output
            hm_basis = self.stages[i](merged_feat)
            hm = self.output_layers[i](hm_basis)
            coord = self.GetCoord(hm)
            res.append((coord, hm))
            pre_input = merged_feat
            pre_output = hm_basis
        return res

    def forward(self, img):
        low_feat = self.pre(img)
        input = low_feat

        res = []
        for i in range(len(self.fine_stages)):
            fine_feat = self.fine_stages[i](input)
            coarse_feat = self.coarse_stages[i](input)

            coarse_hm = self.coarse_output_layers[i](coarse_feat)
            coarse_coord = self.GetCoord(coarse_hm)

            fine_hm = self.fine_output_layers[i](fine_feat)
            fine_coord = self.GetFineCoord(fine_hm, coarse_coord)

            merged_feat = self.merge[i](torch.cat([fine_feat, coarse_feat], dim=1)) + low_feat
            input = merged_feat
            res.append([coarse_coord, coarse_hm, fine_coord])
        return res

class UNetStageCoeff(nn.Module):
    def __init__(
        self,
        lmk_num=98,
        Attn=lambda: UNet([256, 256, 256, 256, 256]),
        nstack=4,
        heatmap_size=32,
        max_depth=256,
        backbone_net=lambda max_depth: HeadingNet([32, 64, max_depth]),
    ):
        super(UNetStageCoeff, self).__init__()
        self.lmk_num = lmk_num
        self.max_depth = max_depth
        # assert heatmap_size == 32
        assert max_depth == 256 or max_depth == 192 or max_depth == 128 or max_depth == 64

        self.pre = backbone_net(max_depth)

        before_stages = []
        stages = []
        coeff_predictor = []
        merge = []
        for i in range(nstack):
            before_stages.append(
                CoordConvTh(heatmap_size, heatmap_size, True, False, max_depth, max_depth, kernel_size=3, padding=1)
            )
            stages.append(Attn())
            coeff_predictor.append(
                nn.Sequential(
                    nn.MaxPool2d(2),
                    UNet([256, 256, 256, 256]),
                    nn.Conv2d(256, lmk_num, kernel_size=1),
                )
            )
            if i > 0:
                merge.append(DoubleConv(max_depth * 2, max_depth, max_depth))
        self.before_stages = nn.ModuleList(before_stages)
        self.stages = nn.ModuleList(stages)
        self.coeff_predictor = nn.ModuleList(coeff_predictor)
        self.merge = nn.ModuleList(merge)

        row_loc, col_loc = self.make_grid("cpu", size=heatmap_size)
        self.register_buffer("xx_loc", col_loc, False)
        self.register_buffer("yy_loc", row_loc, False)

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
        B, C, H, W = img.shape
        if connect_type == 1:
            feat = self.pre(img)
            pre_hm = 0
            res = []
            for i in range(len(self.stages)):
                if i > 0:
                    merge = self.merge[i - 1](torch.cat([feat, pre_hm], dim=1))
                else:
                    merge = feat
                bfore = self.before_stages[i](merge)
                hm_0 = self.stages[i](bfore)
                coeff = self.coeff_predictor[i](merge)
                coeff = coeff.reshape((B, self.lmk_num, -1))
                hm = torch.einsum("bchw,blc->blhw", hm_0, coeff)
                pre_hm = hm_0
                coord = self.GetCoord(hm)
                res.append((coord, hm))

            return res
        elif connect_type == 0:
            return self.forward_NoConnection(img)
        elif connect_type == 2:
            return self.forward_res(img)
        elif connect_type == 3:
            return self.forward_res3(img)

class StackedHG2Head(nn.Module):
    def __init__(
        self,
        lmk_num1=98,
        lmk_num2=70,
        # Attn=lambda: UNet([256, 256, 256, 256, 256]),
        Attn=lambda: Hourglass(3, 256),
        nstack=4,
        heatmap_size=32,
        max_depth=256,
        backbone_net=lambda max_depth: HeadingNet([32, 64, max_depth]),
        num_dvit_per_pred_blk=1,
        merge_oper="cat",
    ):
        super(StackedHG2Head, self).__init__()
        # assert heatmap_size == 32
        # assert max_depth == 256 or max_depth == 192 or max_depth == 128 or max_depth == 64

        self.pre = backbone_net(max_depth)

        stages = []
        output_layers1 = []
        output_layers2 = []
        for i in range(nstack):
            vit_list = []
            for num in range(num_dvit_per_pred_blk):
                vit_list.append(
                    CoordConvTh(heatmap_size, heatmap_size, True, False, max_depth, max_depth, kernel_size=3, padding=1)
                )
                vit_list.append(Attn())
            block = nn.Sequential(*vit_list)
            stages.append(block)
            output_layers1.append(nn.Conv2d(max_depth, lmk_num1, 3, padding=1))
            output_layers2.append(nn.Conv2d(max_depth, lmk_num2, 3, padding=1))

        self.stages = nn.ModuleList(stages)
        self.output_layers1 = nn.ModuleList(output_layers1)
        self.output_layers2 = nn.ModuleList(output_layers2)

        if merge_oper == "cat":
            merge = []
            for i in range(nstack):
                merge.append(DoubleConv(max_depth * 2, max_depth, max_depth))
            self.merge = nn.ModuleList(merge)
        self.merge_oper = merge_oper
        row_loc, col_loc = self.make_grid("cpu", size=heatmap_size)
        self.register_buffer("xx_loc", col_loc, False)
        self.register_buffer("yy_loc", row_loc, False)

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

    def forward_classic_hg(self, img):
        low_feat = self.pre(img)
        pre_feat = low_feat
        res = []
        for i in range(len(self.stages)):
            cur_feat = self.stages[i](pre_feat)
            if self.merge_oper == "cat":
                merged_feat = self.merge[i](torch.cat([pre_feat, cur_feat], dim=1))
            else:
                merged_feat = pre_feat + cur_feat
            hm1 = self.output_layers1[i](merged_feat)
            coord1 = self.GetCoord(hm1)

            hm2 = self.output_layers2[i](merged_feat)
            coord2 = self.GetCoord(hm2)
            res.append((coord1, hm1, coord2, hm2))

            pre_feat = merged_feat
        return res

    def forward_long_skip(self, img):
        low_feat = self.pre(img)
        pre_output = 0
        res = []
        for i in range(len(self.stages)):
            if i == 0:
                merged_feat = low_feat
            else:
                if self.merge_oper == "cat":
                    merged_feat = self.merge[i](torch.cat([low_feat, pre_output], dim=1))
                else:
                    merged_feat = low_feat + pre_output
            hm_basis = self.stages[i](merged_feat)
            hm = self.output_layers[i](hm_basis)
            coord = self.GetCoord(hm)
            res.append((coord, hm))
            pre_output = hm_basis
        return res

    def forward_ResCBSP_type1(self, img):
        # this is the residual connection between two prediction blocks in the paper dual vit
        low_feat = self.pre(img)
        pre_input = low_feat
        pre_output = 0
        res = []
        for i in range(len(self.stages)):
            if i == 0:
                merged_feat = low_feat
            else:
                if self.merge_oper == "cat":
                    merged_feat = self.merge[i](torch.cat([pre_input, pre_output], dim=1))
                else:
                    merged_feat = pre_input + pre_output
            hm_basis = self.stages[i](merged_feat)
            hm = self.output_layers[i](hm_basis)
            coord = self.GetCoord(hm)
            res.append((coord, hm))
            pre_input = merged_feat
            pre_output = hm_basis
        return res

    def forward(self, img, type=0):
        if 0 == type:
            return self.forward_classic_hg(img)
        elif 1 == type:
            return self.forward_ResCBSP_type1(img)
        elif 2 == type:
            return self.forward_long_skip(img)
        else:
            assert 0

class StackedHG2HeadCondition(nn.Module):
    def __init__(
        self,
        lmk_num1=98,
        lmk_num2=70,
        # Attn=lambda: UNet([256, 256, 256, 256, 256]),
        Attn=lambda: Hourglass(3, 256),
        nstack=4,
        heatmap_size=32,
        max_depth=256,
        backbone_net=lambda max_depth: HeadingNet([32, 64, max_depth]),
        num_dvit_per_pred_blk=1,
        merge_oper="cat",
    ):
        super(StackedHG2HeadCondition, self).__init__()
        # assert heatmap_size == 32
        # assert max_depth == 256 or max_depth == 192 or max_depth == 128 or max_depth == 64

        self.pre = backbone_net(max_depth)

        stages = []
        output_layers1 = []

        for i in range(nstack):
            vit_list = []
            for num in range(num_dvit_per_pred_blk):
                vit_list.append(
                    CoordConvTh(heatmap_size, heatmap_size, True, False, max_depth, max_depth, kernel_size=3, padding=1)
                )
                vit_list.append(Attn())
            block = nn.Sequential(*vit_list)
            stages.append(block)
            output_layers1.append(nn.Conv2d(max_depth, lmk_num1, 1))

        self.stages = nn.ModuleList(stages)
        self.output_layers1 = nn.ModuleList(output_layers1)
        self.embeding = nn.Embedding(2, max_depth * heatmap_size * heatmap_size)

        if merge_oper == "cat":
            merge = []
            for i in range(nstack):
                merge.append(DoubleConv(max_depth * 2, max_depth, max_depth))
            self.merge = nn.ModuleList(merge)
        self.merge_oper = merge_oper
        row_loc, col_loc = self.make_grid("cpu", size=heatmap_size)
        self.register_buffer("xx_loc", col_loc, False)
        self.register_buffer("yy_loc", row_loc, False)
        self.heatmap_size = heatmap_size
        self.max_depth = max_depth

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

    def forward_classic_hg(self, img, data_type):
        low_feat = self.pre(img)
        pre_feat = low_feat
        res = []

        embeding = self.embeding(data_type).reshape((-1, self.max_depth, self.heatmap_size, self.heatmap_size))
        for i in range(len(self.stages)):
            cur_feat = self.stages[i](pre_feat)
            if self.merge_oper == "cat":
                merged_feat = self.merge[i](torch.cat([pre_feat, cur_feat], dim=1))
            else:
                merged_feat = pre_feat + cur_feat
            merged_feat = merged_feat + embeding
            hm1 = self.output_layers1[i](merged_feat)
            coord1 = self.GetCoord(hm1)

            res.append((coord1, hm1))

            pre_feat = merged_feat
        return res

    def forward(self, img, data_type):

        return self.forward_classic_hg(img, data_type)

class SelfAttention(nn.Module):
    def __init__(self, channels):
        super(SelfAttention, self).__init__()
        self.channels = channels
        self.mha = nn.MultiheadAttention(channels, 4, batch_first=True)
        self.ln = nn.LayerNorm([channels])
        self.ff_self = nn.Sequential(
            nn.LayerNorm([channels]),
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )

    def forward(self, x):
        x_ln = self.ln(x)
        attention_value, _ = self.mha(x_ln, x_ln, x_ln)
        attention_value = attention_value + x
        attention_value = self.ff_self(attention_value) + attention_value
        return attention_value.contiguous()


class SeparateHM(nn.Module):
    def __init__(self, lmk_num=68):
        super(SeparateHM, self).__init__()
        self.backbone = HeadingNet([32, 64, 256])
        self.unet = UNet([256, 256, 256, 256, 256])
        self.lmk_num = lmk_num
        self.K = 1024 // lmk_num

        # body
        blocks = [
            get_block(in_channel=256, depth=1024, num_units=3),
            # get_block(in_channel=channels[0], depth=channels[1], num_units=3),
        ]
        units = []
        for bottlenecks in blocks:
            for b in bottlenecks:
                units.append(bottleneck_IR_SE(b.in_channel, b.depth, b.stride))
        self.body = nn.Sequential(*units)

        self.stage2 = nn.Sequential(
            HeadingNet((32, 64, 128, 256), 15),
        )

        self.sa = nn.Sequential(
            SelfAttention(256),
            SelfAttention(256),
            SelfAttention(256),
        )
        self.output_layer2 = nn.Linear(256,2)

        self.output_layer = nn.Conv2d(self.K, 1, 1)
        row_loc, col_loc = self.make_grid("cpu", size=16)
        self.register_buffer("xx_loc", col_loc, False)
        self.register_buffer("yy_loc", row_loc, False)

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

    def forward(self, img):
        B, _, _, _ = img.shape
        low_feat = self.backbone(img)
        low_feat = self.unet(low_feat)

        hm = self.body(low_feat)[:, :self.lmk_num * self.K]

        hm_feat = hm.reshape((B * self.lmk_num, self.K, 16, 16))
        hm1 = self.output_layer(hm_feat).reshape((B, self.lmk_num, 16, 16))
        coord1 = self.GetCoord(hm1)

        hm_feat = self.stage2(hm_feat).reshape((B, self.lmk_num, 256))
        hm_feat = self.sa(hm_feat)
        coord2 = self.output_layer2(hm_feat)

        return (coord1, hm1, coord2)
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # import torchsummary

    x = torch.randn((2, 3, 256, 256))
    heatmap_size = 32
    net = StackedHG(
        98,
        nstack=4,
        heatmap_size=heatmap_size,
        backbone_net=lambda max_depth: HeadingNet([32, 64, max_depth]),
        merge_oper="plus",
    )
    r = net.forward_ResCBSP_type1(x)[-1][0]
    print(r.shape)
    print(count_parameters(net) / 1024 / 1024)
