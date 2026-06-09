import torch
import torch.nn as nn

from lib.models.unet import DoubleConv


class PositionalEncoding(nn.Module):  # @save
    """Positional encoding."""

    def __init__(self, num_hiddens, dropout=0, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        # Create a long enough P
        P = torch.zeros((1, max_len, num_hiddens))
        X = torch.arange(max_len, dtype=torch.float32).reshape(-1, 1) / torch.pow(
            10000, torch.arange(0, num_hiddens, 2, dtype=torch.float32) / num_hiddens
        )
        P[:, :, 0::2] = torch.sin(X)
        P[:, :, 1::2] = torch.cos(X)
        self.register_buffer("P", P, False)

    def forward(self, X):
        X = X + self.P[:, : X.shape[1], :]
        return self.dropout(X)


class PositionalEncoding1(nn.Module):
    def __init__(self, num_hidden):
        super(PositionalEncoding1, self).__init__()
        self.linear = nn.Linear(num_hidden + 1, num_hidden)

    def forward(self, x):
        B, C, N = x.shape
        pos = (torch.arange(C, device=x.device, dtype=torch.float) - (C - 1) / 2) / (
            (C - 1) / 2
        )
        pos = pos.reshape((1, C, 1)).repeat((B, 1, 1))
        x = torch.cat([x, pos], dim=2)
        x = self.linear(x)
        return x


class SelfAttention(nn.Module):
    def __init__(self, channels, pos_encoding=0):
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
        self.pos_encoding = None
        if pos_encoding == 1:
            self.pos_encoding = PositionalEncoding1(channels)
        elif pos_encoding == 2:
            self.pos_encoding = PositionalEncoding(channels)

    def forward(self, x):
        size = x.shape[-1]
        x = x.view(-1, self.channels, size * size).swapaxes(1, 2)
        if self.pos_encoding is not None:
            x = self.pos_encoding(x)
        x_ln = self.ln(x)
        attention_value, _ = self.mha(x_ln, x_ln, x_ln)
        attention_value = attention_value + x
        attention_value = self.ff_self(attention_value) + attention_value
        return (
            attention_value.swapaxes(2, 1)
            .view(-1, self.channels, size, size)
            .contiguous()
        )


class SelfAttention_block(nn.Module):
    def __init__(self, channels, pos_encoding=0, win_size=2):
        super(SelfAttention_block, self).__init__()

        self.win_size = win_size
        multiplier = win_size * win_size
        self.mha = nn.MultiheadAttention(channels * multiplier, 4, batch_first=True)
        self.ln = nn.LayerNorm([channels * multiplier])
        self.ff_self = nn.Sequential(
            nn.LayerNorm([channels * multiplier]),
            nn.Linear(channels * multiplier, channels * multiplier),
            nn.GELU(),
            nn.Linear(channels * multiplier, channels * multiplier),
        )
        self.pos_encoding = None
        if pos_encoding == 1:
            self.pos_encoding = PositionalEncoding1(channels * multiplier)
        elif pos_encoding == 2:
            self.pos_encoding = PositionalEncoding(channels * multiplier)

    def forward(self, x):
        B, C, H, W = x.shape

        x = (
            x.reshape(
                (
                    B,
                    C,
                    H // self.win_size,
                    self.win_size,
                    W // self.win_size,
                    self.win_size,
                )
            )
            .permute((0, 2, 4, 1, 3, 5))
            .reshape((B, (H * W) // (self.win_size * self.win_size), -1))
        )

        # x = x.view(-1, self.channels, size * size).swapaxes(1, 2)
        if self.pos_encoding is not None:
            x = self.pos_encoding(x)
        x_ln = self.ln(x)
        attention_value, _ = self.mha(x_ln, x_ln, x_ln)
        attention_value = attention_value + x
        attention_value = self.ff_self(attention_value) + attention_value
        attention_value = (
            attention_value.reshape(
                (
                    B,
                    H // self.win_size,
                    W // self.win_size,
                    C,
                    self.win_size,
                    self.win_size,
                )
            )
            .permute((0, 3, 1, 4, 2, 5))
            .reshape((B, C, H, W))
        )
        return attention_value.contiguous()


class SelfAttention_block2(nn.Module):
    def __init__(self, channels, win_size=2, out_channel=None):
        super(SelfAttention_block2, self).__init__()
        self.win_size = win_size
        if out_channel is None:
            self.out_channel = channels
        else:
            self.out_channel = out_channel
        multiplier = win_size * win_size
        self.mha = nn.MultiheadAttention(
            self.out_channel * multiplier, 4, batch_first=True
        )
        self.ln = nn.LayerNorm([self.out_channel * multiplier])
        self.ff_self = nn.Sequential(
            nn.LayerNorm([self.out_channel * multiplier]),
            nn.Linear(self.out_channel * multiplier, self.out_channel * multiplier),
            nn.GELU(),
            nn.Linear(self.out_channel * multiplier, self.out_channel * multiplier),
        )

        self.patch_embed = nn.Conv2d(
            channels, self.out_channel * multiplier, win_size, win_size
        )

    def forward(self, x):
        B, C, H, W = x.shape
        x = (
            self.patch_embed(x)
            .permute((0, 2, 3, 1))
            .reshape((B, (H * W) // (self.win_size * self.win_size), -1))
        )
        x_ln = self.ln(x)
        attention_value, _ = self.mha(x_ln, x_ln, x_ln)
        attention_value = attention_value + x
        attention_value = self.ff_self(attention_value) + attention_value
        attention_value = (
            attention_value.reshape(
                (
                    B,
                    H // self.win_size,
                    W // self.win_size,
                    self.win_size,
                    self.win_size,
                    self.out_channel,
                )
            )
            .permute((0, 5, 1, 3, 2, 4))
            .reshape((B, self.out_channel, H, W))
        )
        return attention_value.contiguous()


class SelfAttention_block3(nn.Module):
    def __init__(self, channels, pos_encoding=0, win_size=2):
        super(SelfAttention_block3, self).__init__()

        self.win_size = win_size
        multiplier = win_size * win_size
        self.mha = nn.MultiheadAttention(channels * multiplier, 4, batch_first=True)
        self.ln = nn.LayerNorm([channels * multiplier])
        self.ff_self = nn.Sequential(
            nn.LayerNorm([channels * multiplier]),
            nn.Linear(channels * multiplier, channels * multiplier),
            nn.GELU(),
            nn.Linear(channels * multiplier, channels * multiplier),
        )
        self.pos_encoding = None
        if pos_encoding == 1:
            self.pos_encoding = PositionalEncoding1(channels * multiplier)
        elif pos_encoding == 2:
            self.pos_encoding = PositionalEncoding(channels * multiplier)

    def forward(self, x):
        B, C, H, W = x.shape

        x = (
            x.reshape(
                (
                    B,
                    C,
                    H // self.win_size,
                    self.win_size,
                    W // self.win_size,
                    self.win_size,
                )
            )
            .permute((0, 2, 4, 3, 5, 1))
            .reshape((B, (H * W) // (self.win_size * self.win_size), -1))
        )

        # x = x.view(-1, self.channels, size * size).swapaxes(1, 2)
        if self.pos_encoding is not None:
            x = self.pos_encoding(x)
        x_ln = self.ln(x)
        attention_value, _ = self.mha(x_ln, x_ln, x_ln)
        attention_value = attention_value + x
        attention_value = self.ff_self(attention_value) + attention_value
        attention_value = (
            attention_value.reshape(
                (
                    B,
                    H // self.win_size,
                    W // self.win_size,
                    self.win_size,
                    self.win_size,
                    C,
                )
            )
            .permute((0, 5, 1, 3, 2, 4))
            .reshape((B, C, H, W))
        )
        return attention_value.contiguous()


class SelfAttention2(nn.Module):
    def __init__(self, img_size):
        super(SelfAttention2, self).__init__()
        self.img_size = img_size
        self.mha = nn.MultiheadAttention(img_size * img_size, 4, batch_first=True)
        self.ln = nn.LayerNorm([img_size * img_size])
        self.ff_self = nn.Sequential(
            nn.LayerNorm([img_size * img_size]),
            nn.Linear(img_size * img_size, img_size * img_size),
            nn.GELU(),
            nn.Linear(img_size * img_size, img_size * img_size),
        )

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.view(B, C, self.img_size * self.img_size)

        x_ln = self.ln(x)

        K = x_ln

        attention_value, _ = self.mha(x_ln, K, x_ln)
        attention_value = attention_value + x
        attention_value = self.ff_self(attention_value) + attention_value
        res = attention_value.view(B, C, self.img_size, self.img_size).contiguous()
        return res


class SelfAttention2_block(nn.Module):
    def __init__(self, img_size, in_channel, out_channel, win_size=2):
        super(SelfAttention2_block, self).__init__()
        self.img_size = img_size
        self.win_size = win_size
        self.in_channel = in_channel
        self.out_channel = out_channel

        self.mha = nn.MultiheadAttention(
            img_size * img_size // (win_size * win_size), 4, batch_first=True
        )
        self.ln = nn.LayerNorm([img_size * img_size // (win_size * win_size)])
        self.ff_self = nn.Sequential(
            nn.LayerNorm([img_size * img_size // (win_size * win_size)]),
            nn.Linear(
                img_size * img_size // (win_size * win_size),
                img_size * img_size // (win_size * win_size),
            ),
            nn.GELU(),
            nn.Linear(
                img_size * img_size // (win_size * win_size),
                img_size * img_size // (win_size * win_size),
            ),
        )

        self.patch_embed = nn.Conv2d(
            in_channel, out_channel * win_size * win_size, win_size, win_size
        )

    def forward(self, x):
        B, C, H, W = x.shape
        x_embed = self.patch_embed(x)
        x_embed = x_embed.reshape(
            (
                B,
                self.out_channel * self.win_size * self.win_size,
                H * W // (self.win_size * self.win_size),
            )
        )

        x_ln = self.ln(x_embed)
        K = x_ln

        attention_value, _ = self.mha(x_ln, K, x_ln)
        attention_value = attention_value + x_embed
        attention_value = self.ff_self(attention_value) + attention_value
        res = (
            attention_value.reshape(
                (
                    B,
                    self.win_size,
                    self.win_size,
                    self.out_channel,
                    H // self.win_size,
                    W // self.win_size,
                )
            )
            .permute((0, 3, 4, 1, 5, 2))
            .reshape((B, self.out_channel, H, W))
        )
        return res


class SelfAttention2Local(nn.Module):
    def __init__(self, img_size):
        super(SelfAttention2Local, self).__init__()
        self.attn = SelfAttention2(img_size // 2)

    def forward(self, x):
        B, C, H, W = x.shape
        x1 = x[:, :, 0::2, 0::2]
        x2 = x[:, :, 0::2, 1::2]
        x3 = x[:, :, 1::2, 0::2]
        x4 = x[:, :, 1::2, 1::2]

        xx = torch.stack([x1, x2, x3, x4], dim=2).reshape((B * C, 4, H // 2, W // 2))
        xx = self.attn(xx).reshape((B, C, 4, H // 2, W // 2))
        res = torch.zeros_like(x)
        res[:, :, 0::2, 0::2] = xx[:, :, 0]
        res[:, :, 0::2, 1::2] = xx[:, :, 1]
        res[:, :, 1::2, 0::2] = xx[:, :, 2]
        res[:, :, 1::2, 1::2] = xx[:, :, 3]

        return res


class SelfAttention2Local2(nn.Module):
    def __init__(self, channels):
        super(SelfAttention2Local2, self).__init__()
        self.attn = SelfAttention_block3()

    def forward(self, x):
        B, C, H, W = x.shape
        x1 = x[:, :, 0::2, 0::2]
        x2 = x[:, :, 0::2, 1::2]
        x3 = x[:, :, 1::2, 0::2]
        x4 = x[:, :, 1::2, 1::2]

        xx = torch.stack([x1, x2, x3, x4], dim=2).reshape((B * C, 4, H // 2, W // 2))
        xx = self.attn(xx).reshape((B, C, 4, H // 2, W // 2))
        res = torch.zeros_like(x)
        res[:, :, 0::2, 0::2] = xx[:, :, 0]
        res[:, :, 0::2, 1::2] = xx[:, :, 1]
        res[:, :, 1::2, 0::2] = xx[:, :, 2]
        res[:, :, 1::2, 1::2] = xx[:, :, 3]

        return res


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


class SA2SA1(nn.Module):
    def __init__(self, img_size=32, channel_size=256, concat=True):
        super(SA2SA1, self).__init__()
        # self.sa1 = SelfAttention(channel_size)
        self.sa1 = SelfAttention_block2(channel_size)
        self.sa2 = SelfAttention2(img_size)
        self.conv = None
        if concat:
            self.conv = DoubleConv(channel_size * 2, channel_size, channel_size)

    def forward(self, x):
        if not self.conv:
            return self.sa2(x) + self.sa1(x)
        else:
            c = torch.cat([self.sa2(x), self.sa1(x)], dim=1)
            return self.conv(c)


class SA2SA1_2(nn.Module):
    def __init__(self, img_size=32, channel_size=256, win_size=2, concat=True):
        super(SA2SA1_2, self).__init__()
        # self.sa1 = SelfAttention(channel_size)
        self.sa1 = SelfAttention_block2(
            channels=channel_size, win_size=win_size, out_channel=channel_size // 2
        )
        self.sa2 = SelfAttention2_block(
            img_size=img_size,
            in_channel=channel_size,
            out_channel=channel_size // 2,
            win_size=win_size,
        )
        self.conv = None
        if concat:
            self.conv = DoubleConv(channel_size, channel_size)

    def forward(self, x):
        if not self.conv:
            return self.sa2(x) + self.sa1(x)
        else:
            c = torch.cat([self.sa2(x), self.sa1(x)], dim=1)
            return self.conv(c)


class SA2SA1_twins(nn.Module):
    def __init__(self, img_size=32, channel_size=256, win_size=2, concat=True):
        super(SA2SA1_twins, self).__init__()
        # self.sa1 = SelfAttention(channel_size)
        self.sa1 = SelfAttention_block2(
            channels=channel_size, win_size=win_size, out_channel=channel_size // 2
        )
        self.sa2 = SelfAttention_block2(
            channels=channel_size, win_size=win_size, out_channel=channel_size // 2
        )
        self.conv = None
        if concat:
            self.conv = DoubleConv(channel_size, channel_size)

    def forward(self, x):
        if not self.conv:
            return self.sa2(x) + self.sa1(x)
        else:
            c = torch.cat([self.sa2(x), self.sa1(x)], dim=1)
            return self.conv(c)


class SA2SA1_twins2(nn.Module):
    def __init__(self, img_size=32, channel_size=256, win_size=2, concat=True):
        super(SA2SA1_twins2, self).__init__()
        # self.sa1 = SelfAttention(channel_size)
        self.sa1 = SelfAttention2_block(
            img_size=img_size,
            in_channel=channel_size,
            out_channel=channel_size // 2,
            win_size=win_size,
        )
        self.sa2 = SelfAttention2_block(
            img_size=img_size,
            in_channel=channel_size,
            out_channel=channel_size // 2,
            win_size=win_size,
        )
        self.conv = None
        if concat:
            self.conv = DoubleConv(channel_size, channel_size)

    def forward(self, x):
        if not self.conv:
            return self.sa2(x) + self.sa1(x)
        else:
            c = torch.cat([self.sa2(x), self.sa1(x)], dim=1)
            return self.conv(c)


class SelfAttentionGL(nn.Module):
    def __init__(self, img_size=32, channel_size=256, concat=True):
        super(SelfAttentionGL, self).__init__()
        # self.sa1 = SelfAttention(channel_size)
        self.sa1 = SelfAttention_block2(channel_size)
        self.sa2 = SelfAttention2(img_size)
        self.conv = None
        if concat:
            self.conv = DoubleConv(channel_size * 2, channel_size, channel_size)

    def forward(self, x):
        if not self.conv:
            return self.sa2(x) + self.sa1(x)
        else:
            c = torch.cat([self.sa2(x), self.sa1(x)], dim=1)
            return self.conv(c)


class SA2SA1_3(nn.Module):
    def __init__(self, img_size=32, channel_size=256, win_size=2, concat=True):
        super(SA2SA1_3, self).__init__()
        # self.sa1 = SelfAttention(channel_size)
        self.sa1 = MixBlk(in_channel=channel_size, win_size=win_size)
        self.sa2 = SelfAttention2_block(
            img_size=img_size,
            in_channel=channel_size,
            out_channel=channel_size // 2,
            win_size=win_size,
        )
        self.conv = None
        if concat:
            self.conv = DoubleConv(channel_size + (channel_size // 2), channel_size)

    def forward(self, x):
        if not self.conv:
            return self.sa2(x) + self.sa1(x)
        else:
            f2 = self.sa1(x)
            f1 = self.sa2(x)
            c = torch.cat([f1, f2], dim=1)
            return self.conv(c)


class SA2SA1_4(nn.Module):
    def __init__(self, img_size=32, channel_size=256, win_size=2, concat=True):
        super(SA2SA1_4, self).__init__()
        self.sa1 = SelfAttention_block2(
            channels=channel_size, win_size=win_size, out_channel=channel_size // 2
        )
        self.sa2 = SelfAttention2_block(
            img_size=img_size,
            in_channel=channel_size,
            out_channel=channel_size // 2,
            win_size=win_size,
        )
        self.sa3 = nn.Sequential(
            nn.Conv2d(channel_size, channel_size // 2, 1),
            nn.BatchNorm2d(channel_size // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel_size // 2, channel_size // 2, 3, padding=1),
            nn.BatchNorm2d(channel_size // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel_size // 2, channel_size // 2, 1),
        )
        self.conv = None
        if concat:
            self.conv = DoubleConv(channel_size + (channel_size // 2), channel_size)

    def forward(self, x):
        if not self.conv:
            return self.sa2(x) + self.sa1(x)
        else:
            c = torch.cat([self.sa3(x), self.sa2(x), self.sa1(x)], dim=1)
            return self.conv(c)


if __name__ == "__main__":
    x = torch.rand((2, 256, 32, 32))
    sa = SA2SA1_2(32, 256)
    y = sa(x)
    print(y.shape)
