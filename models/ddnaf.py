import torch
import torch.nn as nn
import torch.nn.functional as F

from models.utils.torchblocks import *

# -------------------------
# D-NAFBlock (NAF + Noise-aware Spatial Gate + SCA)
# -------------------------
class DNAFBlock(nn.Module):
    def __init__(
        self,
        c: int,
        dw_expand: int = 2,
        ffn_expand: int = 2,
        drop_out_rate: float = 0.0,
    ):
        super().__init__()
        dw_channels = c * dw_expand

        # ---- branch A (NAF conv branch) ----
        self.norm1 = LayerNorm2d(c)
        self.conv1 = nn.Conv2d(c, dw_channels, kernel_size=1, bias=True)
        self.conv2 = nn.Conv2d(dw_channels, dw_channels, kernel_size=3, padding=1, groups=dw_channels, bias=True)
        self.sg = SimpleGate()

        # noise-aware spatial gate: sigmoid(DWConv3x3(LN(x))) producing C channels
        self.gate_norm = LayerNorm2d(c)
        self.noise_dw = nn.Conv2d(c, c, kernel_size=3, padding=1, groups=c, bias=True)
        self.gate_act = nn.Sigmoid()

        # SCA
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dw_channels // 2, dw_channels // 2, kernel_size=1, bias=True),
        )

        # project back to C
        self.conv3 = nn.Conv2d(dw_channels // 2, c, kernel_size=1, bias=True)
        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0 else nn.Identity()

        # ---- branch B (gated FFN) ----
        self.norm2 = LayerNorm2d(c)
        ffn_channels = c * ffn_expand
        self.conv4 = nn.Conv2d(c, ffn_channels, kernel_size=1, bias=True)  # C -> 2C
        self.conv5 = nn.Conv2d(ffn_channels // 2, c, kernel_size=1, bias=True)      # (after SG) C -> C
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0 else nn.Identity()

        # residual scaling (init 0)
        self.beta  = nn.Parameter(torch.zeros(1, c, 1, 1))
        self.gamma = nn.Parameter(torch.zeros(1, c, 1, 1))

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        # ----- Branch A -----
        x = self.norm1(inp)
        x = self.conv1(x)     # C -> 2C
        x = self.conv2(x)     # DW 3x3 on 2C
        x = self.sg(x)        # 2C -> C

        # noise-aware spatial gate computed from LN(x) (separate LN module)
        g = self.gate_norm(inp)
        g = self.noise_dw(g)  # depthwise 3x3, C channels
        g = self.gate_act(g)  # sigmoid, [0,1]
        g = torch.clamp(g * 2.0, 0.0, 2.0)  # scale to [0, 2] and clip
        x = x * g
        x = x * self.sca(x)
        x = self.conv3(x)
        x = self.dropout1(x)

        y = inp + x * self.beta     # first residual

        # ----- Branch B -----
        x = self.norm2(y)
        x = self.conv4(x)  # C -> 2C
        x = self.sg(x)     # 2C -> C
        x = self.conv5(x)  # C -> C
        x = self.dropout2(x)

        return y + x * self.gamma    # second residual

# -------------------------
# D-NAFNet
# -------------------------
class DNAFNet(nn.Module):
    def __init__(
        self,
        img_channel: int = 3,
        width: int = 32,
        middle_blk_num: int = 12,
        enc_blk_nums=(2, 2, 4, 8),
        dec_blk_nums=(2, 2, 2, 2),
        dw_expand: int = 2,
        ffn_expand: int = 2,
        drop_out_rate: float = 0.0,
    ):
        super().__init__()

        self.intro  = nn.Conv2d(img_channel, width, kernel_size=3, padding=1, bias=True)
        self.ending = nn.Conv2d(width, img_channel, kernel_size=3, padding=1, bias=True)

        self.encoders = nn.ModuleList()
        self.downs    = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.ups      = nn.ModuleList()

        chan = width

        # encoder stages
        for num in enc_blk_nums:
            self.encoders.append(nn.Sequential(*[
                DNAFBlock(chan, dw_expand=dw_expand, ffn_expand=ffn_expand, drop_out_rate=drop_out_rate)
                for _ in range(num)
            ]))
            self.downs.append(nn.Conv2d(chan, 2 * chan, kernel_size=2, stride=2, bias=True))
            chan *= 2

        # middle blocks
        self.middle_blks = nn.Sequential(*[
            DNAFBlock(chan, dw_expand=dw_expand, ffn_expand=ffn_expand, drop_out_rate=drop_out_rate)
            for _ in range(middle_blk_num)
        ])

        # decoder stages
        for num in dec_blk_nums:
            self.ups.append(nn.Sequential(
                nn.Conv2d(chan, chan * 2, kernel_size=1, bias=False),
                nn.PixelShuffle(2),
            ))
            chan //= 2
            self.decoders.append(nn.Sequential(*[
                DNAFBlock(chan, dw_expand=dw_expand, ffn_expand=ffn_expand, drop_out_rate=drop_out_rate)
                for _ in range(num)
            ]))

        self.padder_size = 2 ** len(self.encoders)

    def check_image_size(self, x: torch.Tensor) -> torch.Tensor:
        _, _, h, w = x.size()
        pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        return F.pad(x, (0, pad_w, 0, pad_h))

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        b, c, h, w = inp.shape
        x = self.check_image_size(inp)

        x = self.intro(x)
        encs = []

        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            encs.append(x)
            x = down(x)

        x = self.middle_blks(x)

        for decoder, up, enc_skip in zip(self.decoders, self.ups, encs[::-1]):
            x = up(x)
            x = x + enc_skip
            x = decoder(x)

        x = self.ending(x)
        x = x + self.check_image_size(inp)  # global residual on padded input
        return x[:, :, :h, :w]


def create_dnafnet_sidd_width32(img_channel=3):
    return DNAFNet(
        img_channel=img_channel,
        width=32,
        middle_blk_num=12,
        enc_blk_nums=(2, 2, 4, 8),
        dec_blk_nums=(2, 2, 2, 2),
        drop_out_rate=0.0,
    )