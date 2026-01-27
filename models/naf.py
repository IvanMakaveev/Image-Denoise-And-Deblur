import torch
import torch.nn as nn
import torch.nn.functional as F

from models.utils.torchblocks import *

# -------------------------
# NAFBlock
# -------------------------
class NAFBlock(nn.Module):
    def __init__(self, c: int, dw_expand: int = 2, ffn_expand: int = 2, drop_out_rate: float = 0.0):
        super().__init__()
        dw_channels = c * dw_expand

        # conv branch
        self.norm1 = LayerNorm2d(c)
        self.conv1 = nn.Conv2d(c, dw_channels, kernel_size=1, bias=True)
        self.conv2 = nn.Conv2d(dw_channels, dw_channels, kernel_size=3, padding=1, groups=dw_channels, bias=True)
        self.sg    = SimpleGate()

        # SCA: simplified channel attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dw_channels // 2, dw_channels // 2, kernel_size=1, bias=True),
        )

        self.conv3 = nn.Conv2d(dw_channels // 2, c, kernel_size=1, bias=True)
        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0 else nn.Identity()

        # FFN branch
        self.norm2 = LayerNorm2d(c)
        ffn_channels = c * ffn_expand
        self.conv4 = nn.Conv2d(c, ffn_channels, kernel_size=1, bias=True)
        self.conv5 = nn.Conv2d(ffn_channels // 2, c, kernel_size=1, bias=True)
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0 else nn.Identity()

        # residual scaling (starts at 0 like official)
        self.beta  = nn.Parameter(torch.zeros(1, c, 1, 1))
        self.gamma = nn.Parameter(torch.zeros(1, c, 1, 1))

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        x = self.norm1(inp)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)                 # channels halve here
        x = x * self.sca(x)            # channel attention
        x = self.conv3(x)
        x = self.dropout1(x)

        y = inp + x * self.beta        # first residual

        x = self.conv4(self.norm2(y))
        x = self.sg(x)                 # halve
        x = self.conv5(x)
        x = self.dropout2(x)

        return y + x * self.gamma      # second residual

# -------------------------
# NAFNet
# -------------------------
class NAFNet(nn.Module):
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
                NAFBlock(chan, dw_expand=dw_expand, ffn_expand=ffn_expand, drop_out_rate=drop_out_rate)
                for _ in range(num)
            ]))
            self.downs.append(nn.Conv2d(chan, 2 * chan, kernel_size=2, stride=2, bias=True))
            chan *= 2

        # middle blocks
        self.middle_blks = nn.Sequential(*[
            NAFBlock(chan, dw_expand=dw_expand, ffn_expand=ffn_expand, drop_out_rate=drop_out_rate)
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
                NAFBlock(chan, dw_expand=dw_expand, ffn_expand=ffn_expand, drop_out_rate=drop_out_rate)
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
            x = x + enc_skip          # skip ADD (core NAFNet detail)
            x = decoder(x)

        x = self.ending(x)
        x = x + self.check_image_size(inp)  # global residual on padded input
        return x[:, :, :h, :w]


def create_nafnet_sidd_width32(img_channel=3):
    # Common config used in many NAFNet examples: width=32, enc [2,2,4,8], middle=12, dec [2,2,2,2]
    return NAFNet(img_channel=img_channel, width=32, middle_blk_num=12,
                  enc_blk_nums=(2, 2, 4, 8), dec_blk_nums=(2, 2, 2, 2))