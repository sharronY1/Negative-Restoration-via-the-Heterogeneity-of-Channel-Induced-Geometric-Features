"""Shared NAF encoder with correspondence-guided additive color residuals."""

import torch
from torch import nn
from torch.nn import functional as F

from .blocks import NAFBlock, NAFBlockStack


class TokenResidual(nn.Module):
    def __init__(self, channels, hidden=512, num_layers=4):
        super().__init__()
        # Matches the L19 residual experiment: 1x1 MLP with num_layers>=2,
        # hidden width held constant, last layer zero-init.
        if num_layers < 2:
            raise ValueError(f"num_layers must be >= 2, got {num_layers}")
        layers = [nn.Conv2d(channels + 128, hidden, 1), nn.GELU()]
        for _ in range(num_layers - 2):
            layers.extend([nn.Conv2d(hidden, hidden, 1), nn.GELU()])
        layers.append(nn.Conv2d(hidden, channels, 1))
        self.mlp = nn.Sequential(*layers)
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, x, warped):
        if warped.shape[-2:] != x.shape[-2:]:
            warped = F.interpolate(warped, x.shape[-2:], mode="bilinear", align_corners=False)
        return x + self.mlp(torch.cat([x, warped], dim=1))


class SharedNAFBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        def stack(n, c):
            return NAFBlockStack([NAFBlock(c) for _ in range(n)])
        self.intro = nn.Conv2d(3, 32, 3, padding=1)
        self.enc0 = stack(2, 32)
        self.down2 = nn.Conv2d(32, 64, 2, 2)
        self.enc1 = stack(2, 64)
        self.unshuffle = nn.PixelUnshuffle(7)
        self.align_down_proj = nn.Conv2d(64 * 49, 128, 1)
        self.enc_align = stack(8, 128)
        self.align_up_proj = nn.Conv2d(128, 64 * 49, 1, bias=False)
        self.shuffle = nn.PixelShuffle(7)
        self.dec1 = stack(2, 64)
        self.up2 = nn.Sequential(nn.Conv2d(64, 128, 1, bias=False), nn.PixelShuffle(2))
        self.dec0 = stack(2, 32)
        self.ending = nn.Conv2d(32, 3, 3, padding=1)
        self.res_align = TokenResidual(128)
        self.res_dec1 = TokenResidual(64)
        self.res_dec0 = TokenResidual(32)

    def encode(self, inp):
        h, w = inp.shape[-2:]
        base = F.pad(inp, (0, -w % 14, 0, -h % 14), mode="replicate")
        skip0 = self.enc0(self.intro(base))
        skip1 = self.enc1(self.down2(skip0))
        aligned = self.align_down_proj(self.unshuffle(skip1))
        return aligned, skip0, skip1, base, (h, w)

    def decode(self, enc, warped):
        aligned, skip0, skip1, base, (h, w) = enc
        x = self.enc_align(self.res_align(aligned, warped))
        x = self.shuffle(self.align_up_proj(x)) + skip1
        x = self.dec1(self.res_dec1(x, warped))
        x = self.up2(x) + skip0
        x = self.dec0(self.res_dec0(x, warped))
        return (self.ending(x) + base)[:, :, :h, :w]


class ColorMappingNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = SharedNAFBackbone()

    def forward(self, inp, reference, input_features, reference_features):
        q = F.normalize(input_features.flatten(2).transpose(1, 2), dim=-1)
        k = F.normalize(reference_features.flatten(2).transpose(1, 2), dim=-1)
        # Residual mode in the source uses ALL reference tokens (temperature .01).
        attention = (torch.bmm(q, k.transpose(1, 2)) / 0.01).softmax(dim=-1)
        enc = self.backbone.encode(inp)
        ref = self.backbone.encode(reference)[0]
        ref_hw = reference_features.shape[-2:]
        if ref.shape[-2:] != ref_hw:
            ref = F.interpolate(ref, ref_hw, mode="bilinear", align_corners=False)
        warped = torch.bmm(attention, ref.flatten(2).transpose(1, 2))
        warped = warped.transpose(1, 2).reshape(inp.shape[0], 128, *input_features.shape[-2:])
        if warped.shape[-2:] != enc[0].shape[-2:]:
            warped = F.interpolate(warped, enc[0].shape[-2:], mode="bilinear", align_corners=False)
        return self.backbone.decode(enc, warped)
