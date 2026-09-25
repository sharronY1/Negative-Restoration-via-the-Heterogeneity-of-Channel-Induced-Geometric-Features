"""Fixed L13 restoration network; state names match the released experiment."""

import torch
from torch import nn
from torch.nn import functional as F

from .blocks import NAFBlock, NAFBlockStack


def normalize_u(u):
    lo = u.amin(dim=(-2, -1), keepdim=True)
    hi = u.amax(dim=(-2, -1), keepdim=True)
    return (u - lo) / (hi - lo + 1e-8)


def broadcast_u(u, hw):
    h, w = hw
    uh, uw = u.shape[-2:]
    if h % uh or w % uw:
        raise ValueError(f"Uncertainty grid {(uh, uw)} cannot tile to {hw}")
    return u.repeat_interleave(h // uh, -2).repeat_interleave(w // uw, -1)


class ResidualGuidedBlock(NAFBlock):
    def __init__(self, channels):
        super().__init__(channels)
        # alpha_g was saved by the original experiment, although its feature
        # gate was disabled. Retain the parameter for exact checkpoint loading.
        self.alpha_g = nn.Parameter(torch.tensor(1.0))
        self.alpha_s = nn.Parameter(torch.tensor(1.0))
        self.alpha_s2 = nn.Parameter(torch.tensor(1.0))

    def forward(self, inp, u):
        if u.shape[-2:] != inp.shape[-2:]:
            raise ValueError("Uncertainty and activation sizes must match")
        m = normalize_u(u)
        x = self.sg(self.conv2(self.conv1(self.norm1(inp))))
        x = self.dropout1(self.conv3(x * self.sca(x)))
        y = inp + x * self.beta * (1.0 + self.alpha_s * m)
        x = self.dropout2(self.conv5(self.sg(self.conv4(self.norm2(y)))))
        return y + x * self.gamma * (1.0 + self.alpha_s2 * m)


class GuidedStack(nn.Module):
    def __init__(self, n, channels):
        super().__init__()
        self.blocks = nn.ModuleList([ResidualGuidedBlock(channels) for _ in range(n)])

    def forward(self, x, u):
        for block in self.blocks:
            x = block(x, u)
        return x


class RestorationNet(nn.Module):
    """RGB + joint R/G/B-view L13 features → restored RGB."""

    def __init__(self):
        super().__init__()
        self.n_logit = nn.Parameter(torch.zeros(()))
        self.intro = nn.Conv2d(3, 32, 3, padding=1)
        self.enc0 = NAFBlockStack([NAFBlock(32) for _ in range(2)])
        self.down2 = nn.Conv2d(32, 64, 2, 2)
        self.enc1 = NAFBlockStack([NAFBlock(64) for _ in range(2)])
        self.unshuffle = nn.PixelUnshuffle(7)
        self.align_down_proj = nn.Conv2d(64 * 49, 128, 1)
        self.enc_align = GuidedStack(8, 128)
        self.align_up_proj = nn.Conv2d(128, 64 * 49, 1, bias=False)
        self.shuffle = nn.PixelShuffle(7)
        self.dec1 = GuidedStack(2, 64)
        self.up2 = nn.Sequential(nn.Conv2d(64, 128, 1, bias=False), nn.PixelShuffle(2))
        self.dec0 = GuidedStack(2, 32)
        self.ending = nn.Conv2d(32, 3, 3, padding=1)

    def compute_u(self, feat):
        if feat.ndim != 5 or feat.shape[1] != 3:
            raise ValueError("Expected features [B,3,Hp,Wp,C] in R/G/B order")
        r, g, b = (feat[:, i].permute(0, 3, 1, 2).float() for i in range(3))
        n = self.n_logit.sigmoid()
        ref = n * g + (1.0 - n) * r
        cosine = (F.normalize(b, dim=1, eps=1e-8) *
                  F.normalize(ref, dim=1, eps=1e-8)).sum(1, keepdim=True)
        return 1.0 - cosine.clamp(-1.0, 1.0)

    def forward(self, inp, feat):
        h, w = inp.shape[-2:]
        x_in = F.pad(inp, (0, -w % 28, 0, -h % 28), mode="replicate")
        hp, wp = x_in.shape[-2] // 14, x_in.shape[-1] // 14
        fh, fw = feat.shape[2:4]
        if fh > hp or fw > wp:
            raise ValueError("DA3 features exceed the padded image grid")
        if fh != hp or fw != wp:
            feat = F.pad(feat, (0, 0, 0, wp - fw, 0, hp - fh), mode="replicate")
        u = self.compute_u(feat)
        skip0 = self.enc0(self.intro(x_in))
        skip1 = self.enc1(self.down2(skip0))
        x = self.align_down_proj(self.unshuffle(skip1))
        x = self.enc_align(x, u)
        x = self.shuffle(self.align_up_proj(x)) + skip1
        x = self.dec1(x, broadcast_u(u, x.shape[-2:]))
        x = self.up2(x) + skip0
        x = self.dec0(x, broadcast_u(u, x.shape[-2:]))
        return (self.ending(x) + x_in)[:, :, :h, :w]
