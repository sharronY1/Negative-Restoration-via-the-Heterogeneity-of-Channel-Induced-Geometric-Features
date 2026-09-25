"""PSNR / SSIM / FFT-L1."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def psnr(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-10) -> torch.Tensor:
    pred = pred.clamp(0, 1)
    target = target.clamp(0, 1)
    mse = (pred - target).pow(2).mean(dim=(1, 2, 3))
    return (-10.0 * torch.log10(mse + eps)).mean()


def _gaussian_window(window_size: int = 11, sigma: float = 1.5, channels: int = 3, device=None, dtype=None):
    coords = torch.arange(window_size, dtype=dtype, device=device) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    window = g[:, None] * g[None, :]
    window = window.expand(channels, 1, window_size, window_size).contiguous()
    return window


def ssim(pred: torch.Tensor, target: torch.Tensor, window_size: int = 11) -> torch.Tensor:
    pred = pred.clamp(0, 1)
    target = target.clamp(0, 1)
    c = pred.shape[1]
    window = _gaussian_window(window_size, 1.5, c, pred.device, pred.dtype)
    pad = window_size // 2
    mu1 = F.conv2d(pred, window, padding=pad, groups=c)
    mu2 = F.conv2d(target, window, padding=pad, groups=c)
    mu1_sq, mu2_sq, mu12 = mu1 * mu1, mu2 * mu2, mu1 * mu2
    sigma1_sq = F.conv2d(pred * pred, window, padding=pad, groups=c) - mu1_sq
    sigma2_sq = F.conv2d(target * target, window, padding=pad, groups=c) - mu2_sq
    sigma12 = F.conv2d(pred * target, window, padding=pad, groups=c) - mu12
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    ssim_map = ((2 * mu12 + c1) * (2 * sigma12 + c2)) / (
        (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
    )
    return ssim_map.mean()


def fft_l1(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred_f = torch.fft.fft2(pred, dim=(-2, -1))
    tgt_f = torch.fft.fft2(target, dim=(-2, -1))
    pred_s = torch.stack((pred_f.real, pred_f.imag), dim=-1)
    tgt_s = torch.stack((tgt_f.real, tgt_f.imag), dim=-1)
    return F.l1_loss(pred_s, tgt_s)


def rgb_to_lab_normalized(image: torch.Tensor) -> torch.Tensor:
    """Differentiable D65 sRGB [0,1] → normalized Lab (ver3_r2r_colorbank style)."""
    threshold = 0.04045
    linear = torch.where(
        image <= threshold,
        image / 12.92,
        ((image + 0.055) / 1.055).pow(2.4),
    )

    red, green, blue = linear.unbind(dim=1)
    x = 0.4124564 * red + 0.3575761 * green + 0.1804375 * blue
    y = 0.2126729 * red + 0.7151522 * green + 0.0721750 * blue
    z = 0.0193339 * red + 0.1191920 * green + 0.9503041 * blue

    x = x / 0.95047
    z = z / 1.08883
    delta = 6.0 / 29.0

    def lab_curve(value: torch.Tensor) -> torch.Tensor:
        return torch.where(
            value > delta**3,
            value.clamp_min(0).pow(1.0 / 3.0),
            value / (3.0 * delta**2) + 4.0 / 29.0,
        )

    fx, fy, fz = lab_curve(x), lab_curve(y), lab_curve(z)
    lightness = (116.0 * fy - 16.0) / 100.0
    channel_a = 500.0 * (fx - fy) / 128.0
    channel_b = 200.0 * (fy - fz) / 128.0
    return torch.stack((lightness, channel_a, channel_b), dim=1)


def _wasserstein1_hist(
    hist_a: torch.Tensor,
    hist_b: torch.Tensor,
    bin_width: float,
) -> torch.Tensor:
    """W1 between two 1D histograms (equal-width bins)."""
    pa = hist_a / hist_a.sum().clamp_min(1e-8)
    pb = hist_b / hist_b.sum().clamp_min(1e-8)
    cdfa = torch.cumsum(pa, dim=0)
    cdfb = torch.cumsum(pb, dim=0)
    return torch.abs(cdfa - cdfb).sum() * float(bin_width)


def _channel_hist_wasserstein(
    pred_ch: torch.Tensor,
    tgt_ch: torch.Tensor,
    *,
    bins: int,
    vmin: float,
    vmax: float,
) -> torch.Tensor:
    """Mean batch W1(H(pred), H(tgt)) for one Lab channel."""
    flat_p = pred_ch.reshape(pred_ch.shape[0], -1)
    flat_t = tgt_ch.reshape(tgt_ch.shape[0], -1)
    bin_width = (float(vmax) - float(vmin)) / float(bins)
    losses: list[torch.Tensor] = []
    for i in range(flat_p.shape[0]):
        hp = torch.histc(flat_p[i], bins=bins, min=vmin, max=vmax)
        ht = torch.histc(flat_t[i], bins=bins, min=vmin, max=vmax)
        losses.append(_wasserstein1_hist(hp, ht, bin_width))
    return torch.stack(losses).mean()


def lab_hist_wasserstein_components(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    bins: int = 64,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """W1 between per-channel Lab histograms: L, a, b."""
    pred_lab = rgb_to_lab_normalized(pred.clamp(0.0, 1.0))
    target_lab = rgb_to_lab_normalized(target.clamp(0.0, 1.0))
    loss_l = _channel_hist_wasserstein(
        pred_lab[:, 0], target_lab[:, 0], bins=bins, vmin=0.0, vmax=1.0
    )
    loss_a = _channel_hist_wasserstein(
        pred_lab[:, 1], target_lab[:, 1], bins=bins, vmin=-1.5, vmax=1.5
    )
    loss_b = _channel_hist_wasserstein(
        pred_lab[:, 2], target_lab[:, 2], bins=bins, vmin=-1.5, vmax=1.5
    )
    return loss_l, loss_a, loss_b
