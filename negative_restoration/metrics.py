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
