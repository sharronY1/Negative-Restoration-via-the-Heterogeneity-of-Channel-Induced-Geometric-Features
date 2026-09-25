"""Deterministic crop geometry used by color-mapping training."""

import hashlib
import random
from typing import Any
import numpy as np


def derive_seed(*parts: object) -> int:
    blob = "||".join(str(x) for x in parts).encode("utf-8")
    digest = hashlib.sha256(blob).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


def random_pad_to_min_size(
    img: np.ndarray, min_size: int, rng: random.Random
) -> tuple[np.ndarray, dict[str, int]]:
    h, w = img.shape[:2]
    need_h = max(0, min_size - h)
    need_w = max(0, min_size - w)
    pad_top = rng.randint(0, need_h) if need_h > 0 else 0
    pad_bottom = need_h - pad_top
    pad_left = rng.randint(0, need_w) if need_w > 0 else 0
    pad_right = need_w - pad_left
    if need_h == 0 and need_w == 0:
        return img, {
            "pad_top": 0,
            "pad_bottom": 0,
            "pad_left": 0,
            "pad_right": 0,
        }
    padded = np.pad(
        img,
        ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
        mode="edge",
    )
    return padded, {
        "pad_top": int(pad_top),
        "pad_bottom": int(pad_bottom),
        "pad_left": int(pad_left),
        "pad_right": int(pad_right),
    }


def sample_random_crop(
    img: np.ndarray, crop_size: int, rng: random.Random
) -> tuple[np.ndarray, dict[str, object]]:
    padded, crop_pad = random_pad_to_min_size(img, crop_size, rng)
    h, w = padded.shape[:2]
    max_y = h - crop_size
    max_x = w - crop_size
    top = rng.randint(0, max_y) if max_y > 0 else 0
    left = rng.randint(0, max_x) if max_x > 0 else 0
    crop = np.ascontiguousarray(padded[top : top + crop_size, left : left + crop_size])
    meta = {
        "orig_h": int(img.shape[0]),
        "orig_w": int(img.shape[1]),
        "padded_h": int(padded.shape[0]),
        "padded_w": int(padded.shape[1]),
        "pre_crop_pad": crop_pad,
        "crop_top": int(top),
        "crop_left": int(left),
        "crop_h": int(crop_size),
        "crop_w": int(crop_size),
    }
    return crop, meta


def apply_pre_crop_pad(img: np.ndarray, pad: dict[str, int]) -> np.ndarray:
    pad_top = int(pad.get("pad_top", 0))
    pad_bottom = int(pad.get("pad_bottom", 0))
    pad_left = int(pad.get("pad_left", 0))
    pad_right = int(pad.get("pad_right", 0))
    if pad_top == pad_bottom == pad_left == pad_right == 0:
        return img
    return np.pad(
        img,
        ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
        mode="edge",
    )


def pad_to_cover_crop(
    img: np.ndarray,
    top: int,
    left: int,
    crop_h: int,
    crop_w: int,
) -> np.ndarray:
    """Edge-pad an image so the requested crop box is valid."""
    h, w = img.shape[:2]
    need_h = max(0, top + crop_h - h)
    need_w = max(0, left + crop_w - w)
    if need_h == 0 and need_w == 0:
        return img
    return np.pad(img, ((0, need_h), (0, need_w), (0, 0)), mode="edge")


def replay_crop(
    img: np.ndarray,
    crop_meta: dict[str, Any],
    *,
    pad_if_needed: bool = False,
) -> np.ndarray:
    """Replay the exact crop recorded in extraction metadata.

    For `target` images, the original canvas can differ slightly from `input`.
    When `pad_if_needed=True`, we edge-pad to preserve the same crop box.
    """
    padded = apply_pre_crop_pad(img, crop_meta.get("pre_crop_pad") or {})
    top = int(crop_meta["crop_top"])
    left = int(crop_meta["crop_left"])
    ch = int(crop_meta["crop_h"])
    cw = int(crop_meta["crop_w"])
    if pad_if_needed:
        padded = pad_to_cover_crop(padded, top, left, ch, cw)
    crop = padded[top : top + ch, left : left + cw]
    if crop.shape[0] != ch or crop.shape[1] != cw:
        raise ValueError(
            f"crop replay failed: got {crop.shape[:2]}, expected {(ch, cw)} "
            f"(padded={padded.shape[:2]}, top={top}, left={left})"
        )
    return np.ascontiguousarray(crop)
