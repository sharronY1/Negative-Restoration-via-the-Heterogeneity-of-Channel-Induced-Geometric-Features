"""Portable configuration, image I/O, and strict checkpoint loading."""

from pathlib import Path

import numpy as np
from PIL import Image
import torch
import yaml

from .restoration import RestorationNet
from .color_mapping import ColorMappingNet

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}


def load_config(path):
    with Path(path).open() as f:
        cfg = yaml.safe_load(f)
    if cfg.get("task") not in ("restoration", "color_mapping"):
        raise ValueError("task must be restoration or color_mapping")
    return cfg


def build_model(task):
    if task == "restoration":
        return RestorationNet()
    if task == "color_mapping":
        return ColorMappingNet()
    raise ValueError(f"Unknown task: {task}")


def load_checkpoint(model, path):
    # Historical optimizer learning rates were NumPy float64 scalars. Allow
    # those concrete types without enabling arbitrary pickle execution.
    with torch.serialization.safe_globals([np.core.multiarray.scalar, np.dtype,
                                            type(np.dtype(np.float64))]):
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    state = checkpoint.get("model", checkpoint)
    model.load_state_dict(state, strict=True)
    return checkpoint


def load_rgb(path):
    with Image.open(path) as image:
        return np.array(image.convert("RGB"))


def image_tensor(image):
    return torch.from_numpy(image.transpose(2, 0, 1).copy()).float() / 255.0


def save_image(tensor, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if tensor.ndim == 4:
        tensor = tensor[0]
    rgb = (tensor.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
    Image.fromarray(rgb).save(path)


def select_device(name):
    if name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; select --device cpu explicitly or use a GPU node")
    return torch.device(name)
