"""Inference from raw RGB images using frozen DA3 L13 features."""

import argparse
from pathlib import Path

import torch

from .features import DA3Features
from .io import (IMAGE_EXTENSIONS, build_model, image_tensor, load_checkpoint,
                 load_config, load_rgb, save_image, select_device)


@torch.inference_mode()
def predict(model, extractor, rgb, device, reference=None):
    h, w = rgb.shape[:2]
    if reference is None:
        canvas, feat = extractor.restoration(rgb)
        pred = model(image_tensor(canvas)[None].to(device),
                     torch.from_numpy(feat).float()[None].to(device))
    else:
        canvas, feat = extractor.color(rgb)
        ref, ref_feat = extractor.color(reference)
        pred = model(image_tensor(canvas)[None].to(device),
                     image_tensor(ref)[None].to(device),
                     torch.from_numpy(feat).float()[None].to(device),
                     torch.from_numpy(ref_feat).float()[None].to(device))
    return pred[:, :, :h, :w]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--output", type=Path, required=True, help="Output PNG for a file; output directory for a directory")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    cfg = load_config(args.config)
    if cfg["task"] == "color_mapping":
        if not args.reference or not args.reference.is_file() or not args.input.is_file():
            parser.error("Color mapping requires one --input image and one --reference image")
    elif args.reference is not None:
        parser.error("Restoration does not take a reference image")
    if not args.input.exists():
        raise FileNotFoundError(args.input)
    paths = sorted(p for p in args.input.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS) if args.input.is_dir() else [args.input]
    if not paths:
        raise ValueError("No input images found")
    if args.input.is_dir() and len({p.stem for p in paths}) != len(paths):
        raise ValueError("Input directory contains duplicate image stems; rename them before inference")
    device = select_device(args.device)
    model = build_model(cfg["task"])
    load_checkpoint(model, args.checkpoint)
    model.to(device).eval()
    extractor = DA3Features(cfg["da3"]["model_id"], str(device))
    reference = load_rgb(args.reference) if args.reference else None
    for path in paths:
        output = args.output / f"{path.stem}.png" if args.input.is_dir() else args.output
        if output.resolve() == path.resolve() or (args.reference and output.resolve() == args.reference.resolve()):
            raise ValueError("Output must not overwrite an input image")
        save_image(predict(model, extractor, load_rgb(path), device, reference), output)
        print(output, flush=True)


if __name__ == "__main__":
    main()
