"""Prepare DA3 caches from raw RGB manifests (train or full-image validation)."""

import argparse
import itertools
import json
from pathlib import Path
import random

import numpy as np
from tqdm import tqdm

from .crops import derive_seed, sample_random_crop, replay_crop
from .data import read_manifest
from .features import DA3Features, da3_layer, pad_image
from .io import load_config, load_rgb, select_device


def restoration_variants(inp, target, training):
    if inp.shape != target.shape:
        raise ValueError("Restoration input and target must have identical RGB dimensions")
    if not training:
        yield inp, target
        return
    h, w = inp.shape[:2]
    if min(h, w) < 504:
        raise ValueError("Restoration training images must be at least 504 x 504")
    y, x = (h - 504) // 2, (w - 504) // 2
    inp, target = inp[y:y+504, x:x+504], target[y:y+504, x:x+504]
    for hf, vf, k in itertools.product(range(2), range(2), range(4)):
        def transform(a):
            a = a[:, ::-1] if hf else a
            a = a[::-1] if vf else a
            return np.rot90(a, k).copy()
        yield transform(inp), transform(target)


def color_variants(inp, ref, target, name, seed, training):
    if not training:
        if target.shape != inp.shape:
            raise ValueError("Full-image color input and target must have identical dimensions")
        yield inp, ref, target
        return
    # Preserve the source experiment's independent crop seeds and joint rotations.
    inp, geometry = sample_random_crop(inp, 512, random.Random(derive_seed(seed, "training", name, 0, "input")))
    ref, _ = sample_random_crop(ref, 512, random.Random(derive_seed(seed, "training", name, 0, "reference")))
    target = replay_crop(target, geometry, pad_if_needed=True)
    for k in range(4):
        yield tuple(np.rot90(a, k).copy() for a in (inp, ref, target))


def prepare(cfg, split, device, limit=None):
    task = cfg["task"]
    training = split == "train"
    rows = read_manifest(cfg["data"][f"{split}_manifest"], task)
    if limit is not None:
        if limit < 1:
            raise ValueError("--limit must be positive")
        rows = rows[:limit]
    out = Path(cfg["data"]["cache_dir"]) / split
    # Refuse to silently mix a previous cache with different data or geometry.
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Cache is not empty: {out}. Use a new cache_dir or remove this generated split first.")
    extractor = DA3Features(cfg["da3"]["model_id"], str(select_device(device)))
    out.mkdir(parents=True, exist_ok=True)
    samples = []
    for i, row in enumerate(tqdm(rows, desc=f"Prepare {task}/{split}")):
        inp, target = load_rgb(row["input"]), load_rgb(row["target"])
        if task == "restoration":
            variants = restoration_variants(inp, target, training)
            for j, (image, gt) in enumerate(variants):
                canvas, feat = extractor.restoration(image)
                data = dict(input=canvas, target=pad_image(gt, 28), features=feat,
                            original_hw=np.array(image.shape[:2]))
                filename = f"{i:06d}_{j:02d}.npz"
                np.savez(out / filename, **data)
                samples.append(filename)
        else:
            ref = load_rgb(row["reference"])
            for j, (image, reference, gt) in enumerate(color_variants(inp, ref, target, row["name"], cfg["seed"], training)):
                canvas, feat = extractor.color(image, training=training)
                ref_canvas, ref_feat = extractor.color(reference, training=training)
                data = dict(input=canvas, reference=ref_canvas, target=gt if training else pad_image(gt, 14),
                            input_features=feat, reference_features=ref_feat,
                            original_hw=np.array(image.shape[:2]))
                filename = f"{i:06d}_{j:02d}.npz"
                np.savez(out / filename, **data)
                samples.append(filename)
    index = {"task": task, "layer": da3_layer(task), "split": split, "model_id": cfg["da3"]["model_id"],
             "source_manifest": str(cfg["data"][f"{split}_manifest"]), "samples": samples,
             "source_count": len(rows), "seed": cfg["seed"]}
    (out / "index.json").write_text(json.dumps(index, indent=2) + "\n")
    print(f"Prepared {len(samples)} samples in {out}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val"), required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit", type=int, help="Limit source images for a smoke test only")
    args = parser.parse_args()
    prepare(load_config(args.config), args.split, args.device, args.limit)


if __name__ == "__main__":
    main()
