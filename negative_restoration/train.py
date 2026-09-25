"""Train the fixed restoration or color-mapping network from prepared caches."""

import argparse
import json
import math
from pathlib import Path
import random

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
import yaml

from .data import CachedDataset, forward_batch
from .io import build_model, load_checkpoint, load_config, save_image, select_device
from .metrics import fft_l1, psnr, ssim


def loss_fn(pred, target, task, cfg):
    return F.l1_loss(pred, target) + cfg["fft_weight"] * fft_l1(pred, target)


def learning_rate(step, cfg):
    lr, minimum, warmup = cfg["lr"], cfg["min_lr"], cfg["warmup_iters"]
    if step < warmup:
        return minimum + (lr - minimum) * step / max(warmup, 1)
    t = (step - warmup) / max(cfg["total_iters"] - warmup, 1)
    return minimum + 0.5 * (lr - minimum) * (1 + math.cos(math.pi * t))


def to_device(batch, device):
    return {k: v.to(device) for k, v in batch.items()}


def training_batches(dataset, batch_size, seed, start_step, num_workers):
    # Epoch permutations are reproducible from the completed iteration count,
    # including after a restart; cached samples have no online random transforms.
    per_epoch = len(dataset) // batch_size
    if per_epoch < 1:
        raise ValueError("Training cache has fewer samples than batch_size")
    epoch, offset = divmod(start_step, per_epoch)
    while True:
        generator = torch.Generator().manual_seed(seed + epoch)
        order = torch.randperm(len(dataset), generator=generator).tolist()
        batches = [order[i * batch_size:(i + 1) * batch_size] for i in range(offset, per_epoch)]
        yield from DataLoader(dataset, batch_sampler=batches, num_workers=num_workers)
        epoch, offset = epoch + 1, 0


@torch.no_grad()
def validate(model, loader, device, cfg, output_dir):
    model.eval()
    sums = dict(loss=0.0, psnr=0.0, ssim=0.0)
    if cfg["task"] == "restoration":
        sums.update(psnr_b=0.0, ssim_b=0.0)
    for i, batch in enumerate(loader):
        batch = to_device(batch, device)
        pred = forward_batch(model, batch, cfg["task"])
        h, w = batch["original_hw"][0].tolist()
        pred, target = pred[:, :, :h, :w], batch["target"][:, :, :h, :w]
        sums["loss"] += float(loss_fn(pred, target, cfg["task"], cfg["train"]))
        sums["psnr"] += float(psnr(pred, target))
        sums["ssim"] += float(ssim(pred, target))
        if cfg["task"] == "restoration":
            sums["psnr_b"] += float(psnr(pred[:, 2:3], target[:, 2:3]))
            sums["ssim_b"] += float(ssim(pred[:, 2:3], target[:, 2:3]))
        if i < cfg["train"]["max_val_images"]:
            save_image(pred, output_dir / f"{i:04d}.png")
    model.train()
    return {k: v / len(loader) for k, v in sums.items()}


def save_training(path, model, optimizer, completed, cfg):
    state = {"model": model.state_dict(), "cfg": cfg, "steps_completed": completed,
             "torch_rng": torch.get_rng_state()}
    if torch.cuda.is_available():
        state["cuda_rng"] = torch.cuda.get_rng_state_all()
    if cfg["task"] == "restoration":
        state.update(iter=completed - 1, opt=optimizer.state_dict())
    else:
        state.update(step=completed - 1, optimizer=optimizer.state_dict())
    temporary = path.with_suffix(".tmp")
    torch.save(state, temporary)
    temporary.replace(path)


def resume_training(path, model, optimizer, task):
    checkpoint = load_checkpoint(model, path)
    optimizer.load_state_dict(checkpoint["opt" if task == "restoration" else "optimizer"])
    if "torch_rng" in checkpoint:
        torch.set_rng_state(checkpoint["torch_rng"])
    if torch.cuda.is_available() and "cuda_rng" in checkpoint:
        torch.cuda.set_rng_state_all(checkpoint["cuda_rng"])
    return int(checkpoint.get("steps_completed", checkpoint.get("iter", checkpoint.get("step", -1)) + 1))


def train(cfg, device, resume=None, max_steps=None):
    device = select_device(device)
    torch.manual_seed(cfg["seed"])
    random.seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    task, train_cfg = cfg["task"], cfg["train"]
    root = Path(cfg["data"]["cache_dir"])
    dataset = CachedDataset(root / "train", task, "train")
    val_set = CachedDataset(root / "val", task, "val")
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=cfg["data"]["num_workers"])
    output = Path(train_cfg["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    if (output / "last.pt").exists() and resume is None:
        raise FileExistsError(f"{output / 'last.pt'} exists; use --resume or a new output_dir")
    model = build_model(task).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg["lr"], betas=(0.9, 0.999), weight_decay=0.0)
    start = resume_training(resume, model, optimizer, task) if resume else 0
    total = train_cfg["total_iters"]
    end = min(total, start + max_steps) if max_steps is not None else total
    if start >= end:
        raise ValueError(f"No training steps remain: completed={start}, requested end={end}")
    (output / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    batches = training_batches(dataset, train_cfg["batch_size"], cfg["seed"], start, cfg["data"]["num_workers"])
    print(f"{task}: {len(dataset)} train / {len(val_set)} val samples; steps {start}..{end - 1}", flush=True)
    model.train()
    for step in range(start, end):
        batch = to_device(next(batches), device)
        lr = learning_rate(step, train_cfg)
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.zero_grad(set_to_none=True)
        pred = forward_batch(model, batch, task)
        loss = loss_fn(pred, batch["target"], task, train_cfg)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite loss at step {step}")
        loss.backward()
        optimizer.step()
        completed = step + 1
        if step % train_cfg["log_every"] == 0 or completed == end:
            record = dict(step=step, loss=float(loss.detach()), lr=lr)
            print(json.dumps(record), flush=True)
            with (output / "train.jsonl").open("a") as f:
                f.write(json.dumps(record) + "\n")
        if (step > 0 and step % train_cfg["save_every"] == 0) or completed == end:
            save_training(output / "last.pt", model, optimizer, completed, cfg)
        if (step > 0 and step % train_cfg["val_every"] == 0) or completed == end:
            metrics = validate(model, val_loader, device, cfg, output / f"val_{step}")
            print(json.dumps(dict(step=step, validation=metrics)), flush=True)
            with (output / "val.jsonl").open("a") as f:
                f.write(json.dumps(dict(step=step, **metrics)) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--max-steps", type=int, help="Run at most this many additional steps (smoke test)")
    args = parser.parse_args()
    if args.max_steps is not None and args.max_steps < 1:
        parser.error("--max-steps must be positive")
    train(load_config(args.config), args.device, args.resume, args.max_steps)


if __name__ == "__main__":
    main()
