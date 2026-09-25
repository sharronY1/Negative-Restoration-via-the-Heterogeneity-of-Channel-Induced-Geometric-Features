"""Manifest-based RGB pairs and portable, task-specific feature caches."""

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .io import image_tensor


def read_manifest(path, task):
    path = Path(path)
    rows = json.loads(path.read_text())
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"Expected a non-empty JSON list: {path}")
    keys = ("input", "target") if task == "restoration" else ("input", "reference", "target")
    out = []
    seen = set()
    for row in rows:
        item = {"name": str(row.get("name", Path(row["input"]).stem))}
        if item["name"] in seen:
            raise ValueError(f"Duplicate sample name: {item['name']}")
        seen.add(item["name"])
        for key in keys:
            p = Path(row[key])
            p = p if p.is_absolute() else path.parent / p
            if not p.is_file():
                raise FileNotFoundError(p)
            item[key] = p
        out.append(item)
    return out


class CachedDataset(Dataset):
    def __init__(self, root, task, split):
        self.root = Path(root)
        index = json.loads((self.root / "index.json").read_text())
        if index["task"] != task or index["layer"] != 13 or index["split"] != split:
            raise ValueError("Cache task, layer or split does not match the requested dataset")
        self.task = task
        self.samples = index["samples"]
        if not self.samples:
            raise ValueError(f"Empty cache: {root}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        with np.load(self.root / self.samples[i], allow_pickle=False) as data:
            batch = {k: image_tensor(data[k]) for k in ("input", "target")}
            if self.task == "restoration":
                batch["features"] = torch.from_numpy(data["features"].astype(np.float32))
            else:
                batch["reference"] = image_tensor(data["reference"])
                for k in ("input_features", "reference_features"):
                    batch[k] = torch.from_numpy(data[k].astype(np.float32))
            batch["original_hw"] = torch.from_numpy(data["original_hw"].astype(np.int64))
        return batch


def forward_batch(model, batch, task):
    if task == "restoration":
        return model(batch["input"], batch["features"])
    return model(batch["input"], batch["reference"], batch["input_features"], batch["reference_features"])
