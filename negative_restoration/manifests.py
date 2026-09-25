"""Create portable training manifests from the BlueNeg directory layout."""

import argparse
import json
import os
from pathlib import Path

from .io import IMAGE_EXTENSIONS


def stem_key(path):
    name = path.stem
    for suffix in (".preview_inpainted", ".preview", ".pseudogt", "_preview", "_pseudogt"):
        if name.endswith(suffix):
            return name[:-len(suffix)]
    return name


def image_index(directory):
    if not directory.is_dir():
        raise FileNotFoundError(directory)
    result = {}
    for path in sorted(directory.iterdir()):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            key = stem_key(path)
            if key in result:
                raise ValueError(f"Duplicate stem {key} in {directory}")
            result[key] = path
    if not result:
        raise ValueError(f"No images in {directory}")
    return result


def restoration_rows(root, training, excluded=None):
    rows = []
    buckets = ("training", "gradually_deg_training") if training else ("testing",)
    for bucket in buckets:
        folder = "negative_preview" if training else "negative"
        inputs = image_index(root / bucket / folder)
        targets = image_index(root / bucket / "pseudo_gt")
        for name, inp in inputs.items():
            if name not in targets:
                if excluded is not None:
                    excluded.append({"task": "restoration", "name": f"{bucket}/{name}",
                                     "reason": "missing paired target"})
                print(f"Excluded unpaired restoration input {bucket}/{name}")
                continue
            rows.append(dict(name=f"{bucket}/{name}", input=inp, target=targets[name]))
    if not rows:
        raise ValueError("No complete restoration pairs found")
    return rows


def color_rows(root, training, excluded=None):
    directory = root / ("training" if training else "testing") / "color_transfer"
    mapping = json.loads((directory / "reference_map.json").read_text())
    rows = []
    for row in mapping:
        name = row["stem"]
        if name.startswith(("ps_", "PST")):
            continue
        result = {"name": name}
        missing = []
        for key in ("input", "reference", "target"):
            # Rebase the distributed mapping onto the user's dataset root and
            # use the non-PS reference/target directories of this experiment.
            filename = Path(row.get(key) or f"{name}.png").with_suffix(".png").name
            path = directory / key / filename
            if not path.is_file():
                missing.append(str(path))
            result[key] = path
        if missing:
            entry = {"task": "color_mapping", "name": name, "split": "train" if training else "val", "missing": missing}
            if excluded is not None:
                excluded.append(entry)
            print(f"Excluded incomplete color pair {name}: {', '.join(missing)}")
            continue
        rows.append(result)
    if not rows:
        raise ValueError(f"No color pairs in {directory}")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("data"))
    args = parser.parse_args()
    root = args.data_root.resolve()
    manifests = {}
    excluded = []
    for task, loader in (("restoration", restoration_rows), ("color_mapping", color_rows)):
        for split in ("train", "val"):
            rows = loader(root, split == "train", excluded=excluded)
            manifests[f"{task}_{split}.json"] = [
                {key: os.path.relpath(value, args.output_dir.resolve()) if isinstance(value, Path) else value
                 for key, value in row.items()} for row in rows
            ]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in manifests.items():
        path = args.output_dir / name
        path.write_text(json.dumps(rows, indent=2) + "\n")
        print(f"{path}: {len(rows)} samples")
    (args.output_dir / "manifest_report.json").write_text(json.dumps({"excluded_pairs": excluded}, indent=2) + "\n")


if __name__ == "__main__":
    main()
