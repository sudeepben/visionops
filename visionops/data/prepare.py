"""Normalise a Roboflow/YOLO export into fixed split lists + dataset.yaml.

- Images and labels stay where they were extracted (nothing is copied).
- Split membership is written to data/pcb/splits/{train,val,test}.txt.
- If the export has no test (or val) split, one is carved out of train with a
  fixed seed, GROUPED by source image so that Roboflow augmentations of the same
  photo never leak across splits.
- The test split is then frozen: experiments must never train on it.
"""
from __future__ import annotations

import argparse
import json
import random
import re
from collections import defaultdict
from pathlib import Path

import yaml

from ..boxes import list_images
from ..config import DATASET_DIR, DATASET_YAML, SPLITS_DIR
from .download import RAW_DIR

SPLIT_ALIASES = {"train": ["train"], "val": ["valid", "val", "validation"], "test": ["test"]}


def source_group(path: str | Path) -> str:
    """Group key: Roboflow names augmented copies '<stem>_<ext>.rf.<hash>.<ext>'."""
    name = Path(path).name
    m = re.match(r"(.+?)\.rf\.[0-9a-fA-F]+", name)
    return (m.group(1) if m else Path(name).stem).lower()


def find_raw_root(raw_dir: Path) -> Path:
    """Locate the directory containing data.yaml (exports sometimes nest one level)."""
    cands = sorted(raw_dir.rglob("data.yaml"))
    if not cands:
        raise FileNotFoundError(f"No data.yaml under {raw_dir}. Run visionops.data.download first.")
    return cands[0].parent


def read_names(raw_root: Path) -> list[str]:
    meta = yaml.safe_load((raw_root / "data.yaml").read_text(encoding="utf-8"))
    names = meta.get("names")
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names, key=int)]
    if not names:
        raise ValueError("data.yaml has no 'names'")
    if "nc" in meta and int(meta["nc"]) != len(names):
        raise ValueError(f"data.yaml nc={meta['nc']} but {len(names)} names")
    return [str(n) for n in names]


def grouped_split(images: list[Path], frac: float, seed: int) -> tuple[list[Path], list[Path]]:
    groups: dict[str, list[Path]] = defaultdict(list)
    for p in images:
        groups[source_group(p)].append(p)
    keys = sorted(groups)
    random.Random(seed).shuffle(keys)
    n_hold = max(1, round(len(keys) * frac))
    hold = {k for k in keys[:n_hold]}
    keep = [p for k in keys if k not in hold for p in groups[k]]
    held = [p for k in keys if k in hold for p in groups[k]]
    return sorted(keep), sorted(held)


def prepare(raw_dir: Path = RAW_DIR, out_dir: Path = DATASET_DIR, seed: int = 42,
            val_frac: float = 0.15, test_frac: float = 0.15, merge_val_into_test: bool = False) -> dict:
    raw_root = find_raw_root(raw_dir)
    names = read_names(raw_root)
    splits: dict[str, list[Path]] = {}
    for split, aliases in SPLIT_ALIASES.items():
        for a in aliases:
            imgs = list_images(raw_root / a / "images")
            if imgs:
                splits[split] = imgs
                break
    if "train" not in splits:
        raise FileNotFoundError(f"No train/images under {raw_root}")

    method = {"train": "provided"}
    if merge_val_into_test and "val" in splits and "test" in splits:
        # The provided test split can be too small / miss classes. Use provided val+test (un-augmented
        # originals) as the held-out test set and carve a grouped validation set from train instead.
        splits["test"] = sorted(splits["test"] + splits.pop("val"))
        method["test"] = "provided test + provided valid (merged; un-augmented originals)"
    if "test" not in splits:
        splits["train"], splits["test"] = grouped_split(splits["train"], test_frac, seed)
        method["test"] = f"carved from train (grouped by source image, frac={test_frac}, seed={seed})"
    elif "test" not in method:
        method["test"] = "provided"
    if "val" not in splits:
        splits["train"], splits["val"] = grouped_split(splits["train"], val_frac, seed + 1)
        method["val"] = f"carved from train (grouped by source image, frac={val_frac}, seed={seed + 1})"
    else:
        method["val"] = "provided"

    splits_dir = out_dir / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)
    for split, imgs in splits.items():
        (splits_dir / f"{split}.txt").write_text(
            "\n".join(str(p.resolve()) for p in imgs) + "\n", encoding="utf-8"
        )
    write_dataset_yaml(out_dir / "dataset.yaml", out_dir, names, splits_dir / "train.txt")
    manifest = {
        "raw_root": str(raw_root),
        "names": names,
        "counts": {k: len(v) for k, v in splits.items()},
        "method": method,
        "seed": seed,
    }
    (out_dir / "split_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def write_dataset_yaml(path: Path, root: Path, names: list[str], train_list: Path) -> Path:
    cfg = {
        "path": str(root.resolve()),
        "train": str(Path(train_list).resolve()),
        "val": str((root / "splits" / "val.txt").resolve()),
        "test": str((root / "splits" / "test.txt").resolve()),
        "names": {i: n for i, n in enumerate(names)},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return path


def read_split(split: str, splits_dir: Path = SPLITS_DIR) -> list[Path]:
    f = splits_dir / f"{split}.txt"
    return [Path(l) for l in f.read_text(encoding="utf-8").splitlines() if l.strip()]


def load_names(dataset_yaml: Path = DATASET_YAML) -> list[str]:
    meta = yaml.safe_load(Path(dataset_yaml).read_text(encoding="utf-8"))
    names = meta["names"]
    if isinstance(names, dict):
        return [names[k] for k in sorted(names, key=int)]
    return list(names)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw", type=Path, default=RAW_DIR)
    ap.add_argument("--out", type=Path, default=DATASET_DIR)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--test-frac", type=float, default=0.15)
    ap.add_argument("--merge-val-into-test", action="store_true",
                    help="use provided valid+test as the held-out test set; carve validation from train (grouped)")
    a = ap.parse_args(argv)
    m = prepare(a.raw, a.out, a.seed, a.val_frac, a.test_frac, a.merge_val_into_test)
    print(json.dumps(m, indent=2))
    print(f"Wrote {a.out / 'dataset.yaml'}\nNext: python -m visionops.data.validate")


if __name__ == "__main__":
    main()
