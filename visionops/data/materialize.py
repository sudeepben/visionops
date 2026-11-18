"""Materialise the frozen split lists as a plain YOLO folder layout.

Some tools (e.g. the 3LC integration) expect `<split>/images` + `<split>/labels`
directories instead of `.txt` file lists. This writes hard links (same drive) or
copies into data/pcb_dirs/ and a matching dataset_dirs.yaml. Split membership is
identical to data/pcb/splits/*.txt.

    python -m visionops.data.materialize
"""
from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

import yaml

from ..boxes import image_to_label_path
from ..config import DATA_DIR, DATASET_YAML
from .prepare import load_names, read_split

OUT_DIR = DATA_DIR / "pcb_dirs"


def _link_or_copy(src: Path, dst: Path) -> None:
    if dst.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def materialize(out_dir: Path = OUT_DIR, splits=("train", "val", "test")) -> Path:
    names = load_names(DATASET_YAML)
    counts = {}
    for split in splits:
        imgs = read_split(split)
        for img in imgs:
            _link_or_copy(img, out_dir / split / "images" / img.name)
            lbl = image_to_label_path(img)
            dst = out_dir / split / "labels" / (img.stem + ".txt")
            if lbl.exists():
                _link_or_copy(lbl, dst)
            elif not dst.exists():
                dst.parent.mkdir(parents=True, exist_ok=True)
                dst.write_text("", encoding="utf-8")  # background image
        n_on_disk = len(list((out_dir / split / "images").glob("*")))
        if n_on_disk != len(imgs):
            raise RuntimeError(f"{split}: {n_on_disk} files in {out_dir / split / 'images'} but split list has "
                               f"{len(imgs)} - delete {out_dir} and re-run (stale files or duplicate names)")
        counts[split] = len(imgs)
    cfg = {"path": str(out_dir.resolve()), **{s: f"{s}/images" for s in splits},
           "names": {i: n for i, n in enumerate(names)}}
    y = out_dir / "dataset_dirs.yaml"
    y.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    print(f"{counts} -> {y}")
    return y


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=OUT_DIR)
    a = ap.parse_args(argv)
    materialize(a.out)


if __name__ == "__main__":
    main()
