"""Validate the prepared YOLO dataset and write reports/data_validation.json.

Checks: class names, per-split image counts, unreadable images, missing/empty
label files, malformed lines (wrong field count, polygons, bad class ids,
degenerate or out-of-range boxes), per-class instance counts, exact duplicate
images (MD5) within/across splits, and source-image leakage across splits.
Exit code 1 on fatal problems.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image

from ..boxes import image_to_label_path, read_yolo_labels
from ..config import DATASET_YAML, REPORTS_DIR, SPLITS_DIR
from .prepare import load_names, read_split, source_group

FATAL_ISSUES = {"invalid_class_id", "wrong_field_count", "non_numeric"}


def md5(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        while b := f.read(chunk):
            h.update(b)
    return h.hexdigest()


def validate(dataset_yaml: Path = DATASET_YAML, splits_dir: Path = SPLITS_DIR) -> dict:
    names = load_names(dataset_yaml)
    nc = len(names)
    report: dict = {"names": names, "num_classes": nc, "splits": {}, "fatal": [], "warnings": []}
    hashes: dict[str, list[tuple[str, str]]] = defaultdict(list)
    groups: dict[str, set[str]] = defaultdict(set)

    for split in ("train", "val", "test"):
        try:
            imgs = read_split(split, splits_dir)
        except FileNotFoundError:
            report["fatal"].append(f"missing split list: {split}")
            continue
        s = {
            "images": len(imgs), "missing_label_files": 0, "empty_label_files": 0,
            "unreadable_images": 0, "instances": 0, "class_instances": Counter(),
            "issues": Counter(), "image_sizes": Counter(), "issue_examples": [],
        }
        for img in imgs:
            try:
                with Image.open(img) as im:
                    s["image_sizes"][f"{im.width}x{im.height}"] += 1
            except Exception:
                s["unreadable_images"] += 1
                continue
            lp = image_to_label_path(img)
            if not lp.exists():
                s["missing_label_files"] += 1
            labels, issues = read_yolo_labels(lp, nc)
            if lp.exists() and len(labels) == 0 and not issues:
                s["empty_label_files"] += 1
            for iss in issues:
                s["issues"][iss.issue] += 1
                if len(s["issue_examples"]) < 20:
                    s["issue_examples"].append({"file": str(lp), "line": iss.line_no, "issue": iss.issue})
            for c in labels[:, 0].astype(int):
                s["class_instances"][names[c]] += 1
            s["instances"] += len(labels)
            hashes[md5(img)].append((split, str(img)))
            groups[source_group(img)].add(split)
        if s["images"] == 0:
            report["fatal"].append(f"split '{split}' has no images")
        for k in FATAL_ISSUES & set(s["issues"]):
            report["fatal"].append(f"{split}: {s['issues'][k]} label lines with {k}")
        if s["unreadable_images"]:
            report["warnings"].append(f"{split}: {s['unreadable_images']} unreadable images")
        missing_cls = [n for n in names if s["class_instances"][n] == 0]
        if missing_cls and s["images"]:
            report["warnings"].append(f"{split}: no instances of {missing_cls}")
        s["class_instances"] = dict(s["class_instances"])
        s["issues"] = dict(s["issues"])
        s["image_sizes"] = dict(s["image_sizes"].most_common(5))
        report["splits"][split] = s

    dups = [v for v in hashes.values() if len(v) > 1]
    cross = [v for v in dups if len({sp for sp, _ in v}) > 1]
    leak = {g: sorted(sp) for g, sp in groups.items() if len(sp) > 1}
    report["duplicates"] = {"exact_duplicate_groups": len(dups), "cross_split_exact_duplicates": len(cross),
                            "examples": cross[:10]}
    report["leakage"] = {"source_groups_in_multiple_splits": len(leak), "examples": dict(list(leak.items())[:10])}
    if cross:
        report["warnings"].append(f"{len(cross)} exact duplicate images appear in more than one split")
    if leak:
        report["warnings"].append(
            f"{len(leak)} source images (by Roboflow filename stem) appear in more than one split - "
            "possible test leakage; inspect reports/data_validation.json before trusting test metrics"
        )
    return report


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", type=Path, default=DATASET_YAML)
    ap.add_argument("--out", type=Path, default=REPORTS_DIR / "data_validation.json")
    a = ap.parse_args(argv)
    rep = validate(a.dataset)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(rep, indent=2), encoding="utf-8")
    print(f"Classes ({rep['num_classes']}): {rep['names']}")
    for sp, s in rep["splits"].items():
        print(f"  {sp:5s}: {s['images']} images, {s['instances']} boxes, issues={s['issues']}")
    for w in rep["warnings"]:
        print("WARNING:", w)
    for f in rep["fatal"]:
        print("FATAL:", f)
    print(f"Report: {a.out}")
    sys.exit(1 if rep["fatal"] else 0)


if __name__ == "__main__":
    main()
