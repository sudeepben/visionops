"""Rank TRAINING images for human review.

Signals (combined with weights from visionops.analysis.DEFAULT_WEIGHTS):
  * high-confidence disagreements with the labels (possible missing label / wrong class / loose box)
  * missed objects and other errors on the model's own training data
  * duplicate annotations and per-class geometric outliers in the labels
  * low-confidence correct detections (small weight - never a label verdict on its own)

Caveat: predictions are in-sample (the model saw these images in training), so
errors are under-counted. Disagreements that survive training are still a useful
review signal, but the output is a review PRIORITY, not proof of bad labels.

    python -m visionops.review --weights runs/baseline/weights/best.pt
"""
from __future__ import annotations

import argparse
from pathlib import Path

from .analysis import duplicate_annotations, geometry_outliers, load_ground_truth, rank_for_review, run_matching
from .config import DATASET_YAML, REPORTS_DIR, ROOT, load_config
from .data.prepare import load_names, read_split
from .evaluate import match_cfg


def build_queue(weights: str, cfg: dict, split: str = "train", out_dir: Path = REPORTS_DIR / "review"):
    from .inference import Detector

    ev = cfg.get("eval", {})
    names = load_names(DATASET_YAML)
    images = read_split(split)
    gt = load_ground_truth(images, len(names))
    preds = Detector(weights, imgsz=cfg["imgsz"]).predict_paths(images, conf=ev.get("pred_floor", 0.01))
    events, per_image = run_matching(preds, gt, match_cfg(ev))
    outliers = geometry_outliers(gt)
    dups = duplicate_annotations(gt)
    queue = rank_for_review(per_image, events, outliers, dups)

    out_dir.mkdir(parents=True, exist_ok=True)
    queue.to_csv(out_dir / "review_queue.csv", index=False)
    events.to_csv(out_dir / "train_events.csv", index=False)
    gt.to_csv(out_dir / "train_ground_truth.csv", index=False)
    outliers.to_csv(out_dir / "geometry_outliers.csv", index=False)
    dups.to_csv(out_dir / "duplicate_annotations.csv", index=False)
    print(f"{len(queue)} of {len(images)} training images have review signals; "
          f"{int(queue['possible_annotation_issue'].sum())} with a possible annotation issue.")
    print(f"Queue: {out_dir / 'review_queue.csv'}")
    return queue


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--config", default="configs/baseline.yaml")
    a = ap.parse_args(argv)
    build_queue(a.weights, load_config(ROOT / a.config))


if __name__ == "__main__":
    main()
