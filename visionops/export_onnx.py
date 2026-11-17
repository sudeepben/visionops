"""Export a trained model to ONNX and check prediction consistency against PyTorch.

    python -m visionops.export_onnx --weights runs/baseline/weights/best.pt --n 50
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from .boxes import box_iou
from .config import REPORTS_DIR, ROOT, load_config
from .data.prepare import read_split


def compare_predictions(a, b, iou_thr: float = 0.5) -> dict:
    """a, b: DataFrames with image, cls, conf, x1..y2 from two backends."""
    matched, ious, dconf, n_a, n_b, same_count = 0, [], [], len(a), len(b), 0
    images = sorted(set(a["image"]) | set(b["image"]))
    ga, gb = dict(tuple(a.groupby("image"))), dict(tuple(b.groupby("image")))
    for img in images:
        pa, pb = ga.get(img, a.iloc[0:0]), gb.get(img, b.iloc[0:0])
        same_count += len(pa) == len(pb)
        if len(pa) == 0 or len(pb) == 0:
            continue
        iou = box_iou(pa[["x1", "y1", "x2", "y2"]].to_numpy(), pb[["x1", "y1", "x2", "y2"]].to_numpy())
        iou[pa["cls"].to_numpy()[:, None] != pb["cls"].to_numpy()[None, :]] = 0
        used = set()
        for i in np.argsort(-pa["conf"].to_numpy()):
            row = iou[i].copy()
            row[list(used)] = 0
            j = int(np.argmax(row))
            if row[j] >= iou_thr:
                used.add(j)
                matched += 1
                ious.append(iou[i, j])
                dconf.append(abs(pa["conf"].iloc[i] - pb["conf"].iloc[j]))
    return {
        "images": len(images), "boxes_pytorch": n_a, "boxes_onnx": n_b, "matched_boxes": matched,
        "match_rate": matched / max(n_a, n_b, 1), "mean_iou_matched": float(np.mean(ious)) if ious else None,
        "max_abs_conf_diff": float(np.max(dconf)) if dconf else None,
        "images_with_same_box_count": same_count,
    }


def main(argv=None) -> None:
    from .inference import Detector

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--config", default="configs/baseline.yaml")
    ap.add_argument("--n", type=int, default=50, help="number of test images to compare")
    ap.add_argument("--conf", type=float, default=0.25)
    a = ap.parse_args(argv)
    cfg = load_config(ROOT / a.config)

    from ultralytics import YOLO

    onnx_path = Path(YOLO(a.weights).export(format="onnx", imgsz=cfg["imgsz"], dynamic=False, simplify=False))
    images = read_split("test")[: a.n]
    pt, ox = Detector(a.weights, cfg["imgsz"]), Detector(onnx_path, cfg["imgsz"])
    pt.predict_paths(images[:2], conf=a.conf)  # warm-up
    ox.predict_paths(images[:2], conf=a.conf)
    t0 = time.perf_counter(); p = pt.predict_paths(images, conf=a.conf); t_pt = time.perf_counter() - t0
    t0 = time.perf_counter(); o = ox.predict_paths(images, conf=a.conf); t_ox = time.perf_counter() - t0
    rep = compare_predictions(p, o)
    rep.update({"onnx_path": str(onnx_path), "conf": a.conf, "imgsz": cfg["imgsz"],
                "pytorch_s_per_image": t_pt / max(len(images), 1), "onnx_s_per_image": t_ox / max(len(images), 1),
                "timing_note": "wall-clock on this machine incl. image loading; PyTorch may run on GPU, ONNX Runtime on CPU"})
    out = REPORTS_DIR / "onnx_consistency.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rep, indent=2), encoding="utf-8")
    print(json.dumps(rep, indent=2))


if __name__ == "__main__":
    main()
