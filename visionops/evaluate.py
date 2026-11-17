"""Evaluate a model on the held-out TEST split.

Produces reports/<name>/:
  metrics.json      mAP@50, mAP@50-95, precision, recall (Ultralytics validator, conf=0.001)
                    + operating-point P/R/F1 from our matcher (conf=0.25, IoU=0.5)
  per_class.csv     class-level metrics from both views + instance counts
  ground_truth.csv, predictions.csv, events.csv, per_image.csv  (sample-level analysis)
  plots/            Ultralytics confusion matrix and PR/F1 curves

    python -m visionops.evaluate --weights runs/baseline/weights/best.pt --name baseline
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import pandas as pd

from .analysis import gt_boxes, load_ground_truth, operating_point_metrics, run_matching
from .config import DATASET_YAML, REPORTS_DIR, ROOT, RUNS_DIR, load_config
from .data.prepare import load_names, read_split
from .matching import MatchConfig
from .tracking import log_metrics, track


def match_cfg(ev: dict) -> MatchConfig:
    return MatchConfig(iou_thr=ev.get("iou_match", 0.5), conf_thr=ev.get("conf_operating", 0.25),
                       low_conf_thr=ev.get("low_conf", 0.5), high_conf_thr=ev.get("high_conf", 0.7))


def ultralytics_metrics(weights: str, data: str, split: str, imgsz: int, conf: float, name: str) -> tuple[dict, pd.DataFrame, Path]:
    from ultralytics import YOLO

    model = YOLO(weights, task="detect")
    m = model.val(data=data, split=split, imgsz=imgsz, conf=conf, batch=8, plots=True,
                  project=str((RUNS_DIR / "eval").resolve()), name=name, exist_ok=True, verbose=False)
    names = m.names if isinstance(m.names, dict) else dict(enumerate(m.names))
    overall = {"map50": float(m.box.map50), "map50_95": float(m.box.map), "precision": float(m.box.mp),
               "recall": float(m.box.mr), "speed_ms_per_image": dict(m.speed)}
    rows = []
    for i, c in enumerate(m.box.ap_class_index):
        p, r, ap50, ap = m.box.class_result(i)
        rows.append({"class": names[int(c)], "precision": float(p), "recall": float(r),
                     "map50": float(ap50), "map50_95": float(ap)})
    return overall, pd.DataFrame(rows), Path(m.save_dir)


def evaluate(weights: str, name: str, cfg: dict, split: str = "test", data: str | None = None,
             skip_val: bool = False) -> dict:
    from .inference import Detector

    ev = cfg.get("eval", {})
    data = str((ROOT / (data or cfg["data"])).resolve())
    names = load_names(DATASET_YAML)
    out = REPORTS_DIR / name
    (out / "plots").mkdir(parents=True, exist_ok=True)

    result = {"name": name, "weights": str(weights), "split": split, "names": names}
    per_class = pd.DataFrame({"class": names})
    if not skip_val:
        overall, uc, save_dir = ultralytics_metrics(weights, data, split, cfg["imgsz"], ev.get("conf_map", 0.001), name)
        result["ultralytics"] = overall
        per_class = per_class.merge(uc, on="class", how="left")
        for f in ("confusion_matrix.png", "confusion_matrix_normalized.png", "PR_curve.png", "BoxPR_curve.png",
                  "F1_curve.png", "BoxF1_curve.png"):
            if (save_dir / f).exists():
                shutil.copy(save_dir / f, out / "plots" / f)

    # sample-level predictions + defensible matching
    images = read_split(split)
    gt = load_ground_truth(images, len(names))
    det = Detector(weights, imgsz=cfg["imgsz"])
    preds = det.predict_paths(images, conf=ev.get("pred_floor", 0.01))
    mc = match_cfg(ev)
    events, per_image = run_matching(preds, gt, mc)
    op = operating_point_metrics(events, names)
    result["operating_point"] = {"conf_thr": mc.conf_thr, "iou_thr": mc.iou_thr, **op["overall"]}
    result["counts"] = {"images": len(images), "gt_boxes": int(len(gt_boxes(gt))),
                        "images_fully_correct": int(per_image["correct"].sum()),
                        "fp_by_subtype": events[events.type == "fp"]["subtype"].value_counts().to_dict(),
                        "fn_by_subtype": events[events.type == "fn"]["subtype"].value_counts().to_dict(),
                        "low_confidence_tp": int(per_image["low_conf_tp"].sum()),
                        "annotation_suspicions": events["suspicion"].value_counts().to_dict()}
    inst = gt_boxes(gt)["cls"].value_counts()
    per_class["instances"] = [int(inst.get(i, 0)) for i in range(len(names))]
    opc = pd.DataFrame([{"class": k, **{f"op_{m}": v for m, v in d.items()}} for k, d in op["per_class"].items()])
    per_class = per_class.merge(opc, on="class", how="left")

    gt.to_csv(out / "ground_truth.csv", index=False)
    preds.to_csv(out / "predictions.csv", index=False)
    events.to_csv(out / "events.csv", index=False)
    per_image.to_csv(out / "per_image.csv", index=False)
    per_class.to_csv(out / "per_class.csv", index=False)
    (out / "metrics.json").write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")

    with track(run_name=f"eval-{name}"):
        log_metrics(result.get("ultralytics", {}), prefix=f"{split}_")
        log_metrics(result["operating_point"], prefix=f"{split}_op_")

    u = result.get("ultralytics", {})
    print(f"[{name}] {split}: mAP50={u.get('map50', float('nan')):.4f} mAP50-95={u.get('map50_95', float('nan')):.4f} "
          f"P={u.get('precision', float('nan')):.4f} R={u.get('recall', float('nan')):.4f}")
    print(f"Operating point (conf>={mc.conf_thr}, IoU>={mc.iou_thr}): {result['operating_point']}")
    print(f"Reports: {out}")
    return result


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--name", required=True, help="Report folder name under reports/")
    ap.add_argument("--config", default="configs/baseline.yaml")
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--skip-val", action="store_true", help="Skip the Ultralytics validator (sample-level only)")
    a = ap.parse_args(argv)
    evaluate(a.weights, a.name, load_config(ROOT / a.config), a.split, skip_val=a.skip_val)


if __name__ == "__main__":
    main()
