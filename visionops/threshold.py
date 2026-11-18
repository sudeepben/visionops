"""Choose the confidence threshold on the VALIDATION split, then report it on TEST.

The baseline analysis showed most false negatives are correct-class detections whose
confidence sits below the default 0.25 cut-off. This picks the operating threshold
that maximises F1 on validation (never on test) and reports test P/R/F1 at the
default and the tuned threshold.

    python -m visionops.threshold --weights runs/baseline/weights/best.pt --name baseline
"""
from __future__ import annotations

import argparse
import json
from dataclasses import replace

import numpy as np
import pandas as pd

from .analysis import load_ground_truth, operating_point_metrics, run_matching
from .config import DATASET_YAML, REPORTS_DIR, ROOT, load_config
from .data.prepare import load_names, read_split
from .evaluate import match_cfg

OUT_DIR = REPORTS_DIR / "threshold"


def sweep(preds: pd.DataFrame, gt: pd.DataFrame, thresholds, base_cfg, names: list[str]) -> pd.DataFrame:
    rows = []
    for t in thresholds:
        events, _ = run_matching(preds, gt, replace(base_cfg, conf_thr=float(t)))
        o = operating_point_metrics(events, names)["overall"]
        rows.append({"conf_thr": round(float(t), 3), **o})
    return pd.DataFrame(rows)


def best_threshold(df: pd.DataFrame) -> float:
    """Highest F1; ties broken toward the higher threshold (fewer false alarms)."""
    top = df[df["f1"] == df["f1"].max()]
    return float(top["conf_thr"].max())


def _predict_or_load(weights, cfg, split: str, report_name: str | None):
    if split == "test" and report_name:
        p, g = REPORTS_DIR / report_name / "predictions.csv", REPORTS_DIR / report_name / "ground_truth.csv"
        if p.exists() and g.exists():
            return pd.read_csv(p), pd.read_csv(g)
    from .inference import Detector

    images = read_split(split)
    names = load_names(DATASET_YAML)
    gt = load_ground_truth(images, len(names))
    preds = Detector(weights, imgsz=cfg["imgsz"]).predict_paths(images, conf=cfg.get("eval", {}).get("pred_floor", 0.01))
    return preds, gt


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--name", default="baseline", help="evaluation report to reuse test predictions from")
    ap.add_argument("--config", default="configs/baseline.yaml")
    a = ap.parse_args(argv)
    cfg = load_config(ROOT / a.config)
    names = load_names(DATASET_YAML)
    base = match_cfg(cfg.get("eval", {}))
    grid = np.round(np.arange(0.02, 0.91, 0.01), 2)

    vp, vg = _predict_or_load(a.weights, cfg, "val", None)
    val = sweep(vp, vg, grid, base, names)
    t_star = best_threshold(val)

    tp, tg = _predict_or_load(a.weights, cfg, "test", a.name)
    test = sweep(tp, tg, sorted({base.conf_thr, t_star, *grid}), base, names)

    def at(df, t):
        return df.loc[np.isclose(df["conf_thr"], t)].iloc[0].to_dict()

    result = {
        "selected_on": "val", "default_conf": base.conf_thr, "tuned_conf": t_star, "iou_thr": base.iou_thr,
        "val_at_default": at(val, base.conf_thr), "val_at_tuned": at(val, t_star),
        "test_at_default": at(test, base.conf_thr), "test_at_tuned": at(test, t_star),
        "note": "threshold chosen on validation only; the test curve is reported, never used for selection",
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    val.assign(split="val").to_csv(OUT_DIR / "sweep_val.csv", index=False)
    test.assign(split="test").to_csv(OUT_DIR / "sweep_test.csv", index=False)
    (OUT_DIR / "threshold.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    d, t = result["test_at_default"], result["test_at_tuned"]
    print(f"Tuned threshold (max val F1): {t_star}")
    print(f"TEST @ {base.conf_thr}: P={d['precision']:.3f} R={d['recall']:.3f} F1={d['f1']:.3f}")
    print(f"TEST @ {t_star}: P={t['precision']:.3f} R={t['recall']:.3f} F1={t['f1']:.3f}")


if __name__ == "__main__":
    main()
