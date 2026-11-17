"""Failure analysis and review prioritisation (pure numpy/pandas, no torch).

Data frames used throughout:
  gt_df:    image, width, height, cls, x1, y1, x2, y2           (one row per GT box)
  pred_df:  image, cls, conf, x1, y1, x2, y2                    (one row per prediction)
  events:   image, type, subtype, pred_cls, gt_cls, conf, iou, x1..y2, suspicion, low_confidence
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from .boxes import box_iou, image_to_label_path, read_yolo_labels, xywhn_to_xyxy
from .matching import MatchConfig, match_image

GT_COLS = ["image", "width", "height", "cls", "x1", "y1", "x2", "y2"]
PRED_COLS = ["image", "cls", "conf", "x1", "y1", "x2", "y2"]

# Review score weights. Annotation *suspicions* weigh most; low confidence alone weighs little
# and can never by itself make an image look like a labelling problem.
DEFAULT_WEIGHTS = {
    "possible_missing_label": 3.0,
    "possible_wrong_class": 3.0,
    "possible_loose_or_shifted_box": 2.0,
    "duplicate_annotation": 2.0,
    "unusual_box_geometry": 1.0,
    "fn_missed": 1.5,
    "fn_other": 1.0,
    "fp_other": 0.5,
    "low_conf_tp": 0.25,
}


def load_ground_truth(images: list[Path], num_classes: int | None = None) -> pd.DataFrame:
    rows = []
    for img in images:
        with Image.open(img) as im:
            w, h = im.size
        labels, _ = read_yolo_labels(image_to_label_path(img), num_classes)
        if len(labels) == 0:
            rows.append([str(img), w, h, -1, np.nan, np.nan, np.nan, np.nan])  # placeholder keeps image size
            continue
        xyxy = xywhn_to_xyxy(labels[:, 1:], w, h)
        for c, b in zip(labels[:, 0].astype(int), xyxy):
            rows.append([str(img), w, h, int(c), *b.tolist()])
    df = pd.DataFrame(rows, columns=GT_COLS)
    return df


def gt_boxes(gt_df: pd.DataFrame) -> pd.DataFrame:
    """Only real boxes (drops the size-only placeholder rows)."""
    return gt_df[gt_df["cls"] >= 0]


def run_matching(pred_df: pd.DataFrame, gt_df: pd.DataFrame, cfg: MatchConfig | None = None):
    cfg = cfg or MatchConfig()
    images = gt_df["image"].unique().tolist()
    pred_g = {k: v for k, v in pred_df.groupby("image")} if len(pred_df) else {}
    gt_g = {k: v for k, v in gt_boxes(gt_df).groupby("image")}
    ev_rows, img_rows = [], []
    empty_p = pd.DataFrame(columns=PRED_COLS)
    empty_g = pd.DataFrame(columns=GT_COLS)
    for img in images:
        p = pred_g.get(img, empty_p)
        g = gt_g.get(img, empty_g)
        events = match_image(
            g["cls"].to_numpy(int), g[["x1", "y1", "x2", "y2"]].to_numpy(float),
            p["cls"].to_numpy(int), p["conf"].to_numpy(float), p[["x1", "y1", "x2", "y2"]].to_numpy(float), cfg,
        )
        for e in events:
            d = e.to_dict()
            box = d.pop("box")
            d.pop("gt_box")
            ev_rows.append({"image": img, **d, "x1": box[0], "y1": box[1], "x2": box[2], "y2": box[3]})
        tp = sum(e.type == "tp" for e in events)
        fp = sum(e.type == "fp" for e in events)
        fn = sum(e.type == "fn" for e in events)
        img_rows.append({
            "image": img, "n_gt": len(g), "n_pred": int((p["conf"] >= cfg.conf_thr).sum()) if len(p) else 0,
            "tp": tp, "fp": fp, "fn": fn,
            "low_conf_tp": sum(e.type == "tp" and e.low_confidence for e in events),
            "suspicions": sum(e.suspicion is not None for e in events),
            "correct": fp == 0 and fn == 0,
        })
    ev_cols = ["image", "type", "subtype", "pred_cls", "gt_cls", "conf", "iou", "suspicion", "low_confidence",
               "x1", "y1", "x2", "y2"]
    return pd.DataFrame(ev_rows, columns=ev_cols), pd.DataFrame(img_rows)


def operating_point_metrics(events: pd.DataFrame, names: list[str]) -> dict:
    """Precision/recall at the fixed operating point used for matching (not mAP)."""
    out = {}

    def pr(tp, fp, fn):
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * p * r / (p + r) if p + r else 0.0
        return {"tp": int(tp), "fp": int(fp), "fn": int(fn), "precision": p, "recall": r, "f1": f1}

    t = events["type"]
    out["overall"] = pr((t == "tp").sum(), (t == "fp").sum(), (t == "fn").sum())
    per = {}
    for c, n in enumerate(names):
        tp = ((t == "tp") & (events["pred_cls"] == c)).sum()
        fp = ((t == "fp") & (events["pred_cls"] == c)).sum()
        fn = ((t == "fn") & (events["gt_cls"] == c)).sum()
        per[n] = pr(tp, fp, fn)
    out["per_class"] = per
    return out


def geometry_outliers(gt_df: pd.DataFrame, z_thr: float = 3.5) -> pd.DataFrame:
    """Per-class robust (median/MAD) z-scores of log box area and log aspect ratio."""
    g = gt_boxes(gt_df).copy()
    if g.empty:
        return g.assign(reason=[])
    w = (g["x2"] - g["x1"]).clip(lower=1e-6)
    h = (g["y2"] - g["y1"]).clip(lower=1e-6)
    g["log_area"] = np.log(w * h / (g["width"] * g["height"]))
    g["log_aspect"] = np.log(w / h)
    flags = []
    for _, grp in g.groupby("cls"):
        if len(grp) < 10:  # too few instances to define "unusual"
            continue
        for col in ("log_area", "log_aspect"):
            med = grp[col].median()
            mad = (grp[col] - med).abs().median() * 1.4826
            if mad < 1e-9:
                continue
            z = (grp[col] - med) / mad
            hit = grp[z.abs() > z_thr].assign(reason=f"unusual_{col.replace('log_', '')}", z=z[z.abs() > z_thr])
            flags.append(hit)
    if not flags:
        return g.iloc[0:0].assign(reason=[], z=[])
    return pd.concat(flags)


def duplicate_annotations(gt_df: pd.DataFrame, iou_thr: float = 0.9) -> pd.DataFrame:
    """Pairs of same-class GT boxes in one image overlapping with IoU >= iou_thr."""
    rows = []
    for img, g in gt_boxes(gt_df).groupby("image"):
        if len(g) < 2:
            continue
        b = g[["x1", "y1", "x2", "y2"]].to_numpy(float)
        c = g["cls"].to_numpy(int)
        iou = box_iou(b, b)
        for i in range(len(g)):
            for j in range(i + 1, len(g)):
                if c[i] == c[j] and iou[i, j] >= iou_thr:
                    rows.append({"image": img, "cls": int(c[i]), "iou": float(iou[i, j])})
    return pd.DataFrame(rows, columns=["image", "cls", "iou"])


def rank_for_review(per_image: pd.DataFrame, events: pd.DataFrame, outliers: pd.DataFrame | None = None,
                    dups: pd.DataFrame | None = None, weights: dict | None = None) -> pd.DataFrame:
    """Rank images for human review. Output is a PRIORITY, not a judgement of label quality."""
    w = {**DEFAULT_WEIGHTS, **(weights or {})}
    counts: dict[str, dict[str, float]] = {img: {} for img in per_image["image"]}

    def add(img, key, n=1):
        counts.setdefault(img, {})
        counts[img][key] = counts[img].get(key, 0) + n

    for r in events.itertuples(index=False):
        if isinstance(r.suspicion, str) and r.suspicion:
            add(r.image, r.suspicion)
        elif r.type == "fn":
            add(r.image, "fn_missed" if r.subtype == "missed" else "fn_other")
        elif r.type == "fp":
            add(r.image, "fp_other")
        elif r.type == "tp" and r.low_confidence:
            add(r.image, "low_conf_tp")
    if outliers is not None and len(outliers):
        for img, n in outliers.groupby("image").size().items():
            add(img, "unusual_box_geometry", n)
    if dups is not None and len(dups):
        for img, n in dups.groupby("image").size().items():
            add(img, "duplicate_annotation", n)

    rows = []
    for img, c in counts.items():
        score = sum(w.get(k, 0) * v for k, v in c.items())
        reasons = "; ".join(f"{k} x{int(v)}" for k, v in sorted(c.items(), key=lambda kv: -w.get(kv[0], 0) * kv[1]))
        label_suspicion = any(k in c for k in ("possible_missing_label", "possible_wrong_class",
                                                "possible_loose_or_shifted_box", "duplicate_annotation",
                                                "unusual_box_geometry"))
        rows.append({"image": img, "score": round(score, 3), "reasons": reasons,
                     "possible_annotation_issue": label_suspicion})
    q = pd.DataFrame(rows, columns=["image", "score", "reasons", "possible_annotation_issue"])
    q = q.merge(per_image, on="image", how="left")
    q = q[q["score"] > 0].sort_values(["score", "image"], ascending=[False, True]).reset_index(drop=True)
    q.insert(0, "rank", np.arange(1, len(q) + 1))
    return q
