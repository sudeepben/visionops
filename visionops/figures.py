"""Render README figures from pipeline outputs into docs/images/.

    python -m visionops.figures

Every figure is drawn from files in reports/ - nothing is typed in by hand.
Figures whose inputs are missing are skipped with a message.
"""
from __future__ import annotations

import json

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from .config import DATASET_YAML, REPORTS_DIR, ROOT  # noqa: E402
from .data.prepare import load_names  # noqa: E402
from .inference import GT_COLOR, PRED_COLOR, crop_around, draw_boxes  # noqa: E402

OUT = ROOT / "docs" / "images"
BLUE, ORANGE = "#2a78d6", "#eb6834"      # categorical slots 1-2
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e6e5e0"


def _style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def per_class(name: str = "baseline") -> None:
    p = REPORTS_DIR / name / "per_class.csv"
    if not p.exists():
        print(f"skip per_class: {p} missing")
        return
    df = pd.read_csv(p).sort_values("map50")
    fig, ax = plt.subplots(figsize=(7, 3.4), dpi=150)
    ax.barh(df["class"], df["map50"], color=BLUE, height=0.6)
    for y, (v, n) in enumerate(zip(df["map50"], df["instances"])):
        ax.text(v + 0.01, y, f"{v:.2f}  (n={n})", va="center", fontsize=9, color=INK)
    ax.set_xlim(0, 1)
    ax.set_xlabel("mAP@50 on held-out test set", color=MUTED, fontsize=9)
    ax.set_title("Baseline YOLO11n: per-class mAP@50", loc="left", color=INK, fontsize=11)
    _style(ax)
    fig.tight_layout()
    fig.savefig(OUT / "per_class_map50.png")
    plt.close(fig)


def threshold_curve() -> None:
    d = REPORTS_DIR / "threshold"
    if not (d / "threshold.json").exists():
        print("skip threshold: run python -m visionops.threshold first")
        return
    res = json.loads((d / "threshold.json").read_text(encoding="utf-8"))
    val, test = pd.read_csv(d / "sweep_val.csv"), pd.read_csv(d / "sweep_test.csv")
    fig, ax = plt.subplots(figsize=(7, 3.4), dpi=150)
    ax.plot(val["conf_thr"], val["f1"], color=BLUE, linewidth=2, label="validation (used to choose)")
    ax.plot(test["conf_thr"], test["f1"], color=ORANGE, linewidth=2, label="test (reported only)")
    for x, lab in ((res["default_conf"], f"default {res['default_conf']}"), (res["tuned_conf"], f"tuned {res['tuned_conf']}")):
        ax.axvline(x, color=MUTED, linewidth=1, linestyle="--")
        ax.text(x + 0.01, 0.02, lab, color=MUTED, fontsize=8, rotation=90, va="bottom")
    ax.set_ylim(0, 1)
    ax.set_xlabel("confidence threshold", color=MUTED, fontsize=9)
    ax.set_ylabel("F1 (IoU >= 0.5)", color=MUTED, fontsize=9)
    ax.set_title("Operating threshold chosen on validation", loc="left", color=INK, fontsize=11)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    _style(ax)
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(OUT / "threshold_f1.png")
    plt.close(fig)


def failure_examples(name: str = "baseline", n_each: int = 4) -> None:
    ev_p, gt_p = REPORTS_DIR / name / "events.csv", REPORTS_DIR / name / "ground_truth.csv"
    if not (ev_p.exists() and gt_p.exists()):
        print("skip failure examples: evaluation outputs missing")
        return
    names = load_names(DATASET_YAML)
    ev, gt = pd.read_csv(ev_p), pd.read_csv(gt_p)
    gt = gt[gt["cls"] >= 0]
    picks = [("Missed (below threshold)", ev[(ev.type == "fn") & (ev.subtype == "below_threshold")]),
             ("Missed (no detection)", ev[(ev.type == "fn") & (ev.subtype == "missed")]),
             ("False positive", ev[ev.type == "fp"].sort_values("conf", ascending=False))]
    rows = [(title, df.drop_duplicates("image").head(n_each)) for title, df in picks if len(df)]
    if not rows:
        return
    fig, axes = plt.subplots(len(rows), n_each, figsize=(2.6 * n_each, 2.9 * len(rows)), dpi=130)
    axes = axes.reshape(len(rows), n_each)
    conf_ev = ev[ev.type.isin(["tp", "fp"])]
    for r, (title, df) in enumerate(rows):
        for c in range(n_each):
            ax = axes[r, c]
            ax.axis("off")
            if c >= len(df):
                continue
            e = df.iloc[c]
            img = cv2.imread(e.image)
            if img is None:
                continue
            g = gt[gt.image == e.image]
            p = conf_ev[conf_ev.image == e.image]
            vis = draw_boxes(img, [(b.x1, b.y1, b.x2, b.y2, "") for b in g.itertuples()], GT_COLOR, 2)
            vis = draw_boxes(vis, [(b.x1, b.y1, b.x2, b.y2, "") for b in p.itertuples()], PRED_COLOR, 2)
            vis = draw_boxes(vis, [(e.x1, e.y1, e.x2, e.y2, "")], (0, 200, 255), 2)
            ax.imshow(cv2.cvtColor(crop_around(vis, (e.x1, e.y1, e.x2, e.y2), pad=3, min_size=120), cv2.COLOR_BGR2RGB))
            cls = names[int(e.gt_cls)] if pd.notna(e.gt_cls) else names[int(e.pred_cls)]
            extra = f" conf {e.conf:.2f}" if pd.notna(e.conf) else ""
            ax.set_title(f"{cls}{extra}", fontsize=8, color=INK)
        axes[r, 0].text(-0.05, 0.5, title, transform=axes[r, 0].transAxes, rotation=90, ha="right", va="center",
                        fontsize=9, color=INK)
    fig.suptitle("Test-set failure examples  (green = label, red = prediction, yellow = the error)",
                 fontsize=10, color=INK)
    fig.tight_layout()
    fig.savefig(OUT / "failure_examples.png")
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    per_class()
    threshold_curve()
    failure_examples()
    print(f"Figures in {OUT}")


if __name__ == "__main__":
    main()
