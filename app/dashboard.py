"""VisionOps Streamlit dashboard.

    streamlit run app/dashboard.py

Every tab reads files produced by the pipeline; if they don't exist yet the tab
says which command creates them instead of showing placeholder numbers.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from visionops.config import DATASET_YAML, REPORTS_DIR, load_dotenv  # noqa: E402
from visionops.inference import FP_COLOR, GT_COLOR, PRED_COLOR, crop_around, draw_boxes, find_weights  # noqa: E402

load_dotenv()
st.set_page_config(page_title="VisionOps", layout="wide")


# ---------- helpers ----------
@st.cache_resource(show_spinner="Loading model...")
def get_detector(path: str, imgsz: int):
    from visionops.inference import Detector

    return Detector(path, imgsz=imgsz)


@st.cache_data
def read_csv(path: str, mtime: float) -> pd.DataFrame:
    return pd.read_csv(path)


def csv(path: Path) -> pd.DataFrame | None:
    return read_csv(str(path), path.stat().st_mtime) if path.exists() else None


@st.cache_data(max_entries=64)
def load_image(path: str) -> np.ndarray | None:
    return cv2.imread(path)


def class_names() -> list[str]:
    if DATASET_YAML.exists():
        from visionops.data.prepare import load_names

        return load_names(DATASET_YAML)
    return []


def cname(names, c) -> str:
    try:
        c = int(c)
        return names[c] if 0 <= c < len(names) else str(c)
    except (TypeError, ValueError):
        return "-"


def show_bgr(img, caption=None):
    st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption=caption, use_container_width=True)


def missing(cmd: str):
    st.info(f"No results yet. Run:\n\n```\n{cmd}\n```")


def report_names() -> list[str]:
    return sorted(p.parent.name for p in REPORTS_DIR.glob("*/metrics.json"))


NAMES = class_names()
st.title("VisionOps - PCB defect inspection")
st.caption("Green = ground truth, red = model prediction, orange = false positive. "
           "Review flags are priorities for a human, not verdicts.")

tab_inf, tab_eval, tab_fail, tab_rev, tab_exp = st.tabs(
    ["Inspect image", "Evaluation", "Failure gallery", "Review queue", "Experiments"])

# ---------- Inspect ----------
with tab_inf:
    weights = find_weights()
    if not weights:
        missing("python -m visionops.train --config configs/baseline.yaml")
    else:
        c1, c2, c3 = st.columns([3, 1, 1])
        w = c1.selectbox("Model", weights, format_func=lambda p: str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p))
        th_file = REPORTS_DIR / "threshold" / "threshold.json"
        default_conf = 0.25
        if th_file.exists():
            default_conf = float(json.loads(th_file.read_text(encoding="utf-8"))["tuned_conf"])
        conf = c2.slider("Confidence", 0.01, 0.95, default_conf, 0.01,
                         help="Defaults to the threshold tuned on the validation split, if available")
        imgsz = c3.selectbox("Image size", [640, 960, 1280], index=0)
        up = st.file_uploader("Upload a PCB image", type=["jpg", "jpeg", "png", "bmp"])
        if up is not None:
            img = cv2.imdecode(np.frombuffer(up.read(), np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                st.error("Could not decode image.")
            else:
                try:
                    det = get_detector(str(w), imgsz)
                    dets = det.predict_image(img, conf=conf)
                except Exception as e:
                    st.error(f"Inference failed: {e}")
                    dets = None
                if dets is not None:
                    boxes = [(*d["box_xyxy"], f"{d['class_name']} {d['confidence']:.2f}") for d in dets]
                    show_bgr(draw_boxes(img, boxes, PRED_COLOR), f"{len(dets)} detections")
                    if dets:
                        st.dataframe(pd.DataFrame([{**{k: v for k, v in d.items() if k != "box_xyxy"},
                                                    "box_xyxy": [round(v, 1) for v in d["box_xyxy"]]} for d in dets]),
                                     use_container_width=True)
                    st.download_button("Download JSON", json.dumps(dets, indent=2), "detections.json")

# ---------- Evaluation ----------
with tab_eval:
    runs = report_names()
    if not runs:
        missing("python -m visionops.evaluate --weights runs/baseline/weights/best.pt --name baseline")
    else:
        run = st.selectbox("Report", runs, index=runs.index("baseline") if "baseline" in runs else 0, key="eval_run")
        m = json.loads((REPORTS_DIR / run / "metrics.json").read_text(encoding="utf-8"))
        u, op = m.get("ultralytics", {}), m["operating_point"]
        cols = st.columns(7)
        for col, (label, val) in zip(cols, [("mAP@50", u.get("map50")), ("mAP@50-95", u.get("map50_95")),
                                            ("Precision", u.get("precision")), ("Recall", u.get("recall")),
                                            (f"P @{op['conf_thr']}", op["precision"]), (f"R @{op['conf_thr']}", op["recall"]),
                                            (f"F1 @{op['conf_thr']}", op["f1"])]):
            col.metric(label, "-" if val is None else f"{val:.3f}")
        st.caption(f"Split: {m['split']} | {m['counts']['images']} images, {m['counts']['gt_boxes']} boxes. "
                   "mAP/P/R from the Ultralytics validator; @conf values from VisionOps matching (IoU>=0.5).")
        pc = csv(REPORTS_DIR / run / "per_class.csv")
        if pc is not None:
            st.subheader("Per-class metrics")
            st.dataframe(pc, use_container_width=True)
            metric_cols = [c for c in ["map50", "map50_95", "recall", "op_f1"] if c in pc]
            if metric_cols:
                st.bar_chart(pc.set_index("class")[metric_cols])
        c1, c2 = st.columns(2)
        ev = csv(REPORTS_DIR / run / "events.csv")
        if ev is not None:
            with c1:
                st.subheader("Errors by type")
                err = ev[ev.type != "tp"].groupby(["type", "subtype"]).size().rename("count").reset_index()
                err["kind"] = err["type"].str.upper() + " - " + err["subtype"]
                st.bar_chart(err.set_index("kind")["count"])
        plots = sorted((REPORTS_DIR / run / "plots").glob("*.png"))
        with c2:
            if plots:
                p = st.selectbox("Plot", plots, format_func=lambda x: x.stem)
                st.image(str(p), use_container_width=True)

# ---------- Failure gallery ----------
with tab_fail:
    runs = report_names()
    if not runs:
        missing("python -m visionops.evaluate --weights runs/baseline/weights/best.pt --name baseline")
    else:
        run = st.selectbox("Report", runs, index=runs.index("baseline") if "baseline" in runs else 0, key="fail_run")
        ev, gt = csv(REPORTS_DIR / run / "events.csv"), csv(REPORTS_DIR / run / "ground_truth.csv")
        c1, c2, c3 = st.columns(3)
        kind = c1.selectbox("Show", ["False negatives", "False positives", "Low-confidence true positives",
                                     "Possible annotation issues"])
        cls_filter = c2.selectbox("Class", ["all"] + NAMES)
        per_page = 12
        if kind == "False negatives":
            sel = ev[ev.type == "fn"]
        elif kind == "False positives":
            sel = ev[ev.type == "fp"].sort_values("conf", ascending=False)
        elif kind == "Low-confidence true positives":
            sel = ev[(ev.type == "tp") & ev.low_confidence.astype(bool)].sort_values("conf")
        else:
            sel = ev[ev.suspicion.notna()].sort_values("conf", ascending=False)
        if cls_filter != "all":
            ci = NAMES.index(cls_filter)
            sel = sel[(sel.pred_cls == ci) | (sel.gt_cls == ci)]
        subtypes = sorted(sel["subtype"].dropna().unique())
        sub = c3.multiselect("Subtype", subtypes, default=subtypes)
        sel = sel[sel.subtype.isin(sub)]
        st.write(f"{len(sel)} items")
        pages = max(1, (len(sel) - 1) // per_page + 1)
        page = st.number_input("Page", 1, pages, 1) - 1
        grid = st.columns(4)
        gtb = gt[gt["cls"] >= 0]
        confident = ev[ev.type.isin(["tp", "fp"])]
        for i, (_, r) in enumerate(sel.iloc[page * per_page:(page + 1) * per_page].iterrows()):
            img = load_image(r.image)
            if img is None:
                continue
            g = gtb[gtb.image == r.image]
            p = confident[confident.image == r.image]
            vis = draw_boxes(img, [(b.x1, b.y1, b.x2, b.y2, cname(NAMES, b.cls)) for b in g.itertuples()], GT_COLOR)
            vis = draw_boxes(vis, [(b.x1, b.y1, b.x2, b.y2, f"{cname(NAMES, b.pred_cls)} {b.conf:.2f}")
                                   for b in p.itertuples()], PRED_COLOR)
            vis = draw_boxes(vis, [(r.x1, r.y1, r.x2, r.y2, "")], FP_COLOR if r.type == "fp" else (255, 0, 255), 3)
            cap = (f"{r.type.upper()} {r.subtype} | gt={cname(NAMES, r.gt_cls)} pred={cname(NAMES, r.pred_cls)} "
                   f"conf={'-' if pd.isna(r.conf) else f'{r.conf:.2f}'} iou={'-' if pd.isna(r.iou) else f'{r.iou:.2f}'}"
                   + (f" | {r.suspicion}" if isinstance(r.suspicion, str) else ""))
            with grid[i % 4]:
                show_bgr(crop_around(vis, (r.x1, r.y1, r.x2, r.y2)), cap)
                st.caption(Path(r.image).name)

# ---------- Review queue ----------
with tab_rev:
    q = csv(REPORTS_DIR / "review" / "review_queue.csv")
    if q is None:
        missing("python -m visionops.review --weights runs/baseline/weights/best.pt")
    else:
        dec_path = REPORTS_DIR / "review" / "decisions.csv"
        decisions = csv(dec_path)
        reviewed = set(decisions["image"]) if decisions is not None else set()
        st.write(f"{len(q)} training images with review signals; {len(reviewed)} reviewed. "
                 "Score combines high-confidence disagreements, misses and label-geometry checks. "
                 "Low confidence alone carries little weight and is never treated as a bad label.")
        only_issue = st.checkbox("Only images with a possible annotation issue", value=True)
        hide_done = st.checkbox("Hide reviewed", value=False)
        view = q[q.possible_annotation_issue.astype(bool)] if only_issue else q
        if hide_done:
            view = view[~view.image.isin(reviewed)]
        view = view.assign(file=view.image.map(lambda s: Path(s).name))
        st.dataframe(view[["rank", "score", "file", "reasons", "n_gt", "tp", "fp", "fn"]].head(200),
                     use_container_width=True, hide_index=True)
        if len(view):
            pick = st.selectbox("Inspect", list(view.head(200).itertuples()), format_func=lambda r: f"#{r.rank}  {r.file}  ({r.score})")
            tev = csv(REPORTS_DIR / "review" / "train_events.csv")
            tgt = csv(REPORTS_DIR / "review" / "train_ground_truth.csv")
            img = load_image(pick.image)
            if img is not None and tev is not None and tgt is not None:
                g = tgt[(tgt.image == pick.image) & (tgt["cls"] >= 0)]
                e = tev[(tev.image == pick.image) & tev.type.isin(["tp", "fp"])]
                vis = draw_boxes(img, [(b.x1, b.y1, b.x2, b.y2, cname(NAMES, b.cls)) for b in g.itertuples()], GT_COLOR)
                vis = draw_boxes(vis, [(b.x1, b.y1, b.x2, b.y2, f"{cname(NAMES, b.pred_cls)} {b.conf:.2f}"
                                        + (f" [{b.suspicion}]" if isinstance(b.suspicion, str) else ""))
                                       for b in e.itertuples()], PRED_COLOR)
                show_bgr(vis, pick.reasons)
                fn = tev[(tev.image == pick.image) & (tev.type == "fn")]
                if len(fn):
                    st.write("Missed labelled objects:", ", ".join(f"{cname(NAMES, r.gt_cls)} ({r.subtype})" for r in fn.itertuples()))
            with st.form("decision"):
                d = st.radio("Decision", ["label_ok", "label_issue", "unsure"], horizontal=True)
                note = st.text_input("Note (optional)")
                if st.form_submit_button("Save decision"):
                    row = pd.DataFrame([{"image": pick.image, "decision": d, "note": note,
                                         "timestamp": datetime.now(timezone.utc).isoformat()}])
                    dec_path.parent.mkdir(parents=True, exist_ok=True)
                    row.to_csv(dec_path, mode="a", header=not dec_path.exists(), index=False)
                    st.success("Saved. Use `python -m visionops.experiment build --source reviews` to train on your decisions.")

# ---------- Experiments ----------
with tab_exp:
    comp = csv(REPORTS_DIR / "experiments" / "comparison.csv")
    if comp is None:
        missing("python -m visionops.experiment build --frac 0.05\npython -m visionops.experiment run")
    else:
        st.write("All runs evaluated on the same untouched test split. Only the training list differs.")
        st.dataframe(comp, use_container_width=True, hide_index=True)
        st.bar_chart(comp.set_index("run")[["test_map50", "test_map50_95"]])
        if comp.groupby("variant").size().min() < 2:
            st.warning("One seed per variant: small differences may be run-to-run noise.")
        man = REPORTS_DIR / "experiments" / "manifest.json"
        if man.exists():
            m = json.loads(man.read_text(encoding="utf-8"))
            st.caption(f"Flag source: {m['source']}, removal budget: {m['frac']:.0%} of {m['train_images']} train images.")
