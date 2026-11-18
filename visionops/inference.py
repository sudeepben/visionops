"""Model loading, prediction and drawing helpers shared by CLI, API and dashboard."""
from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from .analysis import PRED_COLS
from .config import ROOT, RUNS_DIR

# BGR colours for drawing
GT_COLOR = (60, 180, 75)
PRED_COLOR = (40, 40, 230)
FP_COLOR = (0, 140, 255)


class Detector:
    """Thin wrapper over an Ultralytics YOLO model (.pt or exported .onnx)."""

    def __init__(self, weights: str | Path, imgsz: int = 640, device: str | None = None):
        from ultralytics import YOLO  # imported lazily so the rest of the package works without torch

        self.weights = str(weights)
        self.imgsz = imgsz
        self.device = device
        self.model = YOLO(self.weights, task="detect")
        if self.weights.lower().endswith(".onnx") and self.device is None:
            self.device = "cpu"  # onnxruntime CPU build
        if self.device is not None:
            # Exported models only learn their class names once a predictor exists. Warm up on the
            # requested device first, otherwise Ultralytics probes the default GPU (fails on some setups).
            self.model.predict(np.zeros((imgsz, imgsz, 3), dtype=np.uint8), device=self.device, verbose=False)
        names = self.model.names
        self.names = [names[i] for i in sorted(names)] if isinstance(names, dict) else list(names)

    def _kw(self, conf: float, iou: float) -> dict:
        kw = {"conf": conf, "iou": iou, "imgsz": self.imgsz, "verbose": False}
        if self.device is not None:
            kw["device"] = self.device
        return kw

    def predict_image(self, img_bgr: np.ndarray, conf: float = 0.25, iou: float = 0.7) -> list[dict]:
        r = self.model.predict(img_bgr, **self._kw(conf, iou))[0]
        return _result_to_dicts(r, self.names)

    def predict_paths(self, paths: list[Path], conf: float = 0.01, iou: float = 0.7, chunk: int = 32) -> pd.DataFrame:
        rows = []
        paths = [str(p) for p in paths]
        if self.weights.lower().endswith(".onnx"):
            chunk = 1  # exported with a static batch size of 1
        for i in range(0, len(paths), chunk):
            batch = paths[i:i + chunk]
            for path, r in zip(batch, self.model.predict(batch, stream=True, **self._kw(conf, iou))):
                for d in _result_to_dicts(r, self.names):
                    rows.append([path, d["class_id"], d["confidence"], *d["box_xyxy"]])
        return pd.DataFrame(rows, columns=PRED_COLS)


def _result_to_dicts(r, names: list[str]) -> list[dict]:
    if r.boxes is None or len(r.boxes) == 0:
        return []
    xyxy = r.boxes.xyxy.cpu().numpy()
    conf = r.boxes.conf.cpu().numpy()
    cls = r.boxes.cls.cpu().numpy().astype(int)
    return [
        {"class_id": int(c), "class_name": names[int(c)], "confidence": float(s),
         "box_xyxy": [float(v) for v in b]}
        for b, s, c in zip(xyxy, conf, cls)
    ]


def find_weights() -> list[Path]:
    """Candidate model files: env override, runs/*/weights/best.pt, models/*.(pt|onnx)."""
    found = []
    env = os.getenv("VISIONOPS_MODEL")
    if env and Path(env).exists():
        found.append(Path(env))
    found += sorted(RUNS_DIR.glob("*/weights/best.pt"))
    found += sorted(RUNS_DIR.glob("*/weights/best.onnx"))
    found += sorted((ROOT / "models").glob("*.pt")) + sorted((ROOT / "models").glob("*.onnx"))
    seen, out = set(), []
    for p in found:
        if p.resolve() not in seen:
            seen.add(p.resolve())
            out.append(p)
    return out


def default_weights() -> Path | None:
    env = os.getenv("VISIONOPS_MODEL")
    if env:
        return Path(env)
    for name in ("curated", "baseline"):
        p = RUNS_DIR / name / "weights" / "best.pt"
        if p.exists():
            return p
    w = find_weights()
    return w[0] if w else None


def draw_boxes(img_bgr: np.ndarray, boxes: list[tuple], color=PRED_COLOR, thickness: int | None = None) -> np.ndarray:
    """boxes: iterable of (x1, y1, x2, y2, label_text)."""
    out = img_bgr.copy()
    t = thickness or max(1, round(max(out.shape[:2]) / 500))
    for x1, y1, x2, y2, text in boxes:
        p1, p2 = (int(x1), int(y1)), (int(x2), int(y2))
        cv2.rectangle(out, p1, p2, color, t)
        if text:
            fs = max(0.4, t * 0.35)
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fs, 1)
            y = max(th + 2, p1[1])
            cv2.rectangle(out, (p1[0], y - th - 4), (p1[0] + tw + 2, y), color, -1)
            cv2.putText(out, text, (p1[0] + 1, y - 2), cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def crop_around(img_bgr: np.ndarray, box, pad: float = 2.0, min_size: int = 160) -> np.ndarray:
    """Crop a context window around a (small) box for gallery thumbnails."""
    h, w = img_bgr.shape[:2]
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    s = max(min_size, (x2 - x1) * (1 + pad), (y2 - y1) * (1 + pad))
    a, b = int(max(0, cx - s / 2)), int(max(0, cy - s / 2))
    c, d = int(min(w, cx + s / 2)), int(min(h, cy + s / 2))
    return img_bgr[b:d, a:c]
