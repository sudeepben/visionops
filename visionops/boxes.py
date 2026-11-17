"""Box utilities: YOLO label I/O, coordinate conversion and IoU."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


@dataclass
class LabelIssue:
    line_no: int
    issue: str
    raw: str


def image_to_label_path(img_path: str | Path) -> Path:
    """Ultralytics convention: .../images/x.jpg -> .../labels/x.txt (last 'images' segment)."""
    p = Path(img_path)
    parts = list(p.parts)
    for i in range(len(parts) - 1, -1, -1):
        if parts[i] == "images":
            parts[i] = "labels"
            break
    return Path(*parts).with_suffix(".txt")


def read_yolo_labels(label_path: str | Path, num_classes: int | None = None):
    """Read a YOLO detection label file.

    Returns (labels, issues) where labels is an (N, 5) float array [cls, xc, yc, w, h]
    (normalised) and issues lists malformed / out-of-range lines.
    A missing file is treated as 'no objects' (valid in YOLO).
    """
    label_path = Path(label_path)
    rows, issues = [], []
    if not label_path.exists():
        return np.zeros((0, 5), dtype=float), issues
    for i, raw in enumerate(label_path.read_text(encoding="utf-8").splitlines(), 1):
        s = raw.strip()
        if not s:
            continue
        parts = s.split()
        if len(parts) != 5:
            kind = "polygon_or_segment" if len(parts) > 5 and len(parts) % 2 == 1 else "wrong_field_count"
            issues.append(LabelIssue(i, kind, raw))
            continue
        try:
            vals = [float(x) for x in parts]
        except ValueError:
            issues.append(LabelIssue(i, "non_numeric", raw))
            continue
        c, xc, yc, w, h = vals
        if c != int(c) or c < 0 or (num_classes is not None and c >= num_classes):
            issues.append(LabelIssue(i, "invalid_class_id", raw))
            continue
        if w <= 0 or h <= 0:
            issues.append(LabelIssue(i, "degenerate_box", raw))
            continue
        if not all(-1e-6 <= v <= 1 + 1e-6 for v in (xc, yc, w, h)):
            issues.append(LabelIssue(i, "coords_out_of_range", raw))
            continue
        x1, y1, x2, y2 = xc - w / 2, yc - h / 2, xc + w / 2, yc + h / 2
        if x1 < -0.01 or y1 < -0.01 or x2 > 1.01 or y2 > 1.01:
            issues.append(LabelIssue(i, "box_exceeds_image", raw))
            # keep it: it is usable after clipping, but report it
        rows.append(vals)
    arr = np.array(rows, dtype=float) if rows else np.zeros((0, 5), dtype=float)
    return arr, issues


def xywhn_to_xyxy(xywhn: np.ndarray, width: int, height: int) -> np.ndarray:
    """Normalised center xywh -> absolute pixel xyxy (clipped to image)."""
    xywhn = np.asarray(xywhn, dtype=float).reshape(-1, 4)
    xc, yc, w, h = xywhn.T
    out = np.stack(
        [(xc - w / 2) * width, (yc - h / 2) * height, (xc + w / 2) * width, (yc + h / 2) * height], axis=1
    )
    out[:, [0, 2]] = out[:, [0, 2]].clip(0, width)
    out[:, [1, 3]] = out[:, [1, 3]].clip(0, height)
    return out


def box_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise IoU between (N,4) and (M,4) xyxy boxes -> (N,M)."""
    a = np.asarray(a, dtype=float).reshape(-1, 4)
    b = np.asarray(b, dtype=float).reshape(-1, 4)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    tl = np.maximum(a[:, None, :2], b[None, :, :2])
    br = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.prod(np.clip(br - tl, 0, None), axis=2)
    area_a = np.prod(a[:, 2:] - a[:, :2], axis=1)
    area_b = np.prod(b[:, 2:] - b[:, :2], axis=1)
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / union, 0.0)


def list_images(folder: str | Path) -> list[Path]:
    folder = Path(folder)
    if not folder.exists():
        return []
    return sorted(p for p in folder.rglob("*") if p.suffix.lower() in IMG_EXTS)
