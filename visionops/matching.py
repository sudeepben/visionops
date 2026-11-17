"""Defensible prediction <-> ground-truth matching for error analysis.

Rules (documented in README / model card):
1. Only predictions with conf >= ``conf_thr`` count as detections.
2. Detections are processed in descending confidence. Each is matched to the
   unmatched ground-truth box of the SAME class with the highest IoU, provided
   IoU >= ``iou_thr`` (greedy, COCO/VOC style). A GT box can be matched once.
3. Unmatched detections are categorised, in this priority order:
   - duplicate:       same-class GT with IoU >= iou_thr exists but was already matched
   - class_confusion: a GT of a DIFFERENT class has IoU >= iou_thr
   - localization:    a same-class GT has loc_iou_min <= IoU < iou_thr
   - background:      none of the above (no meaningful overlap with any GT)
4. Unmatched GT boxes (false negatives) are categorised:
   - below_threshold: a same-class prediction with IoU >= iou_thr exists but its conf < conf_thr
   - class_confusion: a confident prediction of another class overlaps (IoU >= iou_thr)
   - poor_localization: a confident same-class prediction overlaps with loc_iou_min <= IoU < iou_thr
   - missed:          nothing overlaps
5. Annotation *suspicions* are raised only for HIGH-confidence disagreements
   (conf >= high_conf_thr). They are hints for human review, never verdicts.
   Low confidence on its own never produces an annotation suspicion.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .boxes import box_iou


@dataclass
class MatchConfig:
    iou_thr: float = 0.5
    conf_thr: float = 0.25
    low_conf_thr: float = 0.5
    high_conf_thr: float = 0.7
    loc_iou_min: float = 0.1


@dataclass
class Event:
    type: str  # tp | fp | fn
    subtype: str
    pred_cls: int | None = None
    gt_cls: int | None = None
    conf: float | None = None
    iou: float | None = None
    box: list = field(default_factory=list)  # xyxy of pred (tp/fp) or gt (fn)
    gt_box: list | None = None
    suspicion: str | None = None
    low_confidence: bool = False

    def to_dict(self) -> dict:
        return self.__dict__.copy()


def match_image(
    gt_cls: np.ndarray,
    gt_xyxy: np.ndarray,
    pred_cls: np.ndarray,
    pred_conf: np.ndarray,
    pred_xyxy: np.ndarray,
    cfg: MatchConfig | None = None,
) -> list[Event]:
    cfg = cfg or MatchConfig()
    gt_cls = np.asarray(gt_cls, dtype=int).reshape(-1)
    gt_xyxy = np.asarray(gt_xyxy, dtype=float).reshape(-1, 4)
    pred_cls = np.asarray(pred_cls, dtype=int).reshape(-1)
    pred_conf = np.asarray(pred_conf, dtype=float).reshape(-1)
    pred_xyxy = np.asarray(pred_xyxy, dtype=float).reshape(-1, 4)

    iou_all = box_iou(pred_xyxy, gt_xyxy)  # (P, G) for all predictions incl. low-conf
    confident = pred_conf >= cfg.conf_thr
    order = [i for i in np.argsort(-pred_conf, kind="stable") if confident[i]]

    gt_matched = np.zeros(len(gt_cls), dtype=bool)
    events: list[Event] = []
    pred_events: dict[int, Event] = {}

    for p in order:
        same = gt_cls == pred_cls[p]
        ious = iou_all[p] if len(gt_cls) else np.zeros(0)
        cand = np.where(same & ~gt_matched & (ious >= cfg.iou_thr))[0]
        conf = float(pred_conf[p])
        if len(cand):
            g = cand[np.argmax(ious[cand])]
            gt_matched[g] = True
            pred_events[p] = Event(
                "tp", "match", int(pred_cls[p]), int(gt_cls[g]), conf, float(ious[g]),
                pred_xyxy[p].tolist(), gt_xyxy[g].tolist(), None, conf < cfg.low_conf_thr,
            )
            continue
        high = conf >= cfg.high_conf_thr
        ev = Event("fp", "background", int(pred_cls[p]), None, conf, None, pred_xyxy[p].tolist())
        if len(gt_cls):
            dup = np.where(same & gt_matched & (ious >= cfg.iou_thr))[0]
            other = np.where(~same & (ious >= cfg.iou_thr))[0]
            loc = np.where(same & (ious >= cfg.loc_iou_min) & (ious < cfg.iou_thr))[0]
            if len(dup):
                g = dup[np.argmax(ious[dup])]
                ev.subtype, ev.gt_cls, ev.iou, ev.gt_box = "duplicate", int(gt_cls[g]), float(ious[g]), gt_xyxy[g].tolist()
            elif len(other):
                g = other[np.argmax(ious[other])]
                ev.subtype, ev.gt_cls, ev.iou, ev.gt_box = "class_confusion", int(gt_cls[g]), float(ious[g]), gt_xyxy[g].tolist()
                if high:
                    ev.suspicion = "possible_wrong_class"
            elif len(loc):
                g = loc[np.argmax(ious[loc])]
                ev.subtype, ev.gt_cls, ev.iou, ev.gt_box = "localization", int(gt_cls[g]), float(ious[g]), gt_xyxy[g].tolist()
                if high:
                    ev.suspicion = "possible_loose_or_shifted_box"
            else:
                ev.iou = float(ious.max()) if len(ious) else 0.0
        if ev.subtype == "background" and high:
            ev.suspicion = "possible_missing_label"
        ev.low_confidence = conf < cfg.low_conf_thr
        pred_events[p] = ev

    events.extend(pred_events[p] for p in order)

    for g in np.where(~gt_matched)[0]:
        ious = iou_all[:, g] if len(pred_cls) else np.zeros(0)
        same = pred_cls == gt_cls[g]
        ev = Event("fn", "missed", None, int(gt_cls[g]), None, None, gt_xyxy[g].tolist(), gt_xyxy[g].tolist())
        if len(pred_cls):
            below = np.where(same & ~confident & (ious >= cfg.iou_thr))[0]
            other = np.where(~same & confident & (ious >= cfg.iou_thr))[0]
            loc = np.where(same & confident & (ious >= cfg.loc_iou_min) & (ious < cfg.iou_thr))[0]
            if len(below):
                p = below[np.argmax(pred_conf[below])]
                ev.subtype, ev.pred_cls, ev.conf, ev.iou = "below_threshold", int(pred_cls[p]), float(pred_conf[p]), float(ious[p])
            elif len(other):
                p = other[np.argmax(ious[other])]
                ev.subtype, ev.pred_cls, ev.conf, ev.iou = "class_confusion", int(pred_cls[p]), float(pred_conf[p]), float(ious[p])
            elif len(loc):
                p = loc[np.argmax(ious[loc])]
                ev.subtype, ev.pred_cls, ev.conf, ev.iou = "poor_localization", int(pred_cls[p]), float(pred_conf[p]), float(ious[p])
        events.append(ev)
    return events


def summarize(events: list[Event]) -> dict:
    tp = sum(e.type == "tp" for e in events)
    fp = sum(e.type == "fp" for e in events)
    fn = sum(e.type == "fn" for e in events)
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "low_conf_tp": sum(e.type == "tp" and e.low_confidence for e in events),
        "suspicions": sum(e.suspicion is not None for e in events),
    }
