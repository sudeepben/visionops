import numpy as np

from visionops.matching import MatchConfig, match_image, summarize

GT = np.array([[0, 0, 10, 10], [50, 50, 60, 60]], float)
GT_CLS = np.array([0, 1])


def run(pred_cls, conf, boxes, gt_cls=GT_CLS, gt=GT, cfg=None):
    return match_image(gt_cls, gt, np.array(pred_cls), np.array(conf, float), np.array(boxes, float).reshape(-1, 4), cfg)


def by(events, t):
    return [e for e in events if e.type == t]


def test_perfect_match():
    ev = run([0, 1], [0.9, 0.8], [[0, 0, 10, 10], [50, 50, 60, 60]])
    assert summarize(ev) == {"tp": 2, "fp": 0, "fn": 0, "low_conf_tp": 0, "suspicions": 0}


def test_duplicate_detection_is_fp_duplicate():
    ev = run([0, 0, 1], [0.9, 0.8, 0.9], [[0, 0, 10, 10], [0, 0, 10, 11], [50, 50, 60, 60]])
    fps = by(ev, "fp")
    assert len(fps) == 1 and fps[0].subtype == "duplicate"
    assert fps[0].conf == 0.8  # lower-confidence one is the duplicate


def test_class_confusion_high_conf_raises_suspicion_and_fn():
    ev = run([1, 1], [0.95, 0.9], [[0, 0, 10, 10], [50, 50, 60, 60]])
    fp = by(ev, "fp")[0]
    fn = by(ev, "fn")[0]
    assert fp.subtype == "class_confusion" and fp.suspicion == "possible_wrong_class"
    assert fn.subtype == "class_confusion" and fn.gt_cls == 0


def test_localization_error():
    ev = run([0, 1], [0.9, 0.9], [[0, 0, 10, 25], [50, 50, 60, 60]])  # IoU = 0.4
    fp = by(ev, "fp")[0]
    assert fp.subtype == "localization" and fp.suspicion == "possible_loose_or_shifted_box"
    assert by(ev, "fn")[0].subtype == "poor_localization"


def test_background_fp_only_suspicious_when_high_confidence():
    hi = run([0, 0, 1], [0.9, 0.95, 0.9], [[0, 0, 10, 10], [100, 100, 110, 110], [50, 50, 60, 60]])
    lo = run([0, 0, 1], [0.9, 0.4, 0.9], [[0, 0, 10, 10], [100, 100, 110, 110], [50, 50, 60, 60]])
    assert by(hi, "fp")[0].suspicion == "possible_missing_label"
    assert by(lo, "fp")[0].suspicion is None


def test_low_confidence_never_produces_suspicion():
    ev = run([0, 1], [0.3, 0.26], [[0, 0, 10, 10], [50, 50, 60, 60]])
    assert all(e.suspicion is None for e in ev)
    assert summarize(ev)["low_conf_tp"] == 2


def test_fn_below_threshold_vs_missed():
    ev = run([0], [0.1], [[0, 0, 10, 10]])  # below conf_thr=0.25
    fns = {e.gt_cls: e.subtype for e in by(ev, "fn")}
    assert fns == {0: "below_threshold", 1: "missed"}
    assert not by(ev, "tp") and not by(ev, "fp")


def test_empty_inputs():
    assert run([], [], np.zeros((0, 4))) and all(e.type == "fn" for e in run([], [], np.zeros((0, 4))))
    ev = run([0], [0.9], [[0, 0, 5, 5]], gt_cls=np.zeros(0), gt=np.zeros((0, 4)))
    assert len(ev) == 1 and ev[0].subtype == "background"


def test_custom_thresholds():
    cfg = MatchConfig(iou_thr=0.3)
    ev = run([0, 1], [0.9, 0.9], [[0, 0, 10, 25], [50, 50, 60, 60]], cfg=cfg)
    assert summarize(ev)["tp"] == 2
