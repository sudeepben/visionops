import pandas as pd

from conftest import make_image, write_label
from visionops.analysis import (duplicate_annotations, geometry_outliers, load_ground_truth,
                                operating_point_metrics, rank_for_review, run_matching)


def _dataset(tmp_path):
    imgs = []
    for i in range(3):
        p = tmp_path / "images" / f"img{i}.jpg"
        make_image(p, 200, 100, seed=i)
        imgs.append(p)
    write_label(tmp_path / "labels" / "img0.txt", [[0, 0.5, 0.5, 0.1, 0.2]])            # box 90,40,110,60
    write_label(tmp_path / "labels" / "img1.txt", [[1, 0.25, 0.25, 0.1, 0.2],
                                                    [1, 0.25, 0.25, 0.1, 0.2]])          # duplicate annotation
    write_label(tmp_path / "labels" / "img2.txt", [[0, 0.5, 0.5, 0.1, 0.2]])
    return imgs


def test_ground_truth_and_matching(tmp_path):
    imgs = _dataset(tmp_path)
    gt = load_ground_truth(imgs, 2)
    assert set(gt.columns) >= {"image", "width", "height", "cls", "x1"}
    preds = pd.DataFrame([
        [str(imgs[0]), 0, 0.9, 90, 40, 110, 60],      # TP
        [str(imgs[0]), 0, 0.95, 150, 10, 170, 30],    # high-conf background FP -> possible missing label
        [str(imgs[2]), 0, 0.3, 90, 40, 110, 60],      # low-confidence TP only
    ], columns=["image", "cls", "conf", "x1", "y1", "x2", "y2"])
    events, per_image = run_matching(preds, gt)
    assert per_image.set_index("image").loc[str(imgs[0]), "tp"] == 1
    assert (events.suspicion == "possible_missing_label").sum() == 1
    op = operating_point_metrics(events, ["missing_hole", "short"])
    assert op["overall"]["tp"] == 2 and op["overall"]["fp"] == 1 and op["overall"]["fn"] == 2

    dups = duplicate_annotations(gt)
    assert len(dups) == 1 and dups.iloc[0]["image"] == str(imgs[1])

    q = rank_for_review(per_image, events, None, dups)
    order = q["image"].tolist()
    # possible missing label (img0) and duplicate annotation (img1) outrank a low-confidence-only image (img2)
    assert order.index(str(imgs[2])) > order.index(str(imgs[0]))
    assert order.index(str(imgs[2])) > order.index(str(imgs[1]))
    row2 = q.set_index("image").loc[str(imgs[2])]
    assert not row2["possible_annotation_issue"]  # low confidence alone is not a label issue
    assert list(q["rank"]) == list(range(1, len(q) + 1))


def test_geometry_outliers_flags_extreme_box():
    rows = [["a.jpg", 100, 100, 0, 10, 10, 20 + (i % 3), 20] for i in range(30)]
    rows.append(["b.jpg", 100, 100, 0, 0, 0, 100, 5])  # huge, flat box
    gt = pd.DataFrame(rows, columns=["image", "width", "height", "cls", "x1", "y1", "x2", "y2"])
    out = geometry_outliers(gt)
    assert "b.jpg" in set(out["image"])
    assert "a.jpg" not in set(out["image"])
