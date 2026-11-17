from pathlib import Path

import numpy as np

from visionops.boxes import box_iou, image_to_label_path, read_yolo_labels, xywhn_to_xyxy


def test_iou_basic():
    a = np.array([[0, 0, 10, 10]])
    b = np.array([[0, 0, 10, 10], [5, 5, 15, 15], [20, 20, 30, 30]])
    iou = box_iou(a, b)[0]
    assert iou[0] == 1.0
    assert abs(iou[1] - 25 / 175) < 1e-9
    assert iou[2] == 0.0
    assert box_iou(np.zeros((0, 4)), b).shape == (0, 3)


def test_xywhn_to_xyxy_and_clip():
    out = xywhn_to_xyxy(np.array([[0.5, 0.5, 0.2, 0.4], [0.0, 0.0, 0.2, 0.2]]), 100, 50)
    np.testing.assert_allclose(out[0], [40, 15, 60, 35])
    np.testing.assert_allclose(out[1], [0, 0, 10, 5])  # clipped at image border


def test_label_path_uses_last_images_segment():
    p = image_to_label_path(Path("/data/images/train/images/a.jpg"))
    assert p == Path("/data/images/train/labels/a.txt")


def test_read_labels_reports_issues(tmp_path):
    f = tmp_path / "x.txt"
    f.write_text("0 0.5 0.5 0.1 0.1\n5 0.5 0.5 0.1 0.1\n1 0.5 0.5 0 0.1\n0 0.1 0.1 0.2 0.2 0.3 0.3 0.4 0.4\n"
                 "0 1.5 0.5 0.1 0.1\nfoo 1 2 3 4\n0 0.98 0.5 0.1 0.1\n\n")
    labels, issues = read_yolo_labels(f, num_classes=2)
    kinds = sorted(i.issue for i in issues)
    assert kinds == sorted(["invalid_class_id", "degenerate_box", "polygon_or_segment", "coords_out_of_range",
                            "non_numeric", "box_exceeds_image"])
    assert len(labels) == 2  # first line + the one that slightly exceeds the border


def test_missing_label_file_is_background(tmp_path):
    labels, issues = read_yolo_labels(tmp_path / "none.txt")
    assert labels.shape == (0, 5) and not issues
