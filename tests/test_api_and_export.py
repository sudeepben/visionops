import io
import os

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from visionops.export_onnx import compare_predictions


def test_compare_predictions_identical_and_shifted():
    a = pd.DataFrame([["i1", 0, 0.9, 0, 0, 10, 10], ["i1", 1, 0.8, 20, 20, 30, 30], ["i2", 0, 0.7, 5, 5, 9, 9]],
                     columns=["image", "cls", "conf", "x1", "y1", "x2", "y2"])
    r = compare_predictions(a, a.copy())
    assert r["matched_boxes"] == 3 and r["match_rate"] == 1.0 and r["max_abs_conf_diff"] == 0
    b = a.copy()
    b.loc[1, "cls"] = 0  # class changed -> must not match
    r2 = compare_predictions(a, b)
    assert r2["matched_boxes"] == 2


class FakeDetector:
    weights = "fake.pt"
    names = ["missing_hole", "short"]

    def predict_image(self, img, conf=0.25, iou=0.7):
        return [{"class_id": 0, "class_name": "missing_hole", "confidence": 0.9, "box_xyxy": [1.0, 2.0, 3.0, 4.0]}]


def _png():
    buf = io.BytesIO()
    Image.fromarray(np.zeros((32, 48, 3), np.uint8)).save(buf, format="PNG")
    return buf.getvalue()


def test_api_predict_and_errors():
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    from api.main import create_app

    c = TestClient(create_app(FakeDetector()))
    assert c.get("/health").json()["status"] == "ok"
    r = c.post("/predict", files={"file": ("a.png", _png(), "image/png")})
    assert r.status_code == 200
    body = r.json()
    assert body["width"] == 48 and body["height"] == 32 and body["num_detections"] == 1
    assert c.post("/predict", files={"file": ("a.png", b"not an image", "image/png")}).status_code == 400
    assert c.post("/predict?conf=2", files={"file": ("a.png", _png(), "image/png")}).status_code == 422


def test_api_without_model_returns_503(monkeypatch):
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    from api.main import create_app

    monkeypatch.setenv("VISIONOPS_MODEL", os.path.join("does", "not", "exist.pt"))
    c = TestClient(create_app())
    assert c.get("/health").json()["status"] == "no_model"
    assert c.post("/predict", files={"file": ("a.png", _png(), "image/png")}).status_code == 503
