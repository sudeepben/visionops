"""Minimal inference API.

    uvicorn api.main:app --port 8000
    curl -F "file=@board.jpg" "http://localhost:8000/predict?conf=0.25"

Model path: VISIONOPS_MODEL env var, else runs/curated|baseline/weights/best.pt.
"""
from __future__ import annotations

import os
import time

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, Query, UploadFile

from visionops.config import load_dotenv

load_dotenv()

MAX_BYTES = int(os.getenv("VISIONOPS_MAX_UPLOAD_MB", "15")) * 1024 * 1024


def create_app(detector=None) -> FastAPI:
    app = FastAPI(title="VisionOps inference API", version="0.1.0")
    app.state.detector = detector
    app.state.load_error = None

    def get_detector():
        if app.state.detector is None:
            from visionops.inference import Detector, default_weights

            w = default_weights()
            if w is None or not w.exists():
                app.state.load_error = "No model weights found. Train a model or set VISIONOPS_MODEL."
                return None
            try:
                app.state.detector = Detector(w, imgsz=int(os.getenv("VISIONOPS_IMGSZ", "640")))
            except Exception as e:  # pragma: no cover - depends on local install
                app.state.load_error = f"Failed to load model: {e}"
                return None
        return app.state.detector

    @app.get("/health")
    def health():
        det = get_detector()
        return {"status": "ok" if det else "no_model", "model": getattr(det, "weights", None),
                "classes": getattr(det, "names", None), "detail": app.state.load_error if not det else None}

    @app.post("/predict")
    async def predict(file: UploadFile = File(...), conf: float = Query(0.25, ge=0.0, le=1.0),
                      iou: float = Query(0.7, ge=0.0, le=1.0)):
        det = get_detector()
        if det is None:
            raise HTTPException(503, app.state.load_error or "Model not loaded")
        data = await file.read()
        if len(data) > MAX_BYTES:
            raise HTTPException(413, "File too large")
        img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise HTTPException(400, "Could not decode image")
        t0 = time.perf_counter()
        dets = det.predict_image(img, conf=conf, iou=iou)
        return {"filename": file.filename, "width": img.shape[1], "height": img.shape[0],
                "model": det.weights, "inference_ms": round((time.perf_counter() - t0) * 1000, 2),
                "num_detections": len(dets), "detections": dets}

    return app


app = create_app()
