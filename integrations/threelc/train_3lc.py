"""Train the VisionOps baseline through the 3LC Ultralytics integration.

Same model, config, seed and splits as runs/baseline, but training and validation
metrics (per-sample predictions, embeddings) are collected into 3LC Tables/Runs so
they can be explored in the 3LC Dashboard. Afterwards metrics are also collected
on the untouched test split.

Run from the repo root inside .venv-3lc (see integrations/threelc/README.md):
    python integrations/threelc/train_3lc.py
"""
from __future__ import annotations

import argparse
import dataclasses
import os
import sys
from pathlib import Path

os.environ.setdefault("PIN_MEMORY", "False")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from visionops.config import RUNS_DIR, load_config  # noqa: E402
from visionops.data.materialize import materialize  # noqa: E402


def make_settings(Settings, **wanted):
    """Pass only the fields this version of the integration knows about."""
    try:
        known = {f.name for f in dataclasses.fields(Settings)}
    except TypeError:
        known = set(wanted)
    skipped = sorted(set(wanted) - known)
    if skipped:
        print(f"[3lc] Settings fields not supported by this version, skipped: {skipped}")
    return Settings(**{k: v for k, v in wanted.items() if k in known})


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/baseline.yaml")
    ap.add_argument("--project", default="visionops-pcb")
    ap.add_argument("--run-name", default="baseline-3lc")
    ap.add_argument("--epochs", type=int)
    a = ap.parse_args(argv)

    from tlc_ultralytics import YOLO, Settings

    cfg = load_config(ROOT / a.config)
    data = str(materialize())
    settings = make_settings(
        Settings,
        project_name=a.project,
        run_name=a.run_name,
        run_description="VisionOps baseline (YOLO11n, seed 42) - same splits as runs/baseline",
        image_embeddings_dim=2,
        conf_thres=0.1,
    )
    model = YOLO(cfg["model"])
    model.train(data=data, epochs=a.epochs or cfg["epochs"], imgsz=cfg["imgsz"], batch=cfg["batch"],
                seed=cfg["seed"], deterministic=cfg.get("deterministic", True), workers=cfg.get("workers", 1),
                patience=cfg.get("patience", 20), project=str(RUNS_DIR / "3lc"), name=a.run_name, exist_ok=True,
                settings=settings, **(cfg.get("extra_train_args") or {}))
    try:
        model.val(data=data, split="test", imgsz=cfg["imgsz"], batch=8, workers=1, settings=settings)
        print("[3lc] collected metrics on the test split")
    except Exception as e:  # API differences between integration versions
        print(f"[3lc] test-split collection skipped: {e}")
    print("\nNext: run `3lc service` in another terminal, then open https://dashboard.3lc.ai "
          f"and look for project '{a.project}'.")


if __name__ == "__main__":
    main()
