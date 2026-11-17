"""Seeded fine-tuning of a small pretrained YOLO model.

    python -m visionops.train --config configs/baseline.yaml
    python -m visionops.train --config configs/baseline.yaml --name curated --data data/pcb/experiments/curated/dataset.yaml
"""
from __future__ import annotations

import argparse
import json
import platform
import subprocess
import time
from pathlib import Path

import os

# Pinned (page-locked) host memory failed with "CUDA error: out of memory" on a low-RAM Windows PC.
# Ultralytics reads this env var when building dataloaders; set PIN_MEMORY=True to re-enable.
os.environ.setdefault("PIN_MEMORY", "False")

from .config import REPORTS_DIR, ROOT, RUNS_DIR, load_config, seed_everything
from .tracking import log_metrics, log_params, track


def git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                                       stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return None


def env_info() -> dict:
    info = {"python": platform.python_version(), "platform": platform.platform()}
    try:
        import torch

        info["torch"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
    except ImportError:
        pass
    try:
        import ultralytics

        info["ultralytics"] = ultralytics.__version__
    except ImportError:
        pass
    return info


def train(cfg: dict, name: str | None = None, data: str | None = None, epochs: int | None = None) -> dict:
    from ultralytics import YOLO, settings

    try:  # we do our own (optional) MLflow logging; avoid Ultralytics' built-in callback
        settings.update({"mlflow": False})
    except Exception:
        pass

    name = name or cfg.get("name", "baseline")
    data = str((ROOT / (data or cfg["data"])).resolve())
    epochs = epochs or cfg["epochs"]
    seed = int(cfg.get("seed", 42))
    seed_everything(seed)

    args = dict(
        data=data, epochs=epochs, imgsz=cfg["imgsz"], batch=cfg["batch"], seed=seed,
        deterministic=cfg.get("deterministic", True), patience=cfg.get("patience", 20),
        workers=cfg.get("workers", 4), project=str(RUNS_DIR.resolve()), name=name, exist_ok=True,
        plots=True, verbose=True,
    )
    if cfg.get("device") is not None:
        args["device"] = cfg["device"]
    args.update(cfg.get("extra_train_args") or {})

    t0 = time.time()
    with track(run_name=name):
        log_params({"model": cfg["model"], **{k: v for k, v in args.items() if k != "project"}})
        model = YOLO(cfg["model"])
        results = model.train(**args)
        save_dir = Path(model.trainer.save_dir)
        best = Path(model.trainer.best)
        val_metrics = dict(getattr(results, "results_dict", {}) or {})
        log_metrics(val_metrics, prefix="val_")

    info = {
        "name": name, "model": cfg["model"], "data": data, "train_args": {k: v for k, v in args.items()},
        "save_dir": str(save_dir), "best_weights": str(best), "train_minutes": round((time.time() - t0) / 60, 2),
        "val_metrics_best_epoch": val_metrics, "git_commit": git_commit(), "env": env_info(),
    }
    out = REPORTS_DIR / "runs" / f"{name}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(info, indent=2, default=str), encoding="utf-8")
    print(f"Best weights: {best}\nRun info: {out}")
    return info


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/baseline.yaml")
    ap.add_argument("--name")
    ap.add_argument("--data", help="Override dataset yaml (used by curation experiments)")
    ap.add_argument("--epochs", type=int, help="Override epochs (e.g. 1 for a smoke test)")
    a = ap.parse_args(argv)
    train(load_config(ROOT / a.config if not Path(a.config).is_absolute() else a.config), a.name, a.data, a.epochs)


if __name__ == "__main__":  # required on Windows (dataloader workers use spawn)
    main()
