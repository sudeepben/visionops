"""Config loading, project paths and seeding."""
from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.getenv("VISIONOPS_DATA_DIR", ROOT / "data"))
RUNS_DIR = Path(os.getenv("VISIONOPS_RUNS_DIR", ROOT / "runs"))
REPORTS_DIR = Path(os.getenv("VISIONOPS_REPORTS_DIR", ROOT / "reports"))
DATASET_DIR = DATA_DIR / "pcb"          # normalised dataset root
SPLITS_DIR = DATASET_DIR / "splits"     # txt file lists (train/val/test)
DATASET_YAML = DATASET_DIR / "dataset.yaml"


def load_dotenv(path: Path | None = None) -> None:
    """Tiny .env loader so python-dotenv is not a hard dependency."""
    path = path or ROOT / ".env"
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        os.environ.setdefault(key.strip(), val.strip().strip('"').strip("'"))


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    return cfg


def seed_everything(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
