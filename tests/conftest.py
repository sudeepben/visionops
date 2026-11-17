"""Point all VisionOps paths at a temp dir BEFORE the package is imported."""
import os
import sys
import tempfile
from pathlib import Path

_TMP = Path(tempfile.mkdtemp(prefix="visionops_test_"))
os.environ["VISIONOPS_DATA_DIR"] = str(_TMP / "data")
os.environ["VISIONOPS_RUNS_DIR"] = str(_TMP / "runs")
os.environ["VISIONOPS_REPORTS_DIR"] = str(_TMP / "reports")
os.environ["VISIONOPS_MLFLOW"] = "0"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pytest  # noqa: E402
from PIL import Image  # noqa: E402


def make_image(path: Path, w=200, h=100, seed=0):
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.random.default_rng(seed).integers(0, 255, (h, w, 3), dtype=np.uint8)
    Image.fromarray(arr).save(path)


def write_label(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(" ".join(str(v) for v in r) for r in rows) + ("\n" if rows else ""))


@pytest.fixture(scope="session")
def tmp_root():
    return _TMP


@pytest.fixture
def roboflow_export(tmp_path):
    """Fake Roboflow YOLO export with train + valid only (no test split), incl. augmented copies."""
    root = tmp_path / "raw" / "export"
    (root).mkdir(parents=True)
    (root / "data.yaml").write_text("nc: 2\nnames: ['missing_hole', 'short']\ntrain: ../train/images\nval: ../valid/images\n")
    k = 0
    for split, n in (("train", 20), ("valid", 4)):
        for i in range(n):
            for aug in (["a1b2", "c3d4"] if split == "train" else ["e5f6"]):
                name = f"{split}{i:03d}_jpg.rf.{aug}.jpg"
                make_image(root / split / "images" / name, seed=k)
                write_label(root / split / "labels" / name.replace(".jpg", ".txt"), [[i % 2, 0.5, 0.5, 0.2, 0.2]])
                k += 1
    return tmp_path / "raw"
