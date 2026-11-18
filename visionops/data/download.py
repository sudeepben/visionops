"""Download the PCB defect dataset from Roboflow Universe (YOLO format).

Two supported paths:
  A) API:    set ROBOFLOW_API_KEY in .env, then  python -m visionops.data.download
  B) Manual: download the "YOLOv8" (or YOLOv11) ZIP export from the dataset page,
             then  python -m visionops.data.download --zip path/to/export.zip

Dataset: https://universe.roboflow.com/pcbdataset/pcb-defect-detection-9ewqw
License (as shown on the dataset page): CC BY 4.0 - attribution required.
"""
from __future__ import annotations

import argparse
import os
import sys
import zipfile
from pathlib import Path

from ..config import DATA_DIR, load_dotenv

RAW_DIR = DATA_DIR / "raw" / "roboflow"

MANUAL_HELP = """
No ROBOFLOW_API_KEY found. Two options:

 1) Get a free API key: sign in at https://app.roboflow.com -> Settings -> API Keys,
    copy .env.example to .env and set ROBOFLOW_API_KEY=..., then re-run this command.

 2) Manual download: open
    https://universe.roboflow.com/pcbdataset/pcb-defect-detection-9ewqw
    -> Dataset -> choose a version -> Download -> format "YOLOv8" -> "Download zip".
    Then run:  python -m visionops.data.download --zip <path-to-zip>
"""


def extract_zip(zip_path: Path, dest: Path) -> Path:
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        for m in zf.namelist():  # guard against path traversal
            target = (dest / m).resolve()
            if not str(target).startswith(str(dest.resolve())):
                raise ValueError(f"Unsafe path in zip: {m}")
        zf.extractall(dest)
    return dest


def download_roboflow(dest: Path) -> Path:
    api_key = os.getenv("ROBOFLOW_API_KEY", "").strip()
    if not api_key:
        print(MANUAL_HELP)
        sys.exit(2)
    try:
        from roboflow import Roboflow
    except ImportError:
        print("The 'roboflow' package is not installed: pip install roboflow")
        sys.exit(2)
    workspace = os.getenv("ROBOFLOW_WORKSPACE", "pcbdataset")
    project_id = os.getenv("ROBOFLOW_PROJECT", "pcb-defect-detection-9ewqw")
    version = int(os.getenv("ROBOFLOW_VERSION", "1"))
    fmt = os.getenv("ROBOFLOW_FORMAT", "yolov8")
    print(f"Downloading {workspace}/{project_id} v{version} ({fmt}) -> {dest}")
    rf = Roboflow(api_key=api_key)
    project = rf.workspace(workspace).project(project_id)
    ds = project.version(version).download(fmt, location=str(dest), overwrite=True)
    return Path(ds.location)


def main(argv=None) -> None:
    load_dotenv()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--zip", type=Path, help="Path to a manually downloaded Roboflow YOLO export zip")
    ap.add_argument("--dest", type=Path, default=RAW_DIR)
    args = ap.parse_args(argv)
    if args.zip:
        out = extract_zip(args.zip, args.dest)
    else:
        out = download_roboflow(args.dest)
    print(f"Raw dataset at: {out}")
    print("Next: python -m visionops.data.prepare")


if __name__ == "__main__":
    main()
