"""Controlled data-curation experiment.

Question: does removing the training images flagged by the review queue help,
compared with removing the SAME number of random images?

Variants (identical config, seed, val set and untouched test set; only the train list differs):
  baseline        full training split
  curated         training split minus flagged images (whole source groups, so
                  Roboflow-augmented copies of a flagged photo are removed too)
  random_control  training split minus an equal number of random source groups

Flag source:
  --source queue    top-ranked images with a *possible annotation issue* (automatic)
  --source reviews  images a human marked 'label_issue' in the dashboard (reports/review/decisions.csv)

    python -m visionops.experiment build --frac 0.05
    python -m visionops.experiment run  [--seeds 42 43 44] [--epochs N]
    python -m visionops.experiment compare
"""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import pandas as pd

from .config import DATASET_DIR, REPORTS_DIR, ROOT, RUNS_DIR, load_config
from .data.prepare import load_names, read_split, source_group, write_dataset_yaml

EXP_DIR = DATASET_DIR / "experiments"
OUT_DIR = REPORTS_DIR / "experiments"
VARIANTS = ("baseline", "curated", "random_control")


def flagged_images(source: str, queue_csv: Path, decisions_csv: Path) -> list[str]:
    if source == "reviews":
        if not decisions_csv.exists():
            raise FileNotFoundError(f"{decisions_csv} not found - review images in the dashboard first")
        d = pd.read_csv(decisions_csv).drop_duplicates("image", keep="last")
        return d.loc[d["decision"] == "label_issue", "image"].tolist()
    q = pd.read_csv(queue_csv)
    q = q[q["possible_annotation_issue"].astype(bool)].sort_values("score", ascending=False)
    return q["image"].tolist()


def build_variants(frac: float = 0.05, seed: int = 42, source: str = "queue",
                   queue_csv: Path = REPORTS_DIR / "review" / "review_queue.csv",
                   decisions_csv: Path = REPORTS_DIR / "review" / "decisions.csv") -> dict:
    train = [str(p) for p in read_split("train")]
    names = load_names(DATASET_DIR / "dataset.yaml")
    groups: dict[str, list[str]] = defaultdict(list)
    for p in train:
        groups[source_group(p)].append(p)
    budget = max(1, round(frac * len(train))) if source == "queue" else None

    removed_groups, removed = [], set()
    for img in flagged_images(source, queue_csv, decisions_csv):
        g = source_group(img)
        if g in removed_groups or g not in groups:
            continue
        if budget is not None and len(removed) >= budget:
            break
        removed_groups.append(g)
        removed.update(groups[g])
    if not removed:
        raise SystemExit("No flagged images to remove - nothing to compare.")

    keys = sorted(groups)
    random.Random(seed).shuffle(keys)
    rand_removed = set()
    for g in keys:
        if len(rand_removed) >= len(removed):
            break
        rand_removed.update(groups[g])

    manifest = {"source": source, "frac": frac, "seed": seed, "train_images": len(train), "variants": {}}
    for name, drop in (("curated", removed), ("random_control", rand_removed)):
        d = EXP_DIR / name
        (d / "splits").mkdir(parents=True, exist_ok=True)
        keep = [p for p in train if p not in drop]
        (d / "splits" / "train.txt").write_text("\n".join(keep) + "\n", encoding="utf-8")
        # val/test lists are the SAME files as the baseline (never copied or modified)
        yaml_path = write_dataset_yaml(d / "dataset.yaml", DATASET_DIR, names, d / "splits" / "train.txt")
        manifest["variants"][name] = {"dataset": str(yaml_path), "train_images": len(keep),
                                      "removed_images": len(drop), "removed": sorted(drop)}
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"curated: -{len(removed)} images | random_control: -{len(rand_removed)} images "
          f"(of {len(train)}). Manifest: {OUT_DIR / 'manifest.json'}")
    return manifest


def run(config: str, seeds: list[int], epochs: int | None = None) -> None:
    from .evaluate import evaluate
    from .train import train

    cfg = load_config(ROOT / config)
    base_seed = int(cfg.get("seed", 42))
    manifest = json.loads((OUT_DIR / "manifest.json").read_text(encoding="utf-8"))
    for seed in seeds:
        cfg_s = {**cfg, "seed": seed}
        for v in VARIANTS:
            tag = ("" if seed == base_seed else f"_s{seed}") + (f"_e{epochs}" if epochs else "")
            run_name = v + tag
            weights = RUNS_DIR / run_name / "weights" / "best.pt"
            if not weights.exists():
                data = cfg["data"] if v == "baseline" else manifest["variants"][v]["dataset"]
                weights = Path(train(cfg_s, name=run_name, data=data, epochs=epochs)["best_weights"])
            if not (REPORTS_DIR / run_name / "metrics.json").exists():
                evaluate(str(weights), run_name, cfg, split="test")
    compare()


def compare() -> pd.DataFrame:
    manifest = json.loads((OUT_DIR / "manifest.json").read_text(encoding="utf-8"))
    rows = []
    for mpath in sorted(REPORTS_DIR.glob("*/metrics.json")):
        name = mpath.parent.name
        variant = next((v for v in VARIANTS if name == v or name.startswith(v + "_")), None)
        if variant is None:
            continue
        m = json.loads(mpath.read_text(encoding="utf-8"))
        u, op = m.get("ultralytics", {}), m.get("operating_point", {})
        info = manifest["variants"].get(variant, {})
        run_info = REPORTS_DIR / "runs" / f"{name}.json"
        epochs = json.loads(run_info.read_text(encoding="utf-8"))["train_args"]["epochs"] if run_info.exists() else None
        rows.append({"run": name, "variant": variant, "epochs": epochs,
                     "train_images": info.get("train_images", manifest["train_images"]),
                     "removed_images": info.get("removed_images", 0),
                     "test_map50": u.get("map50"), "test_map50_95": u.get("map50_95"),
                     "test_precision": u.get("precision"), "test_recall": u.get("recall"),
                     "op_precision": op.get("precision"), "op_recall": op.get("recall"), "op_f1": op.get("f1")})
    df = pd.DataFrame(rows)
    if df.empty:
        print("No experiment results yet.")
        return df
    summary = df.groupby(["variant", "epochs"], dropna=False)[["test_map50", "test_map50_95", "op_f1"]].agg(["mean", "std", "count"])
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_DIR / "comparison.csv", index=False)
    summary.to_csv(OUT_DIR / "summary.csv")
    print(df.to_string(index=False))
    n = int(df.groupby(["variant", "epochs"], dropna=False).size().min())
    if n < 2:
        print("\nNOTE: one seed per variant - small differences may be run-to-run noise. Use --seeds for a fairer test.")
    return df


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--frac", type=float, default=0.05, help="max fraction of train images to remove (queue source)")
    b.add_argument("--seed", type=int, default=42)
    b.add_argument("--source", choices=["queue", "reviews"], default="queue")
    r = sub.add_parser("run")
    r.add_argument("--config", default="configs/baseline.yaml")
    r.add_argument("--seeds", type=int, nargs="+", default=[42])
    r.add_argument("--epochs", type=int)
    sub.add_parser("compare")
    a = ap.parse_args(argv)
    if a.cmd == "build":
        build_variants(a.frac, a.seed, a.source)
    elif a.cmd == "run":
        run(a.config, a.seeds, a.epochs)
    else:
        compare()


if __name__ == "__main__":
    main()
