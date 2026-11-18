# 3LC integration

This trains the same baseline (YOLO11n, `configs/baseline.yaml`, seed 42, identical
splits) through [3LC's Ultralytics integration](https://docs.3lc.ai/3lc/latest/user-guide/integrations/yolo/yolo.html).
3LC records per-sample predictions and image embeddings as Runs, which you can explore and
edit in the 3LC Dashboard. The VisionOps failure analysis can then be compared with what 3LC
surfaces.

**Status:** run completed (3lc 3.4.2, Ultralytics 8.4.189, RTX 2060, 50 epochs, seed 42).

| Model | Ultralytics | Test mAP@50 | Test mAP@50-95 |
|---|---|---|---|
| VisionOps baseline (`runs/baseline`) | 8.3.253 | 0.601 | 0.249 |
| 3LC run (`baseline-3lc`) | 8.4.189 | 0.583 | 0.205 |

These are **not** a like-for-like comparison. The integration installs a newer Ultralytics with
different training defaults, and each model is a single seed. 3LC is used here to *explore* the
data and errors, not as a competing model.

Findings from the dashboard (screenshots in `docs/images/3lc_*.png`):
- **Per-class F1 vs. confidence** (built-in *F1 (per class)* workflow): F1 peaks at low
  confidence thresholds (about 0.2–0.3) and drops sharply above about 0.5. This independently
  matches VisionOps' finding that the model is under-confident.
- **Find duplicates** workflow: embedding-similar training images form clusters of up to 51
  images, far more than Roboflow's 3 augmented copies per photo. Many photos appear to share board
  layouts. VisionOps splits by source photo; a split by board is the stricter next test.

## Steps (Windows)

```powershell
# 1. Separate environment, so the integration's Ultralytics version can't affect .venv
powershell -ExecutionPolicy Bypass -File .\integrations\threelc\setup_3lc.ps1

# 2. Sign in. Create a free account at https://3lc.ai, then copy your API key.
.\.venv-3lc\Scripts\3lc login <YOUR_3LC_API_KEY>

# 3. Train and collect metrics (about 30-40 min on an RTX 2060)
.\.venv-3lc\Scripts\python integrations\threelc\train_3lc.py

# 4. In a second terminal, start the local Object Service, then open https://dashboard.3lc.ai
.\.venv-3lc\Scripts\3lc service
```

Your data stays on your machine. The Object Service serves it to the dashboard locally.

## What to look for in the dashboard
- Per-sample metrics on train and val: sort by loss or by number of FN/FP and compare with
  `reports/review/review_queue.csv`.
- Embeddings plot: clusters of hard samples and near-duplicates.
- Label editing: fix suspected boxes, retrain on the new table revision, and compare on the
  same test split. This is the same controlled-experiment logic as `visionops.experiment`.
