# 3LC integration

This trains the same baseline (YOLO11n, `configs/baseline.yaml`, seed 42, identical
splits) through [3LC's Ultralytics integration](https://docs.3lc.ai/3lc/latest/user-guide/integrations/yolo/yolo.html).
3LC records per-sample predictions and image embeddings as Runs, which you can explore and
edit in the 3LC Dashboard. The VisionOps failure analysis can then be compared with what 3LC
surfaces.

**Status:** the scripts are written. Results and findings will be added here after the run.
Nothing in this section has been run yet.

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
