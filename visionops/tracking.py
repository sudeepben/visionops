"""Optional MLflow tracking. Every call is a silent no-op if MLflow is missing,
disabled (VISIONOPS_MLFLOW=0) or failing - the pipeline never depends on it.
Local JSON reports under reports/ are always the source of truth."""
from __future__ import annotations

import os
from contextlib import contextmanager


def _enabled() -> bool:
    if os.getenv("VISIONOPS_MLFLOW", "1") == "0":
        return False
    try:
        import mlflow  # noqa: F401

        return True
    except ImportError:
        return False


@contextmanager
def track(run_name: str, experiment: str = "visionops"):
    if not _enabled():
        yield None
        return
    import mlflow

    try:
        mlflow.set_experiment(experiment)
        run = mlflow.start_run(run_name=run_name)
    except Exception as e:  # tracking server down, bad URI, etc.
        print(f"[tracking] MLflow unavailable ({e}); continuing without it")
        yield None
        return
    try:
        yield run
    finally:
        try:
            mlflow.end_run()
        except Exception:
            pass


def log_params(params: dict) -> None:
    if not _enabled():
        return
    try:
        import mlflow

        if mlflow.active_run():
            mlflow.log_params({k: str(v)[:250] for k, v in params.items()})
    except Exception as e:
        print(f"[tracking] log_params skipped: {e}")


def log_metrics(metrics: dict, prefix: str = "") -> None:
    if not _enabled():
        return
    try:
        import mlflow

        if mlflow.active_run():
            clean = {}
            for k, v in metrics.items():
                if isinstance(v, (int, float)):
                    clean[(prefix + k).replace("(", "_").replace(")", "").replace("/", "_")] = float(v)
            mlflow.log_metrics(clean)
    except Exception as e:
        print(f"[tracking] log_metrics skipped: {e}")


def log_artifact(path: str) -> None:
    if not _enabled():
        return
    try:
        import mlflow

        if mlflow.active_run():
            mlflow.log_artifact(path)
    except Exception as e:
        print(f"[tracking] log_artifact skipped: {e}")
