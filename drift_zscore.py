
# Z-Score Drift Detection (Production-friendly baseline)


from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd

try:
    import mlflow
    MLFLOW_AVAILABLE = True
except Exception:
    MLFLOW_AVAILABLE = False


# 1) SAVE BASELINE STATS

def save_baseline_stats(
    X_train: pd.DataFrame,
    baseline_path: str | Path = "artifacts/drift_baseline.json",
    numeric_only: bool = True,
    min_std: float = 1e-8
) -> Path:
    """
    Save train mean/std for drift monitoring.
    Use this AFTER you finalize X_train_encoded (same columns as prod batch).
    """
    baseline_path = Path(baseline_path)
    baseline_path.parent.mkdir(parents=True, exist_ok=True)

    if numeric_only:
        cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    else:
        cols = X_train.columns.tolist()

    stats = {}
    for col in cols:
        s = X_train[col].dropna()
        mu = float(s.mean())
        std = float(s.std())
        if std < min_std:
            std = float(min_std)  # avoid divide-by-zero in monitoring
        stats[col] = {"mean": mu, "std": std}

    payload = {
        "schema_version": "1.0",
        "n_features": len(stats),
        "features": stats,
    }

    baseline_path.write_text(json.dumps(payload, indent=2))
    return baseline_path


# 2) LOAD BASELINE STATS

def load_baseline_stats(baseline_path: str | Path) -> dict:
    baseline_path = Path(baseline_path)
    return json.loads(baseline_path.read_text())

# 3) DRIFT REPORT (Z-SCORE)

def zscore_drift_report(
    X_new: pd.DataFrame,
    baseline: dict,
    z_threshold: float = 3.0
) -> pd.DataFrame:
    """
    Compare new batch mean vs training baseline mean using z-score:
        z = (mu_new - mu_train) / std_train

    Flags drift if abs(z) >= z_threshold
    """
    features = baseline["features"]

    # Ensure X_new has ALL baseline columns (missing -> fill 0)
    baseline_cols = list(features.keys())
    X_new = X_new.reindex(columns=baseline_cols, fill_value=0)

    rows = []
    for col in baseline_cols:
        mu_train = features[col]["mean"]
        std_train = features[col]["std"]

        s_new = X_new[col].dropna()
        mu_new = float(s_new.mean()) if len(s_new) else 0.0

        z = (mu_new - mu_train) / std_train
        drift_flag = abs(z) >= z_threshold

        rows.append({
            "feature": col,
            "train_mean": mu_train,
            "train_std": std_train,
            "new_mean": mu_new,
            "z_score": z,
            "drift_flag": drift_flag
        })

    df = pd.DataFrame(rows)
    df = df.sort_values("z_score", key=lambda s: s.abs(), ascending=False).reset_index(drop=True)
    return df


# 4) SAVE REPORT + OPTIONAL MLFLOW LOGGING

def save_report(
    report_df: pd.DataFrame,
    out_path: str | Path = "artifacts/zscore_drift_report.csv"
) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    report_df.to_csv(out_path, index=False)
    return out_path


def log_report_to_mlflow(report_path: str | Path, artifact_path: str = "drift"):
    """
    Logs drift report CSV to MLflow. Works only if mlflow installed and a run is active.
    """
    if not MLFLOW_AVAILABLE:
        print("⚠️ mlflow not installed; skipping MLflow logging.")
        return
    mlflow.log_artifact(str(report_path), artifact_path=artifact_path)

