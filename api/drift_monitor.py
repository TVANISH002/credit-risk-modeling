import json
from pathlib import Path
from typing import Dict, Any
import pandas as pd


def save_baseline_stats(
    X_train_encoded: pd.DataFrame,
    out_path: str | Path = "artifacts/drift_baseline.json",
    min_std: float = 1e-8,
) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    stats = {}
    for col in X_train_encoded.columns:
        s = X_train_encoded[col].dropna()
        mu = float(s.mean())
        std = float(s.std())
        if std < min_std:
            std = float(min_std)
        stats[col] = {"mean": mu, "std": std}

    payload = {"schema_version": "1.0", "features": stats}
    out_path.write_text(json.dumps(payload, indent=2))
    return out_path


def load_baseline_stats(path: str | Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text())


def zscore_drift(
    X_new_encoded: pd.DataFrame,
    baseline: Dict[str, Any],
    z_threshold: float = 3.0,
) -> pd.DataFrame:
    features = baseline["features"]
    cols = list(features.keys())

    # ensure same columns as baseline
    X_new = X_new_encoded.reindex(columns=cols, fill_value=0)

    rows = []
    for col in cols:
        mu_train = float(features[col]["mean"])
        std_train = float(features[col]["std"])

        s_new = X_new[col].dropna()
        mu_new = float(s_new.mean()) if len(s_new) else 0.0

        z = (mu_new - mu_train) / std_train
        rows.append(
            {
                "feature": col,
                "train_mean": mu_train,
                "train_std": std_train,
                "new_mean": mu_new,
                "z_score": float(z),
                "drift_flag": bool(abs(z) >= z_threshold),
            }
        )

    df = pd.DataFrame(rows)
    df = df.sort_values("z_score", key=lambda s: s.abs(), ascending=False).reset_index(drop=True)
    return df
