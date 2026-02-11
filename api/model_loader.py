from functools import lru_cache
from pathlib import Path
import joblib

# project root: E:\credit risk modelling\
BASE_DIR = Path(__file__).resolve().parents[1]

# model artifact path (versioned)
MODEL_PATH = BASE_DIR / "artifacts" / "model_data_v1.joblib"


@lru_cache(maxsize=1)
def get_model_data():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
    return joblib.load(MODEL_PATH)
