from fastapi import FastAPI
from contextlib import asynccontextmanager
from datetime import datetime, timezone
import pandas as pd

from api.schemas import PredictRequest, PredictResponse
from api.predictor import predict
from api.model_loader import get_model_data

from api.db_sqlite import (
    init_db,
    insert_prediction,
    fetch_logs,
    get_prediction_count,
    fetch_prediction_inputs,
    insert_drift_report,
    fetch_drift_reports,
)

from api.drift_monitor import load_baseline_stats, zscore_drift


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield


app = FastAPI(title="Credit Risk API", version="1.0.0", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/model-info")
def model_info():
    md = get_model_data()
    return {
        "model_type": type(md["model"]).__name__,
        "scaler_type": type(md["scaler"]).__name__,
        "n_features": len(md["features"]),
        "artifact_path": "artifacts/model_data_v1.joblib",
        "cols_scaled": list(md["cols_to_scale"]),
        "model_version": "v1",
    }


@app.post("/predict", response_model=PredictResponse)
def predict_endpoint(req: PredictRequest):
    payload = req.model_dump()
    p, score, rating = predict(payload)

    ts = datetime.now(timezone.utc).isoformat()

    row_id = insert_prediction(
        created_at=ts,
        payload=payload,
        default_probability=float(p),
        credit_score=int(score),
        rating=str(rating),
    )
    
    

    # ✅ AUTO DRIFT CHECK every 100 predictions
    total = get_prediction_count()
    if total % 100 == 0 and total >= 100:
        try:
            baseline = load_baseline_stats("artifacts/drift_baseline.json")
            inputs = fetch_prediction_inputs(limit=100)

            if inputs:
                df_new = pd.DataFrame(inputs)

                # IMPORTANT: Make sure your payload keys match training features logic
                # We will align columns to baseline schema after encoding
                df_new_encoded = pd.get_dummies(df_new)

                drift_df = zscore_drift(df_new_encoded, baseline, z_threshold=3.0)
                drifted_count = int(drift_df["drift_flag"].sum())

                report_payload = {
                    "summary": {
                        "total_features": int(drift_df.shape[0]),
                        "drifted_features": drifted_count,
                        "z_threshold": 3.0,
                    },
                    "top_20": drift_df.head(20).to_dict(orient="records"),
                }

                insert_drift_report(
                    created_at=ts,
                    model_version="v1",
                    z_threshold=3.0,
                    drifted_features_count=drifted_count,
                    report=report_payload,
                )

                print(f"✅ Drift check saved at prediction #{total}")

        except Exception as e:
            print("❌ Drift check failed:", e)

    return {
        "prediction_id": f"sqlite-{row_id}",
        "default_probability": float(p),
        "credit_score": int(score),
        "rating": str(rating),
        "timestamp": ts,
    }


@app.get("/logs")
def logs(limit: int = 20):
    return fetch_logs(limit=limit)

@app.get("/drift-reports")
def drift_reports(limit: int = 5):
    return fetch_drift_reports(limit=limit)



