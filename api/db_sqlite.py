import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, List

# -----------------------------------
# Database Location
# -----------------------------------
BASE_DIR = Path(__file__).resolve().parents[1]
DB_PATH = BASE_DIR / "data" / "predictions.db"


# -----------------------------------
# Connection Helper
# -----------------------------------
def get_conn() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


# -----------------------------------
# Initialize Tables
# -----------------------------------
def init_db() -> None:
    conn = get_conn()
    try:
        # Predictions table
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                input_json TEXT NOT NULL,
                default_probability REAL NOT NULL,
                credit_score INTEGER NOT NULL,
                rating TEXT NOT NULL
            )
            """
        )

        # Drift reports table
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS drift_reports (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                model_version TEXT NOT NULL,
                z_threshold REAL NOT NULL,
                drifted_features_count INTEGER NOT NULL,
                report_json TEXT NOT NULL
            )
            """
        )

        conn.commit()
    finally:
        conn.close()


# -----------------------------------
# Prediction Logging
# -----------------------------------
def insert_prediction(
    created_at: str,
    payload: Dict[str, Any],
    default_probability: float,
    credit_score: int,
    rating: str,
) -> int:
    conn = get_conn()
    try:
        cur = conn.execute(
            """
            INSERT INTO predictions 
            (created_at, input_json, default_probability, credit_score, rating)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                created_at,
                json.dumps(payload),
                float(default_probability),
                int(credit_score),
                str(rating),
            ),
        )
        conn.commit()
        return int(cur.lastrowid)
    finally:
        conn.close()


def fetch_logs(limit: int = 20) -> List[Dict[str, Any]]:
    conn = get_conn()
    try:
        cur = conn.execute(
            """
            SELECT id, created_at, default_probability, credit_score, rating
            FROM predictions
            ORDER BY id DESC
            LIMIT ?
            """,
            (int(limit),),
        )
        rows = cur.fetchall()
        return [
            {
                "prediction_id": f"sqlite-{row['id']}",
                "timestamp": row["created_at"],
                "default_probability": row["default_probability"],
                "credit_score": row["credit_score"],
                "rating": row["rating"],
            }
            for row in rows
        ]
    finally:
        conn.close()


def get_prediction_count() -> int:
    conn = get_conn()
    try:
        cur = conn.execute("SELECT COUNT(*) as cnt FROM predictions")
        row = cur.fetchone()
        return int(row["cnt"])
    finally:
        conn.close()


def fetch_prediction_inputs(limit: int = 100) -> List[Dict[str, Any]]:
    conn = get_conn()
    try:
        cur = conn.execute(
            """
            SELECT input_json
            FROM predictions
            ORDER BY id DESC
            LIMIT ?
            """,
            (int(limit),),
        )
        rows = cur.fetchall()
        return [json.loads(row["input_json"]) for row in rows]
    finally:
        conn.close()


# -----------------------------------
# Drift Reports
# -----------------------------------
def insert_drift_report(
    created_at: str,
    model_version: str,
    z_threshold: float,
    drifted_features_count: int,
    report: Dict[str, Any],
) -> int:
    conn = get_conn()
    try:
        cur = conn.execute(
            """
            INSERT INTO drift_reports
            (created_at, model_version, z_threshold, drifted_features_count, report_json)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                created_at,
                str(model_version),
                float(z_threshold),
                int(drifted_features_count),
                json.dumps(report),
            ),
        )
        conn.commit()
        return int(cur.lastrowid)
    finally:
        conn.close()


def fetch_drift_reports(limit: int = 10) -> List[Dict[str, Any]]:
    conn = get_conn()
    try:
        cur = conn.execute(
            """
            SELECT id, created_at, model_version, z_threshold, drifted_features_count, report_json
            FROM drift_reports
            ORDER BY id DESC
            LIMIT ?
            """,
            (int(limit),),
        )
        rows = cur.fetchall()

        return [
            {
                "drift_id": f"sqlite-drift-{row['id']}",
                "timestamp": row["created_at"],
                "model_version": row["model_version"],
                "z_threshold": row["z_threshold"],
                "drifted_features_count": row["drifted_features_count"],
                "report": json.loads(row["report_json"]),
            }
            for row in rows
        ]
    finally:
        conn.close()
