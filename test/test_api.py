from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_predict_valid_payload():
    payload = {
        "age": 28,
        "income": 1200000,
        "loan_amount": 2560000,
        "loan_tenure_months": 36,
        "avg_dpd_per_delinquency": 20,
        "delinquency_ratio": 30,
        "credit_utilization_ratio": 30,
        "num_open_accounts": 2,
        "residence_type": "Owned",
        "loan_purpose": "Home",
        "loan_type": "Secured",
    }

    r = client.post("/predict", json=payload)
    assert r.status_code == 200

    data = r.json()
    assert "default_probability" in data
    assert 0.0 <= data["default_probability"] <= 1.0
    assert 300 <= data["credit_score"] <= 900
    assert data["rating"] in ["Poor", "Average", "Good", "Excellent", "Undefined"]


def test_predict_rejects_bad_age():
    payload = {
        "age": 10,  # invalid (must be >= 18)
        "income": 1200000,
        "loan_amount": 2560000,
        "loan_tenure_months": 36,
        "avg_dpd_per_delinquency": 20,
        "delinquency_ratio": 30,
        "credit_utilization_ratio": 30,
        "num_open_accounts": 2,
        "residence_type": "Owned",
        "loan_purpose": "Home",
        "loan_type": "Secured",
    }

    r = client.post("/predict", json=payload)
    assert r.status_code == 422
