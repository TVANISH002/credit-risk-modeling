import pandas as pd
from api.model_loader import get_model_data


def prepare_input(payload: dict) -> pd.DataFrame:
    md = get_model_data()
    scaler = md["scaler"]
    features = md["features"]
    cols_to_scale = md["cols_to_scale"]

    income = float(payload["income"])
    loan_amount = float(payload["loan_amount"])

    data = {
        "age": float(payload["age"]),
        "loan_tenure_months": float(payload["loan_tenure_months"]),
        "number_of_open_accounts": float(payload["num_open_accounts"]),
        "credit_utilization_ratio": float(payload["credit_utilization_ratio"]),
        "loan_to_income": (loan_amount / income) if income > 0 else 0.0,
        "delinquency_ratio": float(payload["delinquency_ratio"]),
        "avg_dpd_per_delinquency": float(payload["avg_dpd_per_delinquency"]),

        # ✅ IMPORTANT: use int(…)
        "residence_type_Owned": int(payload["residence_type"] == "Owned"),
        "residence_type_Rented": int(payload["residence_type"] == "Rented"),
        "loan_purpose_Education": int(payload["loan_purpose"] == "Education"),
        "loan_purpose_Home": int(payload["loan_purpose"] == "Home"),
        "loan_purpose_Personal": int(payload["loan_purpose"] == "Personal"),
        "loan_type_Unsecured": int(payload["loan_type"] == "Unsecured"),
    }

    df = pd.DataFrame([data])

    # ✅ Make sure scaler gets exactly what it expects (columns + numeric dtype)
    for c in cols_to_scale:
        if c not in df.columns:
            df[c] = 0.0

    df[cols_to_scale] = scaler.transform(df[cols_to_scale])

    # ✅ Align to training features & force numeric dtype
    df = df.reindex(columns=features, fill_value=0)
    df = df.astype(float)

    return df


def predict(payload: dict):
    md = get_model_data()
    model = md["model"]

    X = prepare_input(payload)

    # ✅ Use sklearn directly (no manual np.exp)
    if hasattr(model, "predict_proba"):
        p_default = float(model.predict_proba(X)[0, 1])
    else:
        # fallback (rare)
        p_default = float(model.predict(X)[0])

    credit_score = int(300 + (1 - p_default) * 600)

    if credit_score < 500:
        rating = "Poor"
    elif credit_score < 650:
        rating = "Average"
    elif credit_score < 750:
        rating = "Good"
    else:
        rating = "Excellent"

    return p_default, credit_score, rating
