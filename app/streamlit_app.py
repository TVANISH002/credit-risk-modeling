import streamlit as st
import requests
import pandas as pd

# -----------------------------
# Config
# -----------------------------
API_URL = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="Finance — Credit Risk Modelling",
    page_icon="📊",
    layout="wide"
)

# -----------------------------
# API helpers
# -----------------------------
def api_get(path, params=None):
    r = requests.get(f"{API_URL}{path}", params=params, timeout=10)
    r.raise_for_status()
    return r.json()

def api_post(path, payload):
    r = requests.post(f"{API_URL}{path}", json=payload, timeout=15)
    r.raise_for_status()
    return r.json()

# -----------------------------
# Styling
# -----------------------------
st.markdown(
    """
    <style>
      .block-container {padding-top: 1.2rem; padding-bottom: 2rem;}
      .card {padding: 1rem; border: 1px solid #e5e7eb; border-radius: 14px; background: #0f172a;}
      .pill {display:inline-block; padding: 0.25rem 0.65rem; border-radius: 999px;
             font-size: 0.85rem; border: 1px solid #334155;}
      .low {background:#064e3b;}
      .mid {background:#78350f;}
      .high {background:#7f1d1d;}
      .muted {color:#94a3b8;}
    </style>
    """,
    unsafe_allow_html=True
)

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.header("⚙️ Model Panel")

api_ok = True
try:
    api_get("/health")
    st.sidebar.success("FastAPI connected ✅")
except Exception:
    api_ok = False
    st.sidebar.error("FastAPI not reachable ❌")
    st.sidebar.caption("Run: uvicorn api.main:app --reload --port 8000")

if api_ok:
    try:
        info = api_get("/model-info")
        st.sidebar.markdown(f"**Model:** `{info['model_type']}`")
        st.sidebar.markdown(f"**Scaler:** `{info['scaler_type']}`")
        st.sidebar.markdown(f"**# Features:** `{info['n_features']}`")
        with st.sidebar.expander("Scaled Columns"):
            st.write(info["cols_scaled"])
    except Exception:
        st.sidebar.warning("/model-info not available")

threshold = st.sidebar.slider("Decision threshold", 0.05, 0.95, 0.50, 0.05)

with st.sidebar.expander("What is threshold?"):
    st.write(
        "If **Default Probability ≥ threshold**, the applicant is flagged as **High Risk**.\n\n"
        "- Lower threshold → stricter\n"
        "- Higher threshold → lenient"
    )

# -----------------------------
# Header
# -----------------------------
st.title("📊 Finance — Credit Risk Modelling")
st.caption("Production-style API inference with persistent prediction logs")

left, right = st.columns([1.35, 1.0], gap="large")

# =============================
# LEFT: Inputs
# =============================
with left:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("🧾 Applicant Inputs")

    r1 = st.columns(3)
    r2 = st.columns(3)
    r3 = st.columns(3)
    r4 = st.columns(3)

    with r1[0]:
        age = st.number_input("Age", 18, 100, 28)
    with r1[1]:
        income = st.number_input("Income", 0, value=1_200_000)
    with r1[2]:
        loan_amount = st.number_input("Loan Amount", 0, value=2_560_000)

    lti = loan_amount / income if income > 0 else 0.0
    with r2[0]:
        st.markdown("**Loan-to-Income Ratio**")
        st.metric("", f"{lti:.2f}")

    with r2[1]:
        loan_tenure_months = st.number_input("Loan Tenure (months)", 0, value=36)
    with r2[2]:
        avg_dpd = st.number_input("Avg DPD per Delinquency", 0, value=20)

    with r3[0]:
        delinquency_ratio = st.number_input("Delinquency Ratio (%)", 0, 100, 30)
    with r3[1]:
        credit_util = st.number_input("Credit Utilization Ratio (%)", 0, 100, 30)
    with r3[2]:
        open_accounts = st.number_input("Open Loan Accounts", 1, 20, 2)

    with r4[0]:
        residence = st.selectbox("Residence Type", ["Owned", "Rented", "Mortgage"])
    with r4[1]:
        purpose = st.selectbox("Loan Purpose", ["Education", "Home", "Auto", "Personal"])
    with r4[2]:
        loan_type = st.selectbox("Loan Type", ["Unsecured", "Secured"])

    calculate = st.button("✅ Calculate Risk", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

# =============================
# RIGHT: Output + Logs
# =============================
with right:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("📌 Risk Output")

    if calculate and api_ok:
        payload = {
            "age": age,
            "income": income,
            "loan_amount": loan_amount,
            "loan_tenure_months": loan_tenure_months,
            "avg_dpd_per_delinquency": avg_dpd,
            "delinquency_ratio": delinquency_ratio,
            "credit_utilization_ratio": credit_util,
            "num_open_accounts": open_accounts,
            "residence_type": residence,
            "loan_purpose": purpose,
            "loan_type": loan_type,
        }

        try:
            out = api_post("/predict", payload)
            p = out["default_probability"]
            score = out["credit_score"]
            rating = out["rating"]

            if p < 0.33:
                band, cls = "Low Risk", "low"
            elif p < 0.66:
                band, cls = "Medium Risk", "mid"
            else:
                band, cls = "High Risk", "high"

            st.markdown(f'<span class="pill {cls}">{band}</span>', unsafe_allow_html=True)
            st.write("")

            c1, c2, c3 = st.columns(3)
            c1.metric("Default Probability", f"{p:.2%}")
            c2.metric("Credit Score", score)
            c3.metric("Rating", rating)

            st.progress(min(max(float(p), 0.0), 1.0))

            if p >= threshold:
                st.error("⚠️ High Risk")
            else:
                st.success("✅ Acceptable Risk")

        except Exception as e:
            st.error("Prediction failed")
            st.code(str(e))

    else:
        st.info("Click **Calculate Risk** to view prediction.")

    st.markdown("---")
    st.subheader("🗂️ Recent Predictions")

    if api_ok:
        try:
            logs = api_get("/logs", params={"limit": 10})
            if logs:
                df = pd.DataFrame(logs)
                st.dataframe(df, use_container_width=True, hide_index=True)
            else:
                st.caption("No logs yet.")
        except Exception:
            st.caption("Logs endpoint not available.")
    else:
        st.caption("Start FastAPI to view logs.")

    st.markdown("</div>", unsafe_allow_html=True)
    
    
    
    
st.subheader("📉 Drift Reports (Latest)")

if api_ok:
    try:
        drift = api_get("/drift-reports", params={"limit": 5})
        if drift:
            # show summary table
            drift_rows = []
            for d in drift:
                drift_rows.append({
                    "timestamp": d["timestamp"],
                    "model_version": d["model_version"],
                    "z_threshold": d["z_threshold"],
                    "drifted_features": d["drifted_features_count"],
                })
            st.dataframe(pd.DataFrame(drift_rows), use_container_width=True, hide_index=True)

            # optional: show detailed top features for the latest report
            with st.expander("See latest drift details"):
                st.json(drift[0]["report"])
        else:
            st.info("No drift reports saved yet.")
    except Exception as e:
        st.error("Could not load drift reports")
        st.code(str(e))
else:
    st.info("Start FastAPI to view drift reports.")
    
    

st.subheader("📊 Analytics (from Logs)")

try:
    logs = api_get("/logs", params={"limit": 500})  # pull enough data for nicer charts
    if logs:
        df = pd.DataFrame(logs)

        # ---- Chart 1: Rating distribution ----
        st.markdown("**1) Rating Distribution**")
        rating_counts = df["rating"].value_counts()
        st.bar_chart(rating_counts)

        # ---- Chart 2: Credit Score distribution (easy + realistic) ----
        st.markdown("**2) Credit Score Distribution**")

        # Create score ranges (bins)
        df["score_range"] = pd.cut(
            df["credit_score"],
            bins=[300, 400, 500, 600, 700, 800, 900],
            include_lowest=True
        )
        
        df["score_range"] = df["score_range"].astype(str)  # Convert to string for better display

        score_dist = df["score_range"].value_counts().sort_index()
        st.bar_chart(score_dist)

    else:
        st.info("No logs yet. Run bulk_calls.py to generate data.")
except Exception as e:
    st.error("Could not load logs for charts.")
    st.code(str(e))
