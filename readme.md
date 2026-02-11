# 📊 Credit Risk Modelling — End-to-End ML System

This project implements a **production-style credit risk prediction system**, covering the full ML lifecycle:

* Data preprocessing & feature engineering
* Model training & experiment tracking (MLflow)
* Model versioning (v1 / v2)
* FastAPI inference service
* SQLite prediction logging
* Automated drift monitoring (Z-score based)
* Streamlit monitoring dashboard
* CI with GitHub Actions

---

# 🏗️ Architecture Overview

```
Training Layer (Notebook)
        ↓
Model Artifacts + Drift Baseline
        ↓
FastAPI Inference API
        ↓
SQLite Logging (Predictions + Drift)
        ↓
Streamlit Monitoring Dashboard
```


# 🧠 Model Development

### 📌 Dataset

Merged:

* Customer data
* Loan data
* Bureau data

Target variable:

```
default (0 = No default, 1 = Default)
```

---

### 📌 Feature Engineering

Created business-relevant features:

* `loan_to_income`
* `delinquency_ratio`
* `avg_dpd_per_delinquency`

Applied:

* VIF for multicollinearity reduction
* WOE/IV for feature selection

---

### 📌 Model Training Attempts

| Attempt | Method                                     |
| ------- | ------------------------------------------ |
| 1       | Logistic Regression, RandomForest, XGBoost |
| 2       | RandomUnderSampler                         |
| 3       | SMOTETomek + Optuna (Logistic)             |
| 4       | SMOTETomek + Optuna (XGBoost)              |

Final models saved as:

```
model_data_v1.joblib
model_data_v2.joblib
```

---

# 📊 Experiment Tracking (MLflow)

Logged:

* Parameters
* Metrics (ROC-AUC, Precision, Recall, F1)
* Model artifacts
* Version tags

Experiment:

```
credit-risk-modelling
```

---

# 🚀 FastAPI Inference Service

### Endpoints

| Endpoint         | Purpose                   |
| ---------------- | ------------------------- |
| `/health`        | API health check          |
| `/model-info`    | Model metadata            |
| `/predict`       | Run inference             |
| `/logs`          | Fetch prediction logs     |
| `/drift-reports` | View latest drift results |

---

### 🔁 Prediction Flow

1. Input received
2. Model artifact loaded
3. Features aligned to training schema
4. Prediction generated
5. Prediction stored in SQLite
6. Drift auto-check every 100 predictions

---

# 🗄️ SQLite Logging Layer

### predictions table

Stores:

* Timestamp
* Full input JSON
* Default probability
* Credit score
* Rating

### drift_reports table

Stores:

* Timestamp
* Model version
* Z-score threshold
* Drifted feature count
* Full drift report JSON

---

# 📉 Data Drift Monitoring

Drift is computed using **Z-score based monitoring**:

```
Z = (new_mean - train_mean) / train_std
```

If:

```
|Z| ≥ 3
```

Feature is flagged as drifted.

### Drift runs:

* Automatically every 100 predictions
* Compares training baseline vs latest live predictions
* Stores results in SQLite

---

# 📊 Streamlit Monitoring Dashboard

The dashboard provides:

* Risk prediction UI
* Rating distribution chart
* Credit score distribution chart
* Recent predictions table
* Latest drift report viewer

---

# 🔄 CI Pipeline (GitHub Actions)

On every push:

* Checkout repository
* Install dependencies
* Run basic FastAPI import checks
* Prepare project for scaling

---

# 🛠️ How to Run Locally

### 1️⃣ Install dependencies

```
pip install -r requirements.txt
```

---

### 2️⃣ Start FastAPI

```
uvicorn api.main:app --reload --port 8000
```

---

### 3️⃣ Start Streamlit

```
streamlit run app/streamlit_app.py
```

---

# 🎯 Production-Style Capabilities

✔ Model versioning (v1 / v2)
✔ Drift monitoring
✔ Auto-trigger drift checks
✔ Prediction logging
✔ Monitoring dashboard
✔ CI pipeline
✔ Modular project structure

---

# 📌 Future Improvements

* Docker containerization
* Cloud storage for artifacts (S3/GCS)
* MLflow Model Registry
* Production database (PostgreSQL)
* Auto-retraining pipeline
* Alerting system (Slack / Email)

---

# 👨‍💻 Author

Anish Tirumala Venkata
M.S. Computer Science — University of Florida

