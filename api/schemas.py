from pydantic import BaseModel, Field
from typing import Literal, Optional


class PredictRequest(BaseModel):
    age: int = Field(..., ge=18, le=100, description="Applicant age (18–100)")
    income: float = Field(..., ge=0, description="Annual income (>=0)")
    loan_amount: float = Field(..., ge=0, description="Loan amount (>=0)")
    loan_tenure_months: int = Field(..., ge=0, le=600, description="Loan tenure in months")
    avg_dpd_per_delinquency: float = Field(..., ge=0, le=365, description="Avg days past due per delinquency")
    delinquency_ratio: float = Field(..., ge=0, le=100, description="Delinquency ratio (%)")
    credit_utilization_ratio: float = Field(..., ge=0, le=100, description="Credit utilization ratio (%)")
    num_open_accounts: int = Field(..., ge=1, le=50, description="Number of open loan accounts")

    residence_type: Literal["Owned", "Rented", "Mortgage"]
    loan_purpose: Literal["Education", "Home", "Auto", "Personal"]
    loan_type: Literal["Unsecured", "Secured"]


class PredictResponse(BaseModel):
    prediction_id: str
    default_probability: float = Field(..., ge=0.0, le=1.0)
    credit_score: int = Field(..., ge=300, le=900)
    rating: Literal["Poor", "Average", "Good", "Excellent", "Undefined"]
    timestamp: str


class ModelInfoResponse(BaseModel):
    model_type: str
    scaler_type: str
    n_features: int
    artifact_path: str
    cols_scaled: list[str]
    model_version: str
    
