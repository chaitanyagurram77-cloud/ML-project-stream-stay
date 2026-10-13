"""StreamStay Production FastAPI Serving Application.
Provides endpoints:
- GET  /health: System health and model readiness check
- POST /predict: Predicts calibrated churn probability and next-month watch hours
- POST /predict/batch: Batch prediction for multiple subscribers
- POST /monitor/drift: Evaluates feature drift (PSI) and retraining triggers
"""

import sys
from pathlib import Path
from typing import List, Optional, Dict, Any
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field
import joblib

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.features import NUMERICAL_FEATURES, CATEGORICAL_FEATURES
from src.drift import run_drift_analysis, evaluate_retraining_trigger

from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

app = FastAPI(
    title="StreamStay Subscriber ML Service",
    description="Real-time Churn Prediction & Next-Month Watch Time Forecasting for Video Streaming",
    version="1.0.0"
)

# Mount static plots directory
PLOTS_DIR = PROJECT_ROOT / "plots"
if PLOTS_DIR.exists():
    app.mount("/plots", StaticFiles(directory=str(PLOTS_DIR)), name="plots")

INDEX_HTML_PATH = Path(__file__).resolve().parent / "index.html"


@app.get("/", response_class=HTMLResponse)
def index_page():
    """Serves the interactive StreamStay Chrome web dashboard."""
    if INDEX_HTML_PATH.exists():
        return HTMLResponse(content=INDEX_HTML_PATH.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>StreamStay Service Online</h1><p>Visit <a href='/docs'>/docs</a> for Swagger UI.</p>")

# Global model references
CHURN_MODEL_PATH = PROJECT_ROOT / "models" / "calibrated_churn_pipeline.joblib"
WATCH_MODEL_PATH = PROJECT_ROOT / "models" / "watch_hours_pipeline.joblib"

clf_pipeline = None
reg_pipeline = None


def load_artifacts():
    """Load pre-trained pipelines from disk."""
    global clf_pipeline, reg_pipeline
    try:
        if CHURN_MODEL_PATH.exists():
            clf_pipeline = joblib.load(CHURN_MODEL_PATH)
        if WATCH_MODEL_PATH.exists():
            reg_pipeline = joblib.load(WATCH_MODEL_PATH)
    except Exception as exc:
        print(f"Warning loading models: {exc}")


@app.on_event("startup")
def startup_event():
    load_artifacts()


class SubscriberInput(BaseModel):
    user_id: Optional[str] = Field("USR_99999", description="Subscriber identifier")
    plan_type: str = Field("standard", description="Subscription tier: basic, standard, premium")
    tenure_months: int = Field(12, ge=1, le=120, description="Months as subscriber")
    avg_weekly_watch_hours: Optional[float] = Field(14.5, ge=0.0, description="Average weekly watch time")
    sessions_per_week: Optional[float] = Field(4.0, ge=0.0, description="Average viewing sessions per week")
    device: Optional[str] = Field("tv", description="Primary streaming device: tv, mobile, web, tablet")
    top_genre: str = Field("Action", description="Favorite content genre")
    payment_failures: int = Field(0, ge=0, le=10, description="Count of recent billing payment failures")
    support_tickets: Optional[float] = Field(0.0, ge=0.0, description="Count of customer support tickets")
    last_login_days_ago: float = Field(2.0, ge=0.0, description="Days since last service login")

    class Config:
        json_schema_extra = {
            "example": {
                "user_id": "USR_10423",
                "plan_type": "standard",
                "tenure_months": 8,
                "avg_weekly_watch_hours": 3.5,
                "sessions_per_week": 2.0,
                "device": "mobile",
                "top_genre": "Action",
                "payment_failures": 1,
                "support_tickets": 2.0,
                "last_login_days_ago": 18.0
            }
        }


class PredictionResponse(BaseModel):
    user_id: str
    churn_probability: float
    predicted_churn: bool
    risk_tier: str
    predicted_next_month_watch_hours: float
    retention_action: str


class HealthResponse(BaseModel):
    status: str
    service: str
    version: str
    models_loaded: Dict[str, bool]


def _format_dataframe(subscribers: List[SubscriberInput]) -> pd.DataFrame:
    records = []
    for s in subscribers:
        records.append({
            "user_id": s.user_id,
            "plan_type": s.plan_type,
            "tenure_months": s.tenure_months,
            "avg_weekly_watch_hours": s.avg_weekly_watch_hours,
            "sessions_per_week": s.sessions_per_week,
            "device": s.device,
            "top_genre": s.top_genre,
            "payment_failures": s.payment_failures,
            "support_tickets": s.support_tickets,
            "last_login_days_ago": s.last_login_days_ago,
        })
    return pd.DataFrame(records)


def _get_risk_and_action(prob: float) -> Tuple[str, str]:
    if prob >= 0.65:
        tier = "Critical Risk"
        action = "Dispatch 30% retention discount coupon + VIP customer success outreach"
    elif prob >= 0.35:
        tier = "Medium Risk"
        action = "Send personalized genre recommendation email & feature highlight"
    else:
        tier = "Low Risk / Healthy"
        action = "Standard organic engagement; recommend premium plan upgrade"
    return tier, action


@app.get("/health", response_model=HealthResponse)
def health_check():
    """Returns system status and pipeline availability."""
    global clf_pipeline, reg_pipeline
    if clf_pipeline is None or reg_pipeline is None:
        load_artifacts()
    return HealthResponse(
        status="healthy",
        service="StreamStay ML Serving API",
        version="1.0.0",
        models_loaded={
            "churn_classifier": clf_pipeline is not None,
            "watch_hours_regressor": reg_pipeline is not None,
        }
    )


@app.post("/predict", response_model=PredictionResponse)
def predict_subscriber(subscriber: SubscriberInput):
    """
    Predict churn probability and next-month watch hours for a single subscriber.
    Shared scikit-learn pipeline handles missing value imputation and scaling automatically.
    """
    global clf_pipeline, reg_pipeline
    if clf_pipeline is None or reg_pipeline is None:
        load_artifacts()

    if clf_pipeline is None or reg_pipeline is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Models not loaded. Please train models first using 'python src/train.py'."
        )

    df_in = _format_dataframe([subscriber])

    try:
        churn_prob = float(clf_pipeline.predict_proba(df_in)[0, 1])
        predicted_churn = bool(churn_prob >= 0.50)
        watch_hours = float(np.maximum(0.0, reg_pipeline.predict(df_in)[0]))
        risk_tier, action = _get_risk_and_action(churn_prob)

        return PredictionResponse(
            user_id=subscriber.user_id,
            churn_probability=round(churn_prob, 4),
            predicted_churn=predicted_churn,
            risk_tier=risk_tier,
            predicted_next_month_watch_hours=round(watch_hours, 2),
            retention_action=action,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference error: {str(exc)}"
        )


@app.post("/predict/batch", response_model=List[PredictionResponse])
def predict_batch(subscribers: List[SubscriberInput]):
    """Predict for a batch of subscribers."""
    global clf_pipeline, reg_pipeline
    if clf_pipeline is None or reg_pipeline is None:
        load_artifacts()

    if clf_pipeline is None or reg_pipeline is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Models not loaded. Run 'python src/train.py' first."
        )

    df_in = _format_dataframe(subscribers)
    try:
        churn_probs = clf_pipeline.predict_proba(df_in)[:, 1]
        watch_hours = np.maximum(0.0, reg_pipeline.predict(df_in))

        results = []
        for i, s in enumerate(subscribers):
            p = float(churn_probs[i])
            w = float(watch_hours[i])
            tier, action = _get_risk_and_action(p)
            results.append(PredictionResponse(
                user_id=s.user_id,
                churn_probability=round(p, 4),
                predicted_churn=bool(p >= 0.50),
                risk_tier=tier,
                predicted_next_month_watch_hours=round(w, 2),
                retention_action=action,
            ))
        return results
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch inference failed: {str(exc)}"
        )


class DriftCheckRequest(BaseModel):
    batch_records: List[SubscriberInput]
    baseline_pr_auc: Optional[float] = 0.82
    current_pr_auc: Optional[float] = None


@app.post("/monitor/drift")
def check_drift(payload: DriftCheckRequest):
    """
    Evaluates Population Stability Index (PSI) between training baseline and batch data.
    Enforces policy: trigger retrain if PSI > 0.20 or PR-AUC drops by > 5%.
    """
    train_data_path = PROJECT_ROOT / "data" / "streamstay_subscribers.csv"
    if not train_data_path.exists():
        raise HTTPException(status_code=404, detail="Training baseline data file not found.")

    ref_df = pd.read_csv(train_data_path)
    curr_df = _format_dataframe(payload.batch_records)

    drift_report = run_drift_analysis(
        ref_df,
        curr_df,
        numerical_features=NUMERICAL_FEATURES,
        categorical_features=CATEGORICAL_FEATURES
    )

    policy_decision = evaluate_retraining_trigger(
        drift_report=drift_report,
        baseline_pr_auc=payload.baseline_pr_auc,
        current_pr_auc=payload.current_pr_auc,
        psi_threshold=0.20,
        pr_auc_drop_threshold=0.05
    )

    return {
        "drift_report": drift_report,
        "retraining_decision": policy_decision
    }
