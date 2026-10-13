"""Automated end-to-end verification suite for StreamStay."""

from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from app.main import app
from src.features import (
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES,
    build_preprocessor,
    split_streamstay_data,
)
from src.drift import run_drift_analysis, evaluate_retraining_trigger
from src.evaluate import (
    compute_classification_metrics,
    compute_regression_metrics,
    mcnemar_test,
)


def test_artifacts_exist():
    required_plots = [
        "plots/calibration_curve.png",
        "plots/clusters_pca_umap.png",
        "plots/feature_importance.png",
        "plots/learning_curve.png",
        "plots/shap_summary.png",
    ]
    for p in required_plots:
        path = Path(p)
        assert path.exists(), f"Missing plot: {p}"
        assert path.stat().st_size > 1000, f"Plot file too small: {p}"
    print("[PASS] All 5 required plots exist and are non-empty.")

    required_models = [
        "models/calibrated_churn_pipeline.joblib",
        "models/watch_hours_pipeline.joblib",
        "models/baseline_churn_pipeline.joblib",
        "models/streamstay_churn.onnx",
        "models/streamstay_watch_hours.onnx",
    ]
    for m in required_models:
        path = Path(m)
        assert path.exists(), f"Missing model: {m}"
        assert path.stat().st_size > 100, f"Model file too small: {m}"
    print("[PASS] All joblib and ONNX model artifacts exist.")

    assert Path("data/streamstay_subscribers.csv").exists(), "Missing dataset"
    assert Path("artifacts/mlflow_runs.json").exists(), "Missing MLflow run record"
    assert Path("notebooks/StreamStay_ML_Project.ipynb").exists(), "Missing notebook"
    assert Path("README.md").exists(), "Missing README.md"
    assert Path("requirements.txt").exists(), "Missing requirements.txt"
    print("[PASS] Data, notebook, requirements, and documentation verified.")


def test_api_serving():
    client = TestClient(app)
    res_health = client.get("/health")
    assert res_health.status_code == 200
    data = res_health.json()
    assert data["status"] == "healthy"
    assert data["models_loaded"]["churn_classifier"] is True
    assert data["models_loaded"]["watch_hours_regressor"] is True
    print("[PASS] GET /health is working.")

    payload = {
        "user_id": "USR_VERIFY",
        "plan_type": "standard",
        "tenure_months": 6,
        "avg_weekly_watch_hours": 4.0,
        "sessions_per_week": 2.0,
        "device": "mobile",
        "top_genre": "Action",
        "payment_failures": 1,
        "support_tickets": 2.0,
        "last_login_days_ago": 15.0
    }
    res_pred = client.post("/predict", json=payload)
    assert res_pred.status_code == 200
    pred = res_pred.json()
    assert 0.0 <= pred["churn_probability"] <= 1.0
    assert isinstance(pred["predicted_churn"], bool)
    assert pred["predicted_next_month_watch_hours"] >= 0.0
    assert "retention_action" in pred
    print(f"[PASS] POST /predict works: churn_prob={pred['churn_probability']:.4f}, watch_hours={pred['predicted_next_month_watch_hours']:.1f}h")


def test_drift_and_retrain():
    df = pd.read_csv("data/streamstay_subscribers.csv")
    splits = split_streamstay_data(df, test_size=0.15, val_size=0.15, random_seed=42)
    train_df = splits["train"]
    test_df = splits["test"]

    drift = run_drift_analysis(train_df, test_df, NUMERICAL_FEATURES, CATEGORICAL_FEATURES)
    assert drift["max_psi"] < 0.10, "In-distribution test set should not exhibit high drift"
    decision = evaluate_retraining_trigger(drift, baseline_pr_auc=0.82)
    assert decision["trigger_retraining"] is False
    print(f"[PASS] Drift calculation verified: max_psi={drift['max_psi']:.4f}, trigger={decision['trigger_retraining']}")


if __name__ == "__main__":
    print("\nStarting StreamStay automated verification suite...")
    test_artifacts_exist()
    test_api_serving()
    test_drift_and_retrain()
    print("\nALL VERIFICATION TESTS PASSED SUCCESSFULLY! [OK]")
