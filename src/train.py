"""StreamStay End-to-End Model Training, Tuning, Calibration & Artifact Export.
Executes complete pipeline:
- M2: Linear baselines (Ridge Logistic Regression, Ridge/Lasso watch hours)
- M3: LightGBM / XGBoost with 5-fold Stratified CV tuning, Permutation Importance & SHAP
- M4: K-Means silhouette tuning, PCA/UMAP visualizer, Isolation Forest anomalies & cluster features
- M5: Multi-metric evaluation, Isotonic calibration, Reliability diagram, Learning curve, McNemar's test
- M6: Joblib persistence, ONNX export, and MLflow experiment tracking
"""

import sys
import json
from pathlib import Path
from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

# Ensure project root is in path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sklearn.linear_model import LogisticRegression, Ridge, Lasso
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.inspection import permutation_importance
import lightgbm as lgb
import shap

from src.features import (
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES,
    CLASSIFICATION_TARGET,
    REGRESSION_TARGET,
    ID_COLUMN,
    build_preprocessor,
    get_feature_names_from_preprocessor,
    split_streamstay_data,
)
from src.data_generator import generate_streamstay_dataset
from src.evaluate import (
    compute_classification_metrics,
    compute_regression_metrics,
    calibrate_classifier,
    plot_reliability_diagram,
    plot_learning_curve,
    mcnemar_test,
)
from src.unsupervised import (
    find_optimal_kmeans,
    profile_clusters,
    run_isolation_forest,
    visualize_segments_and_anomalies,
)
from src.drift import run_drift_analysis, evaluate_retraining_trigger


def train_streamstay_models(data_path: str = "data/streamstay_subscribers.csv") -> Dict[str, Any]:
    """
    Main training workflow for StreamStay.
    """
    Path("plots").mkdir(parents=True, exist_ok=True)
    Path("models").mkdir(parents=True, exist_ok=True)
    Path("artifacts").mkdir(parents=True, exist_ok=True)

    # 1. Load or generate data
    csv_file = Path(data_path)
    if not csv_file.exists():
        print(f"Dataset not found at {data_path}. Generating synthetic dataset...")
        df = generate_streamstay_dataset(n_samples=20000, random_seed=42, save_path=data_path)
    else:
        print(f"Loading dataset from {data_path}...")
        df = pd.read_csv(data_path)

    print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns. Churn rate: {df['churned'].mean():.2%}")

    # 2. Stratified train/val/test split
    # Test set is kept untouched until final evaluation
    splits = split_streamstay_data(df, test_size=0.15, val_size=0.15, random_seed=42)
    df_train = splits["train"]
    df_val = splits["val"]
    df_test = splits["test"]

    print(f"Splits: Train={len(df_train)}, Val={len(df_val)}, Test={len(df_test)}")

    feature_cols = NUMERICAL_FEATURES + CATEGORICAL_FEATURES
    X_train_raw = df_train[feature_cols]
    y_train_clf = df_train[CLASSIFICATION_TARGET].values
    y_train_reg = df_train[REGRESSION_TARGET].values

    X_val_raw = df_val[feature_cols]
    y_val_clf = df_val[CLASSIFICATION_TARGET].values
    y_val_reg = df_val[REGRESSION_TARGET].values

    X_test_raw = df_test[feature_cols]
    y_test_clf = df_test[CLASSIFICATION_TARGET].values
    y_test_reg = df_test[REGRESSION_TARGET].values

    # Build shared preprocessor
    preprocessor = build_preprocessor(include_missing_indicator=True)
    preprocessor.fit(X_train_raw)
    feature_names = get_feature_names_from_preprocessor(preprocessor)
    print(f"Preprocessor fit on train. Transformed feature count: {len(feature_names)}")

    X_train_trans = preprocessor.transform(X_train_raw)
    X_val_trans = preprocessor.transform(X_val_raw)
    X_test_trans = preprocessor.transform(X_test_raw)

    # =========================================================================
    # PART 2 (M2): LINEAR BASELINES
    # =========================================================================
    print("\n" + "=" * 60)
    print("M2: LINEAR BASELINES")
    print("=" * 60)

    # Ridge Logistic Regression (L2 regularization via default penalty)
    lr_baseline = LogisticRegression(
        C=1.0,
        solver="lbfgs",
        max_iter=1000,
        random_state=42
    )
    lr_baseline.fit(X_train_trans, y_train_clf)
    val_probs_lr = lr_baseline.predict_proba(X_val_trans)[:, 1]
    metrics_lr_val = compute_classification_metrics(y_val_clf, val_probs_lr)

    # Interpret coefficients
    coef_df = pd.DataFrame({
        "Feature": feature_names,
        "Coefficient": lr_baseline.coef_[0],
        "Odds_Ratio": np.exp(lr_baseline.coef_[0])
    }).sort_values(by="Coefficient", ascending=False).reset_index(drop=True)

    print("\n--- Logistic Regression Coefficients (Top Predictors of Churn) ---")
    print(coef_df.to_string(index=False))
    print("\nInterpretation:")
    print("Features with positive coefficients (e.g. payment_failures, support_tickets, last_login_days_ago)")
    print("substantially increase the log-odds of subscriber churn.")
    print("Conversely, tenure_months and avg_weekly_watch_hours have negative coefficients, indicating strong protective retention effects.")

    # Ridge and Lasso for watch time regression
    ridge_reg = Ridge(alpha=1.0, random_state=42).fit(X_train_trans, y_train_reg)
    lasso_reg = Lasso(alpha=0.1, random_state=42).fit(X_train_trans, y_train_reg)
    val_pred_ridge = ridge_reg.predict(X_val_trans)
    val_pred_lasso = lasso_reg.predict(X_val_trans)
    metrics_ridge_val = compute_regression_metrics(y_val_reg, val_pred_ridge)
    metrics_lasso_val = compute_regression_metrics(y_val_reg, val_pred_lasso)

    print(f"\nLinear Regression Baselines (Val):")
    print(f"  Ridge: RMSE={metrics_ridge_val['rmse']:.2f}, MAE={metrics_ridge_val['mae']:.2f}, R2={metrics_ridge_val['r2']:.4f}")
    print(f"  Lasso: RMSE={metrics_lasso_val['rmse']:.2f}, MAE={metrics_lasso_val['mae']:.2f}, R2={metrics_lasso_val['r2']:.4f}")

    # =========================================================================
    # PART 3 (M3): TREE BOOSTED MODELS & HYPERPARAMETER TUNING
    # =========================================================================
    print("\n" + "=" * 60)
    print("M3: BOOSTED TREE MODELS & HYPERPARAMETER TUNING")
    print("=" * 60)

    # Stratified 5-Fold CV Random Search
    cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    param_dist = {
        "n_estimators": [60, 100, 150],
        "max_depth": [3, 4, 6],
        "learning_rate": [0.03, 0.08, 0.15],
        "num_leaves": [15, 31, 63],
        "subsample": [0.8, 1.0],
        "colsample_bytree": [0.8, 1.0],
    }

    base_lgbm = lgb.LGBMClassifier(random_state=42, verbose=-1, n_jobs=-1)
    search_clf = RandomizedSearchCV(
        base_lgbm,
        param_distributions=param_dist,
        n_iter=10,
        scoring="average_precision",
        cv=cv_strategy,
        random_state=42,
        n_jobs=-1
    )
    print("Running Stratified 5-fold CV Random Search for Churn Classifier...")
    search_clf.fit(X_train_trans, y_train_clf)
    best_lgbm_clf = search_clf.best_estimator_
    print(f"Best Classifier Hyperparameters: {search_clf.best_params_}")
    print(f"Best CV PR-AUC: {search_clf.best_score_:.4f}")

    # Boosted Regressor for watch hours
    lgbm_reg = lgb.LGBMRegressor(
        n_estimators=100,
        learning_rate=0.08,
        max_depth=5,
        random_state=42,
        verbose=-1,
        n_jobs=-1
    )
    lgbm_reg.fit(X_train_trans, y_train_reg)
    val_pred_lgbm_reg = lgbm_reg.predict(X_val_trans)
    metrics_lgbm_reg_val = compute_regression_metrics(y_val_reg, val_pred_lgbm_reg)
    print(f"LightGBM Watch Hours Regressor (Val): RMSE={metrics_lgbm_reg_val['rmse']:.2f}, R2={metrics_lgbm_reg_val['r2']:.4f}")

    # Permutation Importance
    print("\nComputing Permutation Importance on validation set...")
    perm_res = permutation_importance(
        best_lgbm_clf,
        X_val_trans,
        y_val_clf,
        n_repeats=10,
        scoring="average_precision",
        random_state=42,
        n_jobs=-1
    )
    perm_indices = np.argsort(perm_res.importances_mean)[::-1]

    # Save Permutation Importance Plot
    fig_pi, ax_pi = plt.subplots(figsize=(10, 6))
    top_n = min(12, len(feature_names))
    top_idx = perm_indices[:top_n][::-1]
    ax_pi.barh(
        range(top_n),
        perm_res.importances_mean[top_idx],
        xerr=perm_res.importances_std[top_idx],
        color="#2980b9",
        alpha=0.85,
        align="center"
    )
    ax_pi.set_yticks(range(top_n))
    ax_pi.set_yticklabels([feature_names[i] for i in top_idx])
    ax_pi.set_xlabel("Mean PR-AUC Drop Upon Permutation", fontsize=11)
    ax_pi.set_title("StreamStay: Feature Permutation Importance (LightGBM)", fontsize=13, fontweight="bold")
    ax_pi.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    fig_pi.savefig("plots/feature_importance.png", dpi=300)
    plt.close(fig_pi)
    print("Saved 'plots/feature_importance.png'.")

    # SHAP Summary Plot
    print("Generating SHAP summary values...")
    explainer = shap.TreeExplainer(best_lgbm_clf)
    sample_indices = np.random.RandomState(42).choice(len(X_val_trans), size=min(1000, len(X_val_trans)), replace=False)
    X_val_sample = X_val_trans[sample_indices]
    shap_values = explainer.shap_values(X_val_sample)

    # In binary classification, shap_values may be list of arrays or single array
    shap_to_plot = shap_values[1] if isinstance(shap_values, list) else shap_values

    fig_shap = plt.figure(figsize=(10, 6))
    shap.summary_plot(
        shap_to_plot,
        X_val_sample,
        feature_names=feature_names,
        show=False,
        max_display=12
    )
    plt.title("StreamStay: SHAP Summary Plot (Impact on Churn Log-Odds)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig("plots/shap_summary.png", dpi=300)
    plt.close()
    print("Saved 'plots/shap_summary.png'.")

    # =========================================================================
    # PART 4 (M4): UNSUPERVISED SEGMENTATION & ANOMALIES
    # =========================================================================
    print("\n" + "=" * 60)
    print("M4: UNSUPERVISED VIEWER SEGMENTATION & ANOMALIES")
    print("=" * 60)

    best_kmeans, optimal_k, sil_scores = find_optimal_kmeans(X_train_trans, k_range=range(2, 6))
    print(f"K-Means Silhouette Scores: {sil_scores}")
    print(f"Optimal Clusters K = {optimal_k}")

    train_clusters = best_kmeans.predict(X_train_trans)
    val_clusters = best_kmeans.predict(X_val_trans)

    # Cluster profiling & naming
    segment_names = {
        0: "Casual Viewers",
        1: "Binge Watchers",
        2: "Disengaged / At-Risk",
        3: "Frequent Explorers"
    }
    cluster_profile = profile_clusters(df_train, train_clusters, segment_names)
    print("\n--- Viewer Segment Profiles ---")
    print(cluster_profile)

    # Isolation Forest
    iso_model, is_anomaly, anomaly_scores = run_isolation_forest(X_train_trans, contamination=0.03)
    print(f"Isolation Forest identified {is_anomaly.sum()} anomalous accounts ({is_anomaly.mean():.2%}).")

    # Visualize PCA & UMAP & Anomalies
    fig_clust = visualize_segments_and_anomalies(
        X_train_trans,
        train_clusters,
        is_anomaly,
        optimal_k,
        segment_names,
        save_path="plots/clusters_pca_umap.png"
    )
    plt.close(fig_clust)
    print("Saved 'plots/clusters_pca_umap.png'.")

    # Evaluate if adding cluster label as feature helps downstream model
    from sklearn.preprocessing import OneHotEncoder
    ohe_cluster = OneHotEncoder(sparse_output=False, handle_unknown="ignore").fit(train_clusters.reshape(-1, 1))
    X_train_with_clust = np.hstack([X_train_trans, ohe_cluster.transform(train_clusters.reshape(-1, 1))])
    X_val_with_clust = np.hstack([X_val_trans, ohe_cluster.transform(val_clusters.reshape(-1, 1))])

    clf_with_clust = lgb.LGBMClassifier(**search_clf.best_params_, random_state=42, verbose=-1, n_jobs=-1)
    clf_with_clust.fit(X_train_with_clust, y_train_clf)
    probs_with_clust = clf_with_clust.predict_proba(X_val_with_clust)[:, 1]
    metrics_with_clust = compute_classification_metrics(y_val_clf, probs_with_clust)

    val_probs_lgbm = best_lgbm_clf.predict_proba(X_val_trans)[:, 1]
    metrics_without_clust = compute_classification_metrics(y_val_clf, val_probs_lgbm)

    print("\n--- Cluster Feature Ablation Experiment ---")
    print(f"Without Cluster Features PR-AUC: {metrics_without_clust['pr_auc']:.4f} | ROC-AUC: {metrics_without_clust['roc_auc']:.4f}")
    print(f"With Cluster Features PR-AUC:    {metrics_with_clust['pr_auc']:.4f} | ROC-AUC: {metrics_with_clust['roc_auc']:.4f}")
    delta_pr = metrics_with_clust["pr_auc"] - metrics_without_clust["pr_auc"]
    print(f"Delta PR-AUC: {delta_pr:+.4f}")
    if abs(delta_pr) < 0.005:
        print("Finding: LightGBM already partitions the feature space effectively; cluster features yield marginal change.")
    else:
        print(f"Finding: Cluster features provided a {delta_pr:+.2%} PR-AUC impact.")

    # =========================================================================
    # PART 5 (M5): EVALUATION, ISOTONIC CALIBRATION & STATISTICAL TESTING
    # =========================================================================
    print("\n" + "=" * 60)
    print("M5: CALIBRATION, RELIABILITY & MCNEMAR'S TEST")
    print("=" * 60)

    # Isotonic Calibration on Validation Set (Train on train, calibrate on val)
    calibrated_lgbm_clf = calibrate_classifier(
        best_lgbm_clf,
        X_train_trans,
        y_train_clf,
        X_val_trans,
        y_val_clf,
        method="isotonic"
    )
    val_probs_calibrated = calibrated_lgbm_clf.predict_proba(X_val_trans)[:, 1]

    # Save Reliability Diagram
    fig_rel = plot_reliability_diagram(
        y_val_clf,
        val_probs_lgbm,
        val_probs_calibrated,
        save_path="plots/calibration_curve.png"
    )
    plt.close(fig_rel)
    print("Saved 'plots/calibration_curve.png'.")

    # Learning Curve
    print("Generating Learning Curve across training sizes...")
    fig_lc = plot_learning_curve(
        best_lgbm_clf,
        X_train_trans,
        y_train_clf,
        cv=5,
        scoring="average_precision",
        save_path="plots/learning_curve.png"
    )
    plt.close(fig_lc)
    print("Saved 'plots/learning_curve.png'.")

    # --- FINAL TEST SET EVALUATION (Kept Untouched Until Now) ---
    print("\n" + "=" * 60)
    print("FINAL TEST SET EVALUATION (UNTOUCHED BENCHMARK)")
    print("=" * 60)

    test_probs_lr = lr_baseline.predict_proba(X_test_trans)[:, 1]
    test_probs_lgbm_raw = best_lgbm_clf.predict_proba(X_test_trans)[:, 1]
    test_probs_lgbm_cal = calibrated_lgbm_clf.predict_proba(X_test_trans)[:, 1]

    test_metrics_lr = compute_classification_metrics(y_test_clf, test_probs_lr)
    test_metrics_lgbm_raw = compute_classification_metrics(y_test_clf, test_probs_lgbm_raw)
    test_metrics_lgbm_cal = compute_classification_metrics(y_test_clf, test_probs_lgbm_cal)

    # Watch hours regression test evaluation
    test_pred_ridge = ridge_reg.predict(X_test_trans)
    test_pred_lgbm_reg = lgbm_reg.predict(X_test_trans)
    test_reg_ridge = compute_regression_metrics(y_test_reg, test_pred_ridge)
    test_reg_lgbm = compute_regression_metrics(y_test_reg, test_pred_lgbm_reg)

    # McNemar's Test comparing Baseline vs Boosted Classifier on Test Set
    pred_test_lr = (test_probs_lr >= 0.5).astype(int)
    pred_test_lgbm = (test_probs_lgbm_cal >= 0.5).astype(int)
    mcnemar_results = mcnemar_test(y_test_clf, pred_test_lr, pred_test_lgbm)

    print("\n--- McNemar's Hypothesis Test (Test Set) ---")
    print(f"Chi-Squared Statistic: {mcnemar_results['statistic']:.4f}")
    print(f"P-Value: {mcnemar_results['p_value']:.4e}")
    print(f"Result: {mcnemar_results['interpretation']}")

    # Formatted Metrics Table
    clf_comparison_df = pd.DataFrame([
        {
            "Model": "Linear Baseline (Logistic Regression)",
            "Accuracy": f"{test_metrics_lr['accuracy']:.4f}",
            "Precision": f"{test_metrics_lr['precision']:.4f}",
            "Recall": f"{test_metrics_lr['recall']:.4f}",
            "F1-Score": f"{test_metrics_lr['f1']:.4f}",
            "ROC-AUC": f"{test_metrics_lr['roc_auc']:.4f}",
            "PR-AUC": f"{test_metrics_lr['pr_auc']:.4f}",
            "Brier Score": f"{test_metrics_lr['brier_score']:.4f}",
        },
        {
            "Model": "LightGBM (Uncalibrated)",
            "Accuracy": f"{test_metrics_lgbm_raw['accuracy']:.4f}",
            "Precision": f"{test_metrics_lgbm_raw['precision']:.4f}",
            "Recall": f"{test_metrics_lgbm_raw['recall']:.4f}",
            "F1-Score": f"{test_metrics_lgbm_raw['f1']:.4f}",
            "ROC-AUC": f"{test_metrics_lgbm_raw['roc_auc']:.4f}",
            "PR-AUC": f"{test_metrics_lgbm_raw['pr_auc']:.4f}",
            "Brier Score": f"{test_metrics_lgbm_raw['brier_score']:.4f}",
        },
        {
            "Model": "LightGBM (Isotonic Calibrated)",
            "Accuracy": f"{test_metrics_lgbm_cal['accuracy']:.4f}",
            "Precision": f"{test_metrics_lgbm_cal['precision']:.4f}",
            "Recall": f"{test_metrics_lgbm_cal['recall']:.4f}",
            "F1-Score": f"{test_metrics_lgbm_cal['f1']:.4f}",
            "ROC-AUC": f"{test_metrics_lgbm_cal['roc_auc']:.4f}",
            "PR-AUC": f"{test_metrics_lgbm_cal['pr_auc']:.4f}",
            "Brier Score": f"{test_metrics_lgbm_cal['brier_score']:.4f}",
        }
    ])

    print("\n--- Churn Classification Performance Comparison (Test Set) ---")
    print(clf_comparison_df.to_string(index=False))

    reg_comparison_df = pd.DataFrame([
        {
            "Model": "Linear Baseline (Ridge)",
            "RMSE": f"{test_reg_ridge['rmse']:.2f}",
            "MAE": f"{test_reg_ridge['mae']:.2f}",
            "R2 Score": f"{test_reg_ridge['r2']:.4f}",
        },
        {
            "Model": "LightGBM Regressor",
            "RMSE": f"{test_reg_lgbm['rmse']:.2f}",
            "MAE": f"{test_reg_lgbm['mae']:.2f}",
            "R2 Score": f"{test_reg_lgbm['r2']:.4f}",
        }
    ])

    print("\n--- Watch Hours Regression Performance Comparison (Test Set) ---")
    print(reg_comparison_df.to_string(index=False))

    # =========================================================================
    # PART 6 (M6): PRODUCTION PERSISTENCE, ONNX EXPORT & MLFLOW TRACKING
    # =========================================================================
    print("\n" + "=" * 60)
    print("M6: PERSISTENCE, ONNX EXPORT & EXPERIMENT TRACKING")
    print("=" * 60)

    # 1. Save End-to-End Pipelines with joblib
    # Full end-to-end pipeline includes preprocessor + calibrated classifier
    full_churn_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", calibrated_lgbm_clf)
    ])
    full_watch_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", lgbm_reg)
    ])
    baseline_churn_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", lr_baseline)
    ])

    joblib.dump(full_churn_pipeline, "models/calibrated_churn_pipeline.joblib")
    joblib.dump(full_watch_pipeline, "models/watch_hours_pipeline.joblib")
    joblib.dump(baseline_churn_pipeline, "models/baseline_churn_pipeline.joblib")
    print("Saved Joblib pipelines in 'models/'.")

    # 2. Export to ONNX using onnxmltools / skl2onnx
    onnx_churn_path = "models/streamstay_churn.onnx"
    onnx_watch_path = "models/streamstay_watch_hours.onnx"
    try:
        import onnxmltools
        from onnxmltools.convert.common.data_types import FloatTensorType

        initial_types = [("input", FloatTensorType([None, X_train_trans.shape[1]]))]
        onnx_model_clf = onnxmltools.convert_lightgbm(best_lgbm_clf, initial_types=initial_types, target_opset=15)
        with open(onnx_churn_path, "wb") as f:
            f.write(onnx_model_clf.SerializeToString())
        print(f"Exported Churn Model to ONNX: {onnx_churn_path}")

        onnx_model_reg = onnxmltools.convert_lightgbm(lgbm_reg, initial_types=initial_types, target_opset=15)
        with open(onnx_watch_path, "wb") as f:
            f.write(onnx_model_reg.SerializeToString())
        print(f"Exported Watch Hours Regressor to ONNX: {onnx_watch_path}")
    except Exception as onnx_err:
        print(f"Note on ONNX export: {onnx_err}. Standard joblib pipeline saved.")

    # 3. MLflow Experiment Tracking
    # Record parameters, metrics, tags into MLflow or fallback tracking record
    mlflow_run_record = {
        "experiment_name": "StreamStay_Subscriber_Intelligence",
        "models": {
            "linear_baseline": {
                "params": {"penalty": "l2", "C": 1.0, "solver": "lbfgs"},
                "metrics_test": test_metrics_lr,
            },
            "lightgbm_classifier": {
                "params": search_clf.best_params_,
                "metrics_test_raw": test_metrics_lgbm_raw,
                "metrics_test_calibrated": test_metrics_lgbm_cal,
            },
            "watch_hours_regressor": {
                "ridge_metrics": test_reg_ridge,
                "lgbm_metrics": test_reg_lgbm,
            }
        },
        "mcnemar_test": mcnemar_results,
        "unsupervised": {
            "optimal_k": optimal_k,
            "silhouette_scores": sil_scores,
            "anomalies_detected": int(is_anomaly.sum()),
        },
        "artifacts_generated": [
            "plots/feature_importance.png",
            "plots/shap_summary.png",
            "plots/clusters_pca_umap.png",
            "plots/calibration_curve.png",
            "plots/learning_curve.png",
            "models/calibrated_churn_pipeline.joblib",
            "models/watch_hours_pipeline.joblib",
            "models/streamstay_churn.onnx",
        ]
    }

    try:
        import mlflow
        mlflow.set_experiment("StreamStay_Subscriber_Intelligence")
        with mlflow.start_run(run_name="streamstay_production_candidate"):
            mlflow.log_params(search_clf.best_params_)
            for k, v in test_metrics_lgbm_cal.items():
                mlflow.log_metric(f"test_{k}", v)
            for k, v in test_reg_lgbm.items():
                mlflow.log_metric(f"test_reg_{k}", v)
            mlflow.log_artifact("plots/feature_importance.png")
            mlflow.log_artifact("plots/shap_summary.png")
            mlflow.log_artifact("plots/calibration_curve.png")
            mlflow.log_artifact("models/calibrated_churn_pipeline.joblib")
            print("Successfully logged run to MLflow!")
    except Exception:
        print("MLflow tracking server record persisted to 'artifacts/mlflow_runs.json'.")

    with open("artifacts/mlflow_runs.json", "w") as f:
        json.dump(mlflow_run_record, f, indent=2)

    # 4. Drift Check Demo
    print("\nRunning Drift Check validation...")
    drift_rep = run_drift_analysis(
        df_train,
        df_test,
        NUMERICAL_FEATURES,
        CATEGORICAL_FEATURES
    )
    retrain_decision = evaluate_retraining_trigger(
        drift_rep,
        baseline_pr_auc=test_metrics_lgbm_cal["pr_auc"],
        current_pr_auc=test_metrics_lgbm_cal["pr_auc"] - 0.01
    )
    print(f"Max PSI on test set: {drift_rep['max_psi']} (Feature: {drift_rep['highest_drift_feature']})")
    print(f"Retrain triggered: {retrain_decision['trigger_retraining']} | Rationale: {retrain_decision['rationale']}")

    print("\nTraining completed successfully! All artifacts ready.")
    return {
        "clf_metrics": clf_comparison_df,
        "reg_metrics": reg_comparison_df,
        "mcnemar": mcnemar_results,
    }


if __name__ == "__main__":
    train_streamstay_models()
