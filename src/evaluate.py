"""StreamStay Evaluation & Statistical Validation module.
Computes classification and regression metrics, isotonic probability calibration,
reliability diagrams, learning curves, and McNemar's statistical significance test.
"""

from typing import Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import chi2

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    roc_curve,
    precision_recall_curve,
)
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
from sklearn.model_selection import learning_curve


def compute_classification_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5
) -> Dict[str, float]:
    """
    Compute full classification suite:
    Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC, and Brier Score.
    
    Explains why accuracy is misleading: with an imbalanced target (15% churn),
    a naive classifier predicting 0 for every user yields 85% accuracy
    while failing to detect a single churning customer.
    """
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "brier_score": float(brier_score_loss(y_true, y_prob)),
    }


def compute_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> Dict[str, float]:
    """
    Compute RMSE, MAE, and R^2 for next-month watch hours regression.
    """
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    return {
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
    }


def calibrate_classifier(
    estimator,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    method: str = "isotonic"
) -> CalibratedClassifierCV:
    """
    Applies isotonic calibration using the validation set (without touching test set).
    In scikit-learn 1.4+, cv accepts custom (train_indices, val_indices) splits
    so the base estimator is trained on train, and isotonic regression is fit on validation.
    """
    X_comb = np.vstack([X_train, X_val])
    y_comb = np.concatenate([y_train, y_val])
    train_idx = np.arange(len(X_train))
    val_idx = np.arange(len(X_train), len(X_comb))

    calibrated = CalibratedClassifierCV(
        estimator=estimator,
        method=method,
        cv=[(train_idx, val_idx)]
    )
    calibrated.fit(X_comb, y_comb)
    return calibrated


def plot_reliability_diagram(
    y_true: np.ndarray,
    uncalibrated_probs: np.ndarray,
    calibrated_probs: np.ndarray,
    save_path: Optional[str] = "plots/calibration_curve.png"
) -> plt.Figure:
    """
    Generates and saves a reliability diagram comparing uncalibrated vs calibrated probabilities.
    Includes perfect calibration reference line and confidence histograms.
    """
    prob_true_uncal, prob_pred_uncal = calibration_curve(y_true, uncalibrated_probs, n_bins=10, strategy="uniform")
    prob_true_cal, prob_pred_cal = calibration_curve(y_true, calibrated_probs, n_bins=10, strategy="uniform")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 9), gridspec_kw={"height_ratios": [2.5, 1]})

    # Reliability curve
    ax1.plot([0, 1], [0, 1], "k--", label="Perfect Calibration (y = x)")
    ax1.plot(prob_pred_uncal, prob_true_uncal, "s-", color="#e74c3c", label=f"Uncalibrated (Brier: {brier_score_loss(y_true, uncalibrated_probs):.4f})")
    ax1.plot(prob_pred_cal, prob_true_cal, "o-", color="#2ecc71", label=f"Isotonic Calibrated (Brier: {brier_score_loss(y_true, calibrated_probs):.4f})")
    ax1.set_xlabel("Mean Predicted Probability", fontsize=11)
    ax1.set_ylabel("Fraction of True Positives (Observed)", fontsize=11)
    ax1.set_title("StreamStay Churn Prediction: Reliability Diagram", fontsize=13, fontweight="bold")
    ax1.legend(loc="upper left", framealpha=0.9)
    ax1.grid(True, linestyle=":", alpha=0.6)

    # Predicted probability distribution histograms
    ax2.hist(uncalibrated_probs, bins=25, alpha=0.5, color="#e74c3c", label="Uncalibrated Probs", density=True)
    ax2.hist(calibrated_probs, bins=25, alpha=0.5, color="#2ecc71", label="Calibrated Probs", density=True)
    ax2.set_xlabel("Predicted Probability", fontsize=10)
    ax2.set_ylabel("Density", fontsize=10)
    ax2.legend(loc="upper right")
    ax2.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300)
    return fig


def plot_learning_curve(
    estimator,
    X,
    y,
    cv: int = 5,
    scoring: str = "average_precision",
    save_path: Optional[str] = "plots/learning_curve.png"
) -> plt.Figure:
    """
    Computes and plots 5-fold cross-validated learning curves across training sizes.
    """
    train_sizes, train_scores, test_scores = learning_curve(
        estimator,
        X,
        y,
        cv=cv,
        scoring=scoring,
        train_sizes=np.linspace(0.15, 1.0, 6),
        n_jobs=-1,
        random_state=42
    )

    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(train_sizes, train_mean, "o-", color="#2980b9", label="Training PR-AUC")
    ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.15, color="#2980b9")

    ax.plot(train_sizes, test_mean, "s-", color="#27ae60", label="CV Validation PR-AUC")
    ax.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.15, color="#27ae60")

    ax.set_title("StreamStay LightGBM Classifier: Learning Curve (PR-AUC)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Training Sample Size", fontsize=11)
    ax.set_ylabel("PR-AUC Score", fontsize=11)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="lower right")

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300)
    return fig


def mcnemar_test(
    y_true: np.ndarray,
    y_pred_baseline: np.ndarray,
    y_pred_boosted: np.ndarray
) -> Dict[str, Any]:
    """
    Performs McNemar's Test with Edwards continuity correction
    to statistically compare two classifiers on paired predictions on the test set.
    
    Contingency table:
      - n00: both correct
      - n01: baseline correct, boosted incorrect
      - n10: baseline incorrect, boosted correct
      - n11: both incorrect
      
    Chi-squared statistic = (|n01 - n10| - 1)^2 / (n01 + n10)
    df = 1
    """
    corr_base = (y_pred_baseline == y_true)
    corr_boost = (y_pred_boosted == y_true)

    n00 = int(np.sum(corr_base & corr_boost))
    n01 = int(np.sum(corr_base & (~corr_boost)))
    n10 = int(np.sum((~corr_base) & corr_boost))
    n11 = int(np.sum((~corr_base) & (~corr_boost)))

    discordant = n01 + n10
    if discordant == 0:
        stat = 0.0
        p_value = 1.0
    else:
        stat = ((abs(n01 - n10) - 1.0) ** 2) / float(discordant)
        p_value = float(chi2.sf(stat, df=1))

    return {
        "contingency_table": {
            "both_correct (n00)": n00,
            "baseline_only_correct (n01)": n01,
            "boosted_only_correct (n10)": n10,
            "both_incorrect (n11)": n11,
        },
        "statistic": float(stat),
        "p_value": float(p_value),
        "is_significant_at_005": bool(p_value < 0.05),
        "interpretation": (
            "Statistically significant difference (p < 0.05). Boosted model out-performs baseline."
            if p_value < 0.05 else
            "No statistically significant difference (p >= 0.05)."
        )
    }
