"""StreamStay Drift Detection and Retraining Policy module.
Implements Population Stability Index (PSI) for continuous & categorical features,
detects distribution shifts between training and production inference data,
and evaluates actionable retraining trigger rules.
"""

from typing import Dict, List, Any, Tuple, Optional
import numpy as np
import pandas as pd


def calculate_feature_psi(
    reference: np.ndarray,
    current: np.ndarray,
    num_bins: int = 10,
    epsilon: float = 1e-4
) -> Tuple[float, pd.DataFrame]:
    """
    Compute Population Stability Index (PSI) for a single numerical feature.
    
    Mathematical formula:
        PSI = sum( (Q_i - P_i) * ln(Q_i / P_i) )
    where:
        P_i = fraction of reference population in bin i
        Q_i = fraction of current population in bin i
        
    Rule of thumb thresholds:
        PSI < 0.10: Stable / No significant change
        0.10 <= PSI < 0.20: Moderate shift / Monitor
        PSI >= 0.20: Significant shift / Retraining trigger
    """
    ref_clean = reference[~pd.isna(reference)]
    curr_clean = current[~pd.isna(current)]

    if len(ref_clean) == 0 or len(curr_clean) == 0:
        return 0.0, pd.DataFrame()

    # Create quantile-based bins from reference distribution
    percentiles = np.linspace(0, 100, num_bins + 1)
    bin_edges = np.percentile(ref_clean, percentiles)
    bin_edges[0] = -np.inf
    bin_edges[-1] = np.inf
    bin_edges = np.unique(bin_edges)  # Eliminate duplicate edges for constant values

    if len(bin_edges) <= 1:
        return 0.0, pd.DataFrame()

    ref_counts, _ = np.histogram(ref_clean, bins=bin_edges)
    curr_counts, _ = np.histogram(curr_clean, bins=bin_edges)

    ref_pct = ref_counts / len(ref_clean)
    curr_pct = curr_counts / len(curr_clean)

    # Apply epsilon smoothing for zero bins
    ref_pct = np.where(ref_pct == 0, epsilon, ref_pct)
    curr_pct = np.where(curr_pct == 0, epsilon, curr_pct)

    # Renormalize
    ref_pct /= np.sum(ref_pct)
    curr_pct /= np.sum(curr_pct)

    psi_components = (curr_pct - ref_pct) * np.log(curr_pct / ref_pct)
    psi_total = float(np.sum(psi_components))

    bin_df = pd.DataFrame({
        "bin_lower": bin_edges[:-1],
        "bin_upper": bin_edges[1:],
        "ref_pct": ref_pct,
        "curr_pct": curr_pct,
        "psi_contribution": psi_components,
    })

    return psi_total, bin_df


def calculate_categorical_psi(
    reference: pd.Series,
    current: pd.Series,
    epsilon: float = 1e-4
) -> Tuple[float, pd.DataFrame]:
    """
    Compute PSI for categorical features across categories.
    """
    all_categories = sorted(list(set(reference.dropna().unique()).union(set(current.dropna().unique()))))
    if not all_categories:
        return 0.0, pd.DataFrame()

    ref_counts = reference.value_counts(normalize=True).to_dict()
    curr_counts = current.value_counts(normalize=True).to_dict()

    ref_pcts = np.array([ref_counts.get(cat, epsilon) for cat in all_categories])
    curr_pcts = np.array([curr_counts.get(cat, epsilon) for cat in all_categories])

    ref_pcts /= np.sum(ref_pcts)
    curr_pcts /= np.sum(curr_pcts)

    psi_components = (curr_pcts - ref_pcts) * np.log(curr_pcts / ref_pcts)
    psi_total = float(np.sum(psi_components))

    summary_df = pd.DataFrame({
        "category": all_categories,
        "ref_pct": ref_pcts,
        "curr_pct": curr_pcts,
        "psi_contribution": psi_components
    })
    return psi_total, summary_df


def run_drift_analysis(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    numerical_features: List[str],
    categorical_features: List[str]
) -> Dict[str, Any]:
    """
    Runs multi-feature drift detection across all input features.
    """
    results = {}
    max_psi = 0.0
    highest_drift_feature = None

    for col in numerical_features:
        if col in reference_df.columns and col in current_df.columns:
            psi_val, _ = calculate_feature_psi(
                reference_df[col].values,
                current_df[col].values
            )
            status = "STABLE" if psi_val < 0.1 else ("MONITOR" if psi_val < 0.2 else "DRIFTED")
            results[col] = {
                "type": "numerical",
                "psi": round(psi_val, 4),
                "status": status,
            }
            if psi_val > max_psi:
                max_psi = psi_val
                highest_drift_feature = col

    for col in categorical_features:
        if col in reference_df.columns and col in current_df.columns:
            psi_val, _ = calculate_categorical_psi(
                reference_df[col],
                current_df[col]
            )
            status = "STABLE" if psi_val < 0.1 else ("MONITOR" if psi_val < 0.2 else "DRIFTED")
            results[col] = {
                "type": "categorical",
                "psi": round(psi_val, 4),
                "status": status,
            }
            if psi_val > max_psi:
                max_psi = psi_val
                highest_drift_feature = col

    return {
        "feature_metrics": results,
        "max_psi": round(max_psi, 4),
        "highest_drift_feature": highest_drift_feature,
        "overall_status": "STABLE" if max_psi < 0.1 else ("MONITOR" if max_psi < 0.2 else "DRIFT_DETECTED"),
    }


def evaluate_retraining_trigger(
    drift_report: Dict[str, Any],
    baseline_pr_auc: float,
    current_pr_auc: Optional[float] = None,
    psi_threshold: float = 0.20,
    pr_auc_drop_threshold: float = 0.05
) -> Dict[str, Any]:
    """
    Evaluates production retraining policy:
    Rule: Trigger retrain if max(PSI) > 0.20 OR PR-AUC drops by > 5% relative to baseline.
    
    Parameters
    ----------
    drift_report : dict
        Output from `run_drift_analysis`
    baseline_pr_auc : float
        PR-AUC benchmark achieved during validation
    current_pr_auc : float, optional
        Current production PR-AUC if ground truth labels became available
    psi_threshold : float
        PSI ceiling before flagging distribution shift (default 0.20)
    pr_auc_drop_threshold : float
        Maximum acceptable relative drop in PR-AUC (e.g. 5%)
        
    Returns
    -------
    dict
        Decision flag ('trigger_retraining': bool) and detailed rationale.
    """
    max_psi = drift_report.get("max_psi", 0.0)
    psi_triggered = max_psi >= psi_threshold

    perf_triggered = False
    perf_drop = 0.0
    if current_pr_auc is not None and baseline_pr_auc > 0:
        perf_drop = (baseline_pr_auc - current_pr_auc) / baseline_pr_auc
        perf_triggered = perf_drop >= pr_auc_drop_threshold

    trigger_retraining = psi_triggered or perf_triggered

    reasons = []
    if psi_triggered:
        reasons.append(
            f"Population Stability Index (PSI = {max_psi:.4f}) on '{drift_report.get('highest_drift_feature')}' "
            f"exceeded threshold of {psi_threshold:.2f}."
        )
    if perf_triggered:
        reasons.append(
            f"PR-AUC dropped by {perf_drop:.2%} (from {baseline_pr_auc:.4f} to {current_pr_auc:.4f}), "
            f"exceeding acceptable margin of {pr_auc_drop_threshold:.2%}."
        )

    if not trigger_retraining:
        reasons.append("All features stable (PSI < 0.20) and performance within tolerance.")

    return {
        "trigger_retraining": trigger_retraining,
        "max_psi": max_psi,
        "psi_triggered": psi_triggered,
        "perf_triggered": perf_triggered,
        "pr_auc_drop_pct": round(perf_drop * 100, 2),
        "rationale": " | ".join(reasons),
    }
