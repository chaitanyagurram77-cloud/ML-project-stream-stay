"""StreamStay Unsupervised Learning & Segmentation module.
Implements K-Means clustering with silhouette score selection,
PCA and UMAP manifold visualization, Isolation Forest anomaly detection,
and evaluation of cluster features for downstream classification.
"""

from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


def find_optimal_kmeans(
    X: np.ndarray,
    k_range: range = range(2, 6),
    random_state: int = 42
) -> Tuple[KMeans, int, Dict[int, float]]:
    """
    Evaluates K-Means clustering across k in k_range using silhouette score.
    Returns the optimal fitted KMeans model, optimal K, and scores dictionary.
    """
    scores = {}
    models = {}

    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = kmeans.fit_predict(X)
        score = float(silhouette_score(X, labels, sample_size=min(5000, len(X)), random_state=random_state))
        scores[k] = score
        models[k] = kmeans

    optimal_k = max(scores, key=scores.get)
    best_model = models[optimal_k]
    return best_model, optimal_k, scores


def profile_clusters(
    df: pd.DataFrame,
    cluster_labels: np.ndarray,
    cluster_names: Optional[Dict[int, str]] = None
) -> pd.DataFrame:
    """
    Generates summary statistics for each viewer cluster to define segments
    such as 'Casual Viewers', 'Binge Watchers', and 'At-Risk Users'.
    """
    df_temp = df.copy()
    df_temp["cluster"] = cluster_labels
    if cluster_names:
        df_temp["cluster_name"] = df_temp["cluster"].map(cluster_names)

    cols_to_profile = [
        "avg_weekly_watch_hours",
        "sessions_per_week",
        "tenure_months",
        "payment_failures",
        "support_tickets",
        "last_login_days_ago",
        "churned",
        "next_month_watch_hours",
    ]
    available_cols = [c for c in cols_to_profile if c in df_temp.columns]

    group_col = "cluster_name" if cluster_names else "cluster"
    profile = df_temp.groupby(group_col)[available_cols].agg(["mean", "std"]).round(2)
    return profile


def run_isolation_forest(
    X: np.ndarray,
    contamination: float = 0.03,
    random_state: int = 42
) -> Tuple[IsolationForest, np.ndarray, np.ndarray]:
    """
    Fits an Isolation Forest to flag anomalous streaming accounts
    (e.g., automated bots, credential stuffing, extreme outliers).
    
    Returns
    -------
    model : IsolationForest
    anomaly_flags : np.ndarray
        True if outlier (-1), False if inlier (1)
    anomaly_scores : np.ndarray
        Raw decision function anomaly score
    """
    iso = IsolationForest(
        contamination=contamination,
        random_state=random_state,
        n_estimators=100,
        n_jobs=-1
    )
    preds = iso.fit_predict(X)
    scores = iso.decision_function(X)
    is_anomaly = (preds == -1)
    return iso, is_anomaly, scores


def visualize_segments_and_anomalies(
    X: np.ndarray,
    cluster_labels: np.ndarray,
    is_anomaly: np.ndarray,
    optimal_k: int,
    cluster_names: Optional[Dict[int, str]] = None,
    save_path: Optional[str] = "plots/clusters_pca_umap.png"
) -> plt.Figure:
    """
    Reduces feature space to 2D using PCA and UMAP (or TSNE fallback if UMAP unavailable).
    Visualizes viewer segments and isolation forest anomalies.
    """
    # 1. PCA
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X)
    pca_var = pca.explained_variance_ratio_ * 100

    # 2. UMAP (or t-SNE fallback)
    umap_title = "UMAP 2D Projection"
    try:
        import umap
        reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
        X_2d = reducer.fit_transform(X)
    except Exception:
        from sklearn.manifold import TSNE
        umap_title = "t-SNE 2D Projection (UMAP Fallback)"
        # Sample for fast TSNE rendering
        sub_idx = np.random.RandomState(42).choice(len(X), size=min(3000, len(X)), replace=False)
        reducer = TSNE(n_components=2, random_state=42, perplexity=30)
        X_sub = reducer.fit_transform(X[sub_idx])
        X_2d = np.zeros((len(X), 2))
        X_2d[sub_idx] = X_sub

    palette = ["#3498db", "#e67e22", "#2ecc71", "#9b59b6", "#e74c3c", "#1abc9c"]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    # Panel 1: PCA Clusters
    for k in range(optimal_k):
        mask = (cluster_labels == k)
        label_text = cluster_names.get(k, f"Cluster {k}") if cluster_names else f"Cluster {k}"
        axes[0].scatter(
            X_pca[mask, 0], X_pca[mask, 1],
            c=palette[k % len(palette)],
            label=label_text,
            alpha=0.45,
            s=12,
            edgecolors="none"
        )
    axes[0].set_title(f"PCA Projection (Var: {pca_var[0]:.1f}%, {pca_var[1]:.1f}%)", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("PCA Component 1", fontsize=10)
    axes[0].set_ylabel("PCA Component 2", fontsize=10)
    axes[0].legend(loc="upper right", framealpha=0.9, fontsize=9)
    axes[0].grid(True, linestyle=":", alpha=0.5)

    # Panel 2: UMAP / Manifold Clusters
    for k in range(optimal_k):
        mask = (cluster_labels == k)
        label_text = cluster_names.get(k, f"Cluster {k}") if cluster_names else f"Cluster {k}"
        axes[1].scatter(
            X_2d[mask, 0], X_2d[mask, 1],
            c=palette[k % len(palette)],
            label=label_text,
            alpha=0.45,
            s=12,
            edgecolors="none"
        )
    axes[1].set_title(umap_title, fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Dimension 1", fontsize=10)
    axes[1].set_ylabel("Dimension 2", fontsize=10)
    axes[1].legend(loc="upper right", framealpha=0.9, fontsize=9)
    axes[1].grid(True, linestyle=":", alpha=0.5)

    # Panel 3: Isolation Forest Anomalies on PCA
    axes[2].scatter(
        X_pca[~is_anomaly, 0], X_pca[~is_anomaly, 1],
        c="#95a5a6", label="Normal Accounts", alpha=0.3, s=10, edgecolors="none"
    )
    axes[2].scatter(
        X_pca[is_anomaly, 0], X_pca[is_anomaly, 1],
        c="#e74c3c", label=f"Anomalous ({is_anomaly.sum()} accounts)", alpha=0.85, s=28, marker="x"
    )
    axes[2].set_title("Isolation Forest Anomaly Detection", fontsize=12, fontweight="bold")
    axes[2].set_xlabel("PCA Component 1", fontsize=10)
    axes[2].set_ylabel("PCA Component 2", fontsize=10)
    axes[2].legend(loc="upper right", framealpha=0.9, fontsize=9)
    axes[2].grid(True, linestyle=":", alpha=0.5)

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300)
    return fig
