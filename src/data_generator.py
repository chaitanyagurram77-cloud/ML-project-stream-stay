"""StreamStay synthetic data generator.
Generates realistic streaming platform subscriber dataset (20,000 users).
Includes churn target, next-month watch hours target, and injected missing values.
"""

import numpy as np
import pandas as pd
from pathlib import Path


def generate_streamstay_dataset(
    n_samples: int = 20000,
    random_seed: int = 42,
    missing_rate: float = 0.05,
    save_path: str = None
) -> pd.DataFrame:
    """
    Generate synthetic subscriber data for StreamStay.
    
    Parameters
    ----------
    n_samples : int
        Number of subscriber records to generate.
    random_seed : int
        Seed for reproducibility.
    missing_rate : float
        Proportion of values to inject as NaN into selected columns (~5%).
    save_path : str, optional
        Path where CSV should be written.
        
    Returns
    -------
    pd.DataFrame
        Complete synthetic dataset.
    """
    rng = np.random.RandomState(random_seed)
    
    # User IDs
    user_ids = [f"USR_{i:05d}" for i in range(1, n_samples + 1)]
    
    # Plan Types
    plan_types = rng.choice(
        ["basic", "standard", "premium"],
        size=n_samples,
        p=[0.40, 0.45, 0.15]
    )
    
    # Tenure in months (1 to 60, right-skewed with mixture)
    tenure_base = rng.exponential(scale=18.0, size=n_samples) + 1.0
    tenure_months = np.clip(np.round(tenure_base), 1, 60).astype(int)
    
    # Device
    devices = rng.choice(
        ["tv", "mobile", "web", "tablet"],
        size=n_samples,
        p=[0.45, 0.30, 0.15, 0.10]
    )
    
    # Top Genre
    genres = rng.choice(
        ["Action", "Drama", "Comedy", "Sci-Fi", "Documentary", "Romance"],
        size=n_samples,
        p=[0.28, 0.24, 0.20, 0.14, 0.08, 0.06]
    )
    
    # Average weekly watch hours (bimodal: casual vs binge watchers)
    is_binge = rng.rand(n_samples) < 0.25
    casual_hours = rng.gamma(shape=3.5, scale=2.5, size=n_samples)  # mean ~8.75h
    binge_hours = rng.normal(loc=26.0, scale=6.0, size=n_samples)   # mean ~26h
    avg_weekly_watch_hours = np.where(is_binge, binge_hours, casual_hours)
    # Plan correlation: premium users tend to watch a bit more
    premium_boost = np.where(plan_types == "premium", 3.0, np.where(plan_types == "standard", 1.0, 0.0))
    avg_weekly_watch_hours = np.clip(np.round(avg_weekly_watch_hours + premium_boost, 2), 0.5, 55.0)
    
    # Sessions per week (correlated with watch hours)
    expected_sessions = (avg_weekly_watch_hours / 3.2) + rng.normal(0, 1.2, size=n_samples)
    sessions_per_week = np.clip(np.round(expected_sessions), 1, 28).astype(int)
    
    # Payment failures (0 to 3, most 0)
    payment_failures = rng.choice([0, 1, 2, 3], size=n_samples, p=[0.82, 0.12, 0.045, 0.015])
    
    # Support tickets in past 3 months (Poisson-distributed)
    support_tickets = rng.poisson(lam=0.75, size=n_samples)
    support_tickets = np.clip(support_tickets, 0, 7)
    
    # Last login days ago (inactivity signal)
    # Most users logged in recently (0-7 days), churning/at-risk users inactive (10-40 days)
    inactive_mask = rng.rand(n_samples) < 0.18
    regular_logins = rng.exponential(scale=2.8, size=n_samples)
    inactive_logins = rng.uniform(8.0, 42.0, size=n_samples)
    last_login_days_ago = np.where(inactive_mask, inactive_logins, regular_logins)
    last_login_days_ago = np.clip(np.round(last_login_days_ago), 0, 45).astype(int)
    
    # --- CHURN GENERATION (~15% positive) ---
    # Logit formula: low watch time, short tenure, high payment failures, high support tickets, high last login
    z_churn = (
        -1.75
        - 0.12 * (avg_weekly_watch_hours - 12.0)
        - 0.045 * (tenure_months - 18.0)
        + 0.95 * payment_failures
        + 0.38 * support_tickets
        + 0.09 * (last_login_days_ago - 4.0)
        + 0.25 * (plan_types == "basic").astype(float)
        - 0.30 * (devices == "tv").astype(float)
        + rng.normal(0, 0.70, size=n_samples)
    )
    p_churn = 1.0 / (1.0 + np.exp(-z_churn))
    
    # Thresholding / Bernoulli draw to target approximately 15% churn rate
    churn_threshold = np.percentile(p_churn, 85.0)  # Top 15% highest risk
    # Smooth probabilistic assignment around threshold
    churned = (rng.rand(n_samples) < p_churn).astype(int)
    # Fine-tune to closely match 15.0%
    current_rate = churned.mean()
    if abs(current_rate - 0.15) > 0.02:
        # Align exactly by quantile
        churned = (p_churn >= churn_threshold).astype(int)
        
    # --- REGRESSION TARGET: NEXT-MONTH WATCH HOURS ---
    # Monthly watch hours = approx 4.33 * weekly hours, adjusted by engagement & churn intent
    base_next_month = (avg_weekly_watch_hours * 4.33)
    engagement_factor = (
        1.0
        + 0.003 * tenure_months
        - 0.08 * payment_failures
        - 0.015 * last_login_days_ago
        - 0.35 * churned  # Churned users drop engagement significantly
        + rng.normal(0, 0.15, size=n_samples)
    )
    next_month_watch_hours = np.clip(np.round(base_next_month * np.maximum(engagement_factor, 0.1), 2), 0.0, 240.0)
    
    df = pd.DataFrame({
        "user_id": user_ids,
        "plan_type": plan_types,
        "tenure_months": tenure_months,
        "avg_weekly_watch_hours": avg_weekly_watch_hours,
        "sessions_per_week": sessions_per_week.astype(float),
        "device": devices,
        "top_genre": genres,
        "payment_failures": payment_failures,
        "support_tickets": support_tickets.astype(float),
        "last_login_days_ago": last_login_days_ago.astype(float),
        "churned": churned,
        "next_month_watch_hours": next_month_watch_hours
    })
    
    # --- INJECT ~5% MISSING VALUES IN A FEW COLUMNS ---
    cols_to_miss = ["avg_weekly_watch_hours", "sessions_per_week", "support_tickets", "device"]
    for col in cols_to_miss:
        mask = rng.rand(n_samples) < missing_rate
        df.loc[mask, col] = np.nan
        
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(save_path, index=False)
        print(f"Dataset successfully saved to {save_path} ({len(df)} rows)")
        
    return df


if __name__ == "__main__":
    df = generate_streamstay_dataset(
        n_samples=20000,
        random_seed=42,
        missing_rate=0.05,
        save_path="data/streamstay_subscribers.csv"
    )
    print("\nDataset Summary:")
    print("Shape:", df.shape)
    print("Churn rate:", f"{df['churned'].mean():.2%}")
    print("\nMissing values per column:\n", df.isnull().sum())
    print("\nFirst 3 rows:\n", df.head(3))
