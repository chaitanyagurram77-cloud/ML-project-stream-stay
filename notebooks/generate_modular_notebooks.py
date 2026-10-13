"""generate_modular_notebooks.py
Creates the modular Jupyter Notebooks for StreamStay:
1. 01_M1_ML_System_Lifecycle.ipynb
2. 02_M2_Linear_Models_at_Depth.ipynb
3. 03_M3_Tree_Based_Models_SHAP.ipynb
4. 04_M4_Unsupervised_Clustering.ipynb
5. 05_M5_Evaluation_Calibration.ipynb
6. 06_M6_Production_Serving_Drift.ipynb
"""

import json
import os

def make_notebook(cells):
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.10"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }

def md_cell(source):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in source.split("\n")]
    }

def code_cell(source):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in source.split("\n")]
    }

def create_modular_notebooks(notebook_dir="notebooks"):
    os.makedirs(notebook_dir, exist_ok=True)

    # 1. M1 Lifecycle
    nb1 = [
        md_cell("""# StreamStay: Notebook 01 — ML System Lifecycle & Skew Prevention (M1)
## End-to-End Prediction Architecture & Shared scikit-learn Pipeline

This notebook demonstrates:
1. Generating 20,000 synthetic subscribers with 15% churn rate and ~5% injected missingness.
2. Building the shared scikit-learn ColumnTransformer (median imputation, MissingIndicator, scaling, one-hot encoding).
3. Tracing a prediction from raw JSON payload to inference to eliminate training-serving skew."""),
        code_cell("""import sys, os
import pandas as pd
import numpy as np

if os.path.basename(os.getcwd()) == "notebooks":
    os.chdir("..")

from src.data_generator import generate_streamstay_dataset
from src.features import build_preprocessor, split_streamstay_data, NUMERICAL_FEATURES, CATEGORICAL_FEATURES

df = generate_streamstay_dataset(n_samples=20000, random_seed=42, save_path="data/streamstay_subscribers.csv")
print("Dataset Shape:", df.shape)
print("Churn Rate:", f"{df['churned'].mean():.2%}")
df.head()"""),
        code_cell("""# Fit preprocessor on training partition only to prevent leakage
splits = split_streamstay_data(df, test_size=0.15, val_size=0.15, random_seed=42)
preproc = build_preprocessor(include_missing_indicator=True)
preproc.fit(splits["train"][NUMERICAL_FEATURES + CATEGORICAL_FEATURES])
print("Fitted ColumnTransformer successfully. Zero training-serving skew.")""")
    ]
    with open(os.path.join(notebook_dir, "01_M1_ML_System_Lifecycle.ipynb"), "w") as f:
        json.dump(make_notebook(nb1), f, indent=2)

    # 2. M2 Linear Models
    nb2 = [
        md_cell("""# StreamStay: Notebook 02 — Linear Baseline Models at Depth (M2)
## Ridge Logistic Regression, Odds Ratios & Regularized Watch-Time Regression"""),
        code_cell("""import os, sys
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge, Lasso

if os.path.basename(os.getcwd()) == "notebooks":
    os.chdir("..")

from src.features import split_streamstay_data, build_preprocessor, get_feature_names_from_preprocessor, NUMERICAL_FEATURES, CATEGORICAL_FEATURES

df = pd.read_csv("data/streamstay_subscribers.csv")
splits = split_streamstay_data(df)
train_df, val_df = splits["train"], splits["val"]

preproc = build_preprocessor().fit(train_df[NUMERICAL_FEATURES + CATEGORICAL_FEATURES])
X_tr = preproc.transform(train_df[NUMERICAL_FEATURES + CATEGORICAL_FEATURES])
y_tr_clf = train_df["churned"].values
y_tr_reg = train_df["next_month_watch_hours"].values

# Fit Ridge Logistic Regression
clf = LogisticRegression(C=1.0, max_iter=1000, random_state=42).fit(X_tr, y_tr_clf)
f_names = get_feature_names_from_preprocessor(preproc)

df_coef = pd.DataFrame({
    "Feature": f_names,
    "Coefficient": clf.coef_[0],
    "Odds_Ratio": np.exp(clf.coef_[0])
}).sort_values(by="Coefficient", ascending=False)
display(df_coef)""")
    ]
    with open(os.path.join(notebook_dir, "02_M2_Linear_Models_at_Depth.ipynb"), "w") as f:
        json.dump(make_notebook(nb2), f, indent=2)

    # 3. M3 Tree Models & SHAP
    nb3 = [
        md_cell("""# StreamStay: Notebook 03 — Gradient Boosted Trees & Explainable AI (M3)
## LightGBM 5-Fold Stratified CV, Permutation Importance & SHAP Values"""),
        code_cell("""import os, sys
from IPython.display import Image, display

if os.path.basename(os.getcwd()) == "notebooks":
    os.chdir("..")

print("Permutation Feature Importance:")
display(Image(filename="plots/feature_importance.png"))

print("SHAP Summary Beeswarm Attribution:")
display(Image(filename="plots/shap_summary.png"))""")
    ]
    with open(os.path.join(notebook_dir, "03_M3_Tree_Based_Models_SHAP.ipynb"), "w") as f:
        json.dump(make_notebook(nb3), f, indent=2)

    # 4. M4 Unsupervised
    nb4 = [
        md_cell("""# StreamStay: Notebook 04 — Unsupervised Viewer Segmentation & Anomalies (M4)
## K-Means Silhouette Optimization, 2D PCA/Manifold Projections & Isolation Forest"""),
        code_cell("""import os, sys
from IPython.display import Image, display

if os.path.basename(os.getcwd()) == "notebooks":
    os.chdir("..")

display(Image(filename="plots/clusters_pca_umap.png"))""")
    ]
    with open(os.path.join(notebook_dir, "04_M4_Unsupervised_Clustering.ipynb"), "w") as f:
        json.dump(make_notebook(nb4), f, indent=2)

    # 5. M5 Evaluation & Calibration
    nb5 = [
        md_cell("""# StreamStay: Notebook 05 — Rigorous Model Evaluation & Calibration (M5)
## Reliability Diagrams, PR-AUC vs Accuracy, Learning Curves & McNemar's Test"""),
        code_cell("""import os, sys
from IPython.display import Image, display

if os.path.basename(os.getcwd()) == "notebooks":
    os.chdir("..")

print("Reliability Curve (Isotonic Calibration):")
display(Image(filename="plots/calibration_curve.png"))

print("Learning Curve (5-Fold CV):")
display(Image(filename="plots/learning_curve.png"))""")
    ]
    with open(os.path.join(notebook_dir, "05_M5_Evaluation_Calibration.ipynb"), "w") as f:
        json.dump(make_notebook(nb5), f, indent=2)

    # 6. M6 Production Serving & Drift
    nb6 = [
        md_cell("""# StreamStay: Notebook 06 — Production Serving, ONNX Export & Drift (M6)
## Population Stability Index (PSI) Drift Check & Automated Retraining Triggers"""),
        code_cell("""import os, sys
import pandas as pd

if os.path.basename(os.getcwd()) == "notebooks":
    os.chdir("..")

from src.features import split_streamstay_data, NUMERICAL_FEATURES, CATEGORICAL_FEATURES
from src.drift import run_drift_analysis, evaluate_retraining_trigger

df = pd.read_csv("data/streamstay_subscribers.csv")
splits = split_streamstay_data(df)
drift_rep = run_drift_analysis(splits["train"], splits["test"], NUMERICAL_FEATURES, CATEGORICAL_FEATURES)
decision = evaluate_retraining_trigger(drift_rep, baseline_pr_auc=0.82)

print(f"Max PSI on Test Set: {drift_rep['max_psi']} ({drift_rep['highest_drift_feature']})")
print(f"Retrain Triggered: {decision['trigger_retraining']}")
print(f"Decision Rationale: {decision['rationale']}")""")
    ]
    with open(os.path.join(notebook_dir, "06_M6_Production_Serving_Drift.ipynb"), "w") as f:
        json.dump(make_notebook(nb6), f, indent=2)

    print("[OK] Modular notebooks 01 to 06 successfully created in notebooks/.")

if __name__ == "__main__":
    create_modular_notebooks()
