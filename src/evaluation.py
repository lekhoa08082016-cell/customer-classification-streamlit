"""
Evaluation Module
=================
Metrics computation, cross-validation, and visualization helpers.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    ConfusionMatrixDisplay,
)
from pathlib import Path

FIGURES_DIR = Path(__file__).resolve().parent.parent / "outputs" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------------ #
#  Cross-validation
# ------------------------------------------------------------------ #
def cross_validate_models(
    pipelines: dict,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv_splits: int = 5,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Run StratifiedKFold cross-validation for each model pipeline.
    Returns a DataFrame with mean scores.
    """
    skf = StratifiedKFold(
        n_splits=cv_splits, shuffle=True, random_state=random_state
    )
    scoring = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    results = []

    for name, pipe in pipelines.items():
        cv_res = cross_validate(
            pipe, X_train, y_train,
            cv=skf, scoring=scoring, n_jobs=-1,
        )
        results.append({
            "Model": name,
            "CV Accuracy": cv_res["test_accuracy"].mean(),
            "CV Precision": cv_res["test_precision"].mean(),
            "CV Recall": cv_res["test_recall"].mean(),
            "CV F1": cv_res["test_f1"].mean(),
            "CV ROC-AUC": cv_res["test_roc_auc"].mean(),
        })

    return pd.DataFrame(results)


# ------------------------------------------------------------------ #
#  Test set evaluation
# ------------------------------------------------------------------ #
def evaluate_on_test(
    pipelines: dict,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> pd.DataFrame:
    """
    Fit each pipeline on full training data and evaluate on test set.
    Returns a DataFrame with test metrics.
    """
    results = []
    fitted_pipelines = {}

    for name, pipe in pipelines.items():
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        y_proba = pipe.predict_proba(X_test)[:, 1]

        results.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred),
            "Recall": recall_score(y_test, y_pred),
            "F1": f1_score(y_test, y_pred),
            "ROC-AUC": roc_auc_score(y_test, y_proba),
        })
        fitted_pipelines[name] = pipe

    return pd.DataFrame(results), fitted_pipelines


# ------------------------------------------------------------------ #
#  Training set evaluation (for overfitting check)
# ------------------------------------------------------------------ #
def evaluate_on_train(
    fitted_pipelines: dict,
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> pd.DataFrame:
    """Evaluate fitted pipelines on training data for overfitting analysis."""
    results = []
    for name, pipe in fitted_pipelines.items():
        y_pred = pipe.predict(X_train)
        y_proba = pipe.predict_proba(X_train)[:, 1]
        results.append({
            "Model": name,
            "Train Accuracy": accuracy_score(y_train, y_pred),
            "Train Precision": precision_score(y_train, y_pred),
            "Train Recall": recall_score(y_train, y_pred),
            "Train F1": f1_score(y_train, y_pred),
            "Train ROC-AUC": roc_auc_score(y_train, y_proba),
        })
    return pd.DataFrame(results)


# ------------------------------------------------------------------ #
#  Confusion matrix plot
# ------------------------------------------------------------------ #
def plot_confusion_matrices(
    fitted_pipelines: dict,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    save_dir: Path = FIGURES_DIR,
):
    """Plot and save confusion matrix for each model."""
    for name, pipe in fitted_pipelines.items():
        y_pred = pipe.predict(X_test)
        fig, ax = plt.subplots(figsize=(6, 5))
        ConfusionMatrixDisplay.from_predictions(
            y_test, y_pred,
            display_labels=["No (0)", "Yes (1)"],
            cmap="Blues", ax=ax,
        )
        safe_name = name.lower().replace(" ", "_")
        ax.set_title(f"Confusion Matrix – {name}", fontsize=13)
        fig.tight_layout()
        fig.savefig(save_dir / f"08_confusion_matrix_{safe_name}.png", dpi=150)
        plt.close(fig)


# ------------------------------------------------------------------ #
#  ROC curve plot
# ------------------------------------------------------------------ #
def plot_roc_curves(
    fitted_pipelines: dict,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    save_path: Path = FIGURES_DIR / "09_roc_curve.png",
):
    """Plot ROC curves for all models on the same figure."""
    fig, ax = plt.subplots(figsize=(8, 6))

    for name, pipe in fitted_pipelines.items():
        y_proba = pipe.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        auc_val = roc_auc_score(y_test, y_proba)
        ax.plot(fpr, tpr, label=f"{name} (AUC = {auc_val:.4f})")

    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Random (AUC = 0.5)")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curves – All Models", fontsize=14)
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


# ------------------------------------------------------------------ #
#  Feature importance
# ------------------------------------------------------------------ #
def get_feature_names_after_preprocessing(preprocessor, num_feats, cat_feats):
    """Extract feature names from a fitted ColumnTransformer."""
    ohe = preprocessor.named_transformers_["cat"].named_steps["onehot"]
    cat_names = list(ohe.get_feature_names_out(cat_feats))
    return list(num_feats) + cat_names


def plot_feature_importance(
    fitted_pipelines: dict,
    num_feats: list[str],
    cat_feats: list[str],
    top_n: int = 15,
    save_dir: Path = FIGURES_DIR,
):
    """
    Plot feature importance for tree-based models and
    coefficient magnitude for Logistic Regression.
    """
    importance_dfs = {}

    for name, pipe in fitted_pipelines.items():
        clf = pipe.named_steps["classifier"]
        preprocessor = pipe.named_steps["preprocessor"]
        feature_names = get_feature_names_after_preprocessing(
            preprocessor, num_feats, cat_feats
        )

        if hasattr(clf, "feature_importances_"):
            importances = clf.feature_importances_
        elif hasattr(clf, "coef_"):
            importances = np.abs(clf.coef_[0])
        else:
            continue

        imp_df = pd.DataFrame({
            "Feature": feature_names,
            "Importance": importances,
        }).sort_values("Importance", ascending=False).head(top_n)

        importance_dfs[name] = imp_df

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(
            data=imp_df, x="Importance", y="Feature",
            palette="viridis", ax=ax,
        )
        label = "Coefficient Magnitude" if hasattr(clf, "coef_") else "Feature Importance"
        ax.set_xlabel(label, fontsize=12)
        ax.set_ylabel("Feature", fontsize=12)
        ax.set_title(f"Top {top_n} Features – {name}", fontsize=14)
        fig.tight_layout()
        safe_name = name.lower().replace(" ", "_")
        fig.savefig(save_dir / f"10_feature_importance_{safe_name}.png", dpi=150)
        plt.close(fig)

    return importance_dfs


# ------------------------------------------------------------------ #
#  Model comparison bar chart
# ------------------------------------------------------------------ #
def plot_model_comparison(
    test_results: pd.DataFrame,
    save_path: Path = FIGURES_DIR / "07_model_comparison.png",
):
    """Bar chart comparing all test metrics across models."""
    metrics = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]
    melted = test_results.melt(
        id_vars="Model", value_vars=metrics,
        var_name="Metric", value_name="Score",
    )
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(
        data=melted, x="Model", y="Score", hue="Metric",
        palette="Set2", ax=ax,
    )
    ax.set_title("Model Comparison – Test Set Metrics", fontsize=14)
    ax.set_xlabel("Model", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_ylim(0, 1)
    ax.legend(title="Metric", bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
