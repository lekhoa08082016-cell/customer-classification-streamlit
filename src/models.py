"""
Models Module
=============
Defines classification models and training utilities.
"""

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer


def get_models() -> dict:
    """
    Return a dictionary of model name -> model instance.
    All models use random_state=42 for reproducibility.
    """
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42,
        ),
        "Decision Tree": DecisionTreeClassifier(
            random_state=42,
            class_weight="balanced",
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            random_state=42,
            class_weight="balanced",
            n_jobs=-1,
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            random_state=42,
        ),
    }
    return models


def build_pipeline(
    preprocessor: ColumnTransformer,
    model,
) -> Pipeline:
    """
    Combine a preprocessing ColumnTransformer with a classifier
    into a single sklearn Pipeline to prevent data leakage.
    """
    return Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", model),
    ])
