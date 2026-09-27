"""
Data Preprocessing Module
=========================
Functions for loading, cleaning, and preparing the Bank Marketing dataset
for classification modeling.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer


# ------------------------------------------------------------------ #
#  Constants
# ------------------------------------------------------------------ #
DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "bank-additional-full.csv"

NUMERIC_FEATURES = [
    "age", "duration", "campaign", "pdays", "previous",
    "emp.var.rate", "cons.price.idx", "cons.conf.idx",
    "euribor3m", "nr.employed",
]

CATEGORICAL_FEATURES = [
    "job", "marital", "education", "default", "housing",
    "loan", "contact", "month", "day_of_week", "poutcome",
]


# ------------------------------------------------------------------ #
#  Load data
# ------------------------------------------------------------------ #
def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    """Load the Bank Marketing dataset from a CSV file."""
    df = pd.read_csv(path, sep=";")
    return df


# ------------------------------------------------------------------ #
#  Clean data
# ------------------------------------------------------------------ #
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the dataset:
    - Replace 'unknown' with NaN for specific columns where it makes sense.
    - Keep 'unknown' as a valid category for columns with high unknown rates.
    - Remove exact duplicates.
    - Encode target variable.
    """
    df = df.copy()

    # Remove exact duplicates
    df = df.drop_duplicates().reset_index(drop=True)

    # Identify unknown rates
    unknown_cols = {}
    for col in CATEGORICAL_FEATURES:
        if col in df.columns:
            unknown_rate = (df[col] == "unknown").mean()
            if unknown_rate > 0:
                unknown_cols[col] = unknown_rate

    # For columns with low unknown rate (<20%), replace with NaN
    # For columns with high unknown rate, keep 'unknown' as category
    low_unknown_cols = [c for c, r in unknown_cols.items() if r < 0.20]
    for col in low_unknown_cols:
        df[col] = df[col].replace("unknown", np.nan)

    # Encode target
    df["y"] = df["y"].map({"yes": 1, "no": 0})

    return df


# ------------------------------------------------------------------ #
#  Feature engineering
# ------------------------------------------------------------------ #
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create new features with business meaning:
    - has_previous_contact: whether the client was contacted before
    - campaign_intensity: bucketed campaign contacts
    - age_group: demographic age segments
    """
    df = df.copy()

    # 1. Has previous contact (binary)
    df["has_previous_contact"] = (df["previous"] > 0).astype(int)

    # 2. Campaign intensity
    df["campaign_intensity"] = pd.cut(
        df["campaign"],
        bins=[0, 1, 3, 5, np.inf],
        labels=["single", "low", "medium", "high"],
    )

    # 3. Age group
    df["age_group"] = pd.cut(
        df["age"],
        bins=[0, 30, 40, 55, np.inf],
        labels=["young", "adult", "middle_age", "senior"],
    )

    return df


# ------------------------------------------------------------------ #
#  Build preprocessing pipeline
# ------------------------------------------------------------------ #
def build_preprocessor(
    numeric_features: list[str],
    categorical_features: list[str],
) -> ColumnTransformer:
    """
    Build a ColumnTransformer that:
    - Imputes + scales numeric features
    - Imputes + one-hot encodes categorical features
    """
    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ],
        remainder="drop",
    )
    return preprocessor


# ------------------------------------------------------------------ #
#  Prepare data for modeling
# ------------------------------------------------------------------ #
def prepare_data(
    df: pd.DataFrame,
    test_size: float = 0.3,
    random_state: int = 42,
):
    """
    Full preparation pipeline:
    1. Clean
    2. Feature engineer
    3. Split (stratified)
    Returns X_train, X_test, y_train, y_test, feature lists
    """
    df = clean_data(df)
    df = engineer_features(df)

    # Determine final feature lists (include engineered features)
    num_feats = [f for f in NUMERIC_FEATURES if f in df.columns]
    cat_feats = [f for f in CATEGORICAL_FEATURES if f in df.columns]
    # Add engineered categorical features
    engineered_cat = ["campaign_intensity", "age_group"]
    cat_feats = cat_feats + [f for f in engineered_cat if f in df.columns]
    # Add engineered numeric features
    engineered_num = ["has_previous_contact"]
    num_feats = num_feats + [f for f in engineered_num if f in df.columns]

    X = df[num_feats + cat_feats]
    y = df["y"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    return X_train, X_test, y_train, y_test, num_feats, cat_feats
