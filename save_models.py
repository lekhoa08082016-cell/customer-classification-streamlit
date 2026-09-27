"""
Save trained models and feature importance for the Streamlit app.
Run from the project root: python save_models.py
"""
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

# ── paths ──
DATA_PATH = Path("data/bank-additional-full.csv")
MODELS_DIR = Path("outputs/models")
RESULTS_DIR = Path("outputs/results")
MODELS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ── load & clean ──
df = pd.read_csv(DATA_PATH, sep=";")
df = df.drop_duplicates().reset_index(drop=True)

# Unknown handling
cat_cols = ["job", "marital", "education", "default", "housing",
            "loan", "contact", "month", "day_of_week", "poutcome"]
for col in cat_cols:
    rate = (df[col] == "unknown").mean()
    if 0 < rate < 0.20:
        df[col] = df[col].replace("unknown", np.nan)

# Target encoding
df["y"] = df["y"].map({"yes": 1, "no": 0})

# Feature engineering
df["has_previous_contact"] = (df["previous"] > 0).astype(int)
df["campaign_intensity"] = pd.cut(
    df["campaign"], bins=[0, 1, 3, 5, np.inf],
    labels=["single", "low", "medium", "high"],
)
df["age_group"] = pd.cut(
    df["age"], bins=[0, 30, 40, 55, np.inf],
    labels=["young", "adult", "middle_age", "senior"],
)

# ── feature lists ──
NUMERIC_FEATURES = [
    "age", "duration", "campaign", "pdays", "previous",
    "emp.var.rate", "cons.price.idx", "cons.conf.idx",
    "euribor3m", "nr.employed", "has_previous_contact",
]
CATEGORICAL_FEATURES = [
    "job", "marital", "education", "default", "housing",
    "loan", "contact", "month", "day_of_week", "poutcome",
    "campaign_intensity", "age_group",
]

X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
y = df["y"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y,
)

# ── preprocessor ──
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
        ("num", numeric_transformer, NUMERIC_FEATURES),
        ("cat", categorical_transformer, CATEGORICAL_FEATURES),
    ],
    remainder="drop",
)

# ── models ──
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42, class_weight="balanced"),
    "Random Forest": RandomForestClassifier(n_estimators=300, random_state=42, class_weight="balanced", n_jobs=-1),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
}

# ── train & save all pipelines ──
for name, model in models.items():
    pipe = Pipeline([("preprocessor", preprocessor), ("classifier", model)])
    pipe.fit(X_train, y_train)
    safe_name = name.lower().replace(" ", "_")
    joblib.dump(pipe, MODELS_DIR / f"{safe_name}_pipeline.pkl")
    print(f"Saved: {safe_name}_pipeline.pkl")

# ── determine best model and save as best_model.pkl ──
test_results = pd.read_csv(RESULTS_DIR / "test_results.csv")
best_row = test_results.sort_values("ROC-AUC", ascending=False).iloc[0]
best_name = best_row["Model"]
best_safe = best_name.lower().replace(" ", "_")

# Load the best pipeline we just saved
best_pipeline = joblib.load(MODELS_DIR / f"{best_safe}_pipeline.pkl")
joblib.dump(best_pipeline, MODELS_DIR / "best_model.pkl")
print(f"\nBest model: {best_name} -> saved as best_model.pkl")

# ── save feature importance ──
def get_feature_names(pipe):
    pre = pipe.named_steps["preprocessor"]
    num_names = list(NUMERIC_FEATURES)
    ohe = pre.named_transformers_["cat"].named_steps["onehot"]
    cat_names = list(ohe.get_feature_names_out(CATEGORICAL_FEATURES))
    return num_names + cat_names

# Save for Random Forest (best interpretable tree model)
rf_pipe = joblib.load(MODELS_DIR / "random_forest_pipeline.pkl")
feat_names = get_feature_names(rf_pipe)
rf_clf = rf_pipe.named_steps["classifier"]
imp_df = pd.DataFrame({
    "Feature": feat_names,
    "Importance": rf_clf.feature_importances_,
}).sort_values("Importance", ascending=False)
imp_df.to_csv(RESULTS_DIR / "feature_importance.csv", index=False)
print(f"Feature importance saved ({len(imp_df)} features)")

# Save model metadata
import json
metadata = {
    "best_model_name": best_name,
    "numeric_features": NUMERIC_FEATURES,
    "categorical_features": CATEGORICAL_FEATURES,
    "metrics": {
        "Accuracy": float(best_row["Accuracy"]),
        "Precision": float(best_row["Precision"]),
        "Recall": float(best_row["Recall"]),
        "F1": float(best_row["F1"]),
        "ROC-AUC": float(best_row["ROC-AUC"]),
    },
}
with open(MODELS_DIR / "model_metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)
print("Model metadata saved")

print("\nDone! All models saved to outputs/models/")
