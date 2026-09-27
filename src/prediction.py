"""
Prediction Module
=================
Functions for making predictions on new customer data.
"""

import pandas as pd
import numpy as np


def predict_customer(
    pipeline,
    customer_data: dict,
    feature_columns: list[str],
) -> dict:
    """
    Predict whether a customer will respond positively to a marketing campaign.

    Parameters
    ----------
    pipeline : sklearn.pipeline.Pipeline
        Fitted pipeline (preprocessor + classifier).
    customer_data : dict
        Dictionary with feature values for a single customer.
    feature_columns : list[str]
        Ordered list of feature names the pipeline expects.

    Returns
    -------
    dict with keys: prediction, probability, interpretation
    """
    # Build a single-row DataFrame in the expected column order
    df_input = pd.DataFrame([customer_data])[feature_columns]

    predicted_class = pipeline.predict(df_input)[0]
    predicted_proba = pipeline.predict_proba(df_input)[0, 1]

    # Business-friendly interpretation
    if predicted_proba >= 0.7:
        level = "HIGH"
        msg = ("Customer has a relatively high predicted probability of "
               "positive response. Consider prioritizing this customer "
               "for the marketing campaign.")
    elif predicted_proba >= 0.4:
        level = "MEDIUM"
        msg = ("Customer has a moderate predicted probability of positive "
               "response. May benefit from targeted follow-up.")
    else:
        level = "LOW"
        msg = ("Customer has a low predicted probability of positive "
               "response. Resources may be better allocated elsewhere.")

    return {
        "prediction": "YES" if predicted_class == 1 else "NO",
        "probability": round(float(predicted_proba), 4),
        "confidence_level": level,
        "interpretation": msg,
    }


def prioritize_customers(
    pipeline,
    X: pd.DataFrame,
    budget_size: int,
) -> pd.DataFrame:
    """
    Rank customers by predicted conversion probability and
    select the top-N within the marketing budget.

    Parameters
    ----------
    pipeline : fitted sklearn Pipeline
    X : DataFrame of customer features
    budget_size : number of customers to target

    Returns
    -------
    DataFrame with customer index, probability, and rank.
    """
    probas = pipeline.predict_proba(X)[:, 1]
    ranked = pd.DataFrame({
        "customer_index": X.index,
        "predicted_probability": probas,
    }).sort_values("predicted_probability", ascending=False).reset_index(drop=True)

    ranked["rank"] = range(1, len(ranked) + 1)
    ranked["targeted"] = ranked["rank"] <= budget_size

    return ranked
