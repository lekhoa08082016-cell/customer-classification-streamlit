"""
Customer Conversion Prediction – Streamlit Application
=======================================================
AI-powered customer response prediction using Machine Learning.

Run:  streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path

# ────────────────────────────────────────────────────────
#  Page configuration
# ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Customer Conversion Prediction",
    page_icon="📊",
    layout="wide",
)

# ────────────────────────────────────────────────────────
#  Custom CSS – Modern Data / AI Dashboard style
# ────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Main background */
    .stApp {
        background-color: #F1F5F8;
    }

    /* Header styling */
    .main-header {
        background: linear-gradient(135deg, #26394D 0%, #1a2a3a 100%);
        padding: 2rem 2.5rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        color: white;
    }
    .main-header h1 {
        color: white;
        font-size: 2rem;
        margin-bottom: 0.3rem;
    }
    .main-header p {
        color: #b0c4d8;
        font-size: 1.05rem;
        margin: 0;
    }

    /* Card styling */
    .metric-card {
        background: white;
        border-radius: 10px;
        padding: 1.5rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
        border-left: 4px solid #28BFE8;
        margin-bottom: 1rem;
    }
    .metric-card h3 {
        color: #26394D;
        font-size: 0.95rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 0.5rem;
    }
    .metric-card .value {
        color: #3F4A56;
        font-size: 1.8rem;
        font-weight: 700;
    }

    /* Result cards */
    .result-yes {
        background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
        border-left: 5px solid #28a745;
        padding: 1.5rem;
        border-radius: 10px;
    }
    .result-no {
        background: linear-gradient(135deg, #f8d7da 0%, #f5c6cb 100%);
        border-left: 5px solid #dc3545;
        padding: 1.5rem;
        border-radius: 10px;
    }

    /* Section header */
    .section-header {
        color: #26394D;
        font-size: 1.1rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        border-bottom: 2px solid #28BFE8;
        padding-bottom: 0.5rem;
        margin-bottom: 1rem;
    }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 20px;
        font-weight: 600;
    }

    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ────────────────────────────────────────────────────────
#  Paths
# ────────────────────────────────────────────────────────
MODEL_PATH = Path("outputs/models/best_model.pkl")
METADATA_PATH = Path("outputs/models/model_metadata.json")
FIGURES_DIR = Path("outputs/figures")
RESULTS_DIR = Path("outputs/results")

# ────────────────────────────────────────────────────────
#  Load model and metadata (cached)
# ────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    """Load the trained pipeline from disk."""
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_metadata():
    """Load model metadata (name, features, metrics)."""
    if not METADATA_PATH.exists():
        return None
    with open(METADATA_PATH, "r") as f:
        return json.load(f)


@st.cache_data
def load_test_results():
    """Load test results for all models."""
    path = RESULTS_DIR / "test_results.csv"
    if path.exists():
        return pd.read_csv(path)
    return None


@st.cache_data
def load_cv_results():
    """Load cross-validation results."""
    path = RESULTS_DIR / "cv_results.csv"
    if path.exists():
        return pd.read_csv(path)
    return None


@st.cache_data
def load_feature_importance():
    """Load feature importance data."""
    path = RESULTS_DIR / "feature_importance.csv"
    if path.exists():
        return pd.read_csv(path)
    return None


# ────────────────────────────────────────────────────────
#  Dataset value options (from the Bank Marketing dataset)
# ────────────────────────────────────────────────────────
JOB_OPTIONS = [
    "admin.", "blue-collar", "entrepreneur", "housemaid",
    "management", "retired", "self-employed", "services",
    "student", "technician", "unemployed", "unknown",
]
MARITAL_OPTIONS = ["divorced", "married", "single", "unknown"]
EDUCATION_OPTIONS = [
    "basic.4y", "basic.6y", "basic.9y", "high.school",
    "illiterate", "professional.course", "university.degree", "unknown",
]
DEFAULT_OPTIONS = ["no", "yes", "unknown"]
HOUSING_OPTIONS = ["no", "yes", "unknown"]
LOAN_OPTIONS = ["no", "yes", "unknown"]
CONTACT_OPTIONS = ["cellular", "telephone"]
MONTH_OPTIONS = [
    "jan", "feb", "mar", "apr", "may", "jun",
    "jul", "aug", "sep", "oct", "nov", "dec",
]
DAY_OPTIONS = ["mon", "tue", "wed", "thu", "fri"]
POUTCOME_OPTIONS = ["failure", "nonexistent", "success"]


# ────────────────────────────────────────────────────────
#  Header
# ────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>📊 Customer Conversion Prediction</h1>
    <p>AI-powered customer response prediction using Machine Learning</p>
</div>
""", unsafe_allow_html=True)

# ────────────────────────────────────────────────────────
#  Pre-flight checks
# ────────────────────────────────────────────────────────
model = load_model()
metadata = load_metadata()

if model is None:
    st.error(
        "**Model file not found.**  \n"
        "Please run the training notebook or execute `python save_models.py` first."
    )
    st.stop()

if metadata is None:
    st.warning("Model metadata file not found. Some info will be unavailable.")
    metadata = {
        "best_model_name": "Unknown",
        "numeric_features": [],
        "categorical_features": [],
        "metrics": {},
    }

NUMERIC_FEATURES = metadata["numeric_features"]
CATEGORICAL_FEATURES = metadata["categorical_features"]


# ────────────────────────────────────────────────────────
#  Tabs
# ────────────────────────────────────────────────────────
tab_predict, tab_performance, tab_importance = st.tabs([
    "🔮 Prediction", "📈 Model Performance", "🎯 Feature Importance"
])

# ================================================================
#  TAB 1 – PREDICTION
# ================================================================
with tab_predict:
    # ── Input form ──
    st.markdown('<div class="section-header">Customer Information</div>',
                unsafe_allow_html=True)

    col_profile, col_campaign, col_economic = st.columns(3)

    with col_profile:
        st.markdown("**Customer Profile**")
        age = st.slider("Age", 17, 98, 38)
        job = st.selectbox("Job", JOB_OPTIONS, index=0)
        marital = st.selectbox("Marital Status", MARITAL_OPTIONS, index=1)
        education = st.selectbox("Education", EDUCATION_OPTIONS, index=3)
        default = st.selectbox("Credit Default", DEFAULT_OPTIONS, index=0)
        housing = st.selectbox("Housing Loan", HOUSING_OPTIONS, index=0)
        loan = st.selectbox("Personal Loan", LOAN_OPTIONS, index=0)

    with col_campaign:
        st.markdown("**Campaign Information**")
        contact = st.selectbox("Contact Type", CONTACT_OPTIONS, index=0)
        month = st.selectbox("Last Contact Month", MONTH_OPTIONS, index=4)
        day_of_week = st.selectbox("Last Contact Day", DAY_OPTIONS, index=0)
        duration = st.number_input("Call Duration (seconds)", 0, 5000, 180)
        campaign = st.number_input("Contacts This Campaign", 1, 56, 2)
        pdays = st.number_input("Days Since Last Contact (999=never)", 0, 999, 999)
        previous = st.number_input("Previous Campaign Contacts", 0, 7, 0)
        poutcome = st.selectbox("Previous Outcome", POUTCOME_OPTIONS, index=1)

    with col_economic:
        st.markdown("**Economic Indicators**")
        emp_var_rate = st.number_input("Employment Variation Rate", -3.5, 1.5, 1.1, step=0.1, format="%.1f")
        cons_price_idx = st.number_input("Consumer Price Index", 92.0, 95.0, 93.9, step=0.1, format="%.3f")
        cons_conf_idx = st.number_input("Consumer Confidence Index", -51.0, -26.0, -36.4, step=0.1, format="%.1f")
        euribor3m = st.number_input("Euribor 3 Month Rate", 0.6, 5.1, 4.857, step=0.01, format="%.3f")
        nr_employed = st.number_input("Number of Employees (quarterly)", 4963.0, 5228.0, 5191.0, step=1.0, format="%.1f")

    st.markdown("---")

    # ── Predict button ──
    predict_clicked = st.button("🚀 **PREDICT CUSTOMER RESPONSE**", use_container_width=True, type="primary")

    if predict_clicked:
        try:
            # Build engineered features
            has_previous_contact = 1 if previous > 0 else 0
            if campaign <= 1:
                campaign_intensity = "single"
            elif campaign <= 3:
                campaign_intensity = "low"
            elif campaign <= 5:
                campaign_intensity = "medium"
            else:
                campaign_intensity = "high"

            if age <= 30:
                age_group = "young"
            elif age <= 40:
                age_group = "adult"
            elif age <= 55:
                age_group = "middle_age"
            else:
                age_group = "senior"

            # Build input DataFrame matching the pipeline's expected features
            input_data = {
                # Numeric
                "age": age,
                "duration": duration,
                "campaign": campaign,
                "pdays": pdays,
                "previous": previous,
                "emp.var.rate": emp_var_rate,
                "cons.price.idx": cons_price_idx,
                "cons.conf.idx": cons_conf_idx,
                "euribor3m": euribor3m,
                "nr.employed": nr_employed,
                "has_previous_contact": has_previous_contact,
                # Categorical
                "job": job,
                "marital": marital,
                "education": education,
                "default": default,
                "housing": housing,
                "loan": loan,
                "contact": contact,
                "month": month,
                "day_of_week": day_of_week,
                "poutcome": poutcome,
                "campaign_intensity": campaign_intensity,
                "age_group": age_group,
            }

            input_df = pd.DataFrame([input_data])[NUMERIC_FEATURES + CATEGORICAL_FEATURES]

            # Predict
            prediction = model.predict(input_df)[0]
            probability = model.predict_proba(input_df)[0][1]

            st.markdown("---")

            # ── Results ──
            col_pred, col_prob = st.columns(2)

            with col_pred:
                if prediction == 1:
                    st.markdown("""
                    <div class="result-yes">
                        <h2 style="color:#155724; margin:0;">✅ HIGHER RESPONSE LIKELIHOOD</h2>
                        <p style="color:#155724; font-size:1.05rem; margin-top:0.5rem;">
                            The model predicts that this customer has a relatively high 
                            likelihood of responding positively to the campaign.
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div class="result-no">
                        <h2 style="color:#721c24; margin:0;">⬇️ LOWER RESPONSE LIKELIHOOD</h2>
                        <p style="color:#721c24; font-size:1.05rem; margin-top:0.5rem;">
                            The model predicts a relatively low likelihood of a positive 
                            campaign response.
                        </p>
                    </div>
                    """, unsafe_allow_html=True)

            with col_prob:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>Predicted Probability</h3>
                    <div class="value">{probability:.1%}</div>
                </div>
                """, unsafe_allow_html=True)
                st.progress(probability)

            # ── Business recommendation ──
            st.markdown("---")
            st.markdown('<div class="section-header">Business Recommendation</div>',
                        unsafe_allow_html=True)

            if probability >= 0.70:
                st.success(
                    "**High Propensity Customer**  \n"
                    "Recommended action: Prioritize for targeted marketing, "
                    "personalized offers, or remarketing."
                )
            elif probability >= 0.40:
                st.info(
                    "**Medium Propensity Customer**  \n"
                    "Recommended action: Use moderate-cost communication such as "
                    "email or standard promotional campaigns."
                )
            else:
                st.warning(
                    "**Low Propensity Customer**  \n"
                    "Recommended action: Avoid allocating excessive campaign "
                    "resources without additional evidence of customer interest."
                )

            st.caption(
                "Note: These are illustrative business rules. The predicted probability "
                "should be used as a decision-support indicator, not a guaranteed "
                "prediction of customer behavior."
            )

        except Exception as e:
            st.error(f"Unable to generate prediction. Please check the input values.\n\nError: {e}")

    # ── Model info ──
    st.markdown("---")

    with st.expander("ℹ️ How does the model work?"):
        st.markdown(f"""
**Model:** {metadata['best_model_name']}

The model analyzes customer demographic, campaign interaction, and economic-related 
features to estimate the likelihood of a positive response to a marketing campaign.

The prediction is **probabilistic** and should be used as a **decision-support tool** 
rather than as a guaranteed prediction of customer behavior.

**Key technical details:**
- The model was trained on the UCI Bank Marketing Dataset (41,188 records)
- Preprocessing includes imputation, scaling, and one-hot encoding
- All preprocessing is bundled in an sklearn Pipeline to prevent data leakage
- `random_state=42` ensures reproducibility
        """)

    with st.expander("📊 Model Metrics"):
        if metadata.get("metrics"):
            m = metadata["metrics"]
            mc1, mc2, mc3, mc4, mc5 = st.columns(5)
            mc1.metric("Accuracy", f"{m.get('Accuracy', 0):.4f}")
            mc2.metric("Precision", f"{m.get('Precision', 0):.4f}")
            mc3.metric("Recall", f"{m.get('Recall', 0):.4f}")
            mc4.metric("F1", f"{m.get('F1', 0):.4f}")
            mc5.metric("ROC-AUC", f"{m.get('ROC-AUC', 0):.4f}")


# ================================================================
#  TAB 2 – MODEL PERFORMANCE
# ================================================================
with tab_performance:
    st.markdown('<div class="section-header">Model Comparison</div>',
                unsafe_allow_html=True)

    test_results = load_test_results()
    cv_results = load_cv_results()

    if test_results is not None:
        st.markdown("#### Test Set Results")
        st.dataframe(
            test_results.style.format({
                "Accuracy": "{:.4f}", "Precision": "{:.4f}",
                "Recall": "{:.4f}", "F1": "{:.4f}", "ROC-AUC": "{:.4f}",
            }).highlight_max(
                subset=["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"],
                color="#d4edda",
            ),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("Test results not found. Run the notebook to generate results.")

    if cv_results is not None:
        st.markdown("#### Cross-Validation Results")
        st.dataframe(
            cv_results.style.format({
                c: "{:.4f}" for c in cv_results.columns if c != "Model"
            }).highlight_max(
                subset=[c for c in cv_results.columns if c != "Model"],
                color="#d4edda",
            ),
            use_container_width=True,
            hide_index=True,
        )

    st.markdown("---")

    # ── Charts ──
    st.markdown('<div class="section-header">Evaluation Charts</div>',
                unsafe_allow_html=True)

    col_roc, col_cm = st.columns(2)

    with col_roc:
        st.markdown("#### ROC Curve")
        roc_path = FIGURES_DIR / "09_roc_curve.png"
        if roc_path.exists():
            st.image(str(roc_path), use_container_width=True)
        else:
            st.info("ROC curve image not found.")

    with col_cm:
        st.markdown("#### Confusion Matrices")
        cm_path = FIGURES_DIR / "08_confusion_matrix_all.png"
        if cm_path.exists():
            st.image(str(cm_path), use_container_width=True)
        else:
            st.info("Confusion matrix image not found.")

    st.markdown("---")
    st.markdown("#### Model Comparison Chart")
    mc_path = FIGURES_DIR / "07_model_comparison.png"
    if mc_path.exists():
        st.image(str(mc_path), use_container_width=True)
    else:
        st.info("Model comparison chart not found.")


# ================================================================
#  TAB 3 – FEATURE IMPORTANCE
# ================================================================
with tab_importance:
    st.markdown('<div class="section-header">Feature Importance</div>',
                unsafe_allow_html=True)

    feat_imp = load_feature_importance()

    if feat_imp is not None:
        st.markdown("#### Top 15 Most Important Features (Random Forest)")

        top15 = feat_imp.head(15).copy()
        top15["Importance %"] = (top15["Importance"] / top15["Importance"].sum() * 100).round(2)

        col_table, col_chart = st.columns([1, 1.5])

        with col_table:
            st.dataframe(
                top15[["Feature", "Importance", "Importance %"]].style.format({
                    "Importance": "{:.6f}",
                    "Importance %": "{:.2f}%",
                }).bar(subset=["Importance"], color="#28BFE8"),
                use_container_width=True,
                hide_index=True,
                height=560,
            )

        with col_chart:
            fi_path = FIGURES_DIR / "10_feature_importance_random_forest.png"
            if fi_path.exists():
                st.image(str(fi_path), use_container_width=True)
            else:
                # Fallback: use combined feature importance image
                fi_path_all = FIGURES_DIR / "10_feature_importance.png"
                if fi_path_all.exists():
                    st.image(str(fi_path_all), use_container_width=True)
                else:
                    import matplotlib.pyplot as plt
                    fig, ax = plt.subplots(figsize=(8, 6))
                    top15_plot = top15.iloc[::-1]
                    ax.barh(top15_plot["Feature"], top15_plot["Importance"], color="#28BFE8")
                    ax.set_xlabel("Importance")
                    ax.set_title("Top 15 Features – Random Forest")
                    plt.tight_layout()
                    st.pyplot(fig)

        st.markdown("---")
        st.caption(
            "Feature importance reflects the predictive contribution of each feature "
            "within the model. It does NOT imply causal relationships. "
            "Correlation / predictive importance ≠ causation."
        )
    else:
        st.info("Feature importance data not found. Run `python save_models.py` to generate.")
