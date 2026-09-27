# Customer Classification – Bank Marketing Dataset

## 📋 Project Overview

This project implements a **supervised binary classification** model to predict whether a customer will respond positively to a direct marketing campaign (subscribing to a term deposit).

**Dataset:** UCI Machine Learning Repository – [Bank Marketing Dataset](https://archive.ics.uci.edu/ml/datasets/bank+marketing)

**Business Context:** If a business has customer behavioral and demographic data, a predictive model can help identify customer segments with high conversion probability — enabling targeted advertising, voucher allocation, email marketing, and remarketing strategies.

---

## 📁 Project Structure

```
customer_classification/
│
├── data/                          # Dataset files
│   └── bank-additional-full.csv   # Main dataset (41,188 records)
│
├── notebooks/                     # Jupyter notebooks
│   └── customer_classification.ipynb  # Complete analysis notebook
│
├── src/                           # Python source modules
│   ├── data_preprocessing.py      # Data loading, cleaning, feature engineering
│   ├── models.py                  # Model definitions and pipeline builders
│   ├── evaluation.py              # Metrics, cross-validation, visualizations
│   └── prediction.py              # Prediction functions for new customers
│
├── outputs/                       # Generated outputs
│   ├── figures/                   # EDA and evaluation charts
│   ├── models/                    # Saved model files
│   └── results/                   # CSV result tables
│
├── report/                        # Project report
│   └── report.md                  # Analysis report (1-2 pages)
│
├── requirements.txt               # Python dependencies
├── README.md                      # This file
└── .gitignore                     # Git ignore rules
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.9+ recommended
- pip package manager

### Installation

1. Clone or download the project:
```bash
cd customer_classification
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. (Optional) Dataset is pre-downloaded in `data/`. If missing, the notebook includes instructions to download from UCI.

### Run the Notebook

**Option 1 – VS Code:**
- Open `notebooks/customer_classification.ipynb` in VS Code
- Select a Python kernel
- Run All Cells

**Option 2 – Jupyter:**
```bash
cd notebooks
jupyter notebook customer_classification.ipynb
```

### Run the Streamlit App (Demo UI)

To launch the interactive web interface:
```bash
streamlit run app.py
```
Then open your browser to the URL provided (usually `http://localhost:8501`).

---

## 📊 Dataset

**Source:** UCI Machine Learning Repository – Bank Marketing Dataset

- **Records:** 41,188
- **Features:** 20 (10 numerical + 10 categorical)
- **Target:** `y` (yes/no — whether the client subscribed to a term deposit)
- **Separator:** semicolon (`;`)

**Features include:**
- Client demographics: `age`, `job`, `marital`, `education`
- Campaign info: `contact`, `month`, `day_of_week`, `duration`, `campaign`
- Previous campaign: `pdays`, `previous`, `poutcome`
- Economic indicators: `emp.var.rate`, `cons.price.idx`, `cons.conf.idx`, `euribor3m`, `nr.employed`

---

## 🔬 Methodology

1. **Data Cleaning:** Handle `"unknown"` values, remove duplicates
2. **EDA:** Target distribution, conversion rates, correlation analysis
3. **Feature Engineering:** `has_previous_contact`, `campaign_intensity`, `age_group`
4. **Preprocessing:** `ColumnTransformer` + `Pipeline` (no data leakage)
5. **Models:** Logistic Regression, Decision Tree, Random Forest, Gradient Boosting
6. **Evaluation:** Stratified 5-Fold CV + held-out test set
7. **Metrics:** Accuracy, Precision, Recall, F1, ROC-AUC

---

## 📈 Models Compared

| Model | Description |
|-------|-------------|
| Logistic Regression | Linear baseline with balanced class weights |
| Decision Tree | Interpretable tree model |
| Random Forest | Ensemble of 300 decision trees |
| Gradient Boosting | Sequential boosting approach |

---

## 🔑 Key Features

- ✅ Complete end-to-end ML pipeline
- ✅ No data leakage (Pipeline + ColumnTransformer)
- ✅ Stratified split and cross-validation
- ✅ Multiple evaluation metrics (not just accuracy)
- ✅ Feature importance analysis
- ✅ Business insight interpretation
- ✅ Prediction function for new customers
- ✅ Business scenario simulation

---

## 🔄 Reproducibility

All random operations use:
```python
random_state = 42
```

---

## ⚠️ Limitations

- Dataset represents Portuguese banking, not all e-commerce contexts
- `duration` feature is only known post-contact (not usable for pre-targeting)
- Feature importance ≠ causal relationship
- Model needs periodic retraining as customer behavior changes
- Performance should be validated on real business data before deployment

---

## 📝 License

This project is for educational purposes. Dataset is provided by UCI Machine Learning Repository under their terms.

## 📚 References

- [Moro et al., 2014] S. Moro, P. Cortez and P. Rita. A Data-Driven Approach to Predict the Success of Bank Telemarketing. Decision Support Systems, Elsevier, 62:22-31, June 2014.
- UCI Machine Learning Repository: https://archive.ics.uci.edu/ml/datasets/bank+marketing
