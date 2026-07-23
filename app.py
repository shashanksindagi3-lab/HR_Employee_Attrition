"""
app.py
------
Streamlit app for the HR Attrition dataset.

Two trained pipelines (built by train.py) are loaded and served here:
  1. Decision Tree Classifier -> predicts Attrition (Yes/No) + probability
  2. Linear Regression        -> predicts MonthlyIncome

Run locally:
    streamlit run app.py
"""

import json
import os

import joblib
import pandas as pd
import streamlit as st

MODELS_DIR = "models"

st.set_page_config(
    page_title="HR Attrition & Income Predictor",
    page_icon="📊",
    layout="wide",
)


@st.cache_resource
def load_artifacts():
    clf = joblib.load(os.path.join(MODELS_DIR, "attrition_classifier.pkl"))
    reg = joblib.load(os.path.join(MODELS_DIR, "income_regressor.pkl"))
    with open(os.path.join(MODELS_DIR, "attrition_classifier_meta.json")) as f:
        clf_meta = json.load(f)
    with open(os.path.join(MODELS_DIR, "income_regressor_meta.json")) as f:
        reg_meta = json.load(f)
    return clf, reg, clf_meta, reg_meta


clf, reg, clf_meta, reg_meta = load_artifacts()

# Both models share the same input feature set (same columns dropped)
feature_columns = clf_meta["feature_columns"]
numeric_columns = clf_meta["numeric_columns"]
categorical_columns = clf_meta["categorical_columns"]
categorical_options = clf_meta["categorical_options"]
numeric_ranges = clf_meta["numeric_ranges"]

st.title("📊 HR Attrition & Monthly Income Predictor")
st.caption(
    "Decision Tree classifier predicts whether an employee is likely to leave; "
    "Linear Regression estimates their monthly income. Fill in the employee "
    "profile on the left and see both predictions update on the right."
)

with st.sidebar:
    st.header("Employee Profile")
    st.caption("Adjust the fields to match the employee you want to evaluate.")

    inputs = {}
    for col in feature_columns:
        if col in categorical_columns:
            options = categorical_options[col]
            inputs[col] = st.selectbox(col, options)
        else:
            lo, hi = numeric_ranges[col]
            lo, hi = int(lo), int(hi)
            default = (lo + hi) // 2
            # small integer ranges (ratings 1-4 etc.) get a select slider feel via slider
            inputs[col] = st.slider(col, min_value=lo, max_value=hi, value=default)

    st.button("Predict", type="primary", use_container_width=True)

input_df = pd.DataFrame([inputs])[feature_columns]

col1, col2 = st.columns(2)

with col1:
    st.subheader("🌳 Attrition Risk")
    st.caption(f"Decision Tree · test accuracy {clf_meta['test_accuracy']:.1%}")
    pred = clf.predict(input_df)[0]
    proba = clf.predict_proba(input_df)[0]
    classes = list(clf.classes_)
    yes_idx = classes.index("Yes") if "Yes" in classes else 1
    risk_pct = proba[yes_idx]

    if pred == "Yes":
        st.error(f"Predicted: **Likely to leave** ({risk_pct:.0%} risk)")
    else:
        st.success(f"Predicted: **Likely to stay** ({risk_pct:.0%} risk)")
    st.progress(min(max(risk_pct, 0.0), 1.0))

with col2:
    st.subheader("💰 Estimated Monthly Income")
    st.caption(
        f"Linear Regression · R² {reg_meta['test_r2']:.2f} · MAE ${reg_meta['test_mae']:,.0f}"
    )
    income_pred = reg.predict(input_df)[0]
    st.metric("Predicted Monthly Income", f"${income_pred:,.0f}")
    st.caption(
        f"Typical error margin: ± ${reg_meta['test_mae']:,.0f} (mean absolute error on test set)"
    )

st.divider()
with st.expander("ℹ️ About this app"):
    st.markdown(
        """
        - **Dataset**: IBM HR Employee Attrition dataset (1,470 employees, 35 features).
        - **Classifier**: `DecisionTreeClassifier` (max depth 6, balanced class weights)
          predicting the `Attrition` column.
        - **Regressor**: `LinearRegression` predicting `MonthlyIncome`.
        - Both models are scikit-learn `Pipeline`s that handle scaling and
          one-hot encoding internally, trained via `train.py`.
        - Models are cached with `@st.cache_resource` so they load once per session.
        """
    )
