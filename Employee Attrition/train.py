"""
train.py
--------
Trains two models on the IBM HR Employee Attrition dataset:

1. Decision Tree Classifier -> predicts 'Attrition' (Yes/No)
2. Linear Regression        -> predicts 'MonthlyIncome'

Saves both fitted pipelines (preprocessing + model) to /models as .pkl
files, along with the list of feature columns each expects, so the
Streamlit app can load and reuse them without any retraining.

Run:
    python train.py
"""

import json
import os

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    mean_absolute_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression

DATA_PATH = os.path.join("data", "dataset.csv")
MODELS_DIR = "models"

# Columns that carry no predictive signal (constant or ID-like)
DROP_COLS = ["EmployeeCount", "EmployeeNumber", "Over18", "StandardHours"]


def load_data():
    df = pd.read_csv(DATA_PATH, sep="\t")
    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns])
    return df


def build_preprocessor(X):
    categorical_cols = X.select_dtypes(include="object").columns.tolist()
    numeric_cols = X.select_dtypes(exclude="object").columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ]
    )
    return preprocessor, numeric_cols, categorical_cols


def train_classifier(df):
    print("\n=== Decision Tree Classifier: Attrition ===")
    X = df.drop(columns=["Attrition", "MonthlyIncome"])
    y = df["Attrition"]

    preprocessor, num_cols, cat_cols = build_preprocessor(X)

    pipeline = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            (
                "model",
                DecisionTreeClassifier(
                    max_depth=6,
                    min_samples_leaf=10,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_test)

    acc = accuracy_score(y_test, preds)
    print(f"Accuracy: {acc:.3f}")
    print(classification_report(y_test, preds))

    joblib.dump(pipeline, os.path.join(MODELS_DIR, "attrition_classifier.pkl"))

    metadata = {
        "target": "Attrition",
        "feature_columns": X.columns.tolist(),
        "numeric_columns": num_cols,
        "categorical_columns": cat_cols,
        "categorical_options": {c: sorted(X[c].dropna().unique().tolist()) for c in cat_cols},
        "numeric_ranges": {
            c: [float(X[c].min()), float(X[c].max())] for c in num_cols
        },
        "test_accuracy": round(float(acc), 4),
    }
    with open(os.path.join(MODELS_DIR, "attrition_classifier_meta.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    return metadata


def train_regressor(df):
    print("\n=== Linear Regression: MonthlyIncome ===")
    X = df.drop(columns=["Attrition", "MonthlyIncome"])
    y = df["MonthlyIncome"]

    preprocessor, num_cols, cat_cols = build_preprocessor(X)

    pipeline = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", LinearRegression()),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_test)

    r2 = r2_score(y_test, preds)
    mae = mean_absolute_error(y_test, preds)
    print(f"R^2: {r2:.3f}   MAE: {mae:.1f}")

    joblib.dump(pipeline, os.path.join(MODELS_DIR, "income_regressor.pkl"))

    metadata = {
        "target": "MonthlyIncome",
        "feature_columns": X.columns.tolist(),
        "numeric_columns": num_cols,
        "categorical_columns": cat_cols,
        "categorical_options": {c: sorted(X[c].dropna().unique().tolist()) for c in cat_cols},
        "numeric_ranges": {
            c: [float(X[c].min()), float(X[c].max())] for c in num_cols
        },
        "test_r2": round(float(r2), 4),
        "test_mae": round(float(mae), 1),
    }
    with open(os.path.join(MODELS_DIR, "income_regressor_meta.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    return metadata


def main():
    os.makedirs(MODELS_DIR, exist_ok=True)
    df = load_data()
    train_classifier(df)
    train_regressor(df)
    print("\nSaved models + metadata to ./models/")


if __name__ == "__main__":
    main()
