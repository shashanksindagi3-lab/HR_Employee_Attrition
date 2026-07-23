import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

st.title("📊 HR Employee Attrition Dashboard")

uploaded_file = st.file_uploader("Upload HR Attrition Dataset", type=["xlsx", "csv"])

if uploaded_file:
    # Load data
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    st.success("✅ Data Loaded Successfully")
    st.write("### Preview of Data")
    st.dataframe(df.head())

    # --- Know about the data ---
    st.subheader("Dataset Info")
    st.write(f"Shape: {df.shape}")
    st.write("Columns:", df.columns.tolist())
    st.write("Data Types:", df.dtypes)

    # --- Missing Values ---
    st.subheader("Missing Values")
    st.write(df.isnull().sum())

    # --- Correlation Heatmap ---
    st.subheader("Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(10,6))
    sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

    # --- Box Plot Example ---
    if "MonthlyIncome" in df.columns and "Attrition" in df.columns:
        st.subheader("Box Plot: Monthly Income vs Attrition")
        fig, ax = plt.subplots()
        sns.boxplot(x="Attrition", y="MonthlyIncome", data=df, ax=ax)
        st.pyplot(fig)

    # --- ROC Curve Example ---
    if "Attrition" in df.columns:
        st.subheader("ROC Curve (Logistic Regression Example)")
        df_model = df.select_dtypes(include="number").dropna()
        if "Attrition" in df_model.columns:
            X = df_model.drop("Attrition", axis=1)
            y = df_model["Attrition"]

            if len(y.unique()) == 2:  # binary classification
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
                model = LogisticRegression(max_iter=1000)
                model.fit(X_train, y_train)
                y_prob = model.predict_proba(X_test)[:,1]

                fpr, tpr, _ = roc_curve(y_test, y_prob)
                roc_auc = auc(fpr, tpr)

                fig, ax = plt.subplots()
                ax.plot(fpr, tpr, color="blue", label=f"ROC curve (area = {roc_auc:.2f})")
                ax.plot([0,1], [0,1], color="red", linestyle="--")
                ax.set_xlabel("False Positive Rate")
                ax.set_ylabel("True Positive Rate")
                ax.legend(loc="lower right")
                st.pyplot(fig)
            else:
                st.info("ROC curve requires binary target (Attrition column with 0/1).")
else:
    st.warning("📂 Please upload your HR Attrition dataset to continue.")
