import streamlit as st
import pandas as pd
import requests
import os
import cloudpickle

st.set_page_config(page_title="Retail Stockout Risk Scoring", layout="wide")

MODEL_URL = "https://github.com/robertofernandezmartinez/retail-stockout-risk-scoring/releases/download/v1.0.0/pipe_execution.pkl"
MODEL_PATH = "pipe_execution.pkl"

# ---------------------------------------------------------
# Load model safely with caching
# ---------------------------------------------------------
@st.cache_resource
def load_pipeline():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("📥 Downloading model..."):
            r = requests.get(MODEL_URL)
            with open(MODEL_PATH, "wb") as f:
                f.write(r.content)

    with open(MODEL_PATH, "rb") as f:
        return cloudpickle.load(f)

pipeline = load_pipeline()

# ---------------------------------------------------------
# UI
# ---------------------------------------------------------
st.title("🛒 Retail Stockout Risk Scoring")
st.write("Upload your inventory file to estimate stockout risk probability within 14 days.")

st.subheader("📤 Upload CSV file")
uploaded_file = st.file_uploader("Upload CSV with inventory features", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("📊 Data Preview")
    st.dataframe(df.head())

    try:
        # Ensure correct feature alignment
        expected_cols = pipeline.feature_names_in_
        df = df.reindex(columns=expected_cols, fill_value=0)

        # Ensure date exists if required
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")

        # Predict
        preds = pipeline.predict_proba(df)[:, 1]
        df["stockout_risk"] = preds.round(3)

        st.subheader("📈 Predictions")
        st.dataframe(df)

        st.download_button(
            label="⬇️ Download Results",
            data=df.to_csv(index=False),
            file_name="predictions.csv",
            mime="text/csv"
        )

        # 🔝 Highlight top risk products
        st.subheader("🔥 Top Products at Highest Risk")
        top_risk = df.sort_values(by="stockout_risk", ascending=False).head(10)
        st.table(top_risk[["stockout_risk"] + list(df.columns[:5])])

    except Exception as e:
        st.error(f"❌ Prediction failed: {str(e)}")

else:
    st.info("📎 Upload a CSV file to begin scoring.")
