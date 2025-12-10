import streamlit as st
import pandas as pd
import requests
import cloudpickle

st.set_page_config(page_title="Retail Stockout Risk Scoring", layout="wide")

MODEL_URL = "https://github.com/robertofernandezmartinez/retail-stockout-risk-scoring/releases/download/v1.0.0/pipe_execution.pkl"

@st.cache_resource(show_spinner="Loading model from Release...")
def load_pipeline():
    response = requests.get(MODEL_URL)
    response.raise_for_status()
    return cloudpickle.loads(response.content)

pipeline = load_pipeline()

st.title("🛒 Retail Stockout Risk Scoring")
st.write("Upload your inventory file to estimate **stockout probability within 14 days**.")

uploaded_file = st.file_uploader(
    "Upload CSV with inventory features",
    type=["csv"]
)

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.subheader("📊 Data Preview")
    st.dataframe(df.head())

    # Date conversion if present
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # Run prediction
    with st.spinner("Predicting stockout risk..."):
        probs = pipeline.predict_proba(df)[:, 1]
        df["stockout_risk"] = (probs * 100).round(2)

    st.success("Predictions complete!")

    st.subheader("📈 Risk Results")
    st.dataframe(df)

    st.download_button(
        "⬇ Download Predictions",
        df.to_csv(index=False),
        file_name="stockout_predictions.csv",
        mime="text/csv"
    )

else:
    st.info("Upload a CSV file to begin scoring.")
