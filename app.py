import streamlit as st
import pandas as pd
import requests
import os
import cloudpickle

MODEL_URL = "https://github.com/robertofernandezmartinez/retail-stockout-risk-scoring/releases/download/v1.0.0/pipe_execution_streamlit.pkl"
MODEL_PATH = "pipe_execution_streamlit.pkl"

@st.cache_resource
def load_pipeline():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("📥 Cargando modelo..."):
            r = requests.get(MODEL_URL)
            with open(MODEL_PATH, "wb") as f:
                f.write(r.content)
    with open(MODEL_PATH, "rb") as f:
        return cloudpickle.load(f)

pipeline = load_pipeline()

st.title("🛒 Retail Stockout Risk Scoring")
st.markdown("Upload your inventory file to estimate stockout probability within 14 days.")

uploaded_file = st.file_uploader("📤 Upload CSV file")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("📊 Data Preview")
    st.dataframe(df.head())

    # Predictions
    preds = pipeline.predict_proba(df)[:, 1]
    df["Stockout_Risk"] = preds.round(3)

    # Filters UI
    st.subheader("🎯 Filters")
    category_filter = st.selectbox("Filter by Category:", ["All"] + sorted(df["category"].unique()))
    store_filter = st.selectbox("Filter by Store:", ["All"] + sorted(df["store_id"].unique()))

    filtered_df = df.copy()
    if category_filter != "All":
        filtered_df = filtered_df[filtered_df["category"] == category_filter]
    if store_filter != "All":
        filtered_df = filtered_df[filtered_df["store_id"] == store_filter]

    # Sort by risk
    filtered_df = filtered_df.sort_values(by="Stockout_Risk", ascending=False)

    # Highlight function
    def highlight_risk(val):
        if val >= 0.8: return "background-color: #ff4d4d; color: white;"  # Red
        if val >= 0.5: return "background-color: #ffdc73;"  # Yellow
        return "background-color: #b6fcb6;"  # Green

    st.subheader("🔥 Top Stockout Risks")
    top_risk = filtered_df.head(10)
    st.dataframe(top_risk.style.applymap(highlight_risk, subset=["Stockout_Risk"]))

    # Visualization chart
    st.subheader("📈 Top 5 Critical Products")
    st.bar_chart(top_risk.set_index("product_id")["Stockout_Risk"].head(5))

    # Full scoring table
    st.subheader("📋 Full Predictions")
    st.dataframe(filtered_df.style.applymap(highlight_risk, subset=["Stockout_Risk"]))

    # Download button
    st.download_button(
        label="📥 Download Full Results",
        data=filtered_df.to_csv(index=False),
        file_name="predictions_stockout_risk.csv",
        mime="text/csv"
    )

else:
    st.info("⬆️ Upload your CSV to begin scoring")
