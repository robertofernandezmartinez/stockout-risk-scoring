import streamlit as st
import pandas as pd
import requests
import os
import cloudpickle

MODEL_URL = "https://github.com/robertofernandezmartinez/retail-stockout-risk-scoring/releases/download/v1.0.0/pipe_execution.pkl"
MODEL_PATH = "pipe_execution.pkl"

@st.cache_resource
def load_pipeline():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading model..."):
            r = requests.get(MODEL_URL)
            with open(MODEL_PATH, "wb") as f:
                f.write(r.content)
    with open(MODEL_PATH, "rb") as f:
        return cloudpickle.load(f)

pipeline = load_pipeline()

st.title("Retail Stockout Risk Scoring")
st.write("Upload your inventory file to estimate stockout probability within 14 days. // stockout_risk = 1 is maximum level of risk")

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("📊 Data Preview")
    st.write(df.head())

    # Rename columns to match training
    df = df.rename(columns={
        "Date": "date",
        "Store ID": "store_id",
        "Product ID": "product_id",
        "Category": "category",
        "Region": "region",
        "Inventory Level": "inventory_level",
        "Units Sold": "units_sold",
        "Units Ordered": "units_ordered",
        "Demand Forecast": "demand_forecast",
        "Price": "price",
        "Discount": "discount",
        "Weather Condition": "weather",
        "Holiday/Promotion": "holiday_promo",
        "Competitor Pricing": "competitor_pricing",
        "Seasonality": "seasonality"
    })

    # Convert date column
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # Convert categorical
    if "holiday_promo" in df.columns:
        df["holiday_promo"] = df["holiday_promo"].astype("category")

    # Predict
    try:
        probs = pipeline.predict_proba(df)[:, 1]
        df["stockout_risk"] = probs

        # ==== 💰 Economic Loss Calculation ====
        df["demand_14d"] = df["demand_forecast"] * 14
        df["units_at_risk"] = (df["demand_14d"] - df["inventory_level"]).clip(lower=0)

        df["profit_per_unit"] = df["price"] - (df["price"] * df["discount"] / 100)
        df["economic_loss"] = df["stockout_risk"] * df["units_at_risk"] * df["profit_per_unit"]

        # Show predictions table
        st.subheader("📈 Predictions")
        st.write(df)

        # Download button
        st.download_button("Download Results", df.to_csv(index=False),
                        file_name="stockout_predictions.csv")

        # ==== 🔝 Top Loss Items ====
        high_risk = df.nlargest(10, "economic_loss")[["product_id", "category", "economic_loss"]]
        st.subheader("🔝 Top Items by Expected Loss (€)")
        st.dataframe(high_risk)

    except Exception as e:
        st.error(f"Prediction error: {str(e)}")

else:
    st.info("Upload a CSV to begin scoring.")