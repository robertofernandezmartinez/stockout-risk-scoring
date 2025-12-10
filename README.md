# 🛒 Retail Stockout Risk Scoring

This project predicts **retail stockout risk within 14 days** and estimates the **economic impact** of potential stockouts.  
A trained XGBoost model evaluates products and identifies those that require urgent replenishment.

The solution includes:
- Automated **feature engineering, resampling**, and model training
- A **Streamlit** web app for business users
- A **deployed scoring pipeline** ready for production
- **Expected Loss** analytics to prioritize inventory decisions

---

## 🚀 Live Application

Try the app here 👇  
👉 **https://retail-stockout-risk-scoring.streamlit.app/**

Upload a CSV containing store inventory data and receive:
- Stockout probability per product
- Expected economic loss estimation
- Full ranked table of most critical items

---

## 📌 Key Features

✔ Stockout prediction within 14 days  
✔ Handles **class imbalance** using RandomOverSampler  
✔ **Economic Loss** calculator  
✔ Clean reusable ML pipeline  
✔ Deployment-ready model stored in GitHub Releases  
✔ Interactive UI for planners & business teams  

---

## 📊 Expected Loss Formula

To support business prioritization, we compute:

> **Expected Loss (€) = Stockout_Risk × Daily_Demand × Price × Stockout_Duration**

Where:
- **Stockout_Risk** → model-predicted probability
- **Daily_Demand** → expected units sold per day
- **Price** → product selling price (€)
- **Stockout_Duration** → number of days impacted if stockout occurs (set to 14 in this version)

This allows ranking products not only by probability of shortage,  
but also **financial impact**, improving decision-making.

---

## 📁 Project Structure

📦 retail-stockout-risk-scoring

- 02_Data/
    - 01_Raw - Original inventory dataset

- 03_Notebooks
    - 01_setup.ipynb - Environment setup & library preparation
    - 02_data_quality.ipynb - Data validation & cleaning checks
    - 03_eda.ipynb - Exploratory Data Analysis
    - 04_feature_engineering.ipynb - Feature transformations & new features
    - 05_feature_preselection.ipynb - Feature ranking and selection
    - 06_class_balancing.ipynb - Handling imbalance (SMOTE, undersampling)
    - 07_modeling_classification.ipynb - Model training with XGBoost & tuning
    - 08_preparation_of_production_script.ipynb - Pipeline packaging for execution
    - 09_retraining_script.ipynb - Automated model retraining logic
    - 10_execution_script.ipynb - Test execution with new incoming data

- 04_Models - Model artifacts
    - `pipe_execution.pkl` - Loaded by Streamlit (file stored in Releases)

- 05_Outputs - Auto-generated predictions (ignored in Git)

- `app.py` - Streamlit scoring application

- `requirements.txt` - Python dependencies

- README.md - Documentation (this file)


## 🧠 Model Overview

- Algorithm: **XGBoost Classifier**
- Tuning: RandomizedSearchCV (with recall priority)
- Evaluation metrics: Recall, Precision, ROC-AUC
- Pipeline includes:
  - Column processing & transformations
  - One-Hot / Target Encoding
  - Class balancing
  - Scaling

The pipeline is **fully serialized using cloudpickle** for deployment.

---

## 📦 Installation (Local)

```bash
# Create environment
conda create -n stockout python=3.10
conda activate stockout

# Install dependencies
pip install -r requirements.txt

# Run Streamlit app
streamlit run app.py
