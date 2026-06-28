# 📊 AI Customer Churn Predictor

End-to-end machine learning system for customer churn prediction. Supports per-client training, inference on new data, and automated report generation through a REST API.

---

# 🚀 Overview

This project implements a lightweight SaaS-like ML platform that allows you to:

- upload customer datasets (training)
- train a dedicated model per client (multi-tenant architecture)
- upload new customer data for inference
- generate churn prediction reports
- export results as CSV files

---

# 🧠 Architecture

Training & inference flow:

customers.csv (training data)
        ↓
FastAPI /train
        ↓
ML Pipeline (XGBoost + preprocessing)
        ↓
model.pkl (per client_id)
        ↓
predict_customers.csv (new customers)
        ↓
FastAPI /report
        ↓
churn_report.csv

---

# 📦 Project Structure

AI_Customer_Churn_Predictor/
│
├── api.py                 # FastAPI backend
├── train.py              # ML training pipeline
├── client.py             # simulated client SDK
├── clients_data/         # per-client storage (datasets + reports)
├── models/               # saved trained models
├── clients.db            # client registry database

---

# ⚙️ Tech Stack

Machine Learning:
- pandas
- scikit-learn
- xgboost
- joblib

Backend:
- FastAPI
- uvicorn

Client:
- requests

Storage:
- SQLite (client registry)
- local filesystem (datasets + models)

---

# 🧪 Features

## Multi-client ML system
Each client has:
- isolated dataset
- dedicated trained model
- independent churn reports

---

## Automated training
Endpoint:
POST /train/

Includes:
- automatic feature detection
- preprocessing pipeline (encoding + scaling)
- missing value handling
- class imbalance support

---

## Upload training dataset
POST /upload-dataset/

Uploads historical customer data (customers.csv)

---

## Upload inference dataset
POST /upload-predict-data/

Uploads new customers for churn prediction

---

## Churn report generation
GET /report/

Generates:
- churn probability per customer
- binary prediction (0/1)
- aggregated summary statistics

Output saved to:
clients_data/{client_id}/churn_report.csv

---

# 📊 Dataset Examples

## customers.csv (TRAINING SET)

age,gender,time_on_site,pages_viewed,previous_purchases,cart_added,purchased
22,F,1056,19,4,0,1
58,F,93,17,12,1,1
40,M,560,24,6,1,0

---

## predict_customers.csv (INFERENCE SET)

age,gender,time_on_site,pages_viewed,previous_purchases,cart_added
30,M,400,12,2,0
45,F,800,25,7,1
28,M,150,10,1,0

---

# 🚀 How to Run

## 1. Install dependencies

pip install fastapi uvicorn pandas scikit-learn xgboost joblib requests

---

## 2. Start the API server

uvicorn api:app --host 0.0.0.0 --port 8000 --reload

---

## 3. Run the client

python client.py

---

# 🔁 End-to-end workflow

1. Upload customers.csv
2. Train model
3. Upload predict_customers.csv
4. Run inference
5. Generate churn report
6. Save output locally

---

# 📈 Example output client

{
  "total_customers": 10,
  "avg_churn_probability": 0.62,
  "high_risk_customers": 4
}

---

# ⚠️ Important requirements

- Training and prediction datasets must share identical feature columns
- Column names and data types must be consistent
- Target column exists only in training data
- Prediction dataset must NOT include the target column

---

# 🧠 Technical notes

This system implements:
- automated ML pipeline
- dynamic feature preprocessing
- multi-tenant architecture
- batch inference engine
- persistent model storage per client

---

# 🚀 Possible improvements

- API key authentication
- Streamlit dashboard
- Model versioning system

---

# 📌 Project Summary

AI Customer Churn Predictor is a SaaS-like machine learning platform designed for scalable churn prediction with per-client models and production-style API architecture.
