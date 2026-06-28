from fastapi import FastAPI, UploadFile, File, BackgroundTasks
import pandas as pd
import os
import shutil
import sqlite3
import joblib

from train import train_model

app = FastAPI()

# ===========================
# STORAGE
# ===========================

BASE_DIR = "AI_Customer_Churn_Predictor/clients_data"
MODEL_DIR = "AI_Customer_Churn_Predictor/models"

os.makedirs(BASE_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)


# ===========================
# DATABASE (CLIENTS)
# ===========================

DB_PATH = "AI_Customer_Churn_Predictor/clients.db"


def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    c.execute("""
    CREATE TABLE IF NOT EXISTS clients (
        client_id TEXT PRIMARY KEY,
        dataset_path TEXT,
        model_path TEXT
    )
    """)

    conn.commit()
    conn.close()

init_db()


def save_client(client_id, dataset_path=None, model_path=None):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    c.execute("SELECT client_id FROM clients WHERE client_id=?", (client_id,))
    exists = c.fetchone()

    if exists:
        if dataset_path:
            c.execute("UPDATE clients SET dataset_path=? WHERE client_id=?", (dataset_path, client_id))
        if model_path:
            c.execute("UPDATE clients SET model_path=? WHERE client_id=?", (model_path, client_id))
    else:
        c.execute("""
        INSERT INTO clients (client_id, dataset_path, model_path)
        VALUES (?, ?, ?)
        """, (client_id, dataset_path, model_path))

    conn.commit()
    conn.close()


def get_client(client_id):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    c.execute("SELECT * FROM clients WHERE client_id=?", (client_id,))
    row = c.fetchone()

    conn.close()
    return row


# ===========================
# 1. UPLOAD DATASET
# ===========================

@app.post("/upload-dataset/")
async def upload_dataset(client_id: str, file: UploadFile = File(...)):

    client_folder = os.path.join(BASE_DIR, client_id)
    os.makedirs(client_folder, exist_ok=True)

    file_path = os.path.join(client_folder, "dataset.csv")

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    save_client(client_id, dataset_path=file_path)

    return {
        "status": "uploaded",
        "client_id": client_id,
        "path": file_path
    }


@app.post("/upload-predict-data/")
async def upload_predict(client_id: str, file: UploadFile = File(...)):

    client_folder = os.path.join(BASE_DIR, client_id)
    os.makedirs(client_folder, exist_ok=True)

    file_path = os.path.join(client_folder, "predict_customers.csv")

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    save_client(client_id, dataset_path=file_path)

    return {
        "status": "predict_data_uploaded",
        "path": file_path
    }

# ===========================
# 2. TRAIN MODEL
# ===========================

@app.post("/train/")
async def train(client_id: str, target_column: str, background_tasks: BackgroundTasks):

    client = get_client(client_id)

    if not client:
        return {"error": "client not found"}

    dataset_path = client[1]  # customers.csv

    model_path = os.path.join(MODEL_DIR, f"{client_id}_model.pkl")

    def _train():
        result = train_model(dataset_path, target_column, model_path)
        save_client(client_id, model_path=model_path)
        print("TRAINING DONE:", result)

    background_tasks.add_task(_train)

    return {"status": "training_started"}


# ===========================
# 3. CHURN REPORT
# ===========================

@app.get("/report/")
async def report(client_id: str, target_column: str):

    client = get_client(client_id)

    if not client or not client[2]:
        return {"error": "model not found"}

    model_path = client[2]

    # 👇 QUI CAMBIA TUTTO: usa predict_customers.csv
    predict_path = os.path.join(BASE_DIR, client_id, "predict_customers.csv")

    model = joblib.load(model_path)
    df = pd.read_csv(predict_path)

    # se target esiste lo ignoriamo
    X = df.drop(columns=[target_column], errors="ignore")

    proba = model.predict_proba(X)[:, 1]
    pred = model.predict(X)

    df["churn_probability"] = proba
    df["prediction"] = pred

    report_path = os.path.join(BASE_DIR, client_id, "churn_report.csv")
    df.to_csv(report_path, index=False)

    summary = {
        "total_customers": len(df),
        "avg_churn_probability": float(proba.mean()),
        "high_risk_customers": int((proba > 0.7).sum())
    }

    return {
        "client_id": client_id,
        "summary": summary,
        "report_path": report_path
    }