import requests
import time
import json

BASE_URL = "http://127.0.0.1:8000"


# ===========================
# 1. UPLOAD TRAINING DATASET
# ===========================

def upload_training_data(client_id: str, file_path: str):

    url = f"{BASE_URL}/upload-dataset/"

    with open(file_path, "rb") as f:
        files = {"file": f}
        params = {"client_id": client_id}

        r = requests.post(url, params=params, files=files)

    print("\n[TRAINING DATA UPLOAD]")
    print(r.json())

    return r.json()


# ===========================
# 2. TRAIN MODEL
# ===========================

def train_model(client_id: str, target_column: str):

    url = f"{BASE_URL}/train/"

    params = {
        "client_id": client_id,
        "target_column": target_column
    }

    r = requests.post(url, params=params)

    print("\n[TRAIN REQUEST]")
    print(r.json())

    return r.json()


# ===========================
# 3. UPLOAD PREDICTION DATASET
# ===========================

def upload_prediction_data(client_id: str, file_path: str):

    url = f"{BASE_URL}/upload-predict-data/"

    with open(file_path, "rb") as f:
        files = {"file": f}
        params = {"client_id": client_id}

        r = requests.post(url, params=params, files=files)

    print("\n[PREDICTION DATA UPLOAD]")
    print(r.json())

    return r.json()


# ===========================
# 4. GET CHURN REPORT
# ===========================

def get_report(client_id: str, target_column: str):

    url = f"{BASE_URL}/report/"

    params = {
        "client_id": client_id,
        "target_column": target_column
    }

    r = requests.get(url, params=params)

    print("\n[CHURN REPORT]")
    print(json.dumps(r.json(), indent=2))

    return r.json()


# ===========================
# FULL PIPELINE
# ===========================

def run_pipeline(
    client_id: str,
    training_file: str,
    prediction_file: str,
    target_column: str
):

    # STEP 1 - training data
    upload_training_data(client_id, training_file)

    # STEP 2 - train model
    train_model(client_id, target_column)

    print("\nWaiting for training to finish...")
    time.sleep(10)

    # STEP 3 - prediction data
    upload_prediction_data(client_id, prediction_file)

    # STEP 4 - get report
    report = get_report(client_id, target_column)

    # STEP 5 - save report locally
    output_file = f"{client_id}_churn_report.json"

    with open(output_file, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\nSaved report to {output_file}")


# ===========================
# EXECUTION EXAMPLE
# ===========================

if __name__ == "__main__":

    run_pipeline(
        client_id="client_001",
        training_file="customers.csv",
        prediction_file="predict_customers.csv",
        target_column="purchased"
    )