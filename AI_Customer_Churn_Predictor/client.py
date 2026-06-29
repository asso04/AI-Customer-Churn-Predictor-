import sys
import os
import json
import requests
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QPushButton, QFileDialog, QTextEdit,
    QFrame, QGridLayout, QGroupBox, QMessageBox
)
from PySide6.QtCore import QThread, Signal
from PySide6.QtGui import QFont

BASE_URL = os.getenv("API_DASHBOARD_URL", "http://yourserverip:8000")
HTTP_TIMEOUT = (5.0, 30.0)


# =====================================================================
# 1. API WORKER THREAD (For Main Workflow Actions)
# =====================================================================
class ApiWorker(QThread):
    finished = Signal(str, str, bool)

    def __init__(self, operation: str, **kwargs):
        super().__init__()
        self.operation = operation
        self.kwargs = kwargs

    def run(self):
        global r
        try:
            if self.operation == "train":
                try:
                    requests.get(BASE_URL, timeout=3.0)
                except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
                    error_msg = f"[SERVER UNREACHABLE] Cannot initiate training. The server at {BASE_URL} is offline or unreachable."
                    self.finished.emit(self.operation, error_msg, False)
                    return

            if self.operation == "upload_train":
                url = f"{BASE_URL}/upload-dataset/"
                with open(self.kwargs['file_path'], "rb") as f:
                    r = requests.post(
                        url,
                        params={"client_id": self.kwargs['client_id']},
                        files={"file": f},
                        timeout=HTTP_TIMEOUT
                    )
                r.raise_for_status()
                self.finished.emit("upload_train", json.dumps(r.json(), indent=2), True)

            elif self.operation == "train":
                url = f"{BASE_URL}/train/"
                r = requests.post(
                    url,
                    params={"client_id": self.kwargs['client_id'], "target_column": self.kwargs['target_column']},
                    timeout=HTTP_TIMEOUT
                )
                r.raise_for_status()
                self.finished.emit("train", json.dumps(r.json(), indent=2), True)

            elif self.operation == "upload_predict":
                url = f"{BASE_URL}/upload-predict-data/"
                with open(self.kwargs['file_path'], "rb") as f:
                    r = requests.post(
                        url,
                        params={"client_id": self.kwargs['client_id']},
                        files={"file": f},
                        timeout=HTTP_TIMEOUT
                    )
                r.raise_for_status()
                self.finished.emit("upload_predict", json.dumps(r.json(), indent=2), True)

            elif self.operation == "report":
                url = f"{BASE_URL}/report/"
                r = requests.get(
                    url,
                    params={"client_id": self.kwargs['client_id'], "target_column": self.kwargs['target_column']},
                    timeout=HTTP_TIMEOUT
                )
                r.raise_for_status()
                self.finished.emit("report", json.dumps(r.json(), indent=2), True)

        except requests.exceptions.Timeout:
            error_msg = f"Error: Request timed out (Timeout > {HTTP_TIMEOUT[1]}s)."
            self.finished.emit(self.operation, error_msg, False)
        except requests.exceptions.ConnectionError:
            error_msg = f"Connection Error: Could not reach the server at {BASE_URL}."
            self.finished.emit(self.operation, error_msg, False)
        except requests.exceptions.HTTPError as http_err:
            error_msg = f"HTTP Error: {str(http_err)}\nResponse: {r.text if 'r' in locals() else 'None'}"
            self.finished.emit(self.operation, error_msg, False)
        except Exception as e:
            error_msg = f"Unexpected Error:\n{str(e)}"
            self.finished.emit(self.operation, error_msg, False)


# =====================================================================
# 2. VALIDATION WORKER THREAD (Dedicated to Client ID Check Only)
# =====================================================================
class ValidationWorker(QThread):
    validation_finished = Signal(str, bool)

    def __init__(self, client_id: str):
        super().__init__()
        self.client_id = client_id

    def run(self):
        try:
            url = f"{BASE_URL}/check-client/{self.client_id}"
            r = requests.get(url, timeout=3.0)
            r.raise_for_status()
            self.validation_finished.emit(r.text, True)
        except Exception:
            self.validation_finished.emit("", False)


# =====================================================================
# 3. MAIN DASHBOARD WINDOW WITH STRICT STATE MANAGEMENT
# =====================================================================
class ChurnDashboard(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Enterprise Churn & Analytics Dashboard")
        self.resize(950, 680)

        self.worker = None
        self.val_worker = None

        # State tracking flags
        self.has_selected_train_file = False
        self.is_train_uploaded = False
        self.is_model_trained = False
        self.has_selected_predict_file = False
        self.is_predict_uploaded = False

        self.init_ui()
        self.apply_styles()
        self.update_pipeline_states()

        self.console.append(f"[INFO] API Endpoint configured to: {BASE_URL}\n")

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(15, 15, 15, 15)
        main_layout.setSpacing(15)

        # ================= CONFIGURATION PANEL (LEFT) =================
        left_panel = QVBoxLayout()

        title = QLabel("Predictive Dashboard v1.0")
        title.setFont(QFont("Segoe UI", 16, QFont.Weight.Bold))
        title.setStyleSheet("color: #00adb5; margin-bottom: 5px;")
        left_panel.addWidget(title)

        config_group = QGroupBox("Request Parameters")
        config_layout = QGridLayout(config_group)

        config_layout.addWidget(QLabel("Client ID:"), 0, 0)
        self.client_id_input = QLineEdit("client_001")
        self.client_id_input.editingFinished.connect(self.validate_client_id)
        config_layout.addWidget(self.client_id_input, 0, 1)

        config_layout.addWidget(QLabel("Target Column:"), 1, 0)
        self.target_col_input = QLineEdit("purchased")
        config_layout.addWidget(self.target_col_input, 1, 1)
        left_panel.addWidget(config_group)

        actions_group = QGroupBox("Pipeline Workflow")
        actions_layout = QVBoxLayout(actions_group)
        actions_layout.setSpacing(8)

        # Step 1: Training Data
        self.btn_select_train = QPushButton("Browse Training Dataset...")
        self.btn_select_train.clicked.connect(lambda: self.select_file("train"))
        self.lbl_train_path = QLabel("No file selected")
        self.lbl_train_path.setWordWrap(True)
        self.lbl_train_path.setStyleSheet("color: #aaaaaa; font-style: italic;")
        self.btn_upload_train = QPushButton("1. Upload Training Dataset")
        self.btn_upload_train.clicked.connect(self.action_upload_train)

        actions_layout.addWidget(self.btn_select_train)
        actions_layout.addWidget(self.lbl_train_path)
        actions_layout.addWidget(self.btn_upload_train)
        actions_layout.addWidget(self.create_separator())

        # Step 2: Train Model
        self.btn_train = QPushButton("2. Run Model Training")
        self.btn_train.clicked.connect(self.action_train)
        actions_layout.addWidget(self.btn_train)
        actions_layout.addWidget(self.create_separator())

        # Step 3: Prediction Data
        self.btn_select_predict = QPushButton("Browse Prediction Dataset...")
        self.btn_select_predict.clicked.connect(lambda: self.select_file("predict"))
        self.lbl_predict_path = QLabel("No file selected")
        self.lbl_predict_path.setWordWrap(True)
        self.lbl_predict_path.setStyleSheet("color: #aaaaaa; font-style: italic;")
        self.btn_upload_predict = QPushButton("3. Upload Prediction Dataset")
        self.btn_upload_predict.clicked.connect(self.action_upload_predict)

        actions_layout.addWidget(self.btn_select_predict)
        actions_layout.addWidget(self.lbl_predict_path)
        actions_layout.addWidget(self.btn_upload_predict)
        actions_layout.addWidget(self.create_separator())

        # Step 4: Report Generation
        self.btn_report = QPushButton("4. Request Churn Report")
        self.btn_report.setStyleSheet("background-color: #00adb5; color: #ffffff; font-weight: bold; padding: 10px;")
        self.btn_report.clicked.connect(self.action_get_report)
        actions_layout.addWidget(self.btn_report)

        left_panel.addWidget(actions_group)
        left_panel.addStretch()

        # ================= MONITORING PANEL (RIGHT) =================
        self.console = QTextEdit()
        self.console.setReadOnly(True)
        self.console.setFont(QFont("Consolas", 10))

        right_panel = QVBoxLayout()
        right_panel.addWidget(QLabel("Server Logs & JSON Responses:"))
        right_panel.addWidget(self.console)

        left_widget = QWidget()
        left_widget.setLayout(left_panel)
        main_layout.addWidget(left_widget, 4)
        main_layout.addLayout(right_panel, 6)

    def create_separator(self):
        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setFrameShadow(QFrame.Shadow.Sunken)
        line.setStyleSheet("background-color: #393e46; margin: 4px 0px;")
        return line

    # =====================================================================
    # 4. PIPELINE STATE ENFORCEMENT & ASYNC CHECKS
    # =====================================================================
    def update_pipeline_states(self):
        """Calculates precise interactive states based on workflow progress."""
        self.btn_upload_train.setEnabled(self.has_selected_train_file)
        self.btn_train.setEnabled(self.is_train_uploaded)
        self.btn_select_predict.setEnabled(self.is_model_trained)
        self.btn_upload_predict.setEnabled(self.is_model_trained and self.has_selected_predict_file)
        self.btn_report.setEnabled(self.is_predict_uploaded)

    def set_ui_locked(self, is_locked: bool):
        """Completely disables UI nodes during network transactions."""
        self.btn_select_train.setDisabled(is_locked)
        self.client_id_input.setDisabled(is_locked)
        self.target_col_input.setDisabled(is_locked)

        if is_locked:
            self.btn_upload_train.setDisabled(True)
            self.btn_train.setDisabled(True)
            self.btn_select_predict.setDisabled(True)
            self.btn_upload_predict.setDisabled(True)
            self.btn_report.setDisabled(True)
        else:
            self.update_pipeline_states()

    def validate_client_id(self):
        """Triggers a dedicated validation worker to check the Client ID."""
        client_id = self.client_id_input.text().strip()
        if not client_id:
            return

        self.val_worker = ValidationWorker(client_id)
        self.val_worker.validation_finished.connect(self.on_check_client_finished)
        self.val_worker.start()

    def on_check_client_finished(self, result: str, success: bool):
        """Processes database feedback without overwriting global transaction worker states."""
        if success and result:
            try:
                data = json.loads(result)
                if data.get("exists"):
                    QMessageBox.warning(self, "User Exists", "existing user, try another name")
                    self.client_id_input.setStyleSheet(
                        "background-color: #393e46; color: #eeeeee; border: 1px solid #ff4444; padding: 5px;")
                    self.is_train_uploaded = False
                else:
                    self.client_id_input.setStyleSheet(
                        "background-color: #393e46; color: #eeeeee; border: 1px solid #393e46; padding: 5px;")

                self.update_pipeline_states()
            except Exception as e:
                print(f"Error parsing validation response: {e}")

    def select_file(self, mode: str):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Dataset CSV", "", "CSV Files (*.csv)")
        if file_path:
            if os.path.getsize(file_path) == 0:
                QMessageBox.warning(self, "Empty File", "The selected CSV file is empty and cannot be processed.")
                return

            if mode == "train":
                self.lbl_train_path.setText(file_path)
                self.has_selected_train_file = True
            else:
                self.lbl_predict_path.setText(file_path)
                self.has_selected_predict_file = True

            self.update_pipeline_states()

    def start_worker(self, operation: str, **kwargs):
        self.set_ui_locked(True)
        self.console.append(f"[PROCESSING] Executing: {operation.upper()}...")

        self.worker = ApiWorker(operation, **kwargs)
        self.worker.finished.connect(self.on_worker_finished)
        self.worker.start()

    def on_worker_finished(self, operation: str, result: str, success: bool):
        self.set_ui_locked(False)

        status_tag = "[OK]" if success else "[CRITICAL ERROR]"
        self.console.append(f"{status_tag} == {operation.upper()} ==")
        self.console.append(result)
        self.console.append("\n" + "-" * 50 + "\n")

        if success:
            if operation == "upload_train":
                self.is_train_uploaded = True
            elif operation == "train":
                self.is_model_trained = True
            elif operation == "upload_predict":
                self.is_predict_uploaded = True

            self.update_pipeline_states()
        else:
            QMessageBox.critical(self, "Pipeline Error",
                                 f"Operation '{operation}' could not complete.\nSee control logs for details.")

    def action_upload_train(self):
        path = self.lbl_train_path.text()
        self.start_worker("upload_train", client_id=self.client_id_input.text(), file_path=path)

    def action_train(self):
        if not self.client_id_input.text().strip() or not self.target_col_input.text().strip():
            QMessageBox.warning(self, "Empty Fields", "Client ID and Target Column are mandatory.")
            return
        self.start_worker("train", client_id=self.client_id_input.text(), target_column=self.target_col_input.text())

    def action_upload_predict(self):
        path = self.lbl_predict_path.text()
        self.start_worker("upload_predict", client_id=self.client_id_input.text(), file_path=path)

    def action_get_report(self):
        self.start_worker("report", client_id=self.client_id_input.text(), target_column=self.target_col_input.text())

    # =====================================================================
    # 5. DESIGN APPLICATION (QSS)
    # =====================================================================
    def apply_styles(self):
        dark_stylesheet = """
            QMainWindow { background-color: #222831; }
            QGroupBox {
                color: #00adb5; font-weight: bold;
                border: 2px solid #393e46; border-radius: 6px;
                margin-top: 10px; padding-top: 12px;
            }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px; }
            QLabel { color: #eeeeee; font-size: 12px; }
            QLineEdit {
                background-color: #393e46; color: #eeeeee;
                border: 1px solid #393e46; border-radius: 4px; padding: 5px;
            }
            QLineEdit:focus { border: 1px solid #00adb5; }
            QPushButton {
                background-color: #393e46; color: #eeeeee;
                border: 1px solid #4f5b66; border-radius: 4px;
                padding: 6px; font-size: 11px;
            }
            QPushButton:hover { background-color: #00adb5; color: #222831; font-weight: bold; }
            QPushButton:disabled { background-color: #1a1a24; color: #555555; border: 1px solid #222831; }
            QTextEdit {
                background-color: #1a1a24; color: #39ff14;
                border: 2px solid #393e46; border-radius: 6px; padding: 10px;
            }
        """
        self.setStyleSheet(dark_stylesheet)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    dashboard = ChurnDashboard()
    dashboard.show()
    sys.exit(app.exec())
