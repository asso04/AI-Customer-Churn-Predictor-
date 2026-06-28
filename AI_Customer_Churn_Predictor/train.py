import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score
)

from xgboost import XGBClassifier


# ===========================
# DATASET VALIDATION
# ===========================

def validate_dataset(df: pd.DataFrame, target: str):
    errors = []
    warnings = []

    # 1. Basic checks
    if df is None or df.empty:
        raise ValueError("Dataset is empty or not loaded correctly.")

    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in dataset.")

    # 2. Missing target values
    if df[target].isnull().sum() > 0:
        errors.append(f"Target column contains {df[target].isnull().sum()} missing values.")

    # 3. Minimum rows check
    if len(df) < 50:
        warnings.append("Dataset is very small (<50 rows). Model may not generalize well.")

    # 4. Duplicate rows
    dup = df.duplicated().sum()
    if dup > 0:
        warnings.append(f"Dataset contains {dup} duplicate rows.")

    # 5. Target sanity check (binary classification expected)
    unique_targets = df[target].dropna().unique()

    if len(unique_targets) != 2:
        raise ValueError(
            f"Target must be binary for this model. Found {len(unique_targets)} classes: {unique_targets}"
        )

    # 6. Missing values ratio
    missing_ratio = df.isnull().mean().mean()

    if missing_ratio > 0.3:
        warnings.append(f"High missing values ratio detected: {missing_ratio:.2%}")

    # 7. High cardinality warning (categorical explosion risk)
    cat_cols = df.select_dtypes(include=["object", "category"]).columns

    for col in cat_cols:
        if col == target:
            continue
        if df[col].nunique() > 50:
            warnings.append(f"High cardinality column: {col} ({df[col].nunique()} unique values)")

    # Print results
    if errors:
        print("\n❌ ERRORS:")
        for e in errors:
            print("-", e)
        raise ValueError("Dataset validation failed.")

    if warnings:
        print("\n⚠️ WARNINGS:")
        for w in warnings:
            print("-", w)

    print("\n✅ Dataset validation passed.")


def train_model(DATASET_PATH: str, TARGET_COLUMN: str, MODEL_OUTPUT: str):
    # ===========================
    # LOAD DATA
    # ===========================

    df = pd.read_csv(DATASET_PATH)

    validate_dataset(df, TARGET_COLUMN)

    # ===========================
    # SPLIT FEATURES / TARGET
    # ===========================

    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    # ===========================
    # REMOVE ID-LIKE COLUMNS
    # ===========================

    id_keywords = ["id", "customerid", "clientid", "userid", "uuid"]

    drop_cols = [
        col for col in X.columns
        if any(k in col.lower() for k in id_keywords)
    ]

    if drop_cols:
        print("Removed ID columns:", drop_cols)
        X = X.drop(columns=drop_cols)

    # ===========================
    # REMOVE CONSTANT COLUMNS
    # ===========================

    constant_cols = [c for c in X.columns if X[c].nunique() <= 1]

    if constant_cols:
        print("Removed constant columns:", constant_cols)
        X = X.drop(columns=constant_cols)

    # ===========================
    # AUTO FEATURE DETECTION
    # ===========================

    numeric_features = X.select_dtypes(include=["int64", "float64", "int32", "float32"]).columns.tolist()

    categorical_features = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    # ===========================
    # PREPROCESSING
    # ===========================

    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ])

    # ===========================
    # CLASS BALANCE
    # ===========================

    counts = y.value_counts()
    scale = counts.max() / counts.min() if len(counts) == 2 else 1

    # ===========================
    # MODEL
    # ===========================

    model = XGBClassifier(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale,
        eval_metric="logloss",
        random_state=42
    )

    # ===========================
    # PIPELINE
    # ===========================

    pipeline = Pipeline([
        ("preprocessing", preprocessor),
        ("classifier", model)
    ])

    # ===========================
    # TRAIN / TEST SPLIT
    # ===========================

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # ===========================
    # TRAIN
    # ===========================

    pipeline.fit(X_train, y_train)

    # ===========================
    # EVALUATION
    # ===========================

    pred = pipeline.predict(X_test)

    print("\nAccuracy:", accuracy_score(y_test, pred))
    print("\nClassification Report:\n", classification_report(y_test, pred))
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, pred))

    proba = pipeline.predict_proba(X_test)[:, 1]
    print("\nROC AUC:", roc_auc_score(y_test, proba))

    # ===========================
    # SAVE MODEL
    # ===========================

    joblib.dump(pipeline, MODEL_OUTPUT)

    print(f"\nModel saved as: {MODEL_OUTPUT}")

    return {
        "status": "trained",
        "rows": len(df),
        "features": len(X.columns)
    }