"""Configuration for churn prediction project."""
import os

# Paths (relative to project root)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

# Data
RAW_DATA_PATH = os.path.join(DATA_DIR, "Telco_customer_churn.xlsx")
CLEAN_DATA_PATH = os.path.join(DATA_DIR, "cleaned_churn.csv")
PREDICTIONS_PATH = os.path.join(DATA_DIR, "predictions.csv")
PREDICTIONS_ENRICHED_PATH = os.path.join(DATA_DIR, "predictions_enriched.csv")

# Model
MODEL_PATH = os.path.join(MODELS_DIR, "churn_model.joblib")
ENCODER_PATH = os.path.join(MODELS_DIR, "feature_encoder.joblib")
SCALER_PATH = os.path.join(MODELS_DIR, "feature_scaler.joblib")
PREPROCESS_PATH = os.path.join(MODELS_DIR, "preprocess.joblib")
FEATURE_NAMES_PATH = os.path.join(MODELS_DIR, "feature_names.joblib")
FEATURE_NAMES_JSON = os.path.join(MODELS_DIR, "feature_names.json")
COLUMN_META_PATH = os.path.join(MODELS_DIR, "column_meta.joblib")
MODEL_META_PATH = os.path.join(MODELS_DIR, "model_meta.json")

# Dashboard auth (SQLite)
AUTH_DB_PATH = os.path.join(DATA_DIR, "users.db")

# Email verification (from env; set APP_URL and SMTP_* to enable)
APP_URL = os.environ.get("APP_URL", "http://localhost:8501").rstrip("/")
SMTP_HOST = os.environ.get("SMTP_HOST", "")
SMTP_PORT = int(os.environ.get("SMTP_PORT", "587"))
SMTP_USER = os.environ.get("SMTP_USER", "")
SMTP_PASSWORD = os.environ.get("SMTP_PASSWORD", "")
FROM_EMAIL = os.environ.get("FROM_EMAIL", SMTP_USER or "noreply@example.com")
EMAIL_VERIFICATION_ENABLED = bool(SMTP_HOST and SMTP_USER and SMTP_PASSWORD)

# Explainability (SHAP)
REPORTS_FIGURES_DIR = os.path.join(PROJECT_ROOT, "reports", "figures")
SHAP_TOP_K_DRIVERS = 5  # Top drivers per customer in Top_Drivers string

# Training
RANDOM_STATE = 42
TEST_SIZE = 0.2
TARGET_COL = "Churn Value"

# Columns to drop (identifiers, leakage, or not useful for prediction)
DROP_COLS = [
    "CustomerID",
    "Count",
    "Country",
    "State",
    "City",
    "Zip Code",
    "Lat Long",
    "Latitude",
    "Longitude",
    "Churn Label",
    "Churn Reason",  # Only present for churned; leakage
    "CLTV",  # Customer lifetime value - can be leakage
    "Churn Score",  # IBM dataset: pre-computed score from churn label — leakage; drop and retrain
]

# Categorical columns for encoding (after dropping)
CAT_COLS = [
    "Gender",
    "Senior Citizen",
    "Partner",
    "Dependents",
    "Phone Service",
    "Multiple Lines",
    "Internet Service",
    "Online Security",
    "Online Backup",
    "Device Protection",
    "Tech Support",
    "Streaming TV",
    "Streaming Movies",
    "Contract",
    "Paperless Billing",
    "Payment Method",
    "Tenure Bucket",
]
