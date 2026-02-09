"""Load and clean Telco churn data with feature engineering."""
import os
import pandas as pd
import numpy as np
from .config import (
    RAW_DATA_PATH,
    CLEAN_DATA_PATH,
    DROP_COLS,
    CAT_COLS,
    TARGET_COL,
    PROJECT_ROOT,
)


def load_raw_data(path=None):
    """Load raw Excel/CSV data."""
    path = path or RAW_DATA_PATH
    if not os.path.exists(path):
        # Fallback: project root (e.g. Telco_customer_churn.xlsx)
        alt = os.path.join(PROJECT_ROOT, "Telco_customer_churn.xlsx")
        if os.path.exists(alt):
            path = alt
        else:
            raise FileNotFoundError(f"Data not found: {path}")
    if path.endswith(".xlsx") or path.endswith(".xls"):
        return pd.read_excel(path)
    return pd.read_csv(path)


def clean_total_charges(df):
    """Convert Total Charges to numeric; fill missing with 0 or tenure * monthly."""
    tc = pd.to_numeric(df["Total Charges"], errors="coerce")
    missing = tc.isna()
    if missing.any():
        # Approximate: tenure * monthly charges where available
        approx = df.loc[missing, "Tenure Months"] * df.loc[missing, "Monthly Charges"]
        tc = tc.fillna(approx).fillna(0)
    return tc


def engineer_features(df):
    """Add tenure buckets and engagement metrics."""
    df = df.copy()
    # Tenure buckets (months)
    df["Tenure Bucket"] = pd.cut(
        df["Tenure Months"],
        bins=[-1, 12, 24, 48, 72, 1000],
        labels=["0-12", "13-24", "25-48", "49-72", "73+"],
    )
    # Total spend proxy (if Total Charges was missing we already filled)
    if "Total Charges" in df.columns:
        df["Total Charges"] = clean_total_charges(df)
        df["Avg Monthly Spend"] = np.where(
            df["Tenure Months"] > 0,
            df["Total Charges"] / df["Tenure Months"],
            df["Monthly Charges"],
        )
    return df


def prepare_data(path=None, save_cleaned=True, return_customer_id=False):
    """
    Load, clean, and optionally save cleaned data.
    CustomerID is never used as a feature (it is dropped with DROP_COLS).
    When return_customer_id=True, returns (df, customer_id_series) so callers
    can merge CustomerID into outputs (e.g. predictions.csv for Salesforce sync).
    """
    df = load_raw_data(path)
    # Keep CustomerID separately for output-only use (not for modeling)
    customer_id_series = None
    if return_customer_id and "CustomerID" in df.columns:
        customer_id_series = df["CustomerID"].copy()
    # Drop unused columns (including CustomerID) for features
    to_drop = [c for c in DROP_COLS if c in df.columns]
    df = df.drop(columns=to_drop, errors="ignore")
    # Fix Total Charges
    if "Total Charges" in df.columns:
        df["Total Charges"] = clean_total_charges(df)
    # Feature engineering
    df = engineer_features(df)
    # Ensure target exists
    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column '{TARGET_COL}' not in data")
    # Drop rows with missing target
    df = df.dropna(subset=[TARGET_COL])
    # Align customer id to final row set (same rows as df after dropna)
    if customer_id_series is not None:
        customer_id_series = customer_id_series.loc[df.index]
    if save_cleaned:
        os.makedirs(os.path.dirname(CLEAN_DATA_PATH), exist_ok=True)
        df.to_csv(CLEAN_DATA_PATH, index=False)
    if return_customer_id:
        return df, customer_id_series
    return df


def get_feature_columns(df):
    """Get list of feature columns (exclude target and non-feature)."""
    exclude = {TARGET_COL}
    return [c for c in df.columns if c not in exclude]
