"""
Daily monitoring and drift detection for churn predictions.
Loads data/predictions.csv, computes daily metrics, appends to metrics_history.csv,
computes PSI vs baseline_reference.csv, writes drift_report.json.

Run after predictions: python -m monitoring.monitor
"""
import os
import sys
import json
from datetime import datetime

import pandas as pd
import numpy as np

# Project root (parent of monitoring/)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MONITORING_DIR = os.path.join(PROJECT_ROOT, "monitoring")
PREDICTIONS_PATH = os.path.join(DATA_DIR, "predictions.csv")
BASELINE_PATH = os.path.join(MONITORING_DIR, "baseline_reference.csv")
METRICS_HISTORY_PATH = os.path.join(MONITORING_DIR, "metrics_history.csv")
DRIFT_REPORT_PATH = os.path.join(MONITORING_DIR, "drift_report.json")

# Drift feature columns (must exist in predictions and baseline)
NUMERIC_DRIFT_COLS = ["Tenure Months", "Monthly Charges", "Total Charges"]
CATEGORICAL_DRIFT_COLS = ["Contract", "Internet Service", "Payment Method"]
DRIFT_COLS = NUMERIC_DRIFT_COLS + CATEGORICAL_DRIFT_COLS

# PSI thresholds: < 0.1 OK, 0.1–0.2 WARNING, >= 0.2 ALERT
PSI_WARNING = 0.1
PSI_ALERT = 0.2


def _ensure_monitoring_dir():
    os.makedirs(MONITORING_DIR, exist_ok=True)


def load_predictions(path=None):
    """Load today's predictions CSV. Returns DataFrame or None."""
    p = path or PREDICTIONS_PATH
    if not os.path.exists(p):
        return None
    try:
        df = pd.read_csv(p)
        if "Risk Level" not in df.columns or "Churn Probability" not in df.columns:
            return None
        return df
    except Exception:
        return None


def compute_daily_metrics(df):
    """Return dict: total_customers, high/medium/low counts and %, avg_churn_probability."""
    total = len(df)
    counts = df["Risk Level"].value_counts()
    high = int(counts.get("High", 0))
    medium = int(counts.get("Medium", 0))
    low = int(counts.get("Low", 0))
    avg_prob = float(df["Churn Probability"].mean()) if "Churn Probability" in df.columns else None
    return {
        "date": datetime.now().strftime("%Y-%m-%d"),
        "total_customers": total,
        "high_risk_count": high,
        "medium_risk_count": medium,
        "low_risk_count": low,
        "high_risk_pct": round(100 * high / total, 2) if total else 0,
        "medium_risk_pct": round(100 * medium / total, 2) if total else 0,
        "low_risk_pct": round(100 * low / total, 2) if total else 0,
        "avg_churn_probability": round(avg_prob, 4) if avg_prob is not None else None,
    }


def append_metrics_history(metrics, path=None):
    """Append one row to metrics_history.csv; create file with header if missing."""
    _ensure_monitoring_dir()
    path = path or METRICS_HISTORY_PATH
    row = pd.DataFrame([metrics])
    if os.path.exists(path):
        row.to_csv(path, mode="a", header=False, index=False)
    else:
        row.to_csv(path, mode="w", header=True, index=False)


def psi_numeric(baseline_series, current_series, n_bins=10):
    """
    Population Stability Index for a numeric feature.
    Bin both series the same way (from baseline quantiles), then PSI = sum((curr_pct - base_pct) * ln(curr_pct/base_pct)).
    """
    base = baseline_series.dropna()
    curr = current_series.dropna()
    if len(base) < 2 or len(curr) < 2:
        return 0.0
    try:
        _, bin_edges = np.histogram(base, bins=n_bins)
        bin_edges = np.unique(bin_edges)
        if len(bin_edges) < 2:
            return 0.0
        base_pct = np.histogram(base, bins=bin_edges)[0].astype(float) / len(base)
        curr_hist = np.histogram(curr, bins=bin_edges)[0].astype(float)
        curr_pct = curr_hist / len(curr)
        # Avoid zeros for log
        base_pct = np.clip(base_pct, 1e-6, 1.0)
        curr_pct = np.clip(curr_pct, 1e-6, 1.0)
        psi = np.sum((curr_pct - base_pct) * np.log(curr_pct / base_pct))
        return float(psi)
    except Exception:
        return 0.0


def psi_categorical(baseline_series, current_series):
    """
    PSI for categorical: same formula with category proportions.
    """
    base = baseline_series.astype(str).fillna("__missing__")
    curr = current_series.astype(str).fillna("__missing__")
    all_cats = set(base.unique()) | set(curr.unique())
    if not all_cats:
        return 0.0
    n_base = len(base)
    n_curr = len(curr)
    psi = 0.0
    for c in all_cats:
        base_pct = (base == c).sum() / n_base if n_base else 1e-6
        curr_pct = (curr == c).sum() / n_curr if n_curr else 1e-6
        base_pct = max(base_pct, 1e-6)
        curr_pct = max(curr_pct, 1e-6)
        psi += (curr_pct - base_pct) * np.log(curr_pct / base_pct)
    return float(psi)


def compute_drift(baseline_df, current_df):
    """
    Compute PSI for each drift column. Return dict feature -> psi value.
    """
    result = {}
    for col in NUMERIC_DRIFT_COLS:
        if col in baseline_df.columns and col in current_df.columns:
            result[col] = round(psi_numeric(baseline_df[col], current_df[col]), 4)
        else:
            result[col] = None
    for col in CATEGORICAL_DRIFT_COLS:
        if col in baseline_df.columns and col in current_df.columns:
            result[col] = round(psi_categorical(baseline_df[col], current_df[col]), 4)
        else:
            result[col] = None
    return result


def drift_status(psi_values):
    """
    Overall status from max PSI: OK (< 0.1), WARNING (0.1–0.2), ALERT (>= 0.2).
    """
    valid = [v for v in psi_values if v is not None]
    if not valid:
        return "OK"
    max_psi = max(valid)
    if max_psi >= PSI_ALERT:
        return "ALERT"
    if max_psi >= PSI_WARNING:
        return "WARNING"
    return "OK"


def run_monitor(predictions_path=None):
    """
    Load predictions, compute metrics, append to history, compute drift, write drift_report.json.
    Returns (metrics_dict, drift_report_dict) or (None, None) on failure.
    """
    df = load_predictions(predictions_path)
    if df is None:
        print("Monitor: predictions not found or invalid at {}".format(predictions_path or PREDICTIONS_PATH))
        return None, None

    metrics = compute_daily_metrics(df)
    _ensure_monitoring_dir()
    append_metrics_history(metrics, METRICS_HISTORY_PATH)
    print("Metrics appended to {}".format(METRICS_HISTORY_PATH))

    psi_per_feature = {}
    status = "OK"
    baseline_created_today = False
    if os.path.exists(BASELINE_PATH):
        try:
            baseline_df = pd.read_csv(BASELINE_PATH)
            psi_per_feature = compute_drift(baseline_df, df)
            status = drift_status(psi_per_feature.values())
        except Exception as e:
            print("Drift computation failed: {}".format(e))
            status = "OK"
    else:
        # Create baseline from today's predictions; do NOT compute drift (same data = misleading zeros)
        cols = [c for c in DRIFT_COLS if c in df.columns]
        if cols:
            try:
                df[cols].to_csv(BASELINE_PATH, index=False)
                print("Baseline created from today's predictions at {}.".format(BASELINE_PATH))
                print("Drift skipped this run (baseline vs same file). Run again tomorrow for PSI.")
                baseline_created_today = True
            except Exception as e:
                print("Could not create baseline: {}.".format(e))
        else:
            print("Baseline not found and predictions missing drift columns; skipping drift.")

    report = {
        "run_at": datetime.now().isoformat(),
        "predictions_path": predictions_path or PREDICTIONS_PATH,
        "baseline_path": BASELINE_PATH,
        "psi_per_feature": psi_per_feature,
        "status": status,
        "thresholds": {"warning": PSI_WARNING, "alert": PSI_ALERT},
        "baseline_created_today": baseline_created_today,
    }
    with open(DRIFT_REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2)
    print("Drift report written to {} (status={})".format(DRIFT_REPORT_PATH, status))

    # Generate monitoring plots (trend + PSI bar)
    try:
        from monitoring.plots import generate_all_plots
        generated = generate_all_plots()
        for p in generated:
            print("Plot saved:", p)
    except Exception as e:
        print("Monitoring plots could not be generated: {}".format(e))

    return metrics, report


if __name__ == "__main__":
    pred_path = sys.argv[1] if len(sys.argv) > 1 else None
    run_monitor(pred_path)
