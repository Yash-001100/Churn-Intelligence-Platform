"""
Reusable SHAP explainability: load artifacts, build explainer, compute values,
per-row top drivers, and save global/summary plots. Works with tree-based models;
falls back with clear error if SHAP is not supported.
"""
import os
import sys
import json
import logging
import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import (
    MODEL_PATH,
    PREPROCESS_PATH,
    ENCODER_PATH,
    SCALER_PATH,
    FEATURE_NAMES_JSON,
    FEATURE_NAMES_PATH,
    COLUMN_META_PATH,
    REPORTS_FIGURES_DIR,
)

LOG = logging.getLogger(__name__)

# Default background sample size for TreeExplainer (efficiency)
DEFAULT_BACKGROUND_ROWS = 200


def load_artifacts():
    """
    Load model, preprocess (encoder+scaler), feature names, column_meta.
    Returns (model, encoder, scaler, feature_names, column_meta).
    Raises FileNotFoundError if required files are missing.
    """
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("Model not found at {}. Run train first.".format(MODEL_PATH))
    if not os.path.exists(COLUMN_META_PATH):
        raise FileNotFoundError("Column meta not found at {}. Run train first.".format(COLUMN_META_PATH))
    model = joblib.load(MODEL_PATH)
    if os.path.exists(PREPROCESS_PATH):
        pre = joblib.load(PREPROCESS_PATH)
        encoder = pre["encoder"]
        scaler = pre["scaler"]
    else:
        encoder = joblib.load(ENCODER_PATH)
        scaler = joblib.load(SCALER_PATH)
    if os.path.exists(FEATURE_NAMES_JSON):
        with open(FEATURE_NAMES_JSON) as f:
            feature_names = json.load(f)
    else:
        feature_names = joblib.load(FEATURE_NAMES_PATH)
    column_meta = joblib.load(COLUMN_META_PATH)
    return model, encoder, scaler, feature_names, column_meta


def build_explainer(model, X_background):
    """
    Build SHAP explainer. Prefer TreeExplainer for tree models; else KernelExplainer.
    X_background: 2D array (e.g. 200 rows). Returns (explainer, is_tree).
    Raises RuntimeError with clear message if SHAP fails for this model type.
    """
    import shap
    try:
        explainer = shap.TreeExplainer(model, X_background)
        _ = explainer.shap_values(X_background[:1])
        return explainer, True
    except Exception as tree_err:
        tree_err_msg = str(tree_err)
    try:
        background = X_background[: min(100, len(X_background))]
        explainer = shap.KernelExplainer(model.predict_proba, background)
        return explainer, False
    except Exception as ke_err:
        raise RuntimeError(
            "SHAP is not supported for this model type. TreeExplainer failed: {}. "
            "KernelExplainer failed: {}. Use a tree-based model (XGBoost, LightGBM, CatBoost, RandomForest).".format(
                tree_err_msg, str(ke_err)
            )
        ) from ke_err


def compute_shap_values(explainer, X, is_tree=True):
    """
    Compute SHAP values for X. Returns 2D array (n_samples, n_features) for positive class.
    """
    import shap
    vals = explainer.shap_values(X)
    if isinstance(vals, list):
        vals = vals[1]
    vals = np.asarray(vals)
    if vals.ndim == 3:
        vals = vals[:, :, 1]
    return vals


def _driver_display(row, fname, cat_cols):
    """(display_label, value_str) for one feature and row."""
    if fname in row.index:
        v = row.get(fname)
        if pd.isna(v):
            return fname, ""
        return fname, "{:.2f}".format(v) if isinstance(v, (int, float)) else str(v)
    if "_" in str(fname) and cat_cols:
        parts = str(fname).split("_", 1)
        col = parts[0]
        if col in cat_cols and col in row.index:
            return col, str(row[col]) if pd.notna(row.get(col)) else ""
    return str(fname).replace("_", " "), ""


def get_top_drivers_per_row(shap_values, feature_names, df, column_meta, top_k=5):
    """
    For each row, return a string: "Feature=value (+0.21), TenureMonths=2 (+0.14), ..."
    shap_values: (n_samples, n_features). df: same row order as X used for SHAP.
    """
    cat_cols = column_meta.get("cat_cols") or []
    out = []
    for i in range(shap_values.shape[0]):
        row = df.iloc[i]
        sv = shap_values[i]
        top_idx = np.argsort(np.abs(sv))[::-1][:top_k]
        parts = []
        for j in top_idx:
            if np.abs(sv[j]) < 1e-9:
                continue
            fname = feature_names[j] if j < len(feature_names) else "Feature_{}".format(j)
            label, val = _driver_display(row, fname, cat_cols)
            sign = "+" if sv[j] >= 0 else ""
            parts.append("{}={} ({}{:.2f})".format(label, val, sign, float(sv[j])))
        out.append(", ".join(parts) if parts else "")
    return out


def plot_global_importance(shap_values, feature_names, out_path, dpi=300):
    """
    Save bar chart of mean |SHAP| per feature to out_path.
    Verifies file exists and size > 0; logs path and size. Raises on failure.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    mean_abs = np.abs(shap_values).mean(axis=0)
    order = np.argsort(mean_abs)[::-1]
    names = [feature_names[i].replace("_", " ").title() for i in order]
    vals = mean_abs[order]
    fig, ax = plt.subplots(figsize=(10, 6))
    y_pos = np.arange(len(names))
    ax.barh(y_pos, vals, color="#14B8A6", alpha=0.85)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel("Mean |SHAP| (impact on churn probability)")
    ax.set_title("Global SHAP feature importance")
    ax.invert_yaxis()
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close()
    if not os.path.exists(out_path) or os.path.getsize(out_path) == 0:
        raise RuntimeError("SHAP global importance chart failed to save: {}".format(out_path))
    LOG.info("SHAP global chart saved: %s (%s bytes)", out_path, os.path.getsize(out_path))


def plot_summary(shap_values, X, feature_names, out_path, dpi=300, max_display=20):
    """
    Save SHAP summary plot to out_path. Verifies file exists and size > 0; logs path.
    Raises on failure.
    """
    import shap
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    shap.summary_plot(shap_values, X, feature_names=feature_names, show=False, max_display=max_display)
    plt.tight_layout()
    plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close()
    if not os.path.exists(out_path) or os.path.getsize(out_path) == 0:
        raise RuntimeError("SHAP summary chart failed to save: {}".format(out_path))
    LOG.info("SHAP summary chart saved: %s (%s bytes)", out_path, os.path.getsize(out_path))
