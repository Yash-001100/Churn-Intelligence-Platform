"""Run churn predictions using saved model and encoder; enrich with SHAP Top_Drivers."""
import os
import sys
import logging
import joblib
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import (
    MODEL_PATH,
    ENCODER_PATH,
    SCALER_PATH,
    COLUMN_META_PATH,
    PREDICTIONS_PATH,
    PREDICTIONS_ENRICHED_PATH,
    DATA_DIR,
    FEATURE_NAMES_JSON,
    FEATURE_NAMES_PATH,
    SHAP_TOP_K_DRIVERS,
)
from src.data_loader import prepare_data, engineer_features, clean_total_charges
from src.train import build_features

LOG = logging.getLogger(__name__)


def load_artifacts():
    """Load model, encoder, scaler, and column metadata."""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. Run train.py first."
        )
    model = joblib.load(MODEL_PATH)
    encoder = joblib.load(ENCODER_PATH)
    scaler = joblib.load(SCALER_PATH)
    column_meta = joblib.load(COLUMN_META_PATH)
    return model, encoder, scaler, column_meta


def ensure_columns(df, cat_cols, num_cols):
    """Ensure df has required columns; fill missing with defaults."""
    df = df.copy()
    for c in cat_cols:
        if c not in df.columns:
            df[c] = "Unknown"
        else:
            df[c] = df[c].astype(str).replace("nan", "Unknown")
    for c in num_cols:
        if c not in df.columns:
            df[c] = 0
        else:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
    return df


def predict_proba(df, model=None, encoder=None, scaler=None, column_meta=None):
    """
    Predict churn probability for each row in df.
    df must have the same feature columns as training (after prepare_data + engineer_features).
    Returns array of probabilities (probability of churn = 1).
    """
    if model is None:
        model, encoder, scaler, column_meta = load_artifacts()
    cat_cols = column_meta["cat_cols"]
    num_cols = column_meta["num_cols"]
    df = ensure_columns(df, cat_cols, num_cols)
    X, _, _, _, _, _ = build_features(
        df, encoder=encoder, scaler=scaler, fit=False,
        cat_cols_order=cat_cols, num_cols_order=num_cols,
    )
    proba = model.predict_proba(X)[:, 1]
    return proba


def _enrich_with_shap_drivers(df, model, encoder, scaler, column_meta, feature_names, top_k=SHAP_TOP_K_DRIVERS):
    """
    Compute SHAP for all customers where Risk Level == "High"; add Top_Drivers column.
    Business-aligned: we explain every high-risk customer, not a fixed top-N by probability.
    If SHAP is not installed (ImportError), returns df with empty Top_Drivers and does not raise.
    Other SHAP failures (e.g. unsupported model) still raise.
    """
    try:
        import shap  # noqa: F401
    except ImportError:
        df = df.copy()
        df["Top_Drivers"] = ""
        print("SHAP not installed; Top_Drivers left empty. Install with: pip install shap for per-customer explanations.")
        return df
    from src.shap_utils import (
        build_explainer,
        compute_shap_values,
        get_top_drivers_per_row,
    )
    cat_cols = column_meta["cat_cols"]
    num_cols = column_meta["num_cols"]
    df = ensure_columns(df, cat_cols, num_cols)
    X, _, _, _, _, _ = build_features(
        df, encoder=encoder, scaler=scaler, fit=False,
        cat_cols_order=cat_cols, num_cols_order=num_cols,
    )
    high_risk_mask = df["Risk Level"] == "High"
    explain_idx = df.index[high_risk_mask]
    if len(explain_idx) == 0:
        df = df.copy()
        df["Top_Drivers"] = ""
        LOG.info("No high-risk customers; Top_Drivers left empty.")
        return df
    n_background = min(200, len(X))
    rng = np.random.RandomState(42)
    bg_idx = rng.choice(len(X), size=n_background, replace=False)
    X_background = X[bg_idx]
    explainer, is_tree = build_explainer(model, X_background)
    X_explain = X[df.index.get_indexer(explain_idx)]
    shap_vals = compute_shap_values(explainer, X_explain, is_tree=is_tree)
    df_explain = df.loc[explain_idx]
    drivers_list = get_top_drivers_per_row(shap_vals, feature_names, df_explain, column_meta, top_k=top_k)
    df = df.copy()
    df["Top_Drivers"] = ""
    for i, idx in enumerate(explain_idx):
        df.at[idx, "Top_Drivers"] = drivers_list[i]
    n_high = len(explain_idx)
    LOG.info("SHAP computed for %d high-risk customers (Risk Level == High).", n_high)
    print("SHAP computed for {} high-risk customers (Risk Level == High).".format(n_high))
    return df


def predict_and_save(data_path=None, output_path=None):
    """
    Load cleaned data, run predictions, compute SHAP for all high-risk customers
    (Risk Level == "High"), add Top_Drivers, save to predictions_enriched.csv.
    Report and app read enriched. Returns dataframe with Churn Probability, Risk Level, Top_Drivers.
    """
    model, encoder, scaler, column_meta = load_artifacts()
    if "feature_names" in column_meta:
        feature_names = column_meta["feature_names"]
    else:
        import json
        if os.path.exists(FEATURE_NAMES_JSON):
            with open(FEATURE_NAMES_JSON) as f:
                feature_names = json.load(f)
        else:
            feature_names = joblib.load(FEATURE_NAMES_PATH)
    result = prepare_data(path=data_path, save_cleaned=True, return_customer_id=True)
    if isinstance(result, tuple):
        df, customer_id_series = result
    else:
        df, customer_id_series = result, None
    proba = predict_proba(df, model=model, encoder=encoder, scaler=scaler, column_meta=column_meta)
    df = df.copy()
    df["Churn Probability"] = proba
    df["Risk Level"] = pd.cut(
        proba,
        bins=[-0.01, 0.33, 0.66, 1.01],
        labels=["Low", "Medium", "High"],
    )
    # Include CustomerID in output for sync (e.g. Salesforce); not used as a feature
    if customer_id_series is not None:
        df["CustomerID"] = customer_id_series.values
    out_base = output_path or PREDICTIONS_PATH
    os.makedirs(os.path.dirname(out_base) or ".", exist_ok=True)
    df.to_csv(out_base, index=False)
    print("Predictions saved to {}".format(out_base))
    try:
        df = _enrich_with_shap_drivers(
            df, model, encoder, scaler, column_meta, feature_names,
            top_k=SHAP_TOP_K_DRIVERS,
        )
    except Exception as e:
        df = df.copy()
        df["Top_Drivers"] = ""
        print("SHAP enrichment failed ({}); Top_Drivers left empty. Predictions and report will still run.".format(e))
    out_enriched = os.path.join(os.path.dirname(out_base), "predictions_enriched.csv")
    df.to_csv(out_enriched, index=False)
    print("Enriched predictions (with Top_Drivers) saved to {}".format(out_enriched))
    return df


if __name__ == "__main__":
    predict_and_save()
