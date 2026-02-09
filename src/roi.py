"""
Retention ROI Simulator: revenue at risk, revenue saved, offer cost, net benefit.
Uses predictions (Risk Level, Churn Probability) and optional CLTV.
"""
import numpy as np
import pandas as pd


def _get_segment_mask(pred_df, segment):
    """Return boolean mask for segment: High, Medium, or High+Medium."""
    if pred_df is None or pred_df.empty or "Risk Level" not in pred_df.columns:
        return pd.Series(dtype=bool)
    if segment == "High":
        return pred_df["Risk Level"] == "High"
    if segment == "Medium":
        return pred_df["Risk Level"] == "Medium"
    if segment == "High+Medium":
        return pred_df["Risk Level"].isin(["High", "Medium"])
    return pred_df["Risk Level"] == "High"


def compute_revenue_at_risk(pred_df, avg_cltv, segment="High", cltv_col=None):
    """
    Compute total revenue at risk for the target segment.
    If cltv_col exists and is valid, use sum(CLTV) for segment; else use count(segment) * avg_cltv.
    Returns dict with: revenue_at_risk (float), targeted_customers (int), used_cltv_from_data (bool).
    Handles missing/invalid values: defaults to 0 revenue, 0 customers if invalid.
    """
    if pred_df is None or pred_df.empty:
        return {"revenue_at_risk": 0.0, "targeted_customers": 0, "used_cltv_from_data": False}
    mask = _get_segment_mask(pred_df, segment)
    targeted_customers = int(mask.sum())
    if targeted_customers == 0:
        return {"revenue_at_risk": 0.0, "targeted_customers": 0, "used_cltv_from_data": False}
    avg_cltv = float(avg_cltv) if avg_cltv is not None and not (isinstance(avg_cltv, float) and np.isnan(avg_cltv)) else 0.0
    if avg_cltv < 0:
        avg_cltv = 0.0
    used_cltv_from_data = False
    if cltv_col and cltv_col in pred_df.columns:
        seg_cltv = pred_df.loc[mask, cltv_col]
        seg_cltv = pd.to_numeric(seg_cltv, errors="coerce").fillna(0)
        if (seg_cltv > 0).any():
            revenue_at_risk = float(seg_cltv.sum())
            if revenue_at_risk >= 0:
                used_cltv_from_data = True
            else:
                revenue_at_risk = targeted_customers * avg_cltv
        else:
            revenue_at_risk = targeted_customers * avg_cltv
    else:
        revenue_at_risk = targeted_customers * avg_cltv
    return {
        "revenue_at_risk": max(0.0, revenue_at_risk),
        "targeted_customers": targeted_customers,
        "used_cltv_from_data": used_cltv_from_data,
    }


def compute_roi(revenue_at_risk, save_rate, cost_per_offer, targeted_customers):
    """
    Compute ROI metrics.
    revenue_saved = revenue_at_risk * save_rate (0–1)
    offer_cost = targeted_customers * cost_per_offer
    net_benefit = revenue_saved - offer_cost
    Handles invalid/missing: treats as 0 where appropriate.
    Returns dict with: revenue_at_risk, revenue_saved, offer_cost, net_benefit.
    """
    revenue_at_risk = float(revenue_at_risk) if revenue_at_risk is not None else 0.0
    save_rate = float(save_rate) if save_rate is not None else 0.0
    cost_per_offer = float(cost_per_offer) if cost_per_offer is not None else 0.0
    targeted_customers = int(targeted_customers) if targeted_customers is not None else 0
    if revenue_at_risk < 0:
        revenue_at_risk = 0.0
    if save_rate < 0 or save_rate > 1:
        save_rate = max(0.0, min(1.0, save_rate))
    if cost_per_offer < 0:
        cost_per_offer = 0.0
    if targeted_customers < 0:
        targeted_customers = 0
    revenue_saved = revenue_at_risk * save_rate
    offer_cost = targeted_customers * cost_per_offer
    net_benefit = revenue_saved - offer_cost
    return {
        "revenue_at_risk": revenue_at_risk,
        "revenue_saved": revenue_saved,
        "offer_cost": offer_cost,
        "net_benefit": net_benefit,
    }


def plot_roi_vs_save_rate(revenue_at_risk, targeted_customers, cost_per_offer, save_rates_pct, out_path, dpi=200):
    """
    Plot Net Benefit vs Save Rate (%), save to out_path as PNG.
    save_rates_pct: list of ints (e.g. [0, 5, 10, ..., 50]).
    Returns None. Raises if write fails.
    """
    import os
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    net_benefits = []
    for sr in save_rates_pct:
        r = compute_roi(revenue_at_risk, sr / 100.0, cost_per_offer, targeted_customers)
        net_benefits.append(r["net_benefit"])
    total_offer_cost = targeted_customers * cost_per_offer
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(save_rates_pct, net_benefits, color="#14B8A6", linewidth=2, marker="o", markersize=4, label="Net benefit")
    ax.axhline(0, color="gray", linestyle="--", linewidth=1, label="Break-even")
    if total_offer_cost > 0:
        ax.axhline(-total_offer_cost, color="#f59e0b", linestyle=":", linewidth=1.5, label="Campaign cost (contact all)")
    ax.set_xlabel("Save rate (%)")
    ax.set_ylabel("Net benefit ($)")
    ax.set_title("Retention ROI: Net benefit vs save rate")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close()
