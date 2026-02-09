"""
Monitoring plots: trend charts (high-risk %, avg churn prob) and PSI bar chart.
Saves to reports/figures/monitoring_*.png.
"""
import os
import json

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MONITORING_DIR = os.path.join(PROJECT_ROOT, "monitoring")
REPORTS_FIGURES_DIR = os.path.join(PROJECT_ROOT, "reports", "figures")
METRICS_HISTORY_PATH = os.path.join(MONITORING_DIR, "metrics_history.csv")
DRIFT_REPORT_PATH = os.path.join(MONITORING_DIR, "drift_report.json")


def _ensure_figures_dir():
    os.makedirs(REPORTS_FIGURES_DIR, exist_ok=True)


def plot_high_risk_pct_trend(path=None, output_path=None):
    """
    Trend chart: high-risk % over time.
    path: metrics_history.csv; output_path: e.g. reports/figures/monitoring_high_risk_trend.png
    """
    path = path or METRICS_HISTORY_PATH
    output_path = output_path or os.path.join(REPORTS_FIGURES_DIR, "monitoring_high_risk_trend.png")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    if "date" not in df.columns or "high_risk_pct" not in df.columns or len(df) == 0:
        return None
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "high_risk_pct"]).sort_values("date")
    if len(df) == 0:
        return None
    _ensure_figures_dir()
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(df["date"], df["high_risk_pct"], marker="o", markersize=4, color="#2563eb", linewidth=1.5)
    ax.set_xlabel("Date")
    ax.set_ylabel("High-risk %")
    ax.set_title("High-risk % over time")
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    return output_path


def plot_avg_churn_prob_trend(path=None, output_path=None):
    """
    Trend chart: avg churn probability over time.
    """
    path = path or METRICS_HISTORY_PATH
    output_path = output_path or os.path.join(REPORTS_FIGURES_DIR, "monitoring_avg_churn_trend.png")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    if "date" not in df.columns or "avg_churn_probability" not in df.columns or len(df) == 0:
        return None
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date")
    df["avg_churn_probability"] = pd.to_numeric(df["avg_churn_probability"], errors="coerce")
    df = df.dropna(subset=["avg_churn_probability"])
    if len(df) == 0:
        return None
    _ensure_figures_dir()
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(df["date"], df["avg_churn_probability"], marker="o", markersize=4, color="#059669", linewidth=1.5)
    ax.set_xlabel("Date")
    ax.set_ylabel("Avg churn probability")
    ax.set_title("Avg churn probability over time")
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    return output_path


def plot_psi_bar(path=None, output_path=None):
    """
    Bar chart: PSI per feature (today) from drift_report.json.
    If psi is empty and baseline_created_today, draw a message chart instead of empty bars.
    """
    path = path or DRIFT_REPORT_PATH
    output_path = output_path or os.path.join(REPORTS_FIGURES_DIR, "monitoring_psi_bar.png")
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        report = json.load(f)
    psi = report.get("psi_per_feature", {})
    baseline_created_today = report.get("baseline_created_today", False)
    if not psi and baseline_created_today:
        # Baseline was just created; no drift computed (would be same vs same). Show message.
        _ensure_figures_dir()
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.axis("off")
        ax.text(0.5, 0.6, "Baseline established today.", ha="center", va="center", fontsize=14, fontweight="bold")
        ax.text(0.5, 0.4, "Drift was not computed (baseline vs same file).", ha="center", va="center", fontsize=11)
        ax.text(0.5, 0.2, "Run monitoring again after the next predictions for PSI.", ha="center", va="center", fontsize=10)
        ax.set_title("Drift (PSI) per feature — today vs baseline")
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
        return output_path
    if not psi:
        return None
    features = [k for k, v in psi.items() if v is not None]
    values = [psi[k] for k in features]
    if not features:
        return None
    _ensure_figures_dir()
    fig, ax = plt.subplots(figsize=(8, 4))
    colors = ["#dc2626" if v >= 0.2 else "#f59e0b" if v >= 0.1 else "#22c55e" for v in values]
    ax.barh(features, values, color=colors)
    ax.axvline(0.1, color="#f59e0b", linestyle="--", linewidth=1, label="Warning (0.1)")
    ax.axvline(0.2, color="#dc2626", linestyle="--", linewidth=1, label="Alert (0.2)")
    ax.set_xlabel("PSI")
    ax.set_title("Drift (PSI) per feature — today vs baseline")
    ax.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    return output_path


def generate_all_plots():
    """
    Generate all monitoring plots; return list of saved paths.
    """
    paths = []
    p1 = plot_high_risk_pct_trend()
    if p1:
        paths.append(p1)
    p2 = plot_avg_churn_prob_trend()
    if p2:
        paths.append(p2)
    p3 = plot_psi_bar()
    if p3:
        paths.append(p3)
    return paths


if __name__ == "__main__":
    generated = generate_all_plots()
    for p in generated:
        print("Saved:", p)
    if not generated:
        print("No plots generated (missing metrics_history.csv or drift_report.json).")
