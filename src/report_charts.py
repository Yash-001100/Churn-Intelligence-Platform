"""
Chart builders for churn report. All return PNG bytes (or None).
Professional colors: Low #22C55E, Medium #F59E0B, High #EF4444, Primary #14B8A6.
"""
import io
import os
import joblib
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.config import MODEL_PATH, FEATURE_NAMES_PATH, MODELS_DIR

# Professional palette
COLORS = {
    "primary": "#14B8A6",
    "high_risk": "#EF4444",
    "medium_risk": "#F59E0B",
    "low_risk": "#22C55E",
    "text": "#1E293B",
    "text_muted": "#64748B",
    "bg": "#FFFFFF",
    "border": "#E2E8F0",
}
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["axes.facecolor"] = COLORS["bg"]
# High-res for PDF: 200–300 dpi, figure size suitable for letter page (e.g. 6.5 x 4)
CHART_DPI = 200
CHART_W, CHART_H = 6.5, 4.0


def _fig_to_bytes(fig, dpi=CHART_DPI):
    """Save figure to PNG bytes; tight_layout, bbox_inches=tight, high dpi for PDF."""
    try:
        fig.tight_layout(pad=1.0)
    except Exception:
        pass
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight", facecolor=COLORS["bg"])
    plt.close(fig)
    buf.seek(0)
    data = buf.getvalue()
    if not data or len(data) < 100:
        raise RuntimeError("Chart produced empty or invalid PNG")
    return data


def plot_risk_distribution(stats, width_inch=CHART_W, height_inch=CHART_H):
    """Donut chart: Risk distribution (Low / Medium / High). Returns PNG bytes."""
    fig, ax = plt.subplots(figsize=(width_inch, height_inch))
    counts = [stats["high_risk"], stats["medium_risk"], stats["low_risk"]]
    labels = ["High", "Medium", "Low"]
    colors = [COLORS["high_risk"], COLORS["medium_risk"], COLORS["low_risk"]]
    total = sum(counts)
    pcts = [100 * c / total if total else 0 for c in counts]
    wedges, texts = ax.pie(counts, labels=None, colors=colors, startangle=90, wedgeprops=dict(width=0.5, edgecolor="white", linewidth=2))
    ax.text(0, 0, f"n={total:,}", ha="center", va="center", fontsize=11, fontweight="bold", color=COLORS["text"])
    ax.legend(
        [f"{l} ({p:.1f}%)" for l, p in zip(labels, pcts)],
        loc="center left",
        bbox_to_anchor=(1, 0.5),
        frameon=True,
        fancybox=True,
        shadow=False,
        fontsize=9,
    )
    ax.set_title("Risk distribution", fontsize=14, fontweight="bold", color=COLORS["text"], pad=10)
    return _fig_to_bytes(fig)


def plot_histogram(df, width_inch=CHART_W, height_inch=CHART_H, bins=30):
    """Histogram: Churn probability distribution. Returns PNG bytes. Raises if column missing."""
    if "Churn Probability" not in df.columns:
        raise ValueError("Churn probability distribution chart requires 'Churn Probability' column in predictions.csv")
    ser = df["Churn Probability"].dropna()
    if ser.empty:
        raise ValueError("Churn probability distribution chart: no non-null 'Churn Probability' values")
    fig, ax = plt.subplots(figsize=(width_inch, height_inch))
    ax.hist(ser, bins=bins, color=COLORS["primary"], edgecolor="white", linewidth=0.5, alpha=0.85)
    ax.set_xlabel("Churn probability", fontsize=11, color=COLORS["text_muted"])
    ax.set_ylabel("Count", fontsize=11, color=COLORS["text_muted"])
    ax.set_title("Churn probability distribution", fontsize=14, fontweight="bold", color=COLORS["text"], pad=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_facecolor(COLORS["bg"])
    return _fig_to_bytes(fig)


def plot_contract_chart(df, width_inch=CHART_W, height_inch=CHART_H):
    """Bar chart: Risk by contract type (stacked horizontal). Returns PNG bytes. Raises if columns missing."""
    if "Contract" not in df.columns or "Risk Level" not in df.columns:
        raise ValueError("Risk by contract type chart requires 'Contract' and 'Risk Level' columns in predictions.csv")
    cross = pd.crosstab(df["Contract"], df["Risk Level"], normalize="index").fillna(0)
    for level in ["High", "Medium", "Low"]:
        if level not in cross.columns:
            cross[level] = 0
    cross = cross[["High", "Medium", "Low"]]
    fig, ax = plt.subplots(figsize=(width_inch, height_inch))
    left = None
    for level, color in [("High", COLORS["high_risk"]), ("Medium", COLORS["medium_risk"]), ("Low", COLORS["low_risk"])]:
        vals = (cross[level] * 100).values
        if left is None:
            ax.barh(cross.index, vals, label=level, color=color, height=0.6)
            left = vals.copy()
        else:
            ax.barh(cross.index, vals, left=left, label=level, color=color, height=0.6)
            left = left + vals
    ax.set_xlabel("% of customers", fontsize=11, color=COLORS["text_muted"])
    ax.set_title("Risk by contract type", fontsize=14, fontweight="bold", color=COLORS["text"], pad=10)
    ax.set_xlim(0, 100)
    ax.legend(loc="lower right", fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", alpha=0.3, linestyle="--")
    ax.set_facecolor(COLORS["bg"])
    return _fig_to_bytes(fig)


def _get_feature_importance():
    """Load model and feature names; return (feature_names, importances) or (None, None)."""
    if not os.path.exists(MODEL_PATH) or not os.path.exists(FEATURE_NAMES_PATH):
        return None, None
    try:
        model = joblib.load(MODEL_PATH)
        names = joblib.load(FEATURE_NAMES_PATH)
        if not isinstance(names, (list, np.ndarray)) or len(names) == 0:
            return None, None
        names = list(names)
        # StackingClassifier: use first base estimator that has feature_importances_
        if hasattr(model, "estimators_"):
            for est in model.estimators_:
                if hasattr(est, "feature_importances_"):
                    imp = est.feature_importances_
                    if len(imp) == len(names):
                        return names, imp
                    break
        if hasattr(model, "named_estimators_"):
            for _ename, est in model.named_estimators_.items():
                if hasattr(est, "feature_importances_"):
                    imp = est.feature_importances_
                    if len(imp) == len(names):
                        return names, imp
                    break
        if hasattr(model, "feature_importances_"):
            imp = model.feature_importances_
            if len(imp) == len(names):
                return names, imp
    except Exception:
        pass
    return None, None


def plot_feature_importance(top_n=15, width_inch=CHART_W, height_inch=3.5):
    """Bar chart: Top N feature importance. Returns PNG bytes or None."""
    names, imp = _get_feature_importance()
    if names is None or imp is None or len(names) != len(imp):
        return None
    idx = np.argsort(imp)[::-1][:top_n]
    names = [names[i].replace("_", " ")[:35] for i in idx]
    imp = imp[idx]
    fig, ax = plt.subplots(figsize=(width_inch, height_inch))
    y_pos = np.arange(len(names))[::-1]
    bars = ax.barh(y_pos, imp, color=COLORS["primary"], height=0.65, edgecolor="white", linewidth=0.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel("Importance", fontsize=11, color=COLORS["text_muted"])
    ax.set_title("Top 15 feature importance", fontsize=14, fontweight="bold", color=COLORS["text"], pad=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_facecolor(COLORS["bg"])
    return _fig_to_bytes(fig)


def plot_tenure_chart(df, width_inch=CHART_W, height_inch=CHART_H):
    """Bar chart: Risk by tenure bucket (stacked horizontal). Returns PNG bytes. Raises if columns missing."""
    if "Tenure Bucket" not in df.columns or "Risk Level" not in df.columns:
        raise ValueError("Risk by tenure bucket chart requires 'Tenure Bucket' and 'Risk Level' columns in predictions.csv")
    cross = pd.crosstab(df["Tenure Bucket"], df["Risk Level"], normalize="index").fillna(0)
    for level in ["High", "Medium", "Low"]:
        if level not in cross.columns:
            cross[level] = 0
    cross = cross[["High", "Medium", "Low"]]
    # Sort by typical tenure order if present
    order = ["0-12", "13-24", "25-48", "49-60", "61+"]
    cross = cross.reindex([x for x in order if x in cross.index] + [x for x in cross.index if x not in order])
    fig, ax = plt.subplots(figsize=(width_inch, height_inch))
    left = None
    for level, color in [("High", COLORS["high_risk"]), ("Medium", COLORS["medium_risk"]), ("Low", COLORS["low_risk"])]:
        vals = (cross[level] * 100).values
        if left is None:
            ax.barh(cross.index, vals, label=level, color=color, height=0.6)
            left = vals.copy()
        else:
            ax.barh(cross.index, vals, left=left, label=level, color=color, height=0.6)
            left = left + vals
    ax.set_xlabel("% of customers", fontsize=11, color=COLORS["text_muted"])
    ax.set_title("Risk by tenure bucket", fontsize=14, fontweight="bold", color=COLORS["text"], pad=10)
    ax.set_xlim(0, 100)
    ax.legend(loc="lower right", fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", alpha=0.3, linestyle="--")
    ax.set_facecolor(COLORS["bg"])
    return _fig_to_bytes(fig)


def generate_kpis(stats, width_inch=6.5, height_inch=1.6):
    """KPI cards: Total, High, Medium, Low. Returns PNG bytes."""
    total = stats["total"]
    fig, ax = plt.subplots(figsize=(width_inch, height_inch))
    ax.set_xlim(0, 4)
    ax.set_ylim(0, 1)
    ax.axis("off")
    cards = [
        ("Total", total, COLORS["primary"]),
        ("High", stats["high_risk"], COLORS["high_risk"]),
        ("Medium", stats["medium_risk"], COLORS["medium_risk"]),
        ("Low", stats["low_risk"], COLORS["low_risk"]),
    ]
    for i, (label, value, color) in enumerate(cards):
        x = 0.25 + i * 1.0
        rect = plt.Rectangle((x - 0.45, 0.1), 0.9, 0.8, facecolor=color, edgecolor=COLORS["border"], linewidth=1)
        ax.add_patch(rect)
        ax.text(x, 0.65, f"{value:,}", ha="center", va="center", fontsize=14, fontweight="bold", color="white")
        ax.text(x, 0.3, label, ha="center", va="center", fontsize=10, color="white", alpha=0.95)
    return _fig_to_bytes(fig)
