"""
Churn Prediction & Retention Analytics Report — complete business-ready PDF.
Builds churn_report.pdf from predictions.csv. No placeholders: metrics loaded from
models/eval_metrics.json (written by train.py) or computed from data.

Run: python -m src.report_generator
Output: churn_report.pdf (project root)

Structure: Title → Executive Summary → Key Metrics → Visual Insights (charts + text) →
Insights & Drivers → Risk Segmentation → Recommended Actions → Top High-Risk Table →
Definitions → Method Summary → Limitations → Next Steps.
"""
import os
import io
import json
from datetime import datetime

import numpy as np
import pandas as pd
from reportlab.lib import colors as rl_colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak,
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT

from src.config import (
    PROJECT_ROOT,
    DATA_DIR,
    MODELS_DIR,
    PREDICTIONS_PATH,
    PREDICTIONS_ENRICHED_PATH,
    REPORTS_FIGURES_DIR,
)
MONITORING_DIR = os.path.join(PROJECT_ROOT, "monitoring")
DRIFT_REPORT_PATH = os.path.join(MONITORING_DIR, "drift_report.json")
MONITORING_HIGH_RISK_TREND_PATH = os.path.join(REPORTS_FIGURES_DIR, "monitoring_high_risk_trend.png")
from src.report_charts import (
    COLORS,
    generate_kpis,
    plot_risk_distribution,
    plot_histogram,
    plot_contract_chart,
    plot_feature_importance,
    plot_tenure_chart,
)
from src.roi import compute_revenue_at_risk, compute_roi, plot_roi_vs_save_rate

REPORT_AUTHOR = "Yash Kalra"
OUTPUT_PDF = "churn_report.pdf"


# ---------------------------------------------------------------------------
# Data: load predictions and metrics (no placeholders)
# ---------------------------------------------------------------------------

def load_predictions(predictions_path=None):
    """Load predictions; prefer predictions_enriched.csv if present, else predictions.csv. Return DataFrame or None."""
    if predictions_path is not None:
        path = predictions_path
    else:
        path = PREDICTIONS_ENRICHED_PATH if os.path.exists(PREDICTIONS_ENRICHED_PATH) else PREDICTIONS_PATH
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path)
        if "Risk Level" not in df.columns:
            return None
        return df
    except Exception:
        return None


def load_metrics():
    """
    Load model metrics from models/eval_metrics.json (saved by train.py).
    Returns dict with model_name, accuracy, f1_score, roc_auc; empty dict if file absent.
    """
    path = os.path.join(MODELS_DIR, "eval_metrics.json")
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r") as f:
            m = json.load(f)
        return {
            "model_name": m.get("model_name", "—"),
            "accuracy": m.get("accuracy"),
            "f1_score": m.get("f1_score"),
            "roc_auc": m.get("roc_auc"),
        }
    except Exception:
        return {}


def compute_metrics_from_predictions(df):
    """
    When eval_metrics.json is missing, compute accuracy/F1/ROC-AUC from predictions
    if Churn Value (y_true) and Churn Probability exist. Do not leave blanks.
    Returns dict with model_name, accuracy, f1_score, roc_auc or empty dict.
    """
    if df is None or "Churn Value" not in df.columns or "Churn Probability" not in df.columns:
        return {}
    try:
        y_true = df["Churn Value"].dropna().astype(int)
        y_prob = df["Churn Probability"].dropna()
        idx = y_true.index.intersection(y_prob.index)
        if len(idx) < 10:
            return {}
        y_true = y_true.loc[idx].values
        y_prob = y_prob.loc[idx].values
        y_pred = (y_prob >= 0.5).astype(int)
        from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        auc = roc_auc_score(y_true, y_prob)
        return {
            "model_name": "Computed from predictions",
            "accuracy": round(acc, 4),
            "f1_score": round(f1, 4),
            "roc_auc": round(auc, 4),
        }
    except Exception:
        return {}


def summary_stats(df):
    """Compute total, high/medium/low counts, avg churn probability."""
    counts = df["Risk Level"].value_counts()
    total = len(df)
    avg_prob = df["Churn Probability"].mean() if "Churn Probability" in df.columns else None
    return {
        "total": total,
        "high_risk": int(counts.get("High", 0)),
        "medium_risk": int(counts.get("Medium", 0)),
        "low_risk": int(counts.get("Low", 0)),
        "avg_churn_probability": float(avg_prob) if avg_prob is not None and not pd.isna(avg_prob) else None,
    }


def pct(val, total):
    if total == 0:
        return 0.0
    return round(100 * val / total, 1)


# ---------------------------------------------------------------------------
# Charts (wrap report_charts; add plot_probability_histogram alias)
# ---------------------------------------------------------------------------

def plot_probability_histogram(df, **kwargs):
    """Histogram of churn probability (alias for plot_histogram)."""
    return plot_histogram(df, **kwargs)


# ---------------------------------------------------------------------------
# Tables: segmentation and top 20 high-risk
# ---------------------------------------------------------------------------

def generate_tables(df, stats, total):
    """
    Return (segmentation_table_data, top20_table_data).
    segmentation_table_data: list of rows [Segment, Count, %, Recommended action]
    top20_table_data: list of rows for top 20 high-risk (headers + data)
    """
    pct_hi = pct(stats["high_risk"], total)
    pct_md = pct(stats["medium_risk"], total)
    pct_lo = pct(stats["low_risk"], total)
    seg_data = [
        ["Segment", "Count", "%", "Recommended action"],
        ["High", str(stats["high_risk"]), f"{pct_hi}%", "Proactive outreach, offers, contract upgrade, save desk"],
        ["Medium", str(stats["medium_risk"]), f"{pct_md}%", "Targeted communications, loyalty perks, usage tips"],
        ["Low", str(stats["low_risk"]), f"{pct_lo}%", "Maintain quality, cross-sell where relevant"],
    ]
    # Top 20 high-risk
    display_cols = [c for c in ["Contract", "Tenure Months", "Monthly Charges", "Churn Probability", "Risk Level"] if c in df.columns]
    if not display_cols and "Churn Probability" in df.columns:
        display_cols = ["Churn Probability", "Risk Level"]
    top20_data = None
    if display_cols and "Churn Probability" in df.columns:
        top_high = df[df["Risk Level"] == "High"].nlargest(20, "Churn Probability")[display_cols].head(20)
        table_df = top_high.copy()
        if "Churn Probability" in table_df.columns:
            table_df["Churn Probability"] = table_df["Churn Probability"].apply(lambda x: f"{float(x):.1%}")
        if "Monthly Charges" in table_df.columns:
            table_df["Monthly Charges"] = table_df["Monthly Charges"].apply(lambda x: f"${float(x):,.2f}" if pd.notna(x) else "")
        top20_data = [[str(x) for x in table_df.columns.tolist()]]
        for _, row in table_df.iterrows():
            top20_data.append([str(x) if pd.notna(x) else "" for x in row])
    return seg_data, top20_data


# ---------------------------------------------------------------------------
# PDF styles and table style
# ---------------------------------------------------------------------------

def _styles():
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            name="ReportTitle",
            parent=base["Title"],
            fontSize=22,
            spaceAfter=8,
            textColor=rl_colors.HexColor(COLORS["primary"]),
            alignment=TA_CENTER,
        ),
        "subtitle": ParagraphStyle(
            name="Subtitle",
            parent=base["Normal"],
            fontSize=10,
            spaceAfter=6,
            alignment=TA_CENTER,
            textColor=rl_colors.HexColor(COLORS["text_muted"]),
        ),
        "heading1": ParagraphStyle(
            name="H1",
            parent=base["Heading1"],
            fontSize=14,
            spaceBefore=18,
            spaceAfter=8,
            textColor=rl_colors.HexColor(COLORS["text"]),
        ),
        "heading2": ParagraphStyle(
            name="H2",
            parent=base["Heading2"],
            fontSize=11,
            spaceBefore=12,
            spaceAfter=6,
            textColor=rl_colors.HexColor(COLORS["text"]),
        ),
        "body": ParagraphStyle(
            name="Body",
            parent=base["Normal"],
            fontSize=9,
            spaceAfter=6,
        ),
        "bullet": ParagraphStyle(
            name="Bullet",
            parent=base["Normal"],
            fontSize=9,
            leftIndent=16,
            spaceAfter=4,
        ),
        "small": ParagraphStyle(
            name="Small",
            parent=base["Normal"],
            fontSize=8,
            spaceAfter=4,
            textColor=rl_colors.HexColor(COLORS["text_muted"]),
        ),
    }


def _table_style_header(hex_color):
    return [
        ("BACKGROUND", (0, 0), (-1, 0), rl_colors.HexColor(hex_color)),
        ("TEXTCOLOR", (0, 0), (-1, 0), rl_colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, 0), 9),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("GRID", (0, 0), (-1, -1), 0.5, rl_colors.HexColor("#E2E8F0")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [rl_colors.white, rl_colors.HexColor("#F8FAFC")]),
        ("FONTSIZE", (0, 1), (-1, -1), 8),
    ]


# ---------------------------------------------------------------------------
# Build PDF: all 12 sections in order
# ---------------------------------------------------------------------------

def build_pdf(df, stats, metrics, data_updated, output_path):
    """
    Assemble full report PDF: title, executive summary, KPIs, visual insights
    (each chart + 2–3 sentence explanation), insights & drivers, risk segmentation,
    recommended actions, top 20 table, definitions, method, limitations, next steps.
    """
    total = stats["total"]
    pct_hi = pct(stats["high_risk"], total)
    pct_md = pct(stats["medium_risk"], total)
    pct_lo = pct(stats["low_risk"], total)
    avg_prob = stats.get("avg_churn_probability")
    model_name = metrics.get("model_name") or "—"
    accuracy = metrics.get("accuracy")
    f1_score_val = metrics.get("f1_score")
    roc_auc = metrics.get("roc_auc")
    styles = _styles()
    seg_data, top20_data = generate_tables(df, stats, total)

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=0.65 * inch,
        leftMargin=0.65 * inch,
        topMargin=0.65 * inch,
        bottomMargin=0.65 * inch,
    )
    story = []

    # ----- 1. Title Page -----
    story.append(Spacer(1, 40))
    story.append(Paragraph("Churn Prediction &amp; Retention Analytics Report", styles["title"]))
    story.append(Spacer(1, 12))
    story.append(Paragraph(datetime.now().strftime("%B %d, %Y"), styles["subtitle"]))
    story.append(Paragraph(f"Prepared by: {REPORT_AUTHOR}", styles["subtitle"]))
    story.append(Spacer(1, 30))

    # ----- 2. Executive Summary (5–7 bullets) -----
    story.append(Paragraph("Executive Summary", styles["heading1"]))
    bullets = [
        f"Total customers scored: <b>{total:,}</b>. Risk mix: <b>{pct_hi}%</b> high ({stats['high_risk']:,}), <b>{pct_md}%</b> medium ({stats['medium_risk']:,}), <b>{pct_lo}%</b> low ({stats['low_risk']:,}).",
        "Contract type is a key driver: month-to-month customers show higher churn risk than one- or two-year contracts; two-year contracts have the lowest share of high-risk customers.",
        "Tenure and monthly charges are strong predictors: short tenure and higher monthly spend correlate with elevated churn probability.",
        "Business impact: focusing retention on the high-risk segment (e.g. proactive outreach, contract incentives) can reduce voluntary churn and protect revenue.",
        "Recommended actions: prioritize high-risk month-to-month customers for save-desk and offers; run targeted loyalty campaigns for medium-risk; maintain experience for low-risk.",
    ]
    for b in bullets:
        story.append(Paragraph(f"• {b}", styles["bullet"]))
    story.append(Spacer(1, 16))

    # ----- 3. Key Metrics (KPI cards + table with model metrics) -----
    story.append(Paragraph("Key Metrics", styles["heading1"]))
    kpi_bytes = generate_kpis(stats)
    if not kpi_bytes or len(kpi_bytes) < 100:
        raise RuntimeError("Key Metrics KPI cards chart could not be generated: empty or invalid PNG.")
    buf_kpi = io.BytesIO(kpi_bytes)
    buf_kpi.seek(0)
    story.append(Image(buf_kpi, width=6.5 * inch, height=1.6 * inch))
    story.append(Spacer(1, 8))
    # Metrics table: total, high, medium, low, avg probability, model metrics
    acc_str = f"{accuracy:.4f}" if accuracy is not None else "—"
    f1_str = f"{f1_score_val:.4f}" if f1_score_val is not None else "—"
    auc_str = f"{roc_auc:.4f}" if roc_auc is not None else "—"
    avg_str = f"{avg_prob:.2%}" if avg_prob is not None else "—"
    metrics_rows = [
        ["Metric", "Value"],
        ["Total customers", f"{total:,}"],
        ["High risk", f"{stats['high_risk']:,} ({pct_hi}%)"],
        ["Medium risk", f"{stats['medium_risk']:,} ({pct_md}%)"],
        ["Low risk", f"{stats['low_risk']:,} ({pct_lo}%)"],
        ["Average churn probability", avg_str],
        ["Model", model_name],
        ["Accuracy", acc_str],
        ["F1 score", f1_str],
        ["ROC-AUC", auc_str],
    ]
    t_metrics = Table(metrics_rows, colWidths=[2.8 * inch, 3.2 * inch])
    t_metrics.setStyle(TableStyle(_table_style_header(COLORS["primary"])))
    story.append(t_metrics)
    story.append(Spacer(1, 18))

    # ----- 3b. Monitoring (last 7 days trend + today drift status) -----
    story.append(Paragraph("Monitoring", styles["heading1"]))
    if os.path.exists(MONITORING_HIGH_RISK_TREND_PATH) and os.path.getsize(MONITORING_HIGH_RISK_TREND_PATH) > 0:
        try:
            with open(MONITORING_HIGH_RISK_TREND_PATH, "rb") as f_mon:
                monitoring_bytes = f_mon.read()
            buf_mon = io.BytesIO(monitoring_bytes)
            buf_mon.seek(0)
            story.append(Image(buf_mon, width=6.5 * inch, height=3.0 * inch))
            story.append(Paragraph("High-risk % over time (daily metrics).", styles["small"]))
        except Exception:
            story.append(Paragraph("Monitoring trend chart could not be loaded.", styles["small"]))
    else:
        story.append(Paragraph(
            "Monitoring trend not yet available. Run predictions then: python -m monitoring.monitor",
            styles["small"]
        ))
    story.append(Spacer(1, 6))
    if os.path.exists(DRIFT_REPORT_PATH):
        try:
            with open(DRIFT_REPORT_PATH, "r") as f_drift:
                drift_report = json.load(f_drift)
            status = drift_report.get("status", "OK")
            story.append(Paragraph(
                "Today drift status: <b>{}</b> (PSI vs baseline). "
                "OK &lt; 0.1, WARNING 0.1–0.2, ALERT ≥ 0.2.".format(status),
                styles["body"]
            ))
        except Exception:
            story.append(Paragraph("Drift report could not be loaded.", styles["small"]))
    else:
        story.append(Paragraph("Drift report not yet available. Run: python -m monitoring.monitor", styles["small"]))
    story.append(Spacer(1, 18))

    # ----- 4. Visual Insights (each chart + 2–3 sentence explanation) -----
    # Embed helper: verify PNG bytes, then append Image with consistent size (6.5 x 4 in)
    def _embed_chart(chart_bytes, chart_name, width_inch=6.5, height_inch=4.0):
        if chart_bytes is None or len(chart_bytes) < 100:
            raise RuntimeError(
                f"Visual Insights chart '{chart_name}' could not be generated: "
                "missing data or empty PNG. Check predictions.csv has required columns and model has feature_importances_ if needed."
            )
        buf = io.BytesIO(chart_bytes)
        buf.seek(0)
        story.append(Image(buf, width=width_inch * inch, height=height_inch * inch))
        story.append(Spacer(1, 10))

    story.append(Paragraph("Visual Insights", styles["heading1"]))

    # Donut: Risk distribution
    story.append(Paragraph("Risk distribution", styles["heading2"]))
    donut_bytes = plot_risk_distribution(stats)
    _embed_chart(donut_bytes, "Risk distribution")
    story.append(Paragraph(
        "The donut chart shows how the customer base is split across risk levels. "
        "A large low-risk share indicates a stable base; a material high-risk slice is the priority for retention. "
        "Use this view to size retention effort and set targets.",
        styles["body"]
    ))
    story.append(Spacer(1, 12))

    # Histogram: Churn probability distribution
    story.append(Paragraph("Churn probability distribution", styles["heading2"]))
    hist_bytes = plot_probability_histogram(df)
    _embed_chart(hist_bytes, "Churn probability distribution")
    story.append(Paragraph(
        "The histogram shows how predicted churn probability is distributed across customers. "
        "A bimodal shape (peaks at low and high probability) suggests the model separates likely stayers from at-risk customers. "
        "Use this to calibrate thresholds and campaign sizing.",
        styles["body"]
    ))
    story.append(Spacer(1, 12))

    # Bar: Risk by contract type
    story.append(Paragraph("Risk by contract type", styles["heading2"]))
    contract_bytes = plot_contract_chart(df)
    _embed_chart(contract_bytes, "Risk by contract type")
    story.append(Paragraph(
        "Month-to-month contracts show a higher share of high-risk customers than one- or two-year contracts. "
        "Two-year contracts have the lowest high-risk share. Moving at-risk customers to longer terms (with fair incentives) can reduce churn.",
        styles["body"]
    ))
    story.append(Spacer(1, 12))

    # Bar: Top 15 feature importance (optional: model may not expose feature_importances_)
    story.append(Paragraph("Top 15 feature importance", styles["heading2"]))
    try:
        fi_bytes = plot_feature_importance(top_n=15)
        if fi_bytes and len(fi_bytes) >= 100:
            _embed_chart(fi_bytes, "Top 15 feature importance", height_inch=4.0)
        else:
            story.append(Paragraph(
                "Feature importance not available for this model (saved model does not expose feature_importances_).",
                styles["small"]
            ))
            story.append(Spacer(1, 10))
    except Exception as e:
        story.append(Paragraph(
            f"Feature importance chart could not be generated: {str(e)}",
            styles["small"]
        ))
        story.append(Spacer(1, 10))
    story.append(Paragraph(
        "Feature importance shows which inputs drive the model’s predictions. "
        "Contract type, tenure, and monthly charges often rank high; use this to prioritize data quality and to explain drivers to the business.",
        styles["body"]
    ))
    story.append(Spacer(1, 12))

    # Bar: Risk by tenure bucket
    story.append(Paragraph("Risk by tenure bucket", styles["heading2"]))
    tenure_bytes = plot_tenure_chart(df)
    _embed_chart(tenure_bytes, "Risk by tenure bucket")
    story.append(Paragraph(
        "Short-tenure buckets (e.g. 0–12 months) typically show a higher share of high-risk customers. "
        "New customers and those approaching the end of a commitment are key targets for retention and loyalty programs.",
        styles["body"]
    ))
    story.append(Spacer(1, 12))

    # SHAP global importance and summary (generate and embed; fail clearly if SHAP errors)
    shap_global_path = os.path.join(REPORTS_FIGURES_DIR, "shap_global_importance.png")
    shap_summary_path = os.path.join(REPORTS_FIGURES_DIR, "shap_summary.png")
    try:
        from src.shap_utils import (
            load_artifacts as shap_load_artifacts,
            build_explainer,
            compute_shap_values,
            plot_global_importance,
            plot_summary,
        )
        from src.predict import ensure_columns
        from src.train import build_features as build_features_train
        model_shap, encoder_shap, scaler_shap, feature_names_shap, column_meta_shap = shap_load_artifacts()
        df_shap = ensure_columns(df.copy(), column_meta_shap["cat_cols"], column_meta_shap["num_cols"])
        X_shap, _, _, _, _, _ = build_features_train(
            df_shap,
            encoder=encoder_shap,
            scaler=scaler_shap,
            fit=False,
            cat_cols_order=column_meta_shap["cat_cols"],
            num_cols_order=column_meta_shap["num_cols"],
        )
        n_sample = min(500, len(X_shap))
        rng = np.random.RandomState(42)
        idx_sample = rng.choice(len(X_shap), size=n_sample, replace=False)
        X_sample = X_shap[idx_sample]
        explainer_shap, is_tree = build_explainer(model_shap, X_sample)
        shap_vals = compute_shap_values(explainer_shap, X_sample, is_tree=is_tree)
        os.makedirs(REPORTS_FIGURES_DIR, exist_ok=True)
        plot_global_importance(shap_vals, feature_names_shap, shap_global_path, dpi=300)
        plot_summary(shap_vals, X_sample, feature_names_shap, shap_summary_path, dpi=300, max_display=20)
    except Exception as e:
        raise RuntimeError(
            "SHAP charts could not be generated for the report. Ensure the model is tree-based and SHAP is installed. Error: {}".format(e)
        ) from e
    if not os.path.exists(shap_global_path) or os.path.getsize(shap_global_path) == 0:
        raise RuntimeError("SHAP global importance chart file missing or empty: {}".format(shap_global_path))
    if not os.path.exists(shap_summary_path) or os.path.getsize(shap_summary_path) == 0:
        raise RuntimeError("SHAP summary chart file missing or empty: {}".format(shap_summary_path))

    story.append(Paragraph("SHAP global feature importance", styles["heading2"]))
    with open(shap_global_path, "rb") as f:
        shap_global_bytes = f.read()
    _embed_chart(shap_global_bytes, "SHAP global importance", width_inch=6.5, height_inch=4.0)
    story.append(Paragraph(
        "This chart shows which factors drive churn risk across the whole customer base. "
        "Features with longer bars have a stronger average impact on the model’s predictions. "
        "Use it to prioritize retention levers and to explain model behavior to stakeholders.",
        styles["body"]
    ))
    story.append(Spacer(1, 12))

    story.append(Paragraph("SHAP summary (feature impacts)", styles["heading2"]))
    with open(shap_summary_path, "rb") as f:
        shap_summary_bytes = f.read()
    _embed_chart(shap_summary_bytes, "SHAP summary", width_inch=6.5, height_inch=4.0)
    story.append(Paragraph(
        "Each dot is one customer; position and color show how a feature pushed the prediction up or down. "
        "Red (positive) values increase churn probability; blue (negative) decrease it. "
        "This view helps explain both overall drivers and variation across customers.",
        styles["body"]
    ))
    story.append(Spacer(1, 20))

    # ----- 5. Insights & Drivers -----
    story.append(PageBreak())
    story.append(Paragraph("Insights &amp; Drivers", styles["heading1"]))
    story.append(Paragraph(
        "Contract type strongly drives churn risk: month-to-month customers have the highest share of high-risk predictions, "
        "while two-year contracts have the lowest. Tenure and monthly charges are also strong predictors—short tenure and higher spend "
        "correlate with elevated churn probability. Customer behavior patterns (e.g. no add-ons, paperless billing, payment method) "
        "contribute to the model; use feature importance and segment analysis to prioritize retention levers.",
        styles["body"]
    ))
    story.append(Spacer(1, 12))

    # ----- 6. Risk Segmentation Table -----
    story.append(Paragraph("Risk segmentation", styles["heading1"]))
    t_seg = Table(seg_data, colWidths=[1.0 * inch, 0.9 * inch, 0.7 * inch, 3.4 * inch])
    t_seg.setStyle(TableStyle(_table_style_header(COLORS["primary"])))
    story.append(t_seg)
    story.append(Spacer(1, 16))

    # ----- 7. Recommended Actions (Retention Playbook) -----
    story.append(Paragraph("Recommended actions (retention playbook)", styles["heading1"]))
    actions = [
        "Proactive outreach: target high-risk, month-to-month customers with personalized save offers and contract upgrades.",
        "Contract upgrades and discounts: incentivize one- or two-year commitments for at-risk customers to lock in tenure.",
        "Save desk: route voluntary churn signals and high-risk scores to retention teams with clear scripts and offer rules.",
        "Loyalty campaigns: light-touch campaigns (usage tips, rewards) for medium-risk to prevent slip into high risk.",
        "Protect low-risk: maintain service quality and avoid unnecessary friction; use for referral or cross-sell where appropriate.",
    ]
    for a in actions:
        story.append(Paragraph(f"• {a}", styles["bullet"]))
    story.append(Spacer(1, 12))

    # ----- 7b. Retention ROI Simulator (subsection) -----
    story.append(Paragraph("Retention ROI Simulator", styles["heading2"]))
    roi_avg_cltv = 1200.0
    roi_cost_per_offer = 25.0
    roi_segment = "High"
    rev_roi = compute_revenue_at_risk(df, roi_avg_cltv, segment=roi_segment, cltv_col="CLTV" if "CLTV" in df.columns else None)
    roi_table_data = [["Save rate", "Revenue saved ($)", "Offer cost ($)", "Net benefit ($)"]]
    for sr_pct in [10, 20, 30]:
        r = compute_roi(rev_roi["revenue_at_risk"], sr_pct / 100.0, roi_cost_per_offer, rev_roi["targeted_customers"])
        roi_table_data.append([
            "{}%".format(sr_pct),
            "{:,.0f}".format(r["revenue_saved"]),
            "{:,.0f}".format(r["offer_cost"]),
            "{:,.0f}".format(r["net_benefit"]),
        ])
    t_roi = Table(roi_table_data, colWidths=[1.0 * inch, 1.4 * inch, 1.2 * inch, 1.4 * inch])
    t_roi.setStyle(TableStyle(_table_style_header(COLORS["primary"])))
    story.append(t_roi)
    story.append(Spacer(1, 8))
    roi_chart_path = os.path.join(REPORTS_FIGURES_DIR, "roi_simulator.png")
    try:
        plot_roi_vs_save_rate(
            rev_roi["revenue_at_risk"],
            rev_roi["targeted_customers"],
            roi_cost_per_offer,
            list(range(0, 51, 5)),
            roi_chart_path,
            dpi=200,
        )
        if os.path.exists(roi_chart_path) and os.path.getsize(roi_chart_path) > 0:
            with open(roi_chart_path, "rb") as f_roi:
                roi_bytes = f_roi.read()
            _embed_chart(roi_bytes, "Retention ROI Simulator", width_inch=6.5, height_inch=3.5)
        else:
            story.append(Paragraph("ROI chart could not be generated.", styles["small"]))
    except Exception as e:
        story.append(Paragraph("ROI chart could not be generated: {}.".format(str(e)), styles["small"]))
    story.append(Paragraph(
        "Revenue at risk is estimated from the target segment (high-risk) and average CLTV. "
        "Net benefit = revenue saved (at the chosen save rate) minus offer cost. Use the dashboard ROI Simulator to explore scenarios.",
        styles["body"]
    ))
    story.append(Spacer(1, 16))

    # ----- 8. Top High-Risk Customers Table -----
    story.append(Paragraph("Top high-risk customers", styles["heading1"]))
    story.append(Paragraph("Top 20 by churn probability. Use for immediate retention outreach.", styles["small"]))
    story.append(Spacer(1, 8))
    if top20_data:
        col_w = 6.0 / len(top20_data[0])
        t_top = Table(top20_data, colWidths=[col_w * inch] * len(top20_data[0]))
        t_top.setStyle(TableStyle(_table_style_header(COLORS["primary"])))
        story.append(t_top)
    else:
        story.append(Paragraph("No high-risk records or required columns.", styles["body"]))
    story.append(Spacer(1, 16))

    # ----- 9. Definitions -----
    story.append(Paragraph("Definitions", styles["heading1"]))
    def_data = [
        ["Term", "Definition"],
        ["Churn probability", "Model-generated probability (0–100%) that the customer will churn in the defined period."],
        ["Risk level", "Bucket (High / Medium / Low) derived from churn probability using business-defined thresholds."],
        ["High risk", "Above upper threshold; highest priority for retention actions."],
        ["Medium risk", "Middle band; monitor and light-touch retention."],
        ["Low risk", "Below lower threshold; maintain experience and avoid over-contacting."],
    ]
    t_def = Table(def_data, colWidths=[1.4 * inch, 5.1 * inch])
    t_def.setStyle(TableStyle(_table_style_header(COLORS["primary"])))
    story.append(t_def)
    story.append(Spacer(1, 12))

    # ----- 10. Method Summary -----
    story.append(Paragraph("Method summary", styles["heading1"]))
    story.append(Paragraph(
        f"Model: <b>{model_name}</b>. Preprocessing: one-hot encoding for categorical features, standard scaling for numerical. "
        f"Training: train/test split; best model selected by accuracy. Evaluation: Accuracy {acc_str}, F1 {f1_str}, ROC-AUC {auc_str}. "
        "Output: churn probability per customer mapped to risk levels for business use.",
        styles["body"]
    ))
    story.append(Spacer(1, 12))

    # ----- 11. Limitations & Assumptions -----
    story.append(Paragraph("Limitations &amp; assumptions", styles["heading1"]))
    limits = [
        "Predictions are probabilistic; actual churn depends on factors not in the model (e.g. competitor moves, life events).",
        "Risk thresholds (High/Medium/Low) are set by business; changing them changes segment sizes and actions.",
        f"Results reflect data as of {data_updated}; older data may not reflect current behavior.",
        "Correlations (e.g. contract type vs risk) do not imply causation; test interventions before scaling.",
    ]
    for L in limits:
        story.append(Paragraph(f"• {L}", styles["bullet"]))
    story.append(Spacer(1, 16))
    story.append(Paragraph(
        "<i>Confidential. For internal use. Contact analytics team or report owner for questions.</i>",
        styles["small"]
    ))

    doc.build(story)
    buffer.seek(0)
    pdf_bytes = buffer.getvalue()
    if isinstance(output_path, (str, os.PathLike)):
        with open(output_path, "wb") as f:
            f.write(pdf_bytes)
    else:
        output_path.write(pdf_bytes)


# ---------------------------------------------------------------------------
# Main: load data, compute stats, load metrics, build PDF
# ---------------------------------------------------------------------------

def main(predictions_path=None, output_path=None):
    """
    Load predictions.csv, compute stats, load metrics from eval_metrics.json,
    build full PDF. No placeholders; everything from data or saved metrics.
    """
    df = load_predictions(predictions_path)
    if df is None:
        path = predictions_path or os.path.join(DATA_DIR, "predictions.csv")
        raise FileNotFoundError(f"Predictions not found or invalid: {path}. Run prediction pipeline first.")

    stats = summary_stats(df)
    metrics = load_metrics()
    if not metrics or metrics.get("accuracy") is None:
        metrics = compute_metrics_from_predictions(df)
    data_updated = datetime.now().strftime("%Y-%m-%d")
    pred_path = predictions_path or os.path.join(DATA_DIR, "predictions.csv")
    if os.path.exists(pred_path):
        try:
            data_updated = datetime.fromtimestamp(os.path.getmtime(pred_path)).strftime("%Y-%m-%d")
        except Exception:
            pass

    out = output_path or os.path.join(PROJECT_ROOT, OUTPUT_PDF)
    build_pdf(df, stats, metrics, data_updated, out)
    return out


if __name__ == "__main__":
    import sys
    out_path = os.path.join(PROJECT_ROOT, OUTPUT_PDF)
    if len(sys.argv) > 1:
        out_path = sys.argv[1]
    result = main(output_path=out_path)
    print(f"Report saved: {result}")


# ---------------------------------------------------------------------------
# API compatibility: generate_report_pdf() for run_predict_api.py
# ---------------------------------------------------------------------------

def generate_report_pdf(output_path_or_buffer=None, predictions_path=None):
    """
    Generate full report PDF. If output_path_or_buffer is None, returns (True, pdf_bytes).
    Otherwise writes to path or buffer and returns (True, None). On failure returns (False, None).
    """
    try:
        df = load_predictions(predictions_path)
        if df is None:
            return False, None
        stats = summary_stats(df)
        metrics = load_metrics()
        if not metrics or metrics.get("accuracy") is None:
            metrics = compute_metrics_from_predictions(df)
        data_updated = datetime.now().strftime("%Y-%m-%d")
        pred_path = predictions_path or os.path.join(DATA_DIR, "predictions.csv")
        if os.path.exists(pred_path):
            try:
                data_updated = datetime.fromtimestamp(os.path.getmtime(pred_path)).strftime("%Y-%m-%d")
            except Exception:
                pass
        buffer = io.BytesIO()
        build_pdf(df, stats, metrics, data_updated, buffer)
        buffer.seek(0)
        pdf_bytes = buffer.getvalue()
        if output_path_or_buffer is not None:
            if isinstance(output_path_or_buffer, (str, os.PathLike)):
                with open(output_path_or_buffer, "wb") as f:
                    f.write(pdf_bytes)
            else:
                output_path_or_buffer.write(pdf_bytes)
            return True, None
        return True, pdf_bytes
    except Exception:
        # Re-raise so API can return 500 with real error (e.g. missing column, chart failure)
        raise
