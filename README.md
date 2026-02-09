# Churn Intelligence Platform

An end-to-end **Churn Intelligence Platform** that predicts customer churn, explains risk with SHAP, surfaces insights in an interactive Streamlit dashboard, generates executive PDF reports, syncs high-risk customers to Salesforce with retention tasks, and monitors model health via PSI drift and trend charts. Includes an ROI simulator for retention savings vs. save rate.

---

## Table of Contents

- [Executive Summary](#executive-summary)
- [Key Features](#key-features)
- [Tech Stack](#tech-stack)
- [Dataset](#dataset)
- [System Architecture](#system-architecture)
- [ML Methodology](#ml-methodology)
- [Explainability (SHAP)](#explainability-shap)
- [Dashboard Guide](#dashboard-guide)
- [Report Guide](#report-guide)
- [Automation (n8n)](#automation-n8n)
- [Salesforce Integration](#salesforce-integration)
- [Monitoring](#monitoring)
- [ROI Simulator](#roi-simulator)
- [Folder Structure](#folder-structure)
- [Quickstart (Local Run)](#quickstart-local-run)
- [Configuration](#configuration)
- [Deployment](#deployment)
- [Screenshots](#screenshots)
- [Testing & Validation](#testing--validation)
- [Troubleshooting](#troubleshooting)
- [Security & Privacy](#security--privacy)
- [Roadmap](#roadmap)
- [License & Acknowledgements](#license--acknowledgements)

---

## Executive Summary

**What it is**  
The Churn Intelligence Platform is a production-style Python application that scores customers for churn risk, explains each high-risk customer with SHAP drivers, and turns predictions into actionable retention workflows. It combines ML (tuned XGBoost, LightGBM, CatBoost, and stacking), explainability, a Streamlit dashboard with login, automated PDF reports, n8n scheduling with email alerts, Salesforce CRM sync (Contact upsert + retention Tasks for high-risk), and monitoring (PSI drift + trend charts).

**Why it matters**  
Retention is cheaper than acquisition. Identifying who is likely to churn and why allows teams to prioritize outreach, tailor offers, and measure ROI. This project demonstrates end-to-end ML ops: training, prediction, explainability, reporting, CRM integration, and model monitoring.

**Outcomes**  
- **Risk bands:** Every customer gets a churn probability and a risk level (High / Medium / Low) for prioritization.  
- **Explainability:** High-risk customers get top churn drivers (SHAP) in the dashboard and in the PDF report.  
- **Actions:** Salesforce Contacts are updated with risk and drivers; retention Tasks are created for high-risk customers.  
- **ROI:** The ROI simulator estimates retention savings and net benefit vs. save rate and offer cost.

---

## Key Features

- **ML churn prediction** — Train script with Optuna-tuned XGBoost, LightGBM, CatBoost, and stacking; predict script outputs probabilities and risk bands.  
- **SHAP explainability** — Global feature importance in dashboard and report; per-customer top drivers for all high-risk customers (stored in `Top_Drivers`).  
- **Streamlit dashboard** — Login/signup, filters, risk distribution, risk mix, SHAP summary, single-customer prediction, ROI simulator, upload dataset.  
- **Automated PDF report** — Executive summary, KPIs, charts (risk distribution, contract/tenure, feature importance, ROI curve), monitoring trend + drift status, top high-risk table, definitions, method, limitations.  
- **n8n automation** — Scheduled run (e.g. daily 9am), HTTP calls to run predictions and generate report PDF, Gmail send with summary and PDF attachment.  
- **Salesforce CRM sync** — Upsert Contact by external ID (`CustomerID__c`), write churn probability, risk level, scored-at, top drivers, model version; create retention Tasks for high-risk contacts.  
- **Monitoring** — PSI-based drift vs. training baseline; daily metrics history; trend charts (e.g. high-risk % over time); OK / WARNING / ALERT thresholds.  
- **ROI simulator** — Revenue at risk, revenue saved, offer cost, net benefit vs. save rate (with optional chart).

---

## Tech Stack

| Layer        | Technology |
|-------------|------------|
| Language    | Python 3.10+ |
| ML          | scikit-learn, XGBoost, LightGBM, CatBoost, Optuna |
| Explainability | SHAP |
| Dashboard   | Streamlit, Plotly |
| API         | Flask (for n8n: `/run-predict`, `/report`, `/health`) |
| PDF         | ReportLab, matplotlib |
| CRM         | simple-salesforce |
| Automation  | n8n (Schedule Trigger, HTTP Request, Gmail) |
| Auth        | SQLite + passlib (dashboard login) |
| Config      | python-dotenv, `.env` |

---

## Dataset

**Source**  
The pipeline expects a Telco-style churn dataset. The default input is **`Telco_customer_churn.xlsx`** in the project root (or the path set in config). This matches the IBM Telco Customer Churn dataset schema (customer attributes, tenure, charges, contract, services, and a churn label).

**Schema overview**  
Typical columns used for features (after cleanup and feature engineering):

| Category     | Examples |
|-------------|----------|
| Demographics | Gender, Senior Citizen, Partner, Dependents |
| Services     | Phone Service, Multiple Lines, Internet Service, Online Security, Streaming TV/Movies, etc. |
| Billing      | Contract, Paperless Billing, Payment Method |
| Numeric      | Tenure Months, Monthly Charges, Total Charges |
| Engineered   | Tenure Bucket, Avg Monthly Spend (from Total Charges / Tenure) |

**Excluded from features**  
Identifiers and leakage are dropped: `CustomerID`, `Count`, `Country`, `State`, `City`, `Zip Code`, `Lat Long`, `Latitude`, `Longitude`, `Churn Label`, `Churn Reason`, `CLTV`, `Churn Score`. `CustomerID` is kept only in prediction outputs for CRM sync.

**Target**  
- **Training:** `Churn Value` (binary: 0 = No, 1 = Yes).  
- **Outputs:** `Churn Probability` (0–1) and `Risk Level` (Low / Medium / High).

---

## System Architecture

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│                         CHURN INTELLIGENCE PLATFORM                          │
├─────────────────────────────────────────────────────────────────────────────┤
│  DATA LAYER                                                                  │
│  Telco_customer_churn.xlsx → data/cleaned_churn.csv                          │
│  data/predictions.csv, data/predictions_enriched.csv (CustomerID, Prob,    │
│  Risk Level, Top_Drivers)                                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│  ML LAYER                                                                    │
│  src/train.py → models/churn_model.joblib, encoder, scaler, column_meta,     │
│                 eval_metrics.json, model_meta.json; monitoring/baseline_*   │
│  src/predict.py → predictions + SHAP for high-risk → predictions_enriched   │
├─────────────────────────────────────────────────────────────────────────────┤
│  EXPLAINABILITY                                                              │
│  src/shap_utils.py (global + per-row); reports/figures/shap_*.png            │
├─────────────────────────────────────────────────────────────────────────────┤
│  APPLICATION LAYER                                                           │
│  app.py (Streamlit) ─┬─ Login → Dashboard: filters, charts, ROI, upload    │
│  run_predict_api.py ─┴─ Flask: /run-predict, /report, /health (for n8n)     │
├─────────────────────────────────────────────────────────────────────────────┤
│  REPORTING                                                                   │
│  src/report_generator.py → churn_report.pdf (exec summary, KPIs, charts,     │
│  monitoring trend, drift status, top high-risk table)                       │
├─────────────────────────────────────────────────────────────────────────────┤
│  INTEGRATION                                                                 │
│  src/salesforce_sync.py → Contact upsert (CustomerID__c), Task for High     │
│  monitoring/monitor.py → metrics_history.csv, drift_report.json, plots      │
├─────────────────────────────────────────────────────────────────────────────┤
│  AUTOMATION (n8n)                                                            │
│  Schedule → HTTP /run-predict → HTTP /report (PDF) → Gmail (body + attach)  │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Mermaid (optional)**  
If your GitHub supports Mermaid, you can add:

```mermaid
flowchart LR
  A[Telco Data] --> B[train.py]
  B --> C[models/]
  C --> D[predict.py]
  D --> E[predictions_enriched.csv]
  E --> F[Streamlit app]
  E --> G[report_generator]
  E --> H[salesforce_sync]
  E --> I[monitoring]
  J[n8n Schedule] --> K[/run-predict]
  K --> L[/report]
  L --> M[Gmail]
```

---

## ML Methodology

**Preprocessing**  
- Load raw Excel/CSV; drop identifier and leakage columns; fix `Total Charges` (numeric, fill missing with tenure × monthly where needed).  
- **Feature engineering:** Tenure buckets (0–12, 13–24, 25–48, 49–72, 73+ months), `Avg Monthly Spend`.  
- **Encoding:** Categoricals one-hot encoded (drop first); numerics standardized. Column order and metadata (cat_cols, num_cols, feature_names) are saved for inference.

**Models**  
- **Candidates:** Optuna-tuned XGBoost, LightGBM, CatBoost (accuracy objective), plus a stacking ensemble (base estimators + meta LogisticRegression).  
- **Selection:** Best model by **accuracy** on the holdout set is saved (`churn_model.joblib`). Metrics (accuracy, F1, ROC-AUC) are written to `models/eval_metrics.json`.

**Evaluation metrics**  
| Metric    | Description |
|-----------|-------------|
| Accuracy  | Proportion of correct predictions (threshold 0.5). |
| F1 score  | Harmonic mean of precision and recall (imbalanced class). |
| ROC-AUC   | Area under ROC curve (ranking quality). |

**Thresholding & risk bands**  
Probabilities are bucketed into three bands (configurable in code):

| Risk Level | Churn probability range |
|------------|---------------------------|
| Low        | &lt; 0.33 |
| Medium     | 0.33 – 0.66 |
| High       | &gt; 0.66 |

These drive prioritization in the dashboard, report, Salesforce (e.g. Tasks only for High), and ROI simulator (e.g. revenue at risk for High or High+Medium).

---

## Explainability (SHAP)

**Global**  
- A SHAP summary (e.g. mean |SHAP| or summary plot) is produced and saved under `reports/figures/` (e.g. `shap_summary.png`, `shap_global_importance.png`).  
- The dashboard and PDF report include this chart to show which features drive churn overall.

**Per-customer (high-risk only)**  
- For every customer with **Risk Level = High**, SHAP values are computed; the top-*k* drivers (default *k* = 5) are written to the `Top_Drivers` column in `predictions_enriched.csv`.  
- Format is human-readable (e.g. “Contract_Month-to-month (+0.12), Tenure Months (-0.08), …”).  
- This column is used in the dashboard single-customer view and in the PDF; it is also synced to Salesforce (`Top_Drivers__c`) and used in retention Task descriptions.

**Implementation**  
- `src/shap_utils.py`: build explainer (tree or kernel), compute SHAP values, map to feature names, produce top drivers per row.  
- If SHAP is not installed, prediction still runs; `Top_Drivers` is left empty and a message is printed.

---

## Dashboard Guide

**Entry**  
Run `streamlit run app.py`; open the URL (e.g. `http://localhost:8501`). The first screen is the **login** (or signup) page.

**After login**  
- **Filters & views** — Filter by risk level, contract, tenure, etc.; choose which visualizations to show.  
- **Churn probability distribution** — Histogram of predicted probabilities.  
- **Risk mix** — Proportion of High / Medium / Low (e.g. pie or bar).  
- **What drives churn risk?** — Global SHAP or feature importance chart.  
- **Extra visualizations** — Additional charts (e.g. risk by contract, by tenure).  
- **Why is this customer high risk?** — Select a high-risk customer to see their top SHAP drivers (from `Top_Drivers`).  
- **Customer list** — Table of predictions; expandable “first 500 rows” view.  
- **Retention ROI Simulator** — Inputs: segment (High / Medium / High+Medium), avg CLTV, save rate, cost per offer; outputs: revenue at risk, revenue saved, offer cost, net benefit; optional chart vs. save rate.  
- **Upload Dataset** — Upload a CSV/Excel file, run prediction on it, view results.  
- **Single Customer Prediction** — Enter or select one customer’s attributes and see predicted probability and risk level.

**Data source**  
The dashboard reads `data/predictions_enriched.csv` (or `data/predictions.csv` if enriched is missing). Run `python -m src.predict` (or the API `/run-predict`) to refresh.

---

## Report Guide

**What’s in the PDF**  
The report (`churn_report.pdf`) includes:

1. **Title** — Report name and date.  
2. **Executive summary** — Total scored, risk mix (high/medium/low counts and %), key drivers (contract, tenure, charges), business impact and recommended actions.  
3. **Key metrics** — KPI cards (total, high/medium/low, avg churn probability) and a metrics table (model name, accuracy, F1, ROC-AUC).  
4. **Monitoring** — High-risk % trend (from `reports/figures/monitoring_high_risk_trend.png`) and today’s drift status (OK / WARNING / ALERT) from `monitoring/drift_report.json`.  
5. **Visual insights** — Charts with short explanations: risk distribution, risk by contract, by tenure, feature importance, ROI vs. save rate (from `reports/figures/roi_simulator.png` if present).  
6. **Insights & drivers** — Narrative on main drivers and SHAP.  
7. **Risk segmentation** — Counts and percentages by risk band.  
8. **Recommended actions** — Prioritization and next steps.  
9. **Top high-risk table** — Top 20 high-risk customers (e.g. Contract, Tenure Months, Monthly Charges, Churn Probability, Risk Level).  
10. **Definitions** — Risk level, high/medium/low, thresholds.  
11. **Method summary** — Data, model, thresholds.  
12. **Limitations & next steps** — Short caveats and improvements.

**Where it’s generated**  
- **CLI:** `python -m src.report_generator` → writes `churn_report.pdf` in the project root.  
- **API:** `GET /report` (from `run_predict_api.py`) returns the PDF as a file download (e.g. for n8n attachment).  
- The report uses `data/predictions_enriched.csv` (or `data/predictions.csv`), `models/eval_metrics.json`, `monitoring/drift_report.json`, and figures in `reports/figures/`.

---

## Automation (n8n)

**Workflow overview**  
The included workflow runs on a schedule, calls your prediction API, fetches the PDF report, and emails both a summary and the PDF via Gmail.

**Flow**  
1. **Schedule** — Trigger daily at 9:00 (cron `0 9 * * *`).  
2. **Run Churn Prediction API** — HTTP GET to `https://YOUR-BASE-URL/run-predict` (e.g. ngrok or a deployed Flask app).  
3. **Generate Report PDF** — HTTP GET to `https://YOUR-BASE-URL/report` with `responseFormat: file` to receive the PDF.  
4. **Gmail Send** — Send an email with a body that references the prediction summary (e.g. `$node["Run Churn Prediction API"].json.summary`) and attach the PDF from the previous node (e.g. `attachmentFieldName: "data"`).

**Node-by-node**

| Node                    | Type              | Purpose |
|-------------------------|-------------------|--------|
| Daily at 9am            | Schedule Trigger  | Run once per day at 9:00. |
| Run Churn Prediction API| HTTP Request      | GET `/run-predict`; returns JSON with `success`, `summary` (total, high_risk, medium_risk, low_risk). |
| Generate Report PDF     | HTTP Request      | GET `/report`; response format “file”; output binary as `data` for attachment. |
| Gmail Send              | Gmail             | Send to a list; subject and body use summary from first HTTP node; attach PDF from second HTTP node. |

**How to import**  
1. Open n8n.  
2. Create a new workflow (or open an existing one).  
3. Use **Import from File** (or paste JSON) and select `n8n-workflow-http-with-gmail.json`.  
4. Replace placeholders:  
   - In **Run Churn Prediction API**: set URL to your API base (e.g. `https://your-ngrok-subdomain.ngrok.io/run-predict`).  
   - In **Generate Report PDF**: set URL to the same base with `/report` (e.g. `https://your-ngrok-subdomain.ngrok.io/report`).  
   - In **Gmail Send**: set `toList` and ensure Gmail credentials are configured in n8n.  
5. Save and activate the workflow.

For how to capture workflow and email screenshots, see [Screenshots](#screenshots).

---

## Salesforce Integration

**What gets written**  
- **Contact (upsert):** Matched by external ID `CustomerID__c` (or the field set in `SF_CONTACT_EXTERNAL_ID_FIELD`). Updated/created fields include: `LastName`, `Churn_Probability__c`, `Risk_Level__c`, `Scored_At__c`, `Top_Drivers__c`, `Model_Version__c`.  
- **Task (create):** For each **High**-risk contact, one Task is created: Subject “Retention outreach — High churn risk”, ActivityDate = today + 2 days, Description includes churn probability and top drivers (from `Top_Drivers__c`).

**Required in Salesforce**  
- Custom fields on **Contact**: `CustomerID__c` (External ID, unique), `Churn_Probability__c`, `Risk_Level__c`, `Scored_At__c`, `Top_Drivers__c` (Long Text Area if needed), `Model_Version__c`.  
- **Task** uses standard `WhoId`, `Subject`, `ActivityDate`, `Description`.

**Setup steps**  
1. In Salesforce: Create the Contact custom fields and set `CustomerID__c` as External ID.  
2. In this project: Copy `.env.example` to `.env` and set credentials (see [Configuration](#configuration)).  
3. Ensure predictions CSV has a column matching the external ID (default `CustomerID`); column name can be overridden with `CUSTOMER_ID_COLUMN`.  
4. Run sync: `python -m src.salesforce_sync` (or with `--limit 10` for a test).  
5. Check `logs/salesforce_sync_summary.json` for counts and any errors.

**Environment variables**  
| Variable | Purpose |
|----------|---------|
| `SF_USERNAME` | Salesforce username (SOAP or OAuth password flow). |
| `SF_PASSWORD` | Password (and optionally security token if `SF_PASSWORD_AND_TOKEN=1`). |
| `SF_SECURITY_TOKEN` | Required for SOAP login if not using Connected App. |
| `SF_DOMAIN` | `login` or `test` (sandbox); or My Domain prefix (e.g. `mycompany.my`) for OAuth. |
| `SF_CONSUMER_KEY` / `SF_CONSUMER_SECRET` | Connected App (when SOAP is disabled). |
| `SF_USE_CLIENT_CREDENTIALS` | Set to `1` to use client credentials flow (no username/password). |
| `SF_PASSWORD_AND_TOKEN` | `1` (default) = send password+token; `0` = password only. |
| `CUSTOMER_ID_COLUMN` | CSV column for Contact external ID (default `CustomerID`). |
| `SF_CONTACT_EXTERNAL_ID_FIELD` | Contact external ID field API name (default `CustomerID__c`). |
| `PREDICTIONS_FILE` | Optional path to predictions CSV (default: `data/predictions_enriched.csv` or `data/predictions.csv`). |
| `SF_SYNC_HIGH_RISK_ONLY` | Set to `1` to sync only rows with Risk Level = High. |

---

## Monitoring

**PSI drift**  
- **Baseline:** Training (or first run) writes a snapshot of key columns to `monitoring/baseline_reference.csv` (e.g. Tenure Months, Monthly Charges, Total Charges, Contract, Internet Service, Payment Method).  
- **Current:** Each monitoring run uses the latest predictions (same columns when present).  
- **PSI:** For each feature, Population Stability Index is computed (numeric: binned by baseline quantiles; categorical: proportion shift).  
- **Status:** Overall status is derived from the maximum PSI: **OK** &lt; 0.1, **WARNING** 0.1–0.2, **ALERT** ≥ 0.2.  
- **Output:** `monitoring/drift_report.json` (run_at, psi_per_feature, status, thresholds).

**Trend charts**  
- **Daily metrics:** Each run appends one row to `monitoring/metrics_history.csv` (date, total_customers, high/medium/low counts and %, avg_churn_probability).  
- **Plots:** `monitoring/plots.py` generates trend charts (e.g. high-risk % over time, avg churn probability) and a PSI bar chart; saved under `reports/figures/` (e.g. `monitoring_high_risk_trend.png`, `monitoring_avg_churn_trend.png`, `monitoring_psi_bar.png`).  
- The PDF report includes the high-risk trend and the drift status line.

**Alert thresholds**  
| Status  | PSI range | Action |
|---------|-----------|--------|
| OK      | &lt; 0.1  | No action. |
| WARNING | 0.1–0.2  | Review feature distributions; consider retraining or data checks. |
| ALERT   | ≥ 0.2    | Investigate data or model; plan retrain or threshold update. |

**How to run**  
After predictions exist: `python -m monitoring.monitor` (optionally pass path to predictions CSV). Run daily (e.g. after `/run-predict` in n8n) to build history and drift.

---

## ROI Simulator

**Inputs**  
- **Segment:** High, Medium, or High+Medium (from `Risk Level` in predictions).  
- **Avg CLTV:** Average customer lifetime value (used when no CLTV column in data).  
- **Save rate:** Fraction of at-risk revenue retained by retention actions (0–1).  
- **Cost per offer:** Cost per targeted customer (e.g. discount or incentive).

**Formulas**  
- **Revenue at risk** = sum of CLTV for segment (or segment count × avg CLTV if no CLTV column).  
- **Revenue saved** = revenue at risk × save rate.  
- **Offer cost** = targeted customers × cost per offer.  
- **Net benefit** = revenue saved − offer cost.

**Interpretation**  
- Positive net benefit: retention campaign is profitable at the given save rate and cost.  
- The dashboard (and report) can plot net benefit vs. save rate (e.g. 0%, 5%, …, 50%) for a given cost per offer to support sizing and target save rates.

**Implementation**  
- `src/roi.py`: `compute_revenue_at_risk`, `compute_roi`, `plot_roi_vs_save_rate`.  
- Chart is saved to `reports/figures/roi_simulator.png` when generated for the report.

---

## Folder Structure

```text
.
├── app.py                          # Streamlit dashboard entry
├── run_predict_api.py              # Flask API for n8n (/run-predict, /report, /health)
├── requirements.txt
├── .env.example                    # Template for env vars (copy to .env)
├── .gitignore
├── Telco_customer_churn.xlsx       # Default raw dataset (or your file)
├── churn_report.pdf                # Generated by report_generator (or API)
├── n8n-workflow-http-with-gmail.json
├── DEPLOY_OPTION1.md               # Streamlit Community Cloud deployment steps
│
├── data/
│   ├── cleaned_churn.csv           # Cleaned + engineered training data
│   ├── predictions.csv             # Predictions (CustomerID, Churn Probability, Risk Level)
│   ├── predictions_enriched.csv    # + Top_Drivers (SHAP) for high-risk
│   └── users.db                    # Dashboard auth (SQLite; gitignored)
│
├── models/
│   ├── churn_model.joblib          # Trained model
│   ├── feature_encoder.joblib
│   ├── feature_scaler.joblib
│   ├── preprocess.joblib
│   ├── column_meta.joblib           # cat_cols, num_cols, feature_names
│   ├── feature_names.joblib / feature_names.json
│   ├── eval_metrics.json           # accuracy, f1_score, roc_auc, model_name
│   └── model_meta.json             # version, train_date
│
├── reports/
│   └── figures/                   # SHAP, risk charts, monitoring, ROI
│       ├── shap_summary.png
│       ├── shap_global_importance.png
│       ├── monitoring_high_risk_trend.png
│       ├── monitoring_avg_churn_trend.png
│       ├── monitoring_psi_bar.png
│       └── roi_simulator.png
│
├── monitoring/
│   ├── __init__.py
│   ├── monitor.py                  # Daily metrics + PSI drift + plots
│   ├── plots.py                    # Trend and PSI bar charts
│   ├── baseline_reference.csv     # Baseline for PSI
│   ├── metrics_history.csv        # Daily metrics over time
│   └── drift_report.json          # Latest PSI and status
│
├── logs/
│   └── salesforce_sync_summary.json
│
├── src/
│   ├── __init__.py
│   ├── config.py                  # Paths, TARGET_COL, DROP_COLS, CAT_COLS
│   ├── data_loader.py             # load_raw_data, prepare_data, engineer_features
│   ├── train.py                   # train_and_evaluate, build_features
│   ├── predict.py                 # predict_and_save, predict_proba, SHAP enrich
│   ├── shap_utils.py              # SHAP explainer and top drivers
│   ├── report_charts.py           # KPI and chart generation for report
│   ├── report_generator.py        # Full PDF report
│   ├── roi.py                     # Revenue at risk, ROI, net benefit, plot
│   ├── salesforce_sync.py         # Contact upsert, Task create
│   ├── auth_db.py                 # Dashboard login/signup
│   └── email_sender.py            # Verification email (optional)
│
└── .streamlit/
    └── config.toml                # Streamlit theme/settings
```

---

## Quickstart (Local Run)

**1. Clone and install**

```powershell
cd "C:\Users\kalra\Downloads\My Project"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

On macOS/Linux:

```bash
cd /path/to/My\ Project
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**2. Data**  
Place `Telco_customer_churn.xlsx` in the project root (or set the path in `src/config.py` / env).

**3. Train**

```powershell
python -m src.train
```

This writes artifacts under `models/` and `monitoring/baseline_reference.csv`.

**4. Predict**

```powershell
python -m src.predict
```

This produces `data/predictions.csv` and `data/predictions_enriched.csv` (with `Top_Drivers` for high-risk).

**5. Dashboard**

```powershell
streamlit run app.py
```

Open the URL shown (e.g. `http://localhost:8501`), sign up or log in, then use the dashboard.

**6. Report (optional)**

```powershell
python -m src.report_generator
```

Opens/saves `churn_report.pdf` in the project root.

**7. Monitoring (optional)**

```powershell
python -m monitoring.monitor
```

Updates `monitoring/metrics_history.csv`, `monitoring/drift_report.json`, and `reports/figures/` monitoring plots.

**8. API for n8n (optional)**

```powershell
python run_predict_api.py
```

Then: `GET http://localhost:5050/run-predict`, `GET http://localhost:5050/report`, `GET http://localhost:5050/health`. Use a tunnel (e.g. ngrok) if n8n runs elsewhere.

**9. Salesforce sync (optional)**  
Set `.env` (see [Configuration](#configuration)), then:

```powershell
python -m src.salesforce_sync
```

Use `python -m src.salesforce_sync --limit 10` for a small test.

---

## Configuration

**`.env`**  
Copy `.env.example` to `.env` and fill in values. Never commit `.env` (it is in `.gitignore`).

**`.env.example` summary**  
- **Salesforce:** `SF_USERNAME`, `SF_PASSWORD`, `SF_SECURITY_TOKEN`, `SF_DOMAIN` (or Connected App keys and optional client credentials).  
- **Optional:** `CUSTOMER_ID_COLUMN`, `PREDICTIONS_FILE`, `SF_CONTACT_EXTERNAL_ID_FIELD`, `SF_SYNC_HIGH_RISK_ONLY`, `SF_SYNC_LIMIT`.  
- **Email (verification):** `APP_URL`, `SMTP_HOST`, `SMTP_PORT`, `SMTP_USER`, `SMTP_PASSWORD`, `FROM_EMAIL`.

**Safety**  
- No secrets in code; all sensitive values come from environment (or `.env`).  
- In CI or deployment, use the platform’s secrets (e.g. Streamlit Cloud Secrets, Render env vars).

---

## Deployment

**Streamlit Community Cloud (dashboard)**  
1. Push the repo to GitHub (e.g. `github.com/your-username/your-repo`).  
2. Go to [share.streamlit.io](https://share.streamlit.io), sign in with GitHub.  
3. New app → choose repo, branch `main`, main file path **`app.py`**.  
4. Deploy; then set **Secrets** (e.g. `APP_URL`, `SMTP_*`, Salesforce vars) if you use email or CRM.  
5. Share the app URL.  

Detailed steps are in **DEPLOY_OPTION1.md**.

**API (e.g. Render / Railway)**  
1. Use `run_predict_api.py` as the web process; set start command: `python run_predict_api.py` (and `PORT` if required).  
2. Ensure `data/`, `models/`, and optional `reports/figures/` are present (e.g. build step: run train + predict once, or mount storage).  
3. Set env vars (no `.env` file in production).  
4. Use the deployed base URL in n8n (replace YOUR-NGROK-URL or localhost with this URL).

**n8n**  
- Self-hosted or n8n Cloud. Import `n8n-workflow-http-with-gmail.json`, set the two HTTP node URLs and Gmail credentials, then activate.

---

## Screenshots

The repo does not ship with screenshot assets. To add them (e.g. in `docs/` or `screenshots/`), generate as follows:

| Asset | How to generate |
|-------|------------------|
| **Dashboard login** | Run `streamlit run app.py`, open the app URL, capture the login/signup screen. |
| **Dashboard main** | After login, capture the main view: sidebar + Filters & Views and Churn probability distribution. |
| **ROI Simulator** | In the dashboard, open “Retention ROI Simulator”, set segment (e.g. High), avg CLTV, save rate, cost per offer; capture inputs and net benefit chart. |
| **PDF report** | Run `python -m src.report_generator`, open `churn_report.pdf`, capture first page (title + executive summary) and one chart page (e.g. Key Metrics or Visual Insights). |
| **n8n workflow** | Import `n8n-workflow-http-with-gmail.json` in n8n, open the workflow, capture the canvas (all four nodes and connections). |
| **Email alert** | After an n8n run, open the sent email; capture subject, body (with summary numbers), and PDF attachment. |

Pre-generated charts (SHAP, monitoring, ROI) are written to `reports/figures/` by the predict, report, and monitoring steps; you can use those PNGs in docs or the README if desired.

---

## Testing & Validation

**Basic checks**  
- After `python -m src.train`: `models/churn_model.joblib` and `models/eval_metrics.json` exist; `eval_metrics.json` has `accuracy`, `f1_score`, `roc_auc`.  
- After `python -m src.predict`: `data/predictions.csv` and `data/predictions_enriched.csv` exist; both have `Churn Probability`, `Risk Level`; enriched has `Top_Drivers` (non-empty for some rows if SHAP ran).  
- After `python -m src.report_generator`: `churn_report.pdf` exists and opens; it contains executive summary, KPIs, and at least one chart.  
- After `python -m monitoring.monitor`: `monitoring/drift_report.json` and `monitoring/metrics_history.csv` updated; `reports/figures/` has monitoring PNGs if plots ran.

**Sanity tests**  
- **Train:** Run train twice; second run should overwrite model and metrics without error.  
- **Predict:** Change one feature (e.g. Contract) for one row in cleaned data and confirm probability changes in the expected direction.  
- **Salesforce:** Run `python -m src.salesforce_sync --limit 1` with valid credentials and confirm one Contact upserted and summary in `logs/salesforce_sync_summary.json`.  
- **API:** `GET http://localhost:5050/health` returns 200; after predict, `GET http://localhost:5050/run-predict` returns JSON with `success: true` and `summary`; `GET http://localhost:5050/report` returns PDF bytes.

---

## Troubleshooting

| Issue | What to do |
|-------|------------|
| **Missing columns** (e.g. `Risk Level`, `Churn Probability`) | Ensure you ran `python -m src.predict` and are reading `data/predictions.csv` or `data/predictions_enriched.csv`. If you use a custom CSV, it must include those columns or be produced by the predict pipeline. |
| **SHAP errors** (e.g. “Explainer not supported”) | The pipeline uses tree explainer for tree models and falls back to kernel explainer. If you see errors, ensure `shap` is installed (`pip install shap`) and the saved model is one of the supported types (XGBoost, LightGBM, CatBoost, or ensemble containing them). For kernel explainer, a subset of rows is used as background; reducing background size can help. |
| **Salesforce invalid_grant** | Use OAuth (Connected App): enable “Allow OAuth Username-Password Flows” in the app, set `SF_CONSUMER_KEY`, `SF_CONSUMER_SECRET`, and `SF_DOMAIN` (e.g. My Domain prefix). If using password+token, set `SF_PASSWORD_AND_TOKEN=1` and ensure password is concatenated with security token in `SF_PASSWORD` or that token is in `SF_SECURITY_TOKEN` and the code concatenates. Relax IP restrictions on the Connected App for testing. |
| **Streamlit file paths** | The app resolves paths relative to the project root (where `app.py` lives). If you run from another directory, set the working directory to the project root or set `PROJECT_ROOT` in code. On Streamlit Cloud, the app runs from the repo root; ensure `data/` and `models/` are in the repo or provided at runtime (e.g. secrets or mounted volume). |
| **Report “chart could not be generated”** | Ensure predictions CSV has the columns required by `report_charts` (e.g. `Risk Level`, `Contract`, `Tenure Bucket` for cross-tabs; model or feature names for importance). Run predict and report from the same project root so `reports/figures/` and `models/` are available. |
| **n8n “summary is undefined”** | The Gmail node references `$node["Run Churn Prediction API"].json.summary`. If the API returns an error or no summary, that path is missing. Check that `/run-predict` returns 200 and JSON with `summary` (total, high_risk, medium_risk, low_risk). |
| **PSI / drift “baseline not found”** | Run `python -m src.train` once to create `monitoring/baseline_reference.csv`. The first time you run `python -m monitoring.monitor` after that, it will compare current predictions to baseline; if baseline was just created from the same run, drift is skipped that time. |

---

## Security & Privacy

- **Secrets:** No credentials or API keys in the repository. Use `.env` locally and platform secrets (Streamlit, Render, etc.) in production.  
- **`.env`:** Listed in `.gitignore`; do not commit it.  
- **Least privilege:** Use a dedicated Salesforce user (or Connected App) with only the permissions needed for Contact and Task (read/write).  
- **Data:** Customer data (e.g. predictions CSV) may contain PII; restrict access to `data/` and logs; do not expose `.env` or prediction files via the web server.

---

## Roadmap

- **Threshold tuning:** Make risk-band thresholds (0.33, 0.66) configurable via config or UI.  
- **A/B tests:** Track which retention actions (e.g. offer type) correlate with save rate.  
- **More CRM actions:** Optional Campaign or custom object writes from high-risk list.  
- **Alerting:** Notify on drift ALERT (e.g. email or Slack from n8n when PSI ≥ 0.2).  
- **Model registry:** Version models and link Salesforce `Model_Version__c` to a registry.  
- **Incremental retrain:** Retrain on recent data or when drift exceeds a threshold.

---

## License & Acknowledgements

- **License:** See the repository’s LICENSE file (or specify “MIT”, “Apache-2.0”, etc.).  
- **Dataset:** Default dataset is Telco-style (e.g. IBM Telco Customer Churn); check the data source for terms of use.  
- **Libraries:** This project uses scikit-learn, XGBoost, LightGBM, CatBoost, Optuna, SHAP, Streamlit, Flask, ReportLab, simple-salesforce, and other open-source packages; see `requirements.txt` and respective project licenses.

---

**Summary for recruiters:**  
End-to-end churn platform: train (tuned XGB/LGB/CatBoost + stacking) → predict + SHAP → Streamlit dashboard with login → PDF report → n8n scheduling + email → Salesforce Contact/Task sync → PSI drift + trend monitoring → ROI simulator. Python, production-style structure, no placeholders; runnable in 5–10 minutes with data and `.env` configured.
