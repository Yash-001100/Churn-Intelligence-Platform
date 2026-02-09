"""
Push churn risk outputs to Salesforce: upsert Contact by CustomerID__c, update churn fields,
create retention Tasks for high-risk customers. Run after predictions (e.g. from n8n).
Uses env vars for credentials: SF_USERNAME, SF_PASSWORD, SF_SECURITY_TOKEN, SF_DOMAIN.
"""
import os
import sys
import json
from datetime import datetime, timedelta, timezone

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
sys.path.insert(0, _PROJECT_ROOT)

try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(_PROJECT_ROOT, ".env"))
except ImportError:
    pass

import pandas as pd

from src.config import (
    PROJECT_ROOT,
    PREDICTIONS_ENRICHED_PATH,
    PREDICTIONS_PATH,
    MODEL_META_PATH,
)

# Env vars (no secrets in code; loaded from .env if python-dotenv is installed)
SF_USERNAME = os.environ.get("SF_USERNAME")
SF_PASSWORD = os.environ.get("SF_PASSWORD")
SF_SECURITY_TOKEN = os.environ.get("SF_SECURITY_TOKEN")
SF_CONSUMER_KEY = os.environ.get("SF_CONSUMER_KEY")  # Connected App Consumer Key (use when SOAP is disabled)
SF_CONSUMER_SECRET = os.environ.get("SF_CONSUMER_SECRET")
SF_DOMAIN = os.environ.get("SF_DOMAIN", "login")  # "login" or "test"; for My Domain use e.g. "mycompany.my"
# Set to 1 or true to use Client Credentials flow (no username/password); requires "Enable Client Credentials Flow" on the app
SF_USE_CLIENT_CREDENTIALS = os.environ.get("SF_USE_CLIENT_CREDENTIALS", "0").strip().lower() in ("1", "true", "yes")
# Set to 0 or false to send only password (no token) for OAuth password flow
SF_PASSWORD_AND_TOKEN = os.environ.get("SF_PASSWORD_AND_TOKEN", "1").strip().lower() not in ("0", "false", "no")
CUSTOMER_ID_COLUMN = os.environ.get("CUSTOMER_ID_COLUMN", "CustomerID")
# Salesforce Contact external ID field API name (e.g. CustomerID__c or Customer_ID__c)
SF_CONTACT_EXTERNAL_ID_FIELD = os.environ.get("SF_CONTACT_EXTERNAL_ID_FIELD", "CustomerID__c")
PREDICTIONS_FILE = os.environ.get("PREDICTIONS_FILE", "")
# Process only first N rows (for testing); empty or 0 = no limit
SF_SYNC_LIMIT = os.environ.get("SF_SYNC_LIMIT", "")
# Sync only High-risk rows (saves org storage); set to 1 or true to enable
SF_SYNC_HIGH_RISK_ONLY = os.environ.get("SF_SYNC_HIGH_RISK_ONLY", "0").strip().lower() in ("1", "true", "yes")

LOGS_DIR = os.path.join(PROJECT_ROOT, "logs")
SUMMARY_PATH = os.path.join(LOGS_DIR, "salesforce_sync_summary.json")


def _load_predictions():
    """Load predictions CSV (enriched preferred)."""
    path = PREDICTIONS_FILE.strip() or (
        PREDICTIONS_ENRICHED_PATH if os.path.exists(PREDICTIONS_ENRICHED_PATH) else PREDICTIONS_PATH
    )
    if not os.path.exists(path):
        return None, "Predictions file not found: {}".format(path)
    try:
        df = pd.read_csv(path)
        if "Risk Level" not in df.columns or "Churn Probability" not in df.columns:
            return None, "Predictions file missing Risk Level or Churn Probability"
        return df, None
    except Exception as e:
        return None, str(e)


def _get_model_version():
    """Read model version from model_meta.json if present."""
    if not os.path.exists(MODEL_META_PATH):
        return ""
    try:
        with open(MODEL_META_PATH) as f:
            meta = json.load(f)
        return meta.get("version", "") or meta.get("model_name", "")
    except Exception:
        return ""


def _sf_connection():
    """Create Salesforce connection from env vars. Use Connected App (OAuth) when SOAP is disabled."""
    try:
        from simple_salesforce import Salesforce
    except ImportError:
        raise ImportError("Install simple-salesforce: pip install simple-salesforce")

    # Option 1a: Client Credentials flow (no username/password) — use when app has "Enable Client Credentials Flow"
    if SF_USE_CLIENT_CREDENTIALS and SF_CONSUMER_KEY and SF_CONSUMER_SECRET and SF_DOMAIN and SF_DOMAIN not in ("login", "test"):
        return Salesforce(
            consumer_key=SF_CONSUMER_KEY,
            consumer_secret=SF_CONSUMER_SECRET,
            domain=SF_DOMAIN,
        )

    # Option 1b: Connected App + username/password (OAuth password grant)
    if SF_CONSUMER_KEY and SF_CONSUMER_SECRET and SF_USERNAME and SF_PASSWORD:
        password_to_send = (SF_PASSWORD + (SF_SECURITY_TOKEN or "")) if SF_PASSWORD_AND_TOKEN else SF_PASSWORD
        try:
            return Salesforce(
                username=SF_USERNAME,
                password=password_to_send,
                consumer_key=SF_CONSUMER_KEY,
                consumer_secret=SF_CONSUMER_SECRET,
                domain=SF_DOMAIN,
            )
        except Exception as e:
            if "invalid_grant" in str(e).lower() or "authentication failure" in str(e).lower():
                raise ValueError(
                    "OAuth invalid_grant: Check 1) Connected App has 'Allow OAuth Username-Password Flows' enabled "
                    "(Setup → App Manager → your app → Edit Policies). "
                    "2) Password: use password+security token in .env, or set SF_PASSWORD_AND_TOKEN=0 to try password only. "
                    "3) If your org uses My Domain, set SF_DOMAIN to your prefix (e.g. mycompany.my). "
                    "4) Relax 'IP Relaxation' on the Connected App for testing. Original error: {}".format(e)
                ) from e
            raise

    # Option 2: Username + password + security token (SOAP login)
    if not all([SF_USERNAME, SF_PASSWORD, SF_SECURITY_TOKEN]):
        raise ValueError(
            "Set SF_USERNAME, SF_PASSWORD, SF_SECURITY_TOKEN in .env, OR set SF_CONSUMER_KEY and "
            "SF_CONSUMER_SECRET (Connected App) when SOAP API login is disabled. See README."
        )
    return Salesforce(
        username=SF_USERNAME,
        password=SF_PASSWORD,
        security_token=SF_SECURITY_TOKEN,
        domain=SF_DOMAIN,
    )


def _utc_now():
    """Current UTC time (timezone-aware)."""
    return datetime.now(timezone.utc)


def _scored_at_iso():
    """Current time in ISO format for Scored_At__c."""
    return _utc_now().strftime("%Y-%m-%dT%H:%M:%SZ")


def _activity_date_today_plus_days(days=2):
    """ActivityDate for Task: today + days (YYYY-MM-DD)."""
    return (_utc_now() + timedelta(days=days)).strftime("%Y-%m-%d")


def run_sync(limit=None):
    """
    Load predictions, connect to Salesforce, upsert Contact per row, create Task for High risk.
    Writes logs/salesforce_sync_summary.json. If a record is not found, log and continue.
    limit: process only first N rows (for testing); None = use SF_SYNC_LIMIT env or no limit.
    """
    summary = {
        "run_at": _utc_now().isoformat().replace("+00:00", "Z"),
        "predictions_path": "",
        "customer_id_column": CUSTOMER_ID_COLUMN,
        "row_limit": None,
        "contacts_upserted": 0,
        "contacts_failed": 0,
        "tasks_created": 0,
        "tasks_failed": 0,
        "errors": [],
    }
    df, err = _load_predictions()
    if err:
        summary["errors"].append(err)
        os.makedirs(LOGS_DIR, exist_ok=True)
        with open(SUMMARY_PATH, "w") as f:
            json.dump(summary, f, indent=2)
        print("Salesforce sync failed: {}".format(err))
        return 1
    path = PREDICTIONS_FILE.strip() or (
        PREDICTIONS_ENRICHED_PATH if os.path.exists(PREDICTIONS_ENRICHED_PATH) else PREDICTIONS_PATH
    )
    summary["predictions_path"] = path

    if limit is None and SF_SYNC_LIMIT and str(SF_SYNC_LIMIT).strip().isdigit():
        limit = int(SF_SYNC_LIMIT.strip())
    if SF_SYNC_HIGH_RISK_ONLY and "Risk Level" in df.columns:
        n_before = len(df)
        df = df[df["Risk Level"] == "High"].copy()
        summary["high_risk_only"] = True
        summary["rows_before_filter"] = n_before
        summary["rows_after_filter"] = len(df)
        print("Syncing only High-risk rows: {} of {} total.".format(len(df), n_before))
    else:
        summary["high_risk_only"] = False
    if limit is not None and limit > 0:
        df = df.head(limit)
        summary["row_limit"] = limit
        print("Processing first {} rows only (limit set for testing).".format(limit))

    id_col = CUSTOMER_ID_COLUMN
    if id_col not in df.columns:
        summary["errors"].append(
            "Column '{}' not in predictions. Add it to your pipeline or set CUSTOMER_ID_COLUMN.".format(id_col)
        )
        os.makedirs(LOGS_DIR, exist_ok=True)
        with open(SUMMARY_PATH, "w") as f:
            json.dump(summary, f, indent=2)
        print(summary["errors"][-1])
        return 1

    try:
        sf = _sf_connection()
    except (ValueError, ImportError) as e:
        summary["errors"].append(str(e))
        os.makedirs(LOGS_DIR, exist_ok=True)
        with open(SUMMARY_PATH, "w") as f:
            json.dump(summary, f, indent=2)
        print("Salesforce connection failed: {}".format(e))
        return 1

    model_version = _get_model_version()
    scored_at = _scored_at_iso()
    activity_date = _activity_date_today_plus_days(2)

    for idx, row in df.iterrows():
        ext_id = row.get(id_col)
        if pd.isna(ext_id) or ext_id is None or str(ext_id).strip() == "":
            summary["contacts_failed"] += 1
            summary["errors"].append("Row {}: missing {}".format(idx, id_col))
            continue
        ext_id_str = str(ext_id).strip()
        prob = row.get("Churn Probability")
        prob_val = float(prob) if prob is not None and not pd.isna(prob) else None
        risk = row.get("Risk Level")
        risk_str = str(risk).strip() if risk is not None and not pd.isna(risk) else ""
        top_drivers = row.get("Top_Drivers")
        top_drivers_str = (str(top_drivers).strip() if top_drivers is not None and not pd.isna(top_drivers) else "")[:32768]

        # Do not include the external ID field in the body — it's already in the upsert URL
        contact_data = {
            "LastName": "Customer {}".format(ext_id_str)[:80],
            "Churn_Probability__c": prob_val,
            "Risk_Level__c": risk_str,
            "Scored_At__c": scored_at,
            "Top_Drivers__c": top_drivers_str[:32768] if top_drivers_str else None,
            "Model_Version__c": model_version[:255] if model_version else None,
        }
        try:
            sf.Contact.upsert("{}/{}".format(SF_CONTACT_EXTERNAL_ID_FIELD, ext_id_str), contact_data)
            summary["contacts_upserted"] += 1
        except Exception as e:
            summary["contacts_failed"] += 1
            summary["errors"].append("Row {} ({}): {}".format(idx, ext_id_str, str(e)))
            continue

        contact_id = None
        try:
            soql_value = ext_id_str.replace("\\", "\\\\").replace("'", "''")
            q = sf.query("SELECT Id FROM Contact WHERE {} = '{}'".format(SF_CONTACT_EXTERNAL_ID_FIELD, soql_value))
            if q.get("totalSize", 0) > 0:
                contact_id = q["records"][0]["Id"]
        except Exception:
            pass

        if risk_str == "High" and contact_id:
            desc = "Churn probability: {:.1%}. Top drivers: {}".format(
                prob_val or 0,
                top_drivers_str[:500] if top_drivers_str else "N/A",
            )
            task_data = {
                "WhoId": contact_id,
                "Subject": "Retention outreach — High churn risk",
                "ActivityDate": activity_date,
                "Description": desc[:32000],
            }
            try:
                sf.Task.create(task_data)
                summary["tasks_created"] += 1
            except Exception as e:
                summary["tasks_failed"] += 1
                summary["errors"].append("Task for {}: {}".format(ext_id_str, str(e)))

    os.makedirs(LOGS_DIR, exist_ok=True)
    with open(SUMMARY_PATH, "w") as f:
        json.dump(summary, f, indent=2)
    print(
        "Salesforce sync done: {} contacts upserted, {} failed; {} tasks created, {} failed. Summary: {}".format(
            summary["contacts_upserted"],
            summary["contacts_failed"],
            summary["tasks_created"],
            summary["tasks_failed"],
            SUMMARY_PATH,
        )
    )
    return 0 if summary["contacts_failed"] == 0 and summary["tasks_failed"] == 0 else 0


def test_connection():
    """Try to connect to Salesforce only; print success or error. Use: python -m src.salesforce_sync --test"""
    try:
        sf = _sf_connection()
        # Quick query to confirm session
        sf.query("SELECT Id FROM Contact LIMIT 1")
        print("Salesforce connection OK. Instance: {}.".format(getattr(sf, "sf_instance", "?")))
        return 0
    except Exception as e:
        print("Salesforce connection failed: {}".format(e))
        return 1


if __name__ == "__main__":
    args = [a.strip() for a in sys.argv[1:]]
    if args and args[0].lower() in ("--test", "-t", "test"):
        sys.exit(test_connection())
    limit = None
    if "--limit" in args:
        try:
            i = args.index("--limit")
            limit = int(args[i + 1])
        except (IndexError, ValueError):
            pass
    sys.exit(run_sync(limit=limit))
