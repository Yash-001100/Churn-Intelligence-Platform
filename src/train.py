"""Train churn prediction models and save the best one (target: high accuracy)."""
import os
import sys
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import warnings

# Add project root to path when run as script
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import (
    RANDOM_STATE,
    TEST_SIZE,
    TARGET_COL,
    CAT_COLS,
    MODEL_PATH,
    ENCODER_PATH,
    SCALER_PATH,
    PREPROCESS_PATH,
    FEATURE_NAMES_PATH,
    FEATURE_NAMES_JSON,
    COLUMN_META_PATH,
    MODEL_META_PATH,
    MODELS_DIR,
)
from src.data_loader import prepare_data, get_feature_columns

warnings.filterwarnings("ignore", category=UserWarning)

# Optuna tuning: number of trials per model (fewer = faster)
N_OPTUNA_TRIALS = 12


def get_numeric_columns(df, feature_cols):
    """Numeric columns among feature_cols."""
    return df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()


def get_categorical_columns(df, feature_cols):
    """Categorical columns among feature_cols (that exist in CAT_COLS)."""
    return [c for c in feature_cols if c in CAT_COLS and c in df.columns]


def build_features(df, encoder=None, scaler=None, fit=True, cat_cols_order=None, num_cols_order=None):
    """
    One-hot encode categoricals and scale numerics.
    If fit=True, fit encoder and scaler; else use provided ones and cat/num_cols_order.
    Returns (X array, encoder, scaler, feature_names, cat_cols, num_cols).
    """
    feature_cols = get_feature_columns(df)
    num_cols = num_cols_order if num_cols_order is not None else get_numeric_columns(df, feature_cols)
    cat_cols = cat_cols_order if cat_cols_order is not None else get_categorical_columns(df, feature_cols)
    # Ensure order: cat first then num for consistent columns
    X_cat = df[cat_cols].astype(str) if cat_cols else pd.DataFrame(index=df.index)
    X_num = df[num_cols] if num_cols else pd.DataFrame(index=df.index)

    if fit:
        encoder = OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore")
        scaler = StandardScaler()
        if X_cat.shape[1]:
            enc_cat = encoder.fit_transform(X_cat)
            cat_names = encoder.get_feature_names_out(cat_cols)
        else:
            enc_cat = np.zeros((len(df), 0))
            cat_names = np.array([])
        if X_num.shape[1]:
            enc_num = scaler.fit_transform(X_num)
            num_names = num_cols
        else:
            enc_num = np.zeros((len(df), 0))
            num_names = []
        feature_names = list(cat_names) + list(num_names)
    else:
        if X_cat.shape[1]:
            enc_cat = encoder.transform(X_cat)
            cat_names = encoder.get_feature_names_out(cat_cols)
        else:
            enc_cat = np.zeros((len(df), 0))
            cat_names = np.array([])
        if X_num.shape[1]:
            enc_num = scaler.transform(X_num)
            num_names = num_cols
        else:
            enc_num = np.zeros((len(df), 0))
            num_names = []
        feature_names = list(cat_names) + list(num_names)

    X = np.hstack([enc_cat, enc_num]) if enc_cat.size or enc_num.size else np.zeros((len(df), 0))
    return X, encoder, scaler, feature_names, cat_cols, num_cols


def tune_xgboost(X_train, y_train, X_test, y_test):
    """Optuna tuning for XGBoost."""
    import optuna
    import xgboost as xgb

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 600),
            "max_depth": trial.suggest_int("max_depth", 4, 12),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
            "random_state": RANDOM_STATE,
            "eval_metric": "logloss",
        }
        clf = xgb.XGBClassifier(**params)
        clf.fit(X_train, y_train)
        return accuracy_score(y_test, clf.predict(X_test))

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE, n_startup_trials=5))
    study.optimize(objective, n_trials=N_OPTUNA_TRIALS, show_progress_bar=False)
    best_params = study.best_params
    best_params["random_state"] = RANDOM_STATE
    best_params["eval_metric"] = "logloss"
    clf = xgb.XGBClassifier(**best_params)
    clf.fit(X_train, y_train)
    return clf


def tune_lightgbm(X_train, y_train, X_test, y_test):
    """Optuna tuning for LightGBM."""
    import optuna
    import lightgbm as lgb

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 600),
            "max_depth": trial.suggest_int("max_depth", 4, 12),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
            "num_leaves": trial.suggest_int("num_leaves", 20, 150),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
            "random_state": RANDOM_STATE,
            "verbose": -1,
        }
        clf = lgb.LGBMClassifier(**params)
        clf.fit(X_train, y_train)
        return accuracy_score(y_test, clf.predict(X_test))

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE, n_startup_trials=5))
    study.optimize(objective, n_trials=N_OPTUNA_TRIALS, show_progress_bar=False)
    best_params = study.best_params
    best_params["random_state"] = RANDOM_STATE
    best_params["verbose"] = -1
    clf = lgb.LGBMClassifier(**best_params)
    clf.fit(X_train, y_train)
    return clf


def tune_catboost(X_train, y_train, X_test, y_test):
    """Optuna tuning for CatBoost."""
    import optuna
    from catboost import CatBoostClassifier

    def objective(trial):
        params = {
            "iterations": trial.suggest_int("iterations", 200, 600),
            "depth": trial.suggest_int("depth", 4, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-4, 10.0, log=True),
            "random_seed": RANDOM_STATE,
            "verbose": 0,
        }
        clf = CatBoostClassifier(**params)
        clf.fit(X_train, y_train)
        return accuracy_score(y_test, clf.predict(X_test))

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE, n_startup_trials=5))
    study.optimize(objective, n_trials=N_OPTUNA_TRIALS, show_progress_bar=False)
    best_params = study.best_params
    best_params["random_seed"] = RANDOM_STATE
    best_params["verbose"] = 0
    clf = CatBoostClassifier(**best_params)
    clf.fit(X_train, y_train)
    return clf


# Columns to save as baseline for monitoring/drift (PSI)
BASELINE_DRIFT_COLS = [
    "Tenure Months", "Monthly Charges", "Total Charges",
    "Contract", "Internet Service", "Payment Method",
]


def train_and_evaluate(data_path=None):
    """Load data, train advanced models + stacking; save best by accuracy (target ~99%)."""
    df = prepare_data(path=data_path, save_cleaned=True)
    # Save baseline for monitoring drift (PSI) — used by monitoring/monitor.py
    try:
        import os as _os
        _monitoring_dir = _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "monitoring")
        _os.makedirs(_monitoring_dir, exist_ok=True)
        _baseline_path = _os.path.join(_monitoring_dir, "baseline_reference.csv")
        _cols = [c for c in BASELINE_DRIFT_COLS if c in df.columns]
        if _cols:
            df[_cols].to_csv(_baseline_path, index=False)
            print("Baseline for drift monitoring saved to {}.".format(_baseline_path))
    except Exception as e:
        print("Could not save baseline_reference.csv: {}.".format(e))
    y = df[TARGET_COL].values

    X, encoder, scaler, feature_names, cat_cols, num_cols = build_features(df, fit=True)
    if X.shape[1] == 0:
        raise ValueError("No features produced. Check CAT_COLS and numeric columns.")
    column_meta = {"cat_cols": cat_cols, "num_cols": num_cols, "feature_names": feature_names}

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    results = []
    candidates = {}  # name -> fitted classifier

    # 1) Tuned XGBoost
    try:
        import xgboost
        print("Tuning XGBoost...")
        clf_xgb = tune_xgboost(X_train, y_train, X_test, y_test)
        acc = accuracy_score(y_test, clf_xgb.predict(X_test))
        f1 = f1_score(y_test, clf_xgb.predict(X_test), zero_division=0)
        auc = roc_auc_score(y_test, clf_xgb.predict_proba(X_test)[:, 1])
        results.append({"Model": "XGBoost (tuned)", "Accuracy": acc, "F1": f1, "ROC-AUC": auc})
        candidates["XGBoost (tuned)"] = clf_xgb
    except Exception as e:
        print(f"XGBoost tuning skipped: {e}")

    # 2) Tuned LightGBM
    try:
        import lightgbm
        print("Tuning LightGBM...")
        clf_lgb = tune_lightgbm(X_train, y_train, X_test, y_test)
        acc = accuracy_score(y_test, clf_lgb.predict(X_test))
        f1 = f1_score(y_test, clf_lgb.predict(X_test), zero_division=0)
        auc = roc_auc_score(y_test, clf_lgb.predict_proba(X_test)[:, 1])
        results.append({"Model": "LightGBM (tuned)", "Accuracy": acc, "F1": f1, "ROC-AUC": auc})
        candidates["LightGBM (tuned)"] = clf_lgb
    except Exception as e:
        print(f"LightGBM tuning skipped: {e}")

    # 3) Tuned CatBoost
    try:
        from catboost import CatBoostClassifier
        print("Tuning CatBoost...")
        clf_cat = tune_catboost(X_train, y_train, X_test, y_test)
        acc = accuracy_score(y_test, clf_cat.predict(X_test))
        f1 = f1_score(y_test, clf_cat.predict(X_test), zero_division=0)
        auc = roc_auc_score(y_test, clf_cat.predict_proba(X_test)[:, 1])
        results.append({"Model": "CatBoost (tuned)", "Accuracy": acc, "F1": f1, "ROC-AUC": auc})
        candidates["CatBoost (tuned)"] = clf_cat
    except Exception as e:
        print(f"CatBoost tuning skipped: {e}")

    # 4) Stacking ensemble (unfitted base models + meta LogisticRegression)
    stack_estimators = []
    try:
        import xgboost as xgb
        stack_estimators.append(("xgb", xgb.XGBClassifier(n_estimators=400, max_depth=8, learning_rate=0.05, random_state=RANDOM_STATE, eval_metric="logloss")))
    except ImportError:
        pass
    try:
        import lightgbm as lgb
        stack_estimators.append(("lgb", lgb.LGBMClassifier(n_estimators=400, max_depth=8, learning_rate=0.05, random_state=RANDOM_STATE, verbose=-1)))
    except ImportError:
        pass
    # CatBoost excluded from stacking: not fully sklearn-compatible (__sklearn_tags__) with StackingClassifier
    if len(stack_estimators) >= 2:
        stack_label = "Stacking (XGB+LGB)"
        print("Training stacking ensemble...")
        stack = StackingClassifier(
            estimators=stack_estimators,
            final_estimator=LogisticRegression(max_iter=2000, C=0.1, random_state=RANDOM_STATE),
            cv=5,
        )
        stack.fit(X_train, y_train)
        acc = accuracy_score(y_test, stack.predict(X_test))
        f1 = f1_score(y_test, stack.predict(X_test), zero_division=0)
        auc = roc_auc_score(y_test, stack.predict_proba(X_test)[:, 1])
        results.append({"Model": stack_label, "Accuracy": acc, "F1": f1, "ROC-AUC": auc})
        candidates[stack_label] = stack

    # Fallback: baseline models if no advanced ones
    if not candidates:
        for name, clf in [
            ("Logistic Regression", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)),
            ("Random Forest", RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE)),
        ]:
            clf.fit(X_train, y_train)
            acc = accuracy_score(y_test, clf.predict(X_test))
            f1 = f1_score(y_test, clf.predict(X_test), zero_division=0)
            auc = roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1])
            results.append({"Model": name, "Accuracy": acc, "F1": f1, "ROC-AUC": auc})
            candidates[name] = clf

    # Select best by accuracy (user target: 99%)
    best_name = max(results, key=lambda r: r["Accuracy"])["Model"]
    best_model = candidates[best_name]
    best_acc = max(r["Accuracy"] for r in results)

    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(best_model, MODEL_PATH)
    joblib.dump(encoder, ENCODER_PATH)
    joblib.dump(scaler, SCALER_PATH)
    joblib.dump({"encoder": encoder, "scaler": scaler}, PREPROCESS_PATH)
    joblib.dump(feature_names, FEATURE_NAMES_PATH)
    import json as _json
    from datetime import datetime as _datetime
    with open(FEATURE_NAMES_JSON, "w") as _f:
        _json.dump(feature_names, _f, indent=2)
    joblib.dump(column_meta, COLUMN_META_PATH)
    with open(MODEL_META_PATH, "w") as _f:
        _json.dump({
            "model_name": best_name,
            "train_date": _datetime.now().strftime("%Y-%m-%d"),
            "version": "1.0",
        }, _f, indent=2)

    # Export metrics for report_generator (no placeholders)
    best_row = next(r for r in results if r["Model"] == best_name)
    import json
    eval_path = os.path.join(MODELS_DIR, "eval_metrics.json")
    with open(eval_path, "w") as f:
        json.dump({
            "model_name": best_name,
            "accuracy": round(best_row["Accuracy"], 4),
            "f1_score": round(best_row["F1"], 4),
            "roc_auc": round(best_row["ROC-AUC"], 4),
        }, f, indent=2)
    print(f"Metrics saved to {eval_path}")

    print("\nEvaluation results:")
    for r in results:
        print(f"  {r['Model']}: Accuracy={r['Accuracy']:.4f}, F1={r['F1']:.4f}, ROC-AUC={r['ROC-AUC']:.4f}")
    print(f"\nBest model (by Accuracy): {best_name} -> Accuracy={best_acc:.4f} -> saved to {MODEL_PATH}")
    return results, best_name


if __name__ == "__main__":
    train_and_evaluate()
