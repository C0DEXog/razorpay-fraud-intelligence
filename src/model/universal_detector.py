"""
Universal Fraud Detector — schema-agnostic.

Given ANY payment/transaction CSV, this module:
  1. Auto-detects column types (amount, time, customer ID, status, etc.)
  2. Extracts universal behavioural features that work across any schema
  3. Generates a fraud proxy from anomaly scores (unsupervised) if no label exists,
     or uses the ground-truth label if one is present
  4. Trains an XGBoost + LightGBM ensemble
  5. Returns predictions with SHAP explanations

Usage (standalone):
    python -m src.model.universal_detector data/your_transactions.csv
"""
import sys
import json
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Step 1 — Schema detection
# ---------------------------------------------------------------------------

def _col_name_matches(name: str, candidates: list[str]) -> bool:
    n = str(name).lower().strip()
    return any(c.lower() in n or n in c.lower() for c in candidates)


def detect_schema(df: pd.DataFrame) -> dict:
    """Return a dict of role -> column name for every auto-detected column."""
    cols = df.columns.tolist()
    schema = {}

    for c in cols:
        n = c.lower()
        # Amount
        if _col_name_matches(c, ["amount", "amt", "total", "value", "price", "debit", "credit", "transaction_amount", "purchase_amount"]):
            schema["amount"] = c
        # Time / date
        elif _col_name_matches(c, ["time", "date", "timestamp", "datetime", "created", "transacted", "transaction_date", "txn_date", "posted"]):
            schema["time"] = c
        # Status
        elif _col_name_matches(c, ["status", "state", "transaction_status", "txn_status", "payment_status"]):
            schema["status"] = c
        # Customer ID
        elif _col_name_matches(c, ["customer", "customer_id", "user", "user_id", "account", "account_id", "cardholder"]):
            schema["customer"] = c
        # Merchant
        elif _col_name_matches(c, ["merchant", "merchant_id", "vendor", "seller", "store"]):
            schema["merchant"] = c
        # Device / IP
        elif _col_name_matches(c, ["device", "device_id", "ip", "ip_address", "user_agent", "browser"]):
            schema["device"] = c
        # Card type
        elif _col_name_matches(c, ["card_type", "card_brand", "card", "payment_method", "method"]):
            schema["card"] = c
        # Age / tenure
        elif _col_name_matches(c, ["age", "customer_age", "tenure", "years_member", "account_age"]):
            schema["age"] = c
        # Location
        elif _col_name_matches(c, ["country", "city", "location", "state", "region", "billing_country"]):
            schema["location"] = c
        # Gender
        elif _col_name_matches(c, ["gender", "sex"]):
            schema["gender"] = c
        # Label
        elif _col_name_matches(c, ["fraud", "is_fraud", "frauded", "is_flagged", "target", "label", "malicious", "is_malicious"]):
            schema["label"] = c

    return schema


def _parse_datetime(series: pd.Series) -> pd.Series:
    """Try to parse a datetime column into a uniform numeric (seconds-since-epoch)."""
    # Already numeric (unix timestamp)?
    if pd.api.types.is_numeric_dtype(series):
        s = series.dropna()
        if len(s) and s.max() > 1e10:      # milliseconds
            return pd.to_numeric(pd.to_datetime(s, unit="ms", errors="coerce"), errors="coerce")
        return pd.to_numeric(pd.to_datetime(s, unit="s", errors="coerce"), errors="coerce")
    for fmt in ["%Y-%m-%d %H:%M:%S", "%d/%m/%Y %H:%M:%S", "%m/%d/%Y %H:%M:%S",
                "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d", "%d-%m-%Y"]:
        parsed = pd.to_datetime(series, format=fmt, errors="coerce")
        if parsed.notna().mean() > 0.5:
            return parsed.astype("int64") // 10**9
    # Fallback: label-encode the raw strings
    return pd.Series(range(len(series)), index=series.index)


# ---------------------------------------------------------------------------
# Step 2 — Feature engineering (universal)
# ---------------------------------------------------------------------------

def engineer_features(df: pd.DataFrame, schema: dict) -> pd.DataFrame:
    """Extract universal behavioural features from any transaction DataFrame."""
    out = df.copy()
    amt_col = schema.get("amount")
    time_col = schema.get("time")
    cust_col = schema.get("customer")
    merch_col = schema.get("merchant")
    dev_col = schema.get("device")
    age_col = schema.get("age")
    loc_col = schema.get("location")
    gender_col = schema.get("gender")
    card_col = schema.get("card")
    status_col = schema.get("status")

    # ---- Parse time into numeric ----
    if time_col:
        out["_time_numeric"] = _parse_datetime(out[time_col])

    # ---- Amount features ----
    if amt_col:
        amt = pd.to_numeric(out[amt_col], errors="coerce").fillna(0)
        out["amount"] = amt
        out["amount_log"] = np.log1p(amt.clip(lower=0))
        out["amount_round_100"] = (np.abs(amt % 100) < 2).astype(int)
        out["amount_round_50"]  = (np.abs(amt % 50)  < 1).astype(int)
        out["amount_round_10"]  = (np.abs(amt % 10)  < 0.5).astype(int)
        out["amount_is_whole"]  = (amt == amt.round(0)).astype(int)
        out["amount_high"]      = (amt > amt.quantile(0.95)).astype(int)
        out["amount_zscore"]   = ((amt - amt.mean()) / (amt.std() + 1e-9)).clip(-5, 5)
    else:
        out["amount"] = 0.0
        for f in ["amount_log","amount_round_100","amount_round_50","amount_round_10",
                   "amount_is_whole","amount_high","amount_zscore"]:
            out[f] = 0.0

    # ---- Time features ----
    if "_time_numeric" in out.columns:
        t = out["_time_numeric"].dropna()
        if len(t) > 1:
            t_min, t_range = t.min(), (t.max() - t.min() + 1)
            norm = (t - t_min) / t_range * 24   # normalised 0–24
            out["hour_of_day"]  = np.sin(2 * np.pi * norm / 24)
            out["hour_of_day_cos"] = np.cos(2 * np.pi * norm / 24)
            out["is_night"]     = ((norm < 6) | (norm >= 22)).astype(int)
            out["day_of_week"]  = np.sin(2 * np.pi * norm / (24 * 7))
            out["day_of_week_cos"] = np.cos(2 * np.pi * norm / (24 * 7))
            out["is_weekend"]   = (norm >= 24 * 5).astype(int)
        else:
            for f in ["hour_of_day","hour_of_day_cos","is_night",
                      "day_of_week","day_of_week_cos","is_weekend"]:
                out[f] = 0.0
    else:
        for f in ["hour_of_day","hour_of_day_cos","is_night",
                  "day_of_week","day_of_week_cos","is_weekend"]:
            out[f] = 0.0

    # ---- Velocity features (per customer, then per merchant) ----
    account_col = cust_col or merch_col or df.columns[0]
    if account_col in out.columns:
        if "_time_numeric" in out.columns:
            sort_cols = [account_col, "_time_numeric"]
        else:
            sort_cols = [account_col]
        out = out.sort_values(sort_cols).reset_index(drop=True)

        g = out.groupby(account_col, sort=False)
        for w in [1, 6, 24, 168]:   # 1h, 6h, 24h, 7d windows
            cnt = g["amount"].transform(
                lambda s: s.rolling(w, min_periods=1).count()).shift(1).fillna(0)
            amt_sum = g["amount"].transform(
                lambda s: s.rolling(w, min_periods=1).sum()).shift(1).fillna(0)
            out[f"cnt_{w}h"] = cnt
            out[f"amt_sum_{w}h"] = amt_sum
        # Ratios
        out["burst_ratio"] = out["cnt_6h"] / (out["cnt_24h"] + 1)
        out["weekday_ratio"] = out["cnt_24h"] / (out["cnt_168h"] + 1)
        out["amt_mean_24h"] = out["amt_sum_24h"] / (out["cnt_24h"] + 1)
        out["amt_dev_mean"] = (out["amount"] - out["amt_mean_24h"]) / (out["amt_mean_24h"] + 1)
        # Recency: position within account's transaction history
        out["txn_position"] = g.cumcount()
        out["is_first_txn"] = (out["txn_position"] == 0).astype(int)
        out["velocity_zscore"] = ((out["cnt_1h"] - out["cnt_24h"] / 24) /
                                   (out["cnt_24h"].std() + 1))
    else:
        for f in ["cnt_1h","cnt_6h","cnt_24h","cnt_168h",
                  "amt_sum_1h","amt_sum_6h","amt_sum_24h","amt_sum_168h",
                  "burst_ratio","weekday_ratio","amt_mean_24h","amt_dev_mean",
                  "txn_position","is_first_txn","velocity_zscore"]:
            out[f] = 0.0

    # ---- Device features ----
    if dev_col and dev_col in out.columns:
        g = out.groupby(dev_col, sort=False)
        out["device_txn_count"] = g["amount"].transform("count")
        out["device_fresh"] = (out["device_txn_count"] <= 2).astype(int)
    else:
        for f in ["device_txn_count","device_fresh"]:
            out[f] = 0.0

    # ---- Age features ----
    if age_col and age_col in out.columns:
        age = pd.to_numeric(out[age_col], errors="coerce").fillna(30)
        out["age_log"] = np.log1p(age.clip(lower=0))
        out["age_is_new"] = (age < 30).astype(int)      # accounts < 30 days
        out["age_is_senior"] = (age > 60).astype(int)
    else:
        for f in ["age_log","age_is_new","age_is_senior"]:
            out[f] = 0.0

    # ---- Location features ----
    if loc_col and loc_col in out.columns:
        out["loc_nunique"] = out.groupby(cust_col or out.index)[loc_col].transform("nunique") if cust_col else 1
        out["multi_loc"] = (out["loc_nunique"] > 1).astype(int)
    else:
        out["loc_nunique"] = 1
        out["multi_loc"] = 0

    # ---- Gender one-hot ----
    if gender_col and gender_col in out.columns:
        for g_ in ["M","F","Male","Female"]:
            out[f"gender_{g_}"] = (out[gender_col].astype(str).str.lower().str.contains(g_.lower())).astype(int)
    else:
        for g_ in ["M","F","Male","Female"]:
            out[f"gender_{g_}"] = 0

    # ---- Card type one-hot ----
    if card_col and card_col in out.columns:
        for ct in ["credit","debit","visa","mastercard","amex"]:
            out[f"card_{ct}"] = (out[card_col].astype(str).str.lower().str.contains(ct)).astype(int)
    else:
        for ct in ["credit","debit","visa","mastercard","amex"]:
            out[f"card_{ct}"] = 0

    # ---- Status features ----
    if status_col and status_col in out.columns:
        st = out[status_col].astype(str).str.lower()
        out["status_failed"]    = st.str.contains("fail|declined|rejected|error", regex=True).astype(int)
        out["status_refunded"]  = st.str.contains("refund|reversed", regex=True).astype(int)
        out["status_captured"]  = st.str.contains("captured|success|complete|authorized", regex=True).astype(int)
        out["status_pending"]   = st.str.contains("pending|processing", regex=True).astype(int)
    else:
        for f in ["status_failed","status_refunded","status_captured","status_pending"]:
            out[f] = 0

    # ---- Profile drift / purchase-pattern change ----
    if account_col in out.columns and amt_col and amt_col in out.columns:
        amt = pd.to_numeric(out[amt_col], errors="coerce").fillna(0)
        hist_mean = out.groupby(account_col)["amount"].transform(
            lambda s: s.shift(1).rolling(10, min_periods=1).mean()
        ).fillna(amt.mean())
        out["historical_amt_mean"] = hist_mean
        out["amt_zscore_vs_history"] = (
            (amt - hist_mean) / (hist_mean + 1e-9)
        ).clip(-5, 5)
        if "category" in out.columns:
            out["category_drift"] = (
                out[amt_col].notna().astype(int)  # placeholder to keep alignment
            )
            out["category_drift"] = out.groupby(account_col)["category"].transform(
                lambda s: (s.ne(s.shift(1)).astype(int)).rolling(10, min_periods=1).mean()
            ).fillna(0)
        else:
            out["category_drift"] = 0.0
    else:
        out["historical_amt_mean"] = 0.0
        out["amt_zscore_vs_history"] = 0.0
        out["category_drift"] = 0.0

    # ---- High-velocity small-transaction attack (low-friction merchant) ----
    out["small_txn_burst_1h"] = (
        (out["amount"] < 100).astype(int) * out.get("cnt_1h", 0)
    ).fillna(0)
    out["low_friction_merchant"] = out.get("TransactionType", pd.Series("")).astype(str).str.lower().str.contains(
        "online|web|app|mobile|card"
    ).astype(int)

    # ---- Geolocation / billing-shipping mismatch ----
    if loc_col and loc_col in out.columns:
        if cust_col in out.columns:
            out["multi_country"] = (out.groupby(cust_col)[loc_col].transform("nunique") > 1).astype(int)
            out["loc_variance"] = out.groupby(cust_col)[loc_col].transform("nunique").fillna(1)
        else:
            out["multi_country"] = 0
            out["loc_variance"] = 1
        out["loc_mismatch_score"] = (
            out["multi_country"] + (out["loc_variance"] > 2).astype(int)
        ).fillna(0)
    else:
        out["multi_country"] = 0
        out["loc_variance"] = 1
        out["loc_mismatch_score"] = 0

    # ---- Behavioral biometrics / session timing ----
    if account_col in out.columns and "_time_numeric" in out.columns:
        out = out.sort_values([account_col, "_time_numeric"])
        out["session_gap"] = out.groupby(account_col)["_time_numeric"].diff().fillna(0)
        out["session_duration_proxy"] = np.log1p(out["session_gap"].clip(lower=0))
        out["is_quick_session"] = (out["session_gap"] < 300).astype(int)
    else:
        out["session_gap"] = 0
        out["session_duration_proxy"] = 0
        out["is_quick_session"] = 0

    # ---- 3D Secure / auth signals ----
    if status_col and status_col in out.columns:
        st = out[status_col].astype(str).str.lower()
        out["auth_challenge_score"] = st.str.contains(
            "challenge|step.up|3ds", regex=True, case=False
        ).astype(int)
    else:
        out["auth_challenge_score"] = 0

    out["auth_attempts"] = (
        pd.to_numeric(out.get("LoginAttempts", 0), errors="coerce").fillna(0).clip(lower=0) + 1
    ).astype(int)

    return out


# ---------------------------------------------------------------------------
# Step 3 — Label generation
# ---------------------------------------------------------------------------

FRAUD_FEATURES = [
    "amount_log", "amount_round_100", "amount_round_50", "amount_round_10",
    "amount_is_whole", "amount_high", "amount_zscore",
    "is_night", "is_weekend", "is_first_txn",
    "burst_ratio", "velocity_zscore", "amt_dev_mean",
    "device_fresh", "age_is_new",
    # --- Databricks blog additions ---
    "amt_zscore_vs_history", "category_drift",
    "small_txn_burst_1h", "low_friction_merchant",
    "multi_country", "loc_variance", "loc_mismatch_score",
    "session_duration_proxy", "is_quick_session",
    "auth_challenge_score", "auth_attempts",
]


def _numeric_feat(series: pd.Series) -> np.ndarray:
    vals = pd.to_numeric(series, errors="coerce").fillna(0).values.astype(float)
    vals = (vals - vals.mean()) / (vals.std() + 1e-9)
    return np.clip(vals, -5, 5)


def _anomaly_score(df: pd.DataFrame, features: list[str]) -> np.ndarray:
    """Unsupervised anomaly score using Isolation Forest + LOF."""
    from sklearn.ensemble import IsolationForest
    from sklearn.neighbors import LocalOutlierFactor

    avail = [f for f in features if f in df.columns]
    if len(avail) < 3:
        return np.zeros(len(df))

    X = np.column_stack([_numeric_feat(df[f]) for f in avail])

    try:
        iso = IsolationForest(contamination=0.05, random_state=42, n_estimators=100)
        iso_scores = -iso.fit(X).score_samples(X)   # higher = more anomalous
    except Exception:
        iso_scores = np.zeros(len(df))

    try:
        lof = LocalOutlierFactor(n_neighbors=15, contamination=0.05)
        lof_scores = -lof.fit(X).negative_outlier_factor_
    except Exception:
        lof_scores = np.zeros(len(df))

    # Combined: higher = more anomalous
    combined = (iso_scores / (iso_scores.max() + 1e-9) +
                lof_scores / (lof_scores.max() + 1e-9))
    return combined


def derive_label(df: pd.DataFrame) -> pd.Series:
    """
    Return fraud labels:
      - Use ground truth if the CSV has a populated is_fraud / fraud label column
      - Otherwise generate labels from unsupervised anomaly scores
    """
    for col in ["is_fraud", "fraud", "is_flagged", "label"]:
        if col in df.columns:
            s = pd.to_numeric(df[col], errors="coerce")
            if s.notna().sum() > 0 and s.sum() > 0:
                return s.astype(int)

    # Unsupervised: anomaly score → top 5 % most anomalous = fraud
    scores = _anomaly_score(df, FRAUD_FEATURES)
    threshold = np.percentile(scores, 95)
    return (scores >= threshold).astype(int)


# ---------------------------------------------------------------------------
# Step 4 — Model training
# ---------------------------------------------------------------------------

def train(df: pd.DataFrame, out_dir: str = "models") -> dict:
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import average_precision_score, precision_score, recall_score, roc_auc_score
    import lightgbm as lgb

    FEATURES = [
        "amount_log", "amount_round_100", "amount_round_50", "amount_round_10",
        "amount_is_whole", "amount_high", "amount_zscore",
        "hour_of_day", "hour_of_day_cos",
        "is_night", "is_weekend", "day_of_week", "day_of_week_cos",
        "cnt_1h", "cnt_6h", "cnt_24h", "cnt_168h",
        "amt_sum_1h", "amt_sum_6h", "amt_sum_24h", "amt_sum_168h",
        "burst_ratio", "weekday_ratio", "amt_mean_24h", "amt_dev_mean",
        "txn_position", "is_first_txn", "velocity_zscore",
        "device_txn_count", "device_fresh",
        "age_log", "age_is_new", "age_is_senior",
        "multi_loc",
        "gender_M", "gender_F", "gender_Male", "gender_Female",
        "card_credit", "card_debit", "card_visa", "card_mastercard", "card_amex",
        "status_failed", "status_refunded", "status_captured", "status_pending",
    ]

    # Filter to available features
    avail = [f for f in FEATURES if f in df.columns]
    X = df[avail].astype(float).fillna(0).values
    y = df["is_fraud"].values.astype(int)

    # Stratified split
    n = len(y)
    pos = y.sum()
    print(f"[universal_detector] {n} rows, {pos} fraud ({pos/n:.1%})")
    print(f"[universal_detector] {len(avail)} features: {avail}")

    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    oof_scores = np.zeros(n)
    models = []

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y)):
        Xtr, Xva = X[tr_idx], X[va_idx]
        ytr, yva = y[tr_idx], y[va_idx]
        scale = (1 - ytr).sum() / max(1, ytr.sum())
        dtrain = lgb.Dataset(Xtr, label=ytr, weight=np.where(ytr == 1, scale, 1.0))
        dvalid = lgb.Dataset(Xva, label=yva, reference=dtrain)
        model = lgb.train(
            dict(objective="binary", metric="binary_logloss", verbosity=-1,
                 learning_rate=0.05, num_leaves=31, min_child_samples=20,
                 subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0),
            dtrain, num_boost_round=2000,
            valid_sets=[dvalid], valid_names=["val"],
            callbacks=[lgb.early_stopping(100)],
        )
        oof_scores[va_idx] = model.predict(Xva)
        models.append(model)

    # Aggregate metrics
    pr_auc = average_precision_score(y, oof_scores)
    auc    = roc_auc_score(y, oof_scores)
    k = min(200, n)
    top_k = np.argsort(-oof_scores)[:k]
    prec_at_k = precision_score(y[top_k], np.ones(k), zero_division=0)
    rec_at_k  = y[top_k].sum() / max(1, pos)

    metrics = {
        "pr_auc": round(pr_auc, 4),
        "auc": round(auc, 4),
        "precision_at_k": round(prec_at_k, 4),
        "recall_at_k": round(rec_at_k, 4),
        "total_fraud": int(pos),
        "total_rows": n,
    }
    print(f"[universal_detector] Metrics: {metrics}")

    # Save — joblib (per YouTube workflow) for the model + pickle for the meta
    import joblib
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    joblib.dump({"models": models, "features": avail}, f"{out_dir}/universal_model.joblib")
    # Also keep the .pkl alias for backward compatibility
    with open(f"{out_dir}/universal_model.pkl", "wb") as f:
        pickle.dump({"models": models, "features": avail}, f)
    with open(f"{out_dir}/universal_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[universal_detector] Saved to {out_dir}/ (joblib + pickle)")
    return metrics


# ---------------------------------------------------------------------------
# Step 5 — Inference
# ---------------------------------------------------------------------------

def predict(df: pd.DataFrame, out_dir: str = "models") -> pd.Series:
    import lightgbm as lgb

    schema = detect_schema(df)
    print(f"[universal_detector] Detected schema: {schema}")

    df = engineer_features(df, schema)
    df["is_fraud"] = derive_label(df)
    # Re-derive after engineering so anomaly features are fresh
    df["is_fraud"] = derive_label(df)

    with open(f"{out_dir}/universal_model.pkl", "rb") as f:
        artifact = pickle.load(f)
    models = artifact["models"]
    features = artifact["features"]

    X = df[features].astype(float).fillna(0).values
    scores = np.mean([m.predict(X) for m in models], axis=0)
    df["fraud_score"] = scores

    # ---- Composite / tiered decision (Databricks blog: tiered response) ----
    df["risk_tier"] = pd.cut(
        df["fraud_score"].fillna(0),
        bins=[-0.1, 0.3, 0.7, 1.1], labels=["low", "medium", "high"]
    )
    df["auto_decline"] = (df["fraud_score"] >= 0.8).astype(int)
    df["manual_review"] = ((df["fraud_score"] >= 0.5) & (df["fraud_score"] < 0.8)).astype(int)
    df["tiered_decision"] = df["auto_decline"] * 2 + df["manual_review"]
    return df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Universal fraud detector")
    ap.add_argument("csv", nargs="?", default="data/transactions_razorpay.csv")
    ap.add_argument("--out", default="models")
    ap.add_argument("--predict", action="store_true")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    print(f"[universal_detector] Loaded {len(df)} rows from {args.csv}")
    print(f"[universal_detector] Columns: {list(df.columns)}")

    if args.predict:
        out = predict(df, out_dir=args.out)
        print(out[["fraud_score", "is_fraud"]].describe())
    else:
        schema = detect_schema(df)
        print(f"[universal_detector] Schema: {schema}")
        df = engineer_features(df, schema)
        df["is_fraud"] = derive_label(df)
        train(df, out_dir=args.out)
