"""Razorpay Transaction Risk Dashboard."""
import os, sys, subprocess, pickle, json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import shap
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# .env loading (safe — values never printed or returned)
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
_DOTENV = ROOT / ".env"
if _DOTENV.exists():
    for _line in _DOTENV.read_text().splitlines():
        _line = _line.strip()
        if not _line or _line.startswith("#") or "=" not in _line:
            continue
        _k, _, _v = _line.partition("=")
        os.environ.setdefault(_k.strip(), _v.strip())
sys.path.insert(0, str(ROOT))

# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Razorpay Risk Dashboard", layout="wide")
st.title("Razorpay Risk Dashboard")

FEATURES = [
    "amount_log", "cnt_1", "cnt_6", "cnt_24", "cnt_24_log", "amt_sum_1",
    "amt_sum_6", "amt_sum_24", "amt_mean_24", "amt_ratio_6", "burst_6",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos", "is_night",
    "is_weekend", "amount_round100", "amount_round50",
    "device_age_log", "amt_dev_cat",
    "cat_digital_goods", "cat_travel", "cat_electronics",
]

@st.cache_data(show_spinner=False)
def load_data():
    """Load orders + payments, build features, derive label.

    Auto-detects schema to support both Razorpay (account_col=merchant_id)
    and bank-transaction CSVs (account_col=customer_id) written by
    src/data/load_bank.py.  The column that exists in the loaded CSV is used
    as the velocity grouping key so that bank velocity windows count per
    customer, not per merchant.
    """
    from src.features.engineer import build_features
    from src.model.train import derive_label

    orders = pd.read_csv(ROOT / "data" / "transactions_razorpay.csv")

    # ---- auto-detect velocity grouping key ----
    # Razorpay orders group by merchant_id; bank CSVs (from load_bank.py)
    # group by customer_id.  Prefer merchant_id so the dashboard matches
    # the schema the trained model expects.
    if "merchant_id" in orders.columns:
        account_col = "merchant_id"   # Razorpay orders CSV
    elif "customer_id" in orders.columns:
        account_col = "customer_id"   # bank-transaction CSV from load_bank.py
    else:
        account_col = orders.columns[0]   # fallback: group by first column

    orders = build_features(orders, account_col=account_col)
    orders["is_fraud"] = derive_label(orders)

    payments_path = ROOT / "data" / "payments_razorpay.csv"
    if payments_path.exists():
        payments = pd.read_csv(payments_path)
    else:
        payments = pd.DataFrame(
            columns=["tx_id", "order_id", "amount", "status", "created_at", "local_time"]
        )
    return orders, payments


@st.cache_resource(show_spinner=False)
def load_models():
    """Load LightGBM + RF models and their metrics."""
    with open(ROOT / "models_razorpay" / "model.pkl", "rb") as f:
        lgbm = pickle.load(f)
    with open(ROOT / "models_razorpay" / "metrics.json") as f:
        lgbm_metrics = json.load(f)
    shap_vals = np.load(ROOT / "models_razorpay" / "shap_values.npy")

    rf, rf_metrics = None, None
    rf_path = ROOT / "models_razorpay" / "rf_model.pkl"
    if rf_path.exists():
        with open(rf_path, "rb") as f:
            rf = pickle.load(f)
        with open(ROOT / "models_razorpay" / "rf_metrics.json") as f:
            rf_metrics = json.load(f)
    return lgbm, lgbm_metrics, shap_vals, rf, rf_metrics


df, payments_df = load_data()
lgbm_model, metrics, shap_vals, rf_model, rf_metrics = load_models()

X = df[FEATURES]
scores = lgbm_model.predict(X)
df["risk_score"] = scores
if rf_model is not None:
    df["rf_risk_score"] = rf_model.predict_proba(X)[:, 1]
    df["ensemble_risk"] = 0.5 * df["risk_score"] + 0.5 * df["rf_risk_score"]

# SHAP base value
explainer = shap.TreeExplainer(lgbm_model)
_ev = explainer.expected_value
if hasattr(_ev, "ndim") and _ev.ndim > 0:
    BASE = float(np.array(_ev).item() if _ev.size == 1 else np.array(_ev).ravel()[0])
else:
    BASE = float(_ev)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.caption(f"Model: LightGBM" + (" + RandomForest" if rf_model is not None else ""))
st.sidebar.caption(f"{len(df)} orders loaded")
st.sidebar.caption(f"{len(payments_df)} payments loaded")

# ---------------------------------------------------------------------------
# LIVE REFRESH BUTTON
# ---------------------------------------------------------------------------
st.sidebar.markdown("---")
st.sidebar.subheader("Live Data")

if st.sidebar.button("🔄 Fetch Live from Razorpay API", type="primary", use_container_width=True):
    with st.spinner("Connecting to Razorpay API (fetching orders + payments)..."):
        try:
            from src.data.fetch_razorpay import fetch_orders, fetch_payments
            from src.features.engineer import build_features, feature_columns
            from src.model.train import train
            from src.model.train_ensemble import train_ensemble
            from src.model.generate_shap import main as gen_shap

            df_new = fetch_orders(300)
            fetch_payments(100)
            df_feats = build_features(df_new, account_col="merchant_id")
            train(df_feats, feature_columns(), k=100, out_dir=str(ROOT / "models_razorpay"))
            train_ensemble(data_path=str(DATA_DIR / "transactions_razorpay.csv"), out_dir=str(ROOT / "models_razorpay"))
            gen_shap(data="data/transactions_razorpay.csv", out_dir="models_razorpay")

            st.cache_data.clear()
            st.cache_resource.clear()
            st.sidebar.success(f"Successfully fetched {len(df_new)} orders & retrained model!")
            st.rerun()
        except Exception as e:
            st.sidebar.error(f"Live API Fetch failed: {e}")

if st.sidebar.button("🧠 Run Universal Fraud Detector", use_container_width=True):
    with st.spinner("Analyzing any CSV with universal model..."):
        from src.model.universal_detector import predict, detect_schema, engineer_features
        try:
            # Always reload fresh from data/ (in case upload changed it)
            orders = pd.read_csv(DATA_DIR / "transactions_razorpay.csv")
            orders = engineer_features(orders, detect_schema(orders))
            orders = predict(orders, out_dir=str(ROOT / "models"))
            # Save predictions so the dashboard can reference them
            orders.to_csv(DATA_DIR / "transactions_razorpay.csv", index=False)
            st.sidebar.success(f"Universal model: PR-AUC = {json.load(open(str(ROOT/'models'/'universal_metrics.json'))).get('pr_auc', 'N/A')}")
            st.sidebar.info(f"Detected fraud rate: {orders['is_fraud'].mean():.1%}")
            # Force reload on next interaction
            st.cache_data.clear()
            st.cache_resource.clear()
            st.session_state["orders_uploaded"] = True
            st.rerun()
        except Exception as e:
            st.sidebar.error(f"Universal detector failed: {e}")

# ---------------------------------------------------------------------------
# MANUAL CSV UPLOAD (add Razorpay data file from folder)
# ---------------------------------------------------------------------------
st.sidebar.markdown("---")
st.sidebar.subheader("Upload CSV")
orders_file = st.sidebar.file_uploader(
    "Orders CSV (transactions_razorpay.csv)",
    type=["csv"],
    help="Upload a Razorpay orders CSV to replace the current dataset",
)
payments_file = st.sidebar.file_uploader(
    "Payments CSV (payments_razorpay.csv)",
    type=["csv"],
    help="Upload a Razorpay payments CSV",
)
if orders_file is not None:
    try:
        uploaded_df = pd.read_csv(orders_file)
        uploaded_df.to_csv(DATA_DIR / "transactions_razorpay.csv", index=False)
        st.session_state["orders_uploaded"] = True
        st.sidebar.success("Orders CSV saved to data/ — click Reload to apply")
    except Exception as e:
        st.sidebar.error(f"Failed to save orders CSV: {e}")
if payments_file is not None:
    try:
        uploaded_payments = pd.read_csv(payments_file)
        uploaded_payments.to_csv(DATA_DIR / "payments_razorpay.csv", index=False)
        st.session_state["payments_uploaded"] = True
        st.sidebar.success("Payments CSV saved to data/ — click Reload to apply")
    except Exception as e:
        st.sidebar.error(f"Failed to save payments CSV: {e}")

# ---------------------------------------------------------------------------
# RELOAD BUTTON — flush caches after upload so the new CSV is read in
# ---------------------------------------------------------------------------
uploaded_any = st.session_state.get("orders_uploaded") or st.session_state.get("payments_uploaded")
if uploaded_any:
    st.sidebar.markdown("**New CSV detected. Reload to see the new data.**")
    if st.sidebar.button("Reload data", type="primary", key="reload_uploaded"):
        st.cache_data.clear()
        st.cache_resource.clear()
        st.session_state["orders_uploaded"] = False
        st.session_state["payments_uploaded"] = False
        st.rerun()

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab_overview, tab_ensemble, tab_merchant, tab_payments = st.tabs(
    ["Overview", "Ensemble", "Merchant Risk", "Payments"]
)

# ============================ OVERVIEW TAB ==============================
with tab_overview:
    # ---- Metrics ----
    st.subheader("Performance Metrics")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("PR-AUC", f"{metrics['pr_auc']}")
    c2.metric("Precision@K", f"{metrics['precision_at_k']}")
    c3.metric("Recall@K", f"{metrics['recall_at_k']}")
    c4.metric("Precision@5%", f"{metrics['precision_at_k_5pct']}")
    c5.metric("FP Rate @K", f"{metrics['fp_rate_at_k']}")

    # ---- Risk score distribution ----
    st.subheader("Risk Score Distribution")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(scores[df.is_fraud == 0], bins=40, alpha=0.7, label="Legit", color="steelblue")
    axes[0].hist(scores[df.is_fraud == 1], bins=40, alpha=0.7, label="Fraud", color="crimson")
    axes[0].set_xlabel("Risk Score"); axes[0].set_ylabel("Count"); axes[0].legend()
    scores_sorted = np.sort(scores)[::-1]
    axes[1].plot(range(len(scores_sorted)), scores_sorted, color="darkorange")
    axes[1].set_xlabel("Transaction rank"); axes[1].set_ylabel("Risk score")
    axes[1].set_title("Sorted risk scores")
    st.pyplot(fig)

    # ---- THRESHOLD SLIDER (in addition to K) ----
    st.subheader("Threshold-Based Escalation")
    th = st.slider("Risk-score threshold", 0.0, 1.0, 0.5, 0.01)
    above = df[df["risk_score"] >= th]
    bc1, bc2, bc3 = st.columns(3)
    bc1.metric("Escalated (n)", len(above))
    bc2.metric("% of portfolio", f"{len(above)/max(1,len(df)):.1%}")
    bc3.metric("Precision @ threshold",
               f"{above.is_fraud.mean():.1%}" if len(above) else "—")

    # ---- Top-K escalation ----
    K = st.slider("Review capacity (K)", 5, 100, 30)
    topk = df.nlargest(K, "risk_score")
    # Columns to display in Top-K table — only include those that exist
    display_cols = [c for c in ["tx_id", "order_id", "category", "amount", "hour",
                                 "is_fraud", "risk_score"] if c in topk.columns]
    st.subheader(f"Top {K} Transactions (by risk score)")
    st.dataframe(
        topk[display_cols].assign(
            flagged=lambda d: d["is_fraud"].map({0: "Legit", 1: "Fraud"}) if "is_fraud" in d.columns else "—"),
        use_container_width=True,
    )
    st.caption(f"Precision: {topk.is_fraud.mean():.1%} ({int(topk.is_fraud.sum())} fraud / {K} reviewed)")

    # ---- FEATURE IMPORTANCE BAR CHART ----
    st.subheader("Feature Importance (LightGBM, top 12)")
    try:
        booster = lgbm_model.booster_ if hasattr(lgbm_model, "booster_") else lgbm_model
        importances = booster.feature_importance(importance_type="gain")
    except Exception:
        importances = lgbm_model.feature_importances_
    fi = pd.DataFrame({"feature": FEATURES, "importance": importances})
    fi = fi.sort_values("importance", ascending=False).head(12)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(fi["feature"][::-1], fi["importance"][::-1], color="teal")
    ax.set_xlabel("Gain importance")
    ax.set_title("Top 12 features driving LightGBM risk score")
    st.pyplot(fig)

    # ---- TIME-SERIES RISK TREND ----
    st.subheader("Portfolio Risk Over Time")
    bin_unit = st.selectbox("Bin unit", ["minutes", "hours"], index=0)
    if "local_time" in df.columns and len(df) > 0:
        ts = pd.to_numeric(df["local_time"], errors="coerce")
        if bin_unit == "hours":
            bins = (ts / 3600.0).round().astype(int)
        else:
            bins = (ts / 60.0).round().astype(int)
        grouped = df.groupby(bins).agg(
            mean_risk=("risk_score", "mean"),
            n=("risk_score", "size"),
            fraud_rate=("is_fraud", "mean"),
        )
        fig, ax = plt.subplots(figsize=(11, 4))
        ax.plot(grouped.index, grouped["mean_risk"], color="crimson", marker="o", label="Mean risk")
        ax.set_xlabel(f"Time bin ({bin_unit} since first tx)")
        ax.set_ylabel("Mean risk score")
        ax.set_title("Portfolio risk over time")
        ax.grid(alpha=0.3)
        ax.legend()
        st.pyplot(fig)
        with st.expander("View raw bins"):
            st.dataframe(grouped, use_container_width=True)
    else:
        st.info("`local_time` missing — cannot build time-series.")

    # ---- SHAP audit trail ----
    st.subheader("SHAP Audit Trail")
    idx = st.selectbox("Select transaction", list(range(len(df))),
                       format_func=lambda i: f"TX {i} — {df.get('category', pd.Series(['—'])).iloc[i] if 'category' in df.columns else i} — score={scores[i]:.3f}")
    row = df.iloc[idx]
    shap_row = shap_vals[idx]
    exp = shap.Explanation(
        base_values=BASE,
        values=shap_row, data=row[FEATURES].values, feature_names=FEATURES,
    )
    fig, ax = plt.subplots(figsize=(10, 5))
    shap.plots.bar(exp, max_display=12, ax=ax)
    st.pyplot(fig)
    label = ("Fraud" if ("is_fraud" in row.index and row.is_fraud == 1)
             else ("Legit" if "is_fraud" in row.index else "—"))
    amount_str = f"Rs {row.get('amount', 0)/100:.2f}" if "amount" in row.index else "—"
    cat_str = row.get("category", "—") if "category" in row.index else "—"
    st.markdown(
        f"**TX {idx}** | Score: {scores[idx]:.4f} | "
        f"{label} | {amount_str} | {cat_str}"
    )

    # ---- SHAP waterfall (top case) ----
    st.subheader("SHAP Waterfall — Highest Risk")
    top_idx = int(df.nlargest(1, "risk_score").index[0])
    top_exp = shap.Explanation(
        values=shap_vals[top_idx],
        base_values=BASE,
        data=X.iloc[top_idx].values, feature_names=FEATURES,
    )
    fig, ax = plt.subplots(figsize=(8, 5))
    shap.plots.waterfall(top_exp, max_display=12, show=False)
    st.pyplot(plt.gcf())
    plt.clf()

    # ---- Escalation audit log ----
    st.subheader("Escalation Log")
    audit_cols = ["tx_id", "order_id", "category", "amount"]
    audit = []
    for rank, (_, row) in enumerate(df.nlargest(K, "risk_score").iterrows(), 1):
        top_factors = sorted(zip(FEATURES, shap_vals[row.name]), key=lambda x: -abs(x[1]))[:3]
        audit.append({
            "rank": rank,
            "tx_id": row.get("tx_id", f"TX{row.name}"),
            "order_id": str(row.get("order_id", "—"))[:16],
            "category": row.get("category", "—"),
            "amount": f"Rs {row.get('amount', 0)/100:.2f}" if "amount" in row.index else "—",
            "risk_score": round(float(row.get("risk_score", 0)), 4),
            "actual": "Fraud" if ("is_fraud" in row.index and row.is_fraud == 1) else ("Legit" if "is_fraud" in row.index else "—"),
            "top_factors": " | ".join(f"{f}={v:+.3f}" for f, v in top_factors),
        })
    audit_df = pd.DataFrame(audit)
    st.dataframe(audit_df, use_container_width=True)

# ============================ ENSEMBLE TAB ==============================
with tab_ensemble:
    st.subheader("Ensemble: LightGBM + RandomForest")
    if rf_model is None:
        st.warning("RandomForest model not found. Run `python -m src.model.train_ensemble`.")
    else:
        cmp = pd.DataFrame({
            "model": ["LightGBM", "RandomForest"],
            "PR-AUC": [metrics["pr_auc"], rf_metrics["pr_auc"]],
            "Precision@K": [metrics["precision_at_k"], rf_metrics["precision_at_k"]],
            "Recall@K": [metrics["recall_at_k"], rf_metrics["recall_at_k"]],
            "Precision@5%": [metrics["precision_at_k_5pct"], rf_metrics["precision_at_k_5pct"]],
            "FP@K": [metrics["fp_rate_at_k"], rf_metrics["fp_rate_at_k"]],
        })
        st.dataframe(cmp, use_container_width=True)

        st.markdown("**Prediction agreement**")
        st.metric("Pearson r (LGBM vs RF)",
                  f"{np.corrcoef(df['risk_score'], df['rf_risk_score'])[0,1]:.3f}")

        # Side-by-side histogram
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.hist(df["risk_score"], bins=40, alpha=0.55, label="LightGBM", color="steelblue")
        ax.hist(df["rf_risk_score"], bins=40, alpha=0.55, label="RandomForest", color="darkorange")
        ax.set_xlabel("Risk score"); ax.set_ylabel("Count")
        ax.set_title("Model score distributions")
        ax.legend()
        st.pyplot(fig)

        st.markdown("**Ensemble score (0.5·LGBM + 0.5·RF) — top 20**")
        ensemble_cols = ["tx_id", "order_id", "category", "amount", "risk_score",
                         "rf_risk_score", "ensemble_risk", "is_fraud"]
        available_ens = [c for c in ensemble_cols if c in df.columns]
        st.dataframe(
            df.nlargest(20, "ensemble_risk")[available_ens],
            use_container_width=True,
        )

# =========================== MERCHANT RISK TAB ===========================
with tab_merchant:
    st.subheader("Risk by Merchant")
    if "merchant_id" in df.columns:
        msum = df.groupby("merchant_id").agg(
            n=("tx_id", "size"),
            mean_risk=("risk_score", "mean"),
            max_risk=("risk_score", "max"),
            fraud_count=("is_fraud", "sum"),
            mean_amount=("amount", "mean"),
        ).sort_values("mean_risk", ascending=False)
        st.dataframe(msum, use_container_width=True)

        fig, ax = plt.subplots(figsize=(9, 4))
        ax.bar(msum.index.astype(str), msum["mean_risk"], color="indianred")
        ax.set_xlabel("merchant_id")
        ax.set_ylabel("Mean risk score")
        ax.set_title("Mean risk score by merchant (sorted desc)")
        st.pyplot(fig)
    else:
        st.info("`merchant_id` missing.")

# =========================== PAYMENTS TAB ===============================
with tab_payments:
    st.subheader("Razorpay Payments (live API)")
    if payments_df.empty:
        st.info("No payments file yet. Click **Refresh from API** in the sidebar.")
    else:
        st.dataframe(payments_df, use_container_width=True)
        if "status" in payments_df.columns and "amount" in payments_df.columns:
            cap = payments_df.loc[payments_df['status']=='captured','amount'].sum()
            fail = payments_df.loc[payments_df['status'].isin(['failed','refunded']),'amount'].sum()
            st.metric("Total captured amount (Rs)", f"{cap:.2f}")
            st.metric("Total failed/refunded (Rs)", f"{fail:.2f}")
        else:
            st.info("Payments CSV missing required columns: status, amount.")
        fig, ax = plt.subplots(figsize=(7, 3))
        vc = payments_df["status"].value_counts()
        ax.bar(vc.index.astype(str), vc.values, color="slateblue")
        ax.set_title("Payment status counts")
        st.pyplot(fig)

        # Merge with orders on order_id for richer signal
        st.markdown("**Merged with orders (by order_id)**")
        pay_cols = [c for c in ["order_id", "status", "amount"] if c in payments_df.columns]
        if "order_id" in payments_df.columns and "amount" in payments_df.columns and "status" in payments_df.columns:
            merged = df.merge(
                payments_df[pay_cols].rename(columns={"amount": "paid_amount", "status": "pay_status"}),
                on="order_id", how="left", suffixes=("", "_pay"),
            )
            merge_cols = ["tx_id", "order_id", "category", "amount", "paid_amount",
                          "pay_status", "risk_score", "is_fraud"]
            merge_show = [c for c in merge_cols if c in merged.columns]
            st.dataframe(merged.nlargest(20, "risk_score")[merge_show], use_container_width=True)
        else:
            st.info("Payments file missing required columns: order_id, amount, status.")
