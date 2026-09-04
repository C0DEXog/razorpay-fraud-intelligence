"""
Universal Fraud Detector — standalone Streamlit page.

Upload any transaction CSV → auto-detect columns → predict fraud on each row.

Run with:
    streamlit run app/detect.py --server.port 8502
"""
import os, sys, pickle, json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

st.set_page_config(page_title="Universal Fraud Detector", layout="wide")
st.title("Universal Fraud Detector")
st.markdown("""
**Upload any transaction CSV** (Kaggle, Razorpay, Stripe, bank statement, etc.) and
this model will auto-detect the schema, train a LightGBM ensemble, and tell you
which transactions are fraud vs. legitimate.

It works whether your file already has a `fraud`/`is_fraud` column or not —
without labels, it uses anomaly detection to derive them.
""")

uploaded = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded is not None:
    # Persist upload to a known location and reset
    safe_name = "uploaded_transactions.csv"
    save_path = ROOT / "data" / safe_name
    df = pd.read_csv(uploaded)
    df.to_csv(save_path, index=False)
    st.success(f"Loaded {len(df):,} rows × {df.shape[1]} columns from `{uploaded.name}`")
    st.write("**Columns detected:**", list(df.columns))

    if st.button("Run fraud detection", type="primary"):
        with st.spinner("Training model and detecting fraud..."):
            from src.model.universal_detector import (
                detect_schema, engineer_features, derive_label, train
            )

            schema = detect_schema(df)
            st.write("**Auto-detected schema:**", schema)

            df_feat = engineer_features(df, schema)
            df_feat["is_fraud"] = derive_label(df_feat)
            n_fraud = int(df_feat["is_fraud"].sum())
            st.info(f"Found {n_fraud} suspicious transactions "
                    f"({n_fraud / max(1, len(df)):.1%} of {len(df):,})")

            # Train on this dataset
            metrics = train(df_feat, out_dir=str(ROOT / "models"))
            st.success(
                f"Model trained. PR-AUC: {metrics['pr_auc']}, "
                f"AUC: {metrics['auc']}, Recall@K: {metrics['recall_at_k']}"
            )

            # Predict on full data
            from src.model.universal_detector import predict
            df_pred = predict(df_feat, out_dir=str(ROOT / "models"))
            df_pred = df_pred.sort_values("fraud_score", ascending=False).reset_index(drop=True)
            df_pred["verdict"] = df_pred["fraud_score"].apply(
                lambda s: "FRAUD" if s >= 0.5 else "Legit"
            )

            # Save predictions
            df_pred.to_csv(ROOT / "data" / "predictions.csv", index=False)

            # Show results
            st.subheader("Top 50 highest-risk transactions")
            display_cols = [c for c in [
                "verdict", "fraud_score", "risk_tier", "tiered_decision",
                "TransactionID", "tx_id", "AccountID", "customer",
                "TransactionAmount", "amount", "TransactionDate", "time",
                "MerchantID", "merchant", "Location", "device",
                "TransactionType", "is_fraud",
                # new Databricks features
                "amt_zscore_vs_history", "multi_country", "loc_mismatch_score",
                "session_duration_proxy", "auth_challenge_score", "auth_attempts",
                "small_txn_burst_1h", "low_friction_merchant",
            ] if c in df_pred.columns]
            st.dataframe(df_pred[display_cols].head(50), use_container_width=True)

            # Tiered decision summary
            st.subheader("Tiered Decision Summary")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total transactions", f"{len(df_pred):,}")
            n_fraud_pred = int((df_pred["verdict"] == "FRAUD").sum())
            n_review = int(df_pred["manual_review"].sum()) if "manual_review" in df_pred.columns else 0
            n_decline = int(df_pred["auto_decline"].sum()) if "auto_decline" in df_pred.columns else 0
            c2.metric("Auto-Decline (score >= 0.8)", n_decline)
            c3.metric("Manual Review (0.5–0.8)", n_review)
            c4.metric("Auto-Approve (< 0.5)", len(df_pred) - n_decline - n_review)
            st.caption(
                "Auto-Decline: high-confidence fraud → block immediately. "
                "Manual Review: borderline → send to analyst queue. "
                "Auto-Approve: low risk → no friction."
            )

            # Geo/session/auth breakdown
            if "multi_country" in df_pred.columns:
                st.subheader("Geolocation / Billing-Shipping Mismatch")
                g1, g2, g3 = st.columns(3)
                g1.metric("Multi-country activity",
                          int(df_pred["multi_country"].sum()) if df_pred["multi_country"].dtype != object else 0)
                g2.metric("High loc mismatch",
                          int((df_pred.get("loc_mismatch_score", 0) > 1).sum()))
                g3.metric("3D Secure challenges triggered",
                          int(df_pred["auth_challenge_score"].sum()) if "auth_challenge_score" in df_pred.columns else 0)

            if "session_duration_proxy" in df_pred.columns:
                st.subheader("Session & Behavioural Signals")
                s1, s2, s3 = st.columns(3)
                s1.metric("Quick sessions (<5 min gap)",
                          int(df_pred["is_quick_session"].sum()) if "is_quick_session" in df_pred.columns else 0)
                s2.metric("High login attempts",
                          int((df_pred.get("auth_attempts", pd.Series(0)) > 2).sum()))
                s3.metric("Small-txn burst (1h)",
                          int(df_pred["small_txn_burst_1h"].sum()) if "small_txn_burst_1h" in df_pred.columns else 0)

            # Summary metrics
            st.subheader("Model Performance")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total transactions", f"{len(df_pred):,}")
            c2.metric("Predicted fraud", n_fraud_pred)
            c3.metric("Predicted legit", int((df_pred["verdict"] == "Legit").sum()))
            c4.metric("Model PR-AUC", f"{metrics['pr_auc']}")

            # Distribution chart
            st.subheader("Risk score distribution")
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.hist(df_pred["fraud_score"], bins=50, color="steelblue", edgecolor="white")
            ax.axvline(0.5, color="crimson", linestyle="--", label="Fraud threshold")
            ax.set_xlabel("Fraud risk score (0=legit, 1=fraud)")
            ax.set_ylabel("Count")
            ax.legend()
            st.pyplot(fig)

            st.download_button(
                "Download all predictions as CSV",
                df_pred.to_csv(index=False).encode("utf-8"),
                "fraud_predictions.csv",
                "text/csv",
            )

st.markdown("---")
st.markdown("**Try a sample:**")
if st.button("Run on demo data (Razorpay test transactions)"):
    demo = ROOT / "data" / "transactions_razorpay.csv"
    if demo.exists():
        df = pd.read_csv(demo)
        st.info(f"Loaded {len(df):,} demo rows. Click 'Run fraud detection' above.")
    else:
        st.error("Demo data not found.")
