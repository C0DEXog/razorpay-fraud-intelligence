"""
Fraud Chat — manual transaction input (like the video workflow).

User enters transaction details via Streamlit widgets → features engineered →
loaded joblib model predicts Fraud / Legit with score.

Run: streamlit run app/fraud_chat.py --server.port 8503
"""
import os, sys, pickle, json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

st.set_page_config(page_title="Fraud Chat — Manual Check", layout="centered")
st.title("Fraud Chat")
st.markdown("Enter a transaction manually — the model predicts **Fraud** or **Legit** instantly.")

# Load universal model
model_path = ROOT / "models" / "universal_model.pkl"
features_path = ROOT / "models" / "features.json"

# Try joblib first (from video pipeline), fall back to pickle
try:
    import joblib
    model = joblib.load(str(model_path)) if model_path.exists() else None
    if model and isinstance(model, dict) and "models" in model:
        pass  # our universal artifact
except Exception:
    model = None

# Feature engineering import
from src.features.engineer import build_features
from src.model.universal_detector import detect_schema, engineer_features

with st.form("transaction_form"):
    st.subheader("Transaction Details")
    amount = st.number_input("Amount (Rs)", min_value=0.0, value=150.0, step=10.0)
    transaction_type = st.selectbox("Transaction Type", ["Transfer", "Cash Out", "Payment", "Debit", "Credit"])
    hour = st.slider("Hour of day (0-23)", 0, 23, 14)
    old_bal = st.number_input("Old Balance", min_value=0.0, value=5000.0)
    new_bal = st.number_input("New Balance", min_value=0.0, value=4850.0)
    device_age = st.number_input("Device Age (days)", min_value=0, value=30)
    login_attempts = st.number_input("Login Attempts", min_value=0, value=1)
    category = st.selectbox("Category", ["electronics", "digital_goods", "travel", "other"])
    merchant = st.selectbox("Merchant", ["M001", "M002", "M003", "M004"])
    submitted = st.form_submit_button("Predict Fraud / Legit")

if submitted:
    # Build a single-row DataFrame matching universal schema
    row = pd.DataFrame([{
        "TransactionID": "MANUAL_001",
        "AccountID": "MANUAL_USER",
        "TransactionAmount": amount,
        "TransactionDate": "2024-01-01 14:00:00",
        "TransactionType": transaction_type,
        "Location": "IN",
        "DeviceID": "D001",
        "IP Address": "192.168.1.1",
        "MerchantID": merchant,
        "Channel": "online",
        "CustomerAge": 35,
        "CustomerOccupation": "Engineer",
        "TransactionDuration": 60,
        "LoginAttempts": login_attempts,
        "AccountBalance": new_bal,
        "PreviousTransactionDate": "2024-01-01 13:00:00",
    }])

    schema = detect_schema(row)
    feat_df = engineer_features(row, schema)

    # Load model predictions
    try:
        from src.model.universal_detector import predict
        pred_df = predict(feat_df, out_dir=str(ROOT / "models"))
        score = float(pred_df["fraud_score"].iloc[0])
        tier = pred_df["risk_tier"].iloc[0] if "risk_tier" in pred_df.columns else "medium"
        verdict = "FRAUD" if score >= 0.5 else "Legit"

        st.subheader("Prediction")
        col1, col2, col3 = st.columns(3)
        col1.metric("Verdict", verdict)
        col2.metric("Risk Score", f"{score:.3f}")
        col3.metric("Tier", str(tier))

        if verdict == "FRAUD":
            st.error("This transaction was flagged as FRAUD — review before authorization.")
        else:
            st.success("This transaction appears LEGIT — low risk.")

        # Show feature values that drove the score (explanation)
        st.markdown("**Key drivers**")
        feature_importance = {
            "Amount z-score vs history": feat_df.get("amt_zscore_vs_history", pd.Series([0])).iloc[0],
            "Quick session flag": feat_df.get("is_quick_session", pd.Series([0])).iloc[0],
            "Velocity 1h count": feat_df.get("cnt_1h", pd.Series([0])).iloc[0],
            "Small txn burst": feat_df.get("small_txn_burst_1h", pd.Series([0])).iloc[0],
            "Multi-country": feat_df.get("multi_country", pd.Series([0])).iloc[0],
            "Auth attempts": feat_df.get("auth_attempts", pd.Series([1])).iloc[0],
        }
        for k, v in feature_importance.items():
            st.write(f"- {k}: {v}")

        # Show the feature vector (for debugging / transparency)
        with st.expander("View feature vector (48 dims)"):
            feat_cols = [c for c in feat_df.columns if c not in [
                "TransactionID", "AccountID", "TransactionDate", "TransactionType",
                "Location", "DeviceID", "IP Address", "MerchantID", "Channel",
                "CustomerAge", "CustomerOccupation", "TransactionDuration",
                "PreviousTransactionDate", "amount", "is_fraud", "fraud_score",
                "verdict", "risk_tier", "tiered_decision"
            ]]
            st.dataframe(feat_df[feat_cols].T.rename(columns={0: "value"}), use_container_width=True)

    except Exception as e:
        st.error(f"Prediction error: {e}")
        import traceback
        st.text(traceback.format_exc())

st.markdown("---")
st.info("Based on the YouTube workflow: Kaggle dataset → EDA (0.13% fraud) → Pipeline → Joblib → Streamlit form. We already have the pipeline; this page is the missing interactive form.")
