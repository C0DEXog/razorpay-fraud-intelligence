"""Load real bank transactions CSV, adapt to Razorpay schema, save.

Maps the bank CSV columns to the schema the dashboard expects:
  TransactionID    -> order_id
  AccountID        -> customer_id
  TransactionAmount-> amount (rupees, NOT paise)
  TransactionDate  -> created_at (epoch) + local_time
  MerchantID       -> merchant_id
  DeviceID         -> device_id
  Channel          -> category
  LoginAttempts    -> (preserved, used for synthetic fraud label)
  CustomerAge      -> (preserved)
"""
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "transactions_razorpay.csv"

CHANNEL_TO_CATEGORY = {
    "Online": "digital_goods",
    "ATM":    "electronics",
    "Branch": "travel",
}


def adapt_bank_csv(src: str = None) -> pd.DataFrame:
    """Read /Users/tanujkhanna/Downloads/bank_transactions_data_2.csv
    and return a DataFrame in the Razorpay schema."""
    if src is None:
        src = "/Users/tanujkhanna/Downloads/bank_transactions_data_2.csv"
    df = pd.read_csv(src)
    out = pd.DataFrame({
        "tx_id":   np.arange(len(df)),
        "order_id":  df["TransactionID"].values,
        "customer_id": df["AccountID"].values,
        "merchant_id": df["MerchantID"].values,
        "amount":   df["TransactionAmount"].astype(float),  # keep as rupees
        "category": df["Channel"].map(CHANNEL_TO_CATEGORY).fillna("other").values,
        "status":   df["TransactionType"].values,  # Debit/Credit
        "device_id": df["DeviceID"].values,
        "device_age_days": 30.0,  # bank has no device-age field
        "LoginAttempts": df["LoginAttempts"].astype(int).values,
        "CustomerAge": df["CustomerAge"].astype(int).values,
        "AccountBalance": df["AccountBalance"].astype(float).values,
        "TransactionDuration": df["TransactionDuration"].astype(int).values,
    })
    # Parse timestamp
    ts = pd.to_datetime(df["TransactionDate"], errors="coerce")
    out["created_at"] = ts.astype("int64") // 10**9  # epoch seconds
    out["hour"]      = ts.dt.hour.fillna(12).astype(int)
    out["dayofweek"] = ts.dt.dayofweek.fillna(0).astype(int)
    # local_time: seconds since first transaction
    out["local_time"] = out["created_at"] - out["created_at"].min()
    return out


def synthetic_label(df: pd.DataFrame) -> pd.Series:
    """Behavioural fingerprint for fraud proxy:
    - High login attempts (suspicious)
    - Very high amount (vs balance)
    - Very low balance (drained account)
    - Short transaction duration (automated)
    """
    amt_to_bal = df["amount"] / (df["AccountBalance"] + 1.0)
    flags = (
        (df["LoginAttempts"] >= 3).astype(int) +
        (amt_to_bal > 0.8).astype(int) +
        (df["AccountBalance"] < 100).astype(int) +
        (df["TransactionDuration"] < 30).astype(int)
    )
    return (flags >= 2).astype(int)


if __name__ == "__main__":
    out = adapt_bank_csv()
    out["is_fraud"] = synthetic_label(out)
    OUT.parent.mkdir(exist_ok=True)
    out.to_csv(OUT, index=False)
    print(f"Loaded {len(out)} bank rows → {OUT}")
    print(f"  Fraud proxy: {out['is_fraud'].sum()} flagged "
          f"({out['is_fraud'].mean():.1%})")
    print(f"  Categories: {out['category'].value_counts().to_dict()}")
    print(f"  Amount range: Rs {out['amount'].min():.0f} – Rs {out['amount'].max():.0f}")
