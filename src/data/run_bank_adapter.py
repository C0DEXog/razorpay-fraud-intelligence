#!/usr/bin/env python3
"""Convert bank_transactions_data_2.csv → dashboard-ready CSV.

No synthetic fraud labels are added.  The output CSV uses the internal
schema (tx_id, order_id, amount in paise, local_time, hour,
merchant_id/customer_id, category, device_age_days, etc.) but leaves
`is_fraud` empty / NaN so the model trains on real behavioral patterns only.

Usage:
    python3 src/data/run_bank_adapter.py
"""
from pathlib import Path

# Bank CSV path
BANK_CSV = Path("/Users/tanujkhanna/Downloads/bank_transactions_data_2.csv")
OUT_CSV  = Path(__file__).resolve().parents[2] / "data" / "transactions_razorpay.csv"

if __name__ == "__main__":
    from src.data.load_bank import load_bank

    print("Converting bank CSV → dashboard schema (no synthetic labels)…")
    df = load_bank(BANK_CSV)

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)

    print(f"\nSaved {len(df)} rows to:\n  {OUT_CSV}")
    print(f"Columns ({len(df.columns)}): {list(df.columns)}")
    print("\nNo 'is_fraud' column added — model will train on behavioral patterns only.")
    print("Velocity grouping: customer_id (per-customer tx count, not per-merchant).")
    print("\nRestart the dashboard to score the new data.")
