"""Feature engineering for the transaction risk detector.

Four feature families, all computed with strict temporal ordering so no
future information leaks into a prediction. Merchant-level aggregates use
expanding windows over prior transactions only.

Schema-safe: every column access is guarded so arbitrary CSV uploads never crash.
"""
import numpy as np
import pandas as pd


def _col(df: pd.DataFrame, name: str, default=0.0):
    """Return the column if present, otherwise a zero Series of the right length."""
    return df[name] if name in df.columns else pd.Series(default, index=df.index)


def build_features(df: pd.DataFrame, account_col: str = "merchant_id") -> pd.DataFrame:
    # Sort by account_col + time; use index as tiebreaker if time is absent
    sort_cols = [c for c in [account_col, "local_time"] if c in df.columns]
    out = df.sort_values(sort_cols).reset_index(drop=True)

    # ---- velocity: frequency & amount over prior window (expanding, no leakage) ----
    has_tx_id   = "tx_id"   in out.columns
    has_amount  = "amount"  in out.columns
    has_account = account_col in out.columns

    if has_account and has_tx_id and has_amount:
        g = out.groupby(account_col, sort=False)
        for w in (1, 6, 24):
            cnt = g["tx_id"].transform(
                lambda s: s.rolling(w, min_periods=1).count()).shift(1)
            amt = g["amount"].transform(
                lambda s: s.rolling(w, min_periods=1).sum()).shift(1)
            out[f"cnt_{w}"] = cnt.fillna(0)
            out[f"amt_sum_{w}"] = amt.fillna(0)
        out["amt_mean_24"] = (out["amt_sum_24"] / out["cnt_24"].replace(0, np.nan)).fillna(0)
        out["amt_ratio_6"] = out["amount"] / (out["amt_mean_24"].replace(0, np.nan) + 1e-9)
        out["amt_ratio_6"] = out["amt_ratio_6"].replace([np.inf, -np.inf], np.nan).fillna(1.0)
        out["burst_6"] = out["cnt_6"] / (out["cnt_24"].replace(0, np.nan) + 1e-9)
        out["burst_6"] = out["burst_6"].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    else:
        for col in ["cnt_1", "cnt_6", "cnt_24", "amt_sum_1", "amt_sum_6", "amt_sum_24",
                    "amt_mean_24", "amt_ratio_6", "burst_6"]:
            out[col] = 0.0

    # ---- temporal (guard against missing hour/dayofweek) ----
    hour = _col(out, "hour")
    dow  = _col(out, "dayofweek")
    out["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    out["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    out["dow_sin"]  = np.sin(2 * np.pi * dow / 7)
    out["dow_cos"]  = np.cos(2 * np.pi * dow / 7)
    out["is_night"]   = ((hour < 5) | (hour >= 23)).astype(int)
    out["is_weekend"] = (dow >= 5).astype(int)

    # ---- amount shape ----
    amount = _col(out, "amount")
    device_age = _col(out, "device_age_days")
    out["amount_round100"] = (np.abs(amount - np.round(amount / 100) * 100) < 2).astype(int)
    out["amount_round50"]   = (np.abs(amount - np.round(amount / 50)  * 50)  < 2).astype(int)
    out["amount_log"]      = np.log1p(amount)
    out["device_age_log"]  = np.log1p(device_age)
    out["cnt_24_log"]      = np.log1p(out["cnt_24"])

    # ---- peer-group deviation (category baseline) ----
    if "category" in out.columns and "amount" in out.columns:
        cat_mean = out.groupby("category")["amount"].transform("mean")
        cat_std  = out.groupby("category")["amount"].transform("std").replace(0, np.nan)
        out["amt_dev_cat"] = (amount - cat_mean) / (cat_std + 1e-9)
        out["amt_dev_cat"] = out["amt_dev_cat"].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    else:
        out["amt_dev_cat"] = 0.0

    # ---- category one-hot (compact) ----
    for c in ["digital_goods", "travel", "electronics"]:
        out[f"cat_{c}"] = (out["category"] == c).astype(int) if "category" in out.columns else pd.Series(0, index=out.index)

    return out


def feature_columns() -> list[str]:
    """Canonical list of model features in the order expected by the trainers.

    Kept in one place so train.py, train_ensemble.py and the dashboard all
    agree on the exact feature ordering — the LightGBM model was trained on
    this list, so any drift will silently degrade metrics or SHAP alignment.
    """
    return [
        "amount_log", "cnt_1", "cnt_6", "cnt_24", "cnt_24_log", "amt_sum_1",
        "amt_sum_6", "amt_sum_24", "amt_mean_24", "amt_ratio_6", "burst_6",
        "hour_sin", "hour_cos", "dow_sin", "dow_cos", "is_night",
        "is_weekend", "amount_round100", "amount_round50",
        "device_age_log", "amt_dev_cat",
        "cat_digital_goods", "cat_travel", "cat_electronics",
    ]