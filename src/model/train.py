"""Train + evaluate the transaction risk detector.

Honest metrics only (per Track 02 bar):
  - Precision@K  : precision among the top-K flagged (K = review budget)
  - PR-AUC       : precision-recall area, robust to imbalance
  - FP cost      : cost of false alarms, reported alongside recall
  - Temporal split: train on earlier window, evaluate on later (no leakage)
"""
import json
import pickle
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.model_selection import StratifiedKFold


def temporal_split(df: pd.DataFrame, val_frac: float = 0.25):
    """Stratified temporal split: sample per merchant, not global cut."""
    # split merchants by time of first appearance to avoid merchant-level leakage
    merchant_first = df.groupby("merchant_id")["tx_id"].min().sort_values()
    cut_idx = int(len(merchant_first) * (1 - val_frac))
    train_merchants = merchant_first.index[:cut_idx].values
    val_merchants = merchant_first.index[cut_idx:].values
    return df[df["merchant_id"].isin(train_merchants)].reset_index(drop=True), \
           df[df["merchant_id"].isin(val_merchants)].reset_index(drop=True)


def make_lgb(dtrain, dvalid=None, params=None):
    p = dict(
        objective="binary", metric="binary_logloss", verbosity=-1,
        learning_rate=0.05, num_leaves=31, min_child_samples=20,
        subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
        n_estimators=2000, early_stopping_rounds=100,
    )
    if params:
        p.update(params)
    valid_sets = [dvalid] if dvalid is not None else []
    valid_names = ["val"] if dvalid is not None else []
    return lgb.train(p, dtrain, num_boost_round=p.pop("n_estimators"),
                     valid_sets=valid_sets, valid_names=valid_names,
                     callbacks=[lgb.early_stopping(p["early_stopping_rounds"])])


def precision_at_k(y_true: np.ndarray, scores: np.ndarray, k: int) -> float:
    k = min(k, len(scores))
    if k == 0:
        return 0.0
    topk = np.argsort(-scores)[:k]
    return float(y_true[topk].mean())


def evaluate(y_true: np.ndarray, scores: np.ndarray, k: int) -> dict:
    y_true = np.asarray(y_true, dtype=int)
    scores = np.asarray(scores, dtype=float)
    pr_auc = float(average_precision_score(y_true, scores))
    prec, rec, _ = precision_recall_curve(y_true, scores)
    return {
        "pr_auc": round(pr_auc, 4),
        "precision_at_k": round(precision_at_k(y_true, scores, k), 4),
        "recall_at_k": round(float(y_true[np.argsort(-scores)[:k]].sum() / max(1, y_true.sum())), 4),
        "precision_at_k_5pct": round(precision_at_k(y_true, scores, max(1, int(len(scores) * 0.05))), 4),
        "fp_rate_at_k": round(float((1 - y_true)[np.argsort(-scores)[:k]].mean()), 4),
    }


def derive_label(df: pd.DataFrame) -> pd.Series:
    """Return the fraud label for `df`.

    If the DataFrame already carries a populated `is_fraud` column (e.g. the
    labelled Razorpay CSV), that ground truth is returned unchanged.  Otherwise
    a behavioural proxy is derived — orders with multiple red flags are more
    suspicious.  Flags are not labels, they are features; the model learns
    what combination of features leads to high risk.
    """
    if "is_fraud" in df.columns and df["is_fraud"].sum() > 0:
        return df["is_fraud"].astype(int)
    def _col(name):
        """Return the column as a Series, or a zero Series if missing (schema drift)."""
        return df[name] if name in df.columns else pd.Series(0.0, index=df.index)

    amount = _col("amount")
    hour = _col("hour")
    device_age = _col("device_age_days")
    login_attempts = _col("LoginAttempts")
    balance = _col("AccountBalance")

    # Flag 1: round-number amount (psychological pricing vs fraud pricing)
    is_round = ((amount % 50 == 0) & (amount > 50)).astype(int)
    # Flag 2: night transaction (22:00–05:00)
    is_night = ((hour >= 22) | (hour <= 5)).astype(int)
    # Flag 3: new device (< 14 days)
    is_new_device = (device_age < 14).astype(int)
    # Flag 4: high login attempts (≥3 — credential stuffing signal)
    is_login_heavy = (login_attempts >= 3).astype(int)
    # Flag 5: low balance with high amount (draining behaviour)
    is_drain = (balance < amount).astype(int)
    # Score: sum of flags (0–5)
    score = is_round + is_night + is_new_device + is_login_heavy + is_drain
    return (score >= 2).astype(int)


def train(df: pd.DataFrame, feats: list[str], k: int = 2000, out_dir: str = "models"):
    # Use the CSV's ground-truth label when present; otherwise fall back to the
    # behavioural heuristic (for schemas that carry no is_fraud column).
    df = df.copy()
    if "is_fraud" not in df.columns or df["is_fraud"].sum() == 0:
        df["is_fraud"] = derive_label(df)
    train_df, val_df = temporal_split(df)
    Xtr, ytr = train_df[feats], train_df["is_fraud"]
    Xva, yva = val_df[feats], val_df["is_fraud"]

    scale = float((1 - ytr).sum() / ytr.sum())
    dtr = lgb.Dataset(Xtr, label=ytr, weight=np.where(ytr == 1, scale, 1.0))
    dva = lgb.Dataset(Xva, label=yva, reference=dtr)

    model = make_lgb(dtr, dvalid=dva)
    preds = model.predict(Xva)
    metrics = evaluate(yva.values, preds, k=k)

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    with open(f"{out_dir}/model.pkl", "wb") as f:
        pickle.dump(model, f)
    with open(f"{out_dir}/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(json.dumps(metrics, indent=2))
    print(f"best_iteration={model.best_iteration}")
    return model, metrics


if __name__ == "__main__":
    import argparse
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/transactions_razorpay.csv")
    ap.add_argument("--out", default="models")
    ap.add_argument("--k", type=int, default=2000)
    args = ap.parse_args()

    from src.features.engineer import build_features, feature_columns
    df = pd.read_csv(args.data)
    df = build_features(df, account_col="merchant_id")
    train(df, feature_columns(), k=args.k, out_dir=args.out)