"""Train RandomForest ensemble for Razorpay risk detection."""
import json, pickle
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, precision_recall_curve

sys_path_insert = False
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.features.engineer import build_features, feature_columns
from src.model.train import temporal_split, derive_label, precision_at_k

def evaluate(y_true, scores, k=200):
    y_true = np.asarray(y_true, dtype=int); scores = np.asarray(scores, dtype=float)
    pr_auc = float(average_precision_score(y_true, scores))
    prec, rec, _ = precision_recall_curve(y_true, scores)
    return {
        "pr_auc": round(pr_auc, 4),
        "precision_at_k": round(precision_at_k(y_true, scores, k), 4),
        "recall_at_k": round(float(y_true[np.argsort(-scores)[:k]].sum() / max(1, y_true.sum())), 4),
        "precision_at_k_5pct": round(precision_at_k(y_true, scores, max(1, int(len(scores)*0.05))), 4),
        "fp_rate_at_k": round(float((1-y_true)[np.argsort(-scores)[:k]].mean()), 4),
    }

def train_ensemble(data_path="data/transactions_razorpay.csv", out_dir="models_razorpay"):
    df = pd.read_csv(data_path)
    df = build_features(df, account_col="merchant_id")
    df["is_fraud"] = derive_label(df)
    train_df, val_df = temporal_split(df)
    feats = feature_columns()
    Xtr, ytr = train_df[feats], train_df["is_fraud"]
    Xva, yva = val_df[feats], val_df["is_fraud"]
    rf = RandomForestClassifier(n_estimators=300, max_depth=10, min_samples_leaf=5, random_state=42, n_jobs=-1)
    rf.fit(Xtr, ytr)
    preds = rf.predict_proba(Xva)[:,1]
    metrics = evaluate(yva.values, preds, k=200)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    with open(f"{out_dir}/rf_model.pkl", "wb") as f: pickle.dump(rf, f)
    with open(f"{out_dir}/rf_metrics.json", "w") as f: json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))
    return rf, metrics

if __name__ == "__main__":
    train_ensemble()
