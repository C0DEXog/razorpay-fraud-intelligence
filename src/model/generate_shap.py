"""Compute SHAP values for every row and cache them to models_razorpay/shap_values.npy.

The dashboard loads this file at startup so SHAP explanations render instantly
without recomputing the explainer on every load.  Run after training:
    python -m src.model.generate_shap
"""
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import shap

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main(data: str = "data/transactions_razorpay.csv", out_dir: str = "models_razorpay"):
    from src.features.engineer import build_features, feature_columns

    df = pd.read_csv(ROOT / data)
    df = build_features(df, account_col="merchant_id")
    feats = feature_columns()
    X = df[feats].astype(float)

    with open(ROOT / out_dir / "model.pkl", "rb") as f:
        model = pickle.load(f)

    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(X, check_additivity=False)
    if isinstance(sv, list):          # older shap returns per-class list
        sv = sv[1]
    sv = np.asarray(sv, dtype=float)

    out_path = ROOT / out_dir / "shap_values.npy"
    np.save(out_path, sv)
    print(f"Saved {out_path}: shape={sv.shape}")


if __name__ == "__main__":
    main()