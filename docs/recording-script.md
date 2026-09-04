# 2-Minute Demo Script

## 0:00 — 0:20  Opening
Open `localhost:8502`. Show:
- Title: "Razorpay Risk Dashboard"
- 5 metric cards (PR-AUC, Precision@K, Recall@K, Precision@5%, FP Rate@K)
- Sidebar: "300 orders loaded, LightGBM"
- "Real Razorpay test-mode data only — no simulated fraud labels"

## 0:20 — 0:50  Data Pipeline
Voiceover:
- Fetches live from `https://api.razorpay.com/v1/orders` (test mode)
- Engineers 24 features: velocity (1/6/24hr rolling), temporal, amount shape, peer deviation
- Trains LightGBM with temporal split (no future leakage)
- Defensive design: predicts risk, humans take action

## 0:50 — 1:20  Explainability
- Click on a high-risk transaction
- SHAP bar plot: top-3 drivers
- SHAP waterfall: base → score
- Show how amount + velocity + hour explain the risk

## 1:20 — 1:50  Bounded Escalation
- Move K slider: 5 → 30 → 100
- Show escalation list updates
- Point out: defense-only, top-K, human audit
- Top factors column in audit log

## 1:50 — 2:00  Closing
- Click "Refresh from API" → spinner → live update
- Show: open source, .env protected, ready for submission
- Mention: real Razorpay API, no synthetic data
