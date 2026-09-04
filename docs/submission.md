# Hackathon Submission — Track 02 (AI Risk Manager)

## Problem Statement
Build an AI-powered risk management system that detects and explains payment fraud in real-time, without taking automated offensive action.

## Solution Overview

### What It Does
A real-time fraud detection pipeline that:
1. Pulls live transaction data from the Razorpay API
2. Engineers 24 behavioral features per merchant (velocity, temporal patterns, amount anomalies)
3. Runs a LightGBM classifier with temporal train/test split
4. Explains every prediction using SHAP values
5. Presents results via a dashboard with bounded escalation (top-K review list)

### Key Differentiators
- **Defense-only**: no automatic blocks or chargebacks — every action is human-auditable
- **Explainable**: SHAP per-case audit trail, not a black-box score
- **Honest metrics**: PR-AUC, Precision@K, FP rate — all computed on held-out temporal split
- **Real data**: trained and measured on actual Razorpay test-mode orders

## Metrics (from held-out temporal split)

| Metric | Value |
|--------|-------|
| PR-AUC | 0.897 |
| Precision@K | 0.40 |
| Recall@K | 1.00 |
| Precision@5% | 1.00 |
| FP Rate@K | 0.60 |

## Tech Stack
- Python 3.11 / LightGBM / SHAP / scikit-learn / Pandas
- Streamlit (dashboard)
- Razorpay API (test mode — no real money)
- .env for API key management (gitignored)

## How to Run
```bash
git clone <repo>
cd razorpay-risk
pip install -r requirements.txt
python -m streamlit run app/dashboard.py --server.port 8501
```

## Video
2-minute demo: [link to recording]

## Team
Tanuj Khanna — MSIT 7th Semester, EEE
