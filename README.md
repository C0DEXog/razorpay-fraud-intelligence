Razorpay Real-Time Risk & Fraud Intelligence System 

> **Sub-25ms Explainable Machine Learning Engine for Transaction Anomaly Detection & Chargeback Prevention**  
> *Built with LightGBM, Random Forest Ensemble, TreeSHAP, and Direct Razorpay REST API Integration.*

---

## Executive Summary

Digital payment gateways like Razorpay process millions of transactions daily across UPI, Cards, and Net Banking. In high-velocity payment ecosystems, traditional rule engines produce high false-decline rates (>15%), directly hurting merchant Gross Merchandise Value (GMV), while opaque deep neural networks fail regulatory auditability requirements.

This project delivers an **end-to-end, production-ready, and explainable risk intelligence engine** that:
1. **Synchronizes Live Data**: Direct ingestion from Razorpay REST API endpoints (`/v1/orders` and `/v1/payments`).
2. **Extracts 24 High-Signal Features**: Multi-window velocity counters, cyclical trigonometric temporal encodings, and category price deviations.
3. **Dual-Model ML Architecture**: Ensembles LightGBM (gradient boosting) and Random Forest (bagging) with strict temporal splitting to prevent future data leakage.
4. **Game-Theoretic XAI (TreeSHAP)**: Provides mathematical waterfall audit trails for every flagged transaction, complying with RBI regulatory guidelines.
5. **Interactive Operations Cockpit**: A Streamlit dashboard with dynamic threshold calibration and Top-K escalation queues.

---

## Architecture & Data Flow

```
+-------------------------------------------------------------------------+
|                          1. INGESTION & GATEWAY                         |
|  Razorpay REST API (GET /v1/orders, GET /v1/payments) with Test API Key |
+------------------------------------+------------------------------------+
                                     |
                                     v
+-------------------------------------------------------------------------+
|                  2. REAL-TIME FEATURE ENGINEERING (24 Signals)          |
|  - Velocity: cnt_1, cnt_6, cnt_24, amt_sum_24, burst_6                  |
|  - Cyclical Time: hour_sin, hour_cos, dow_sin, dow_cos, is_night        |
|  - Behavioral: amount_round100, amt_dev_cat, device_age_log             |
+------------------------------------+------------------------------------+
                                     |
                                     v
+-------------------------------------------------------------------------+
|                         3. MACHINE LEARNING ENGINE                      |
|  LightGBM (Leaf-wise GBDT) + Random Forest (300 Trees) Weighted Average |
|  - Evaluated on PR-AUC & Precision@K (Strict Temporal Past-to-Future Split)|
+------------------------------------+------------------------------------+
                                     |
                                     v
+-------------------------------------------------------------------------+
|                    4. EXPLAINABILITY & OPERATIONS COCKPIT               |
|  - TreeSHAP Local Feature Attributions (Waterfall Plots & Risk Drivers) |
|  - Streamlit Dashboard: Dynamic Threshold Slider & Top-K Escalation Queue|
+-------------------------------------------------------------------------+
```

---

## Key Evaluation Metrics (Imbalanced Fraud Data)

| Metric | Benchmark Score | Why It Matters |
|---|---|---|
| **PR-AUC** | **~0.67** | Gold standard metric for severe class imbalance (<2% fraud rate). |
| **Precision@30** | **88%** | Guarantees human review teams only inspect high-probability fraud. |
| **Recall@30** | **82%** | Captures the vast majority of bad actors in the test partition. |
| **False Positive Rate** | **< 5%** | Protects legitimate merchant checkout conversion and customer GMV. |

---

## Quick Start Guide

### 1. Clone & Setup Virtual Environment
```bash
git clone https://github.com/C0DEXog/razorpay-fraud-intelligence.git
cd razorpay-fraud-intelligence

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure Environment Variables
Create a `.env` file in the root directory:
```bash
cp .env.example .env
```
Add your Razorpay Test API Key and Secret:
```env
RZP_KEY=rzp_test_your_key_here
RZP_SECRET=your_secret_here
```

### 3. Fetch Live Orders & Train Models
```bash
# Ingest live orders from Razorpay API
python src/data/fetch_razorpay.py 300

# Train LightGBM + Random Forest Models & Precompute SHAP values
python src/model/train.py
python src/model/train_ensemble.py
python src/model/generate_shap.py
```

### 4. Launch the Interactive Dashboard
```bash
streamlit run app/dashboard.py --server.port 8501
```
Open **`http://localhost:8501`** in your browser.

---

## Dashboard Capabilities

- ** One-Click Live Sync**: Sidebar button instantly queries Razorpay API, recalculates features, retrains the models, and updates visualizations.
- ** Dynamic Escalation Slider**: Allows risk compliance managers to adjust sensitivity ($0.0 \to 1.0$) according to daily team capacity.
- ** Real-Time SHAP Waterfall**: Unpacks individual transaction risk scores into mathematical feature contributions ($+/-$ points).
- ** Merchant Risk Tab**: Highlights seller accounts showing abnormal chargeback and anomaly clusters.

---

##  Repository Structure

```
razorpay-risk/
├── app/
│   ├── dashboard.py               # Main Streamlit Risk Cockpit
│   └── fraud_chat.py              # Interactive fraud reasoning assistant
├── src/
│   ├── data/
│   │   └── fetch_razorpay.py      # Razorpay REST API client
│   ├── features/
│   │   └── engineer.py            # 24 velocity, cyclical & pricing signals
│   └── model/
│       ├── train.py               # LightGBM training pipeline
│       ├── train_ensemble.py      # Random Forest + Ensemble blender
│       └── generate_shap.py       # TreeSHAP matrix generator
├── data/
│   └── transactions_razorpay.csv  # Normalized transaction records
├── models_razorpay/
│   ├── model.pkl                  # LightGBM serialized model
│   ├── rf_model.pkl               # Random Forest serialized model
│   ├── metrics.json               # Evaluation metrics
│   └── shap_values.npy            # Precomputed TreeSHAP attributions
├── requirements.txt               # Python package dependencies
├── .env.example                   # Template for API credentials
└── README.md                      # Project documentation
```

---

##  License & Attribution
Developed for Razorpay Technical Review & Hackathon Demonstration by **Tanuj Khanna** (MSIT EEE).
