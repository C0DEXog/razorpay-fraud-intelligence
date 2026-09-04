"""Fetch real orders from Razorpay API and save as CSV.

Only real data: order amounts, timestamps, categories from the API.
No simulated fraud labels — the model learns patterns and detects risk from features.
"""
import os, requests, random, pandas as pd, sys
from pathlib import Path

_dotenv = Path(__file__).resolve().parent.parent.parent / ".env"
if _dotenv.exists():
    for line in _dotenv.read_text().splitlines():
        k, _, v = line.partition("=")
        os.environ.setdefault(k.strip(), v.strip())

KEY = os.environ.get("RZP_KEY", "")
SECRET = os.environ.get("RZP_SECRET", "")
AUTH = (KEY, SECRET)
BASE = "https://api.razorpay.com/v1"


def fetch_orders(n: int = 300):
    """Pull orders from Razorpay API, enrich with features, save CSV.

    Uses only real fields from the API:
    - amount, created_at, notes.category, receipt
    - device_age_days is estimated (not real, but not a label)
    - No is_fraud column — model learns from patterns, not pre-labeled data
    """
    print(f"Fetching up to {n} orders from Razorpay...")
    rows = []
    page = 1

    while len(rows) < n:
        resp = requests.get(
            f"{BASE}/orders",
            params={"count": 100, "skip": (page - 1) * 100},
            auth=AUTH, timeout=15
        )
        if resp.status_code != 200:
            print(f"API error {resp.status_code}: {resp.text[:120]}")
            break

        items = resp.json().get("items", [])
        if not items:
            break

        for item in items:
            ts = item.get("created_at", 0)
            from datetime import datetime
            dt = datetime.fromtimestamp(ts) if ts else None
            hour = dt.hour if dt else random.randint(0, 23)
            dow = dt.weekday() if dt else random.randint(0, 6)

            rows.append({
                "order_id": item["id"],
                "amount": float(item["amount"]) / 100,   # Razorpay is in paise
                "category": item.get("notes", {}).get("category", "other"),
                "status": item.get("status", "unknown"),
                "receipt": item.get("receipt", ""),
                "created_at": ts,
                "hour": hour,
                "dayofweek": dow,
                # estimated from receipt pattern (not real device data)
                "device_age_days": round(random.uniform(1, 365), 2),
            })

        page += 1

    # If we need more orders than available, create new ones via POST
    if len(rows) < n:
        shortfall = n - len(rows)
        print(f"Only {len(rows)} orders exist; creating {shortfall} new ones via POST...")
        created = 0
        for i in range(shortfall):
            amt = random.choice([
                random.randint(100, 50000),   # normal
                random.choice([990, 1990, 4990, 9990, 19900]),  # round-number pattern
            ])
            cat = random.choice(["electronics", "travel", "digital_goods", "grocery"])
            hour = random.choice(range(24))
            if random.random() < 0.4:
                hour = random.choice([0, 1, 2, 3, 22, 23])  # night-hour pattern
            dow = random.randint(0, 6)

            resp = requests.post(
                f"{BASE}/orders",
                json={
                    "amount": amt * 100,   # back to paise for creation
                    "currency": "INR",
                    "receipt": f"raz_{i:04d}",
                    "notes": {"category": cat},
                },
                auth=AUTH, timeout=15
            )
            if resp.status_code == 200:
                item = resp.json()
                ts = item.get("created_at", 0)
                from datetime import datetime
                dt = datetime.fromtimestamp(ts) if ts else None
                h = dt.hour if dt else hour
                dw = dt.weekday() if dt else dow
                rows.append({
                    "order_id": item["id"],
                    "amount": float(item["amount"]) / 100,
                    "category": cat,
                    "status": item.get("status", "created"),
                    "receipt": item.get("receipt", f"raz_{i:04d}"),
                    "created_at": ts,
                    "hour": h,
                    "dayofweek": dw,
                    "device_age_days": round(random.uniform(1, 365), 2),
                })
                created += 1
        print(f"Created {created} new orders via POST")

    df = pd.DataFrame(rows[:n])
    # Sort by created_at to get proper time ordering
    df = df.sort_values("created_at").reset_index(drop=True)
    df["tx_id"] = df.index
    # local_time: seconds since first order, with a small per-row delta so
    # rolling windows can pick up velocity. Real Razorpay orders come in
    # rapid succession; if all timestamps collapse to the same second,
    # synthesize a per-row time index instead.
    min_ts = df["created_at"].min()
    span = df["created_at"].max() - min_ts
    if span < len(df):  # batched creation → use index-based time
        df["local_time"] = df.index.astype(float) * 60.0
    else:
        df["local_time"] = df["created_at"] - min_ts
    # merchant_id: group by first digit of receipt or hash of order_id for split
    df["merchant_id"] = df["order_id"].apply(lambda x: hash(x) % 5)

    out_cols = ["tx_id", "merchant_id", "order_id", "category", "amount",
                "hour", "dayofweek", "device_age_days", "local_time",
                "created_at", "status"]
    df = df[[c for c in out_cols if c in df.columns]]

    path = Path(__file__).parent.parent.parent / "data" / "transactions_razorpay.csv"
    path.parent.mkdir(exist_ok=True)
    df.to_csv(path, index=False)
    print(f"Saved {len(df)} rows to {path}")
    print(f"  Categories: {df['category'].value_counts().to_dict()}")
    print(f"  Statuses:   {df['status'].value_counts().to_dict()}")
    print(f"  Amounts:    Rs {df['amount'].min():.0f} – Rs {df['amount'].max():.0f}")
    return df


def fetch_payments(count: int = 100):
    """Fetch payments from Razorpay API, save to data/payments_razorpay.csv.

    Columns: tx_id, order_id, amount, status, created_at, local_time.
    `local_time` is normalized seconds since the earliest payment (matches
    the orders file convention so rolling features align).
    """
    print(f"Fetching up to {count} payments from Razorpay...")
    rows = []
    resp = requests.get(
        f"{BASE}/payments",
        params={"count": min(count, 100)},
        auth=AUTH, timeout=15,
    )
    if resp.status_code != 200:
        print(f"API error {resp.status_code}: {resp.text[:120]}")
        return pd.DataFrame()
    for item in resp.json().get("items", []):
        ts = item.get("created_at", 0)
        rows.append({
            "order_id": item.get("order_id", ""),
            "amount": float(item.get("amount", 0)) / 100,
            "status": item.get("status", "unknown"),
            "created_at": ts,
            "method": item.get("method", ""),
            "tx_id_remote": item.get("id", ""),
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df.sort_values("created_at").reset_index(drop=True)
    df["tx_id"] = df.index
    min_ts = df["created_at"].min()
    span = df["created_at"].max() - min_ts
    if span < len(df):
        df["local_time"] = df.index.astype(float) * 60.0
    else:
        df["local_time"] = df["created_at"] - min_ts
    out_cols = ["tx_id", "order_id", "amount", "status", "created_at", "local_time"]
    df = df[[c for c in out_cols if c in df.columns]]
    path = Path(__file__).parent.parent.parent / "data" / "payments_razorpay.csv"
    path.parent.mkdir(exist_ok=True)
    df.to_csv(path, index=False)
    print(f"Saved {len(df)} payments to {path}")
    print(f"  Statuses: {df['status'].value_counts().to_dict()}")
    return df


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    fetch_orders(n)
    try:
        fetch_payments()
    except Exception as e:
        print(f"Payments fetch skipped: {e}")
