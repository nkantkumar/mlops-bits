"""DVC Pipeline Stage 1: Generate or prepare reproducible transaction data."""

import csv
import random
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import yaml


def load_params() -> dict[str, Any]:
    """Load configuration parameters from params.yaml."""
    params_file = Path("params.yaml")
    if not params_file.exists():
        raise FileNotFoundError(f"Missing {params_file}")
    with open(params_file, "r", encoding="utf-8") as f:
        params: dict[str, Any] = yaml.safe_load(f)
    return params.get("prepare", {})


def generate_transactions(
    num_samples: int, fraud_ratio: float, seed: int
) -> list[dict[str, Any]]:
    """Deterministically generate synthetic transactions for baseline data."""
    random.seed(seed)
    records = []
    base_dt = datetime(2026, 1, 1, 0, 0, 0, tzinfo=UTC)

    fraud_target_count = int(num_samples * fraud_ratio)

    for i in range(num_samples):
        is_synthetic_fraud = i < fraud_target_count

        if is_synthetic_fraud:
            # Synthetic fraud pattern: untrusted device, mismatched country, elevated amount
            amount = round(random.uniform(5000.0, 15000.0), 2)
            merchant_category = random.choice(["crypto", "gambling", "electronics"])
            device_trust = round(random.uniform(0.01, 0.25), 2)
            loc_country = random.choice(["NG", "FR", "DE"])
            bill_country = "US"
            velocity = random.randint(6, 12)
        else:
            # Normal consumer pattern
            amount = round(random.lognormvariate(3.9, 0.6), 2)
            merchant_category = random.choice(["retail", "grocery", "travel"])
            device_trust = round(random.uniform(0.70, 0.99), 2)
            loc_country = "US"
            bill_country = "US"
            velocity = random.randint(1, 3)

        tx_dt = base_dt + timedelta(minutes=i)
        records.append(
            {
                "transaction_id": f"tx-dvc-{i:06d}",
                "user_id": f"usr-{random.randint(1000, 9999)}",
                "amount": max(1.0, amount),
                "currency": "USD",
                "merchant_category": merchant_category,
                "device_trust_score": device_trust,
                "location_country": loc_country,
                "billing_country": bill_country,
                "velocity_1h": velocity,
                "timestamp": tx_dt.isoformat(),
            }
        )

    # Shuffle to mix fraud cases throughout dataset
    random.shuffle(records)
    return records


def main() -> None:
    """Execute data preparation stage."""
    params = load_params()
    num_samples = int(params.get("num_samples", 1000))
    fraud_ratio = float(params.get("fraud_ratio", 0.05))
    seed = int(params.get("random_seed", 42))
    output_path = Path(params.get("output_path", "data/raw/transactions.csv"))

    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(
        f"📦 Generating {num_samples} transactions (fraud ratio: {fraud_ratio:.1%}, seed: {seed})..."
    )
    records = generate_transactions(num_samples, fraud_ratio, seed)

    fieldnames = [
        "transaction_id",
        "user_id",
        "amount",
        "currency",
        "merchant_category",
        "device_trust_score",
        "location_country",
        "billing_country",
        "velocity_1h",
        "timestamp",
    ]

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    print(f"✅ Data written to {output_path} ({len(records)} rows)")


if __name__ == "__main__":
    main()
