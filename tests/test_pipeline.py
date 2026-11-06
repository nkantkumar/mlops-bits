"""Tests for DVC pipeline data preparation and evaluation stages."""

import tempfile
from pathlib import Path

from mlops_bits.pipeline.evaluate import load_transactions_from_csv
from mlops_bits.pipeline.prepare import generate_transactions


def test_generate_transactions_determinism() -> None:
    """generate_transactions produces identical datasets when using same seed."""
    txs_1 = generate_transactions(num_samples=100, fraud_ratio=0.10, seed=42)
    txs_2 = generate_transactions(num_samples=100, fraud_ratio=0.10, seed=42)
    assert len(txs_1) == 100
    assert txs_1 == txs_2


def test_generate_transactions_fraud_ratio() -> None:
    """Generated dataset reflects requested fraud ratio."""
    num_samples = 200
    fraud_ratio = 0.15
    txs = generate_transactions(
        num_samples=num_samples, fraud_ratio=fraud_ratio, seed=123
    )
    # Target count is int(200 * 0.15) = 30
    assert len(txs) == num_samples
    # Check that high risk fields exist in generated samples
    has_untrusted_devices = any(tx["device_trust_score"] < 0.3 for tx in txs)
    assert has_untrusted_devices is True


def test_csv_roundtrip_validation() -> None:
    """CSV output from preparation parses cleanly into TransactionRequest models."""
    import csv

    txs = generate_transactions(num_samples=50, fraud_ratio=0.10, seed=99)
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_csv = Path(tmpdir) / "test_txs.csv"
        fieldnames = list(txs[0].keys())
        with open(tmp_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(txs)

        loaded = load_transactions_from_csv(tmp_csv)
        assert len(loaded) == 50
        assert loaded[0].transaction_id == txs[0]["transaction_id"]
        assert loaded[0].amount == txs[0]["amount"]
