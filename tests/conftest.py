"""Shared pytest fixtures for enterprise fraud detection test suite."""

import random

import pytest

from mlops_bits.config import FraudDetectorConfig
from mlops_bits.detector import LowLatencyFraudDetector
from mlops_bits.drift import DistributionDriftMonitor
from mlops_bits.schemas import TransactionRequest


@pytest.fixture
def detector() -> LowLatencyFraudDetector:
    """Fixture providing initialized fraud detector with default production config."""
    return LowLatencyFraudDetector(FraudDetectorConfig())


@pytest.fixture
def drift_monitor() -> DistributionDriftMonitor:
    """Fixture providing distribution drift monitor."""
    return DistributionDriftMonitor()


@pytest.fixture
def legitimate_transaction() -> TransactionRequest:
    """Standard legitimate consumer transaction."""
    return TransactionRequest(
        transaction_id="tx-legit-001",
        user_id="user-123",
        amount=45.50,
        currency="USD",
        merchant_category="grocery",
        device_trust_score=0.95,
        location_country="US",
        billing_country="US",
        velocity_1h=1,
    )


@pytest.fixture
def high_risk_fraud_transaction() -> TransactionRequest:
    """High-risk fraudulent transaction with multiple compounding risk factors."""
    return TransactionRequest(
        transaction_id="tx-fraud-999",
        user_id="user-456",
        amount=14_500.00,
        currency="USD",
        merchant_category="crypto",
        device_trust_score=0.05,
        location_country="NG",
        billing_country="US",
        velocity_1h=8,
    )


@pytest.fixture
def baseline_distribution() -> list[float]:
    """Synthetic baseline reference distribution of normal transaction amounts."""
    random.seed(42)
    # Baseline: lognormal distribution around $50-$150
    return [round(random.lognormvariate(4.2, 0.5), 2) for _ in range(500)]
