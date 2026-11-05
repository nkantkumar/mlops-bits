"""Unit and boundary constraint tests for the fraud detection engine."""

import pytest
from pydantic import ValidationError

from mlops_bits.detector import LowLatencyFraudDetector
from mlops_bits.schemas import (
    BatchTransactionRequest,
    RiskDecision,
    TransactionRequest,
)


def test_legitimate_transaction_approved(
    detector: LowLatencyFraudDetector, legitimate_transaction: TransactionRequest
) -> None:
    """A low-risk consumer transaction should be approved with minimal fraud score."""
    response = detector.predict(legitimate_transaction)
    assert response.transaction_id == "tx-legit-001"
    assert response.decision == RiskDecision.APPROVE
    assert not response.is_fraud
    assert response.fraud_score < 0.20
    assert len(response.risk_factors) == 0
    assert response.inference_latency_ms >= 0.0


def test_high_risk_transaction_rejected(
    detector: LowLatencyFraudDetector, high_risk_fraud_transaction: TransactionRequest
) -> None:
    """A transaction triggering multiple high-risk factors must be rejected."""
    response = detector.predict(high_risk_fraud_transaction)
    assert response.transaction_id == "tx-fraud-999"
    assert response.decision == RiskDecision.REJECT
    assert response.is_fraud
    assert response.fraud_score >= detector.config.reject_threshold
    assert len(response.risk_factors) >= 4
    # Ensure key risk factors were identified
    risk_text = " ".join(response.risk_factors)
    assert "COUNTRY_MISMATCH" in risk_text
    assert "UNTRUSTED_DEVICE" in risk_text
    assert "HIGH_RISK_MERCHANT_CATEGORY" in risk_text
    assert "HIGH_VELOCITY" in risk_text


def test_critical_velocity_burst_hard_boundary(
    detector: LowLatencyFraudDetector, legitimate_transaction: TransactionRequest
) -> None:
    """Velocity exceeding critical threshold triggers immediate rejection without exception."""
    critical_tx = legitimate_transaction.model_copy(update={"velocity_1h": 12})
    response = detector.predict(critical_tx)

    assert response.decision == RiskDecision.REJECT
    assert response.is_fraud
    assert response.fraud_score >= 0.99
    assert any("CRITICAL_VELOCITY_BURST" in r for r in response.risk_factors)


def test_deterministic_scoring(
    detector: LowLatencyFraudDetector, high_risk_fraud_transaction: TransactionRequest
) -> None:
    """Scoring the exact same payload repeatedly must produce strictly identical outputs."""
    first_res = detector.predict(high_risk_fraud_transaction)
    for _ in range(10):
        subsequent_res = detector.predict(high_risk_fraud_transaction)
        assert subsequent_res.fraud_score == first_res.fraud_score
        assert subsequent_res.decision == first_res.decision
        assert subsequent_res.risk_factors == first_res.risk_factors


def test_extreme_amount_boundary(
    detector: LowLatencyFraudDetector, legitimate_transaction: TransactionRequest
) -> None:
    """Extreme transaction amount triggers specific risk factor."""
    normal_amt_tx = legitimate_transaction.model_copy(update={"amount": 9_999.0})
    normal_res = detector.predict(normal_amt_tx)
    assert not any("EXTREME_AMOUNT" in r for r in normal_res.risk_factors)

    extreme_tx = legitimate_transaction.model_copy(update={"amount": 10_000.0})
    extreme_res = detector.predict(extreme_tx)
    assert any("EXTREME_AMOUNT" in r for r in extreme_res.risk_factors)
    assert extreme_res.fraud_score > normal_res.fraud_score


def test_device_trust_boundaries(
    detector: LowLatencyFraudDetector, legitimate_transaction: TransactionRequest
) -> None:
    """Device trust score at 0.0, 0.3, and 1.0 boundaries are handled cleanly."""
    tx_untrusted = legitimate_transaction.model_copy(update={"device_trust_score": 0.0})
    res_untrusted = detector.predict(tx_untrusted)
    assert any("UNTRUSTED_DEVICE" in r for r in res_untrusted.risk_factors)

    tx_trusted = legitimate_transaction.model_copy(update={"device_trust_score": 1.0})
    res_trusted = detector.predict(tx_trusted)
    assert not any("UNTRUSTED_DEVICE" in r for r in res_trusted.risk_factors)
    assert res_trusted.fraud_score < res_untrusted.fraud_score


def test_pydantic_schema_boundary_validation() -> None:
    """Schema must reject negative amounts, invalid currencies, and out-of-range trust scores."""
    # Negative amount
    with pytest.raises(ValidationError):
        TransactionRequest(
            transaction_id="tx-err-1",
            user_id="u1",
            amount=-10.0,
            location_country="US",
            billing_country="US",
        )

    # Zero amount
    with pytest.raises(ValidationError):
        TransactionRequest(
            transaction_id="tx-err-2",
            user_id="u1",
            amount=0.0,
            location_country="US",
            billing_country="US",
        )

    # Device trust score > 1.0
    with pytest.raises(ValidationError):
        TransactionRequest(
            transaction_id="tx-err-3",
            user_id="u1",
            amount=10.0,
            device_trust_score=1.5,
            location_country="US",
            billing_country="US",
        )

    # Negative velocity
    with pytest.raises(ValidationError):
        TransactionRequest(
            transaction_id="tx-err-4",
            user_id="u1",
            amount=10.0,
            velocity_1h=-1,
            location_country="US",
            billing_country="US",
        )


def test_batch_scoring(
    detector: LowLatencyFraudDetector,
    legitimate_transaction: TransactionRequest,
    high_risk_fraud_transaction: TransactionRequest,
) -> None:
    """Batch prediction correctly processes multiple transactions with throughput metrics."""
    batch_req = BatchTransactionRequest(
        transactions=[legitimate_transaction, high_risk_fraud_transaction]
    )
    batch_res = detector.predict_batch(batch_req)

    assert batch_res.total_processed == 2
    assert len(batch_res.results) == 2
    assert batch_res.results[0].decision == RiskDecision.APPROVE
    assert batch_res.results[1].decision == RiskDecision.REJECT
    assert batch_res.throughput_rps > 0.0
