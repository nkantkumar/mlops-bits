"""Low-latency enterprise fraud detection engine."""

import math
import time

from mlops_bits.config import FraudDetectorConfig
from mlops_bits.schemas import (
    BatchTransactionRequest,
    BatchTransactionResponse,
    RiskDecision,
    TransactionRequest,
    TransactionResponse,
)


class LowLatencyFraudDetector:
    """Production fraud scoring engine optimized for sub-millisecond inference."""

    def __init__(self, config: FraudDetectorConfig | None = None) -> None:
        self.config = config or FraudDetectorConfig()

    def _extract_and_score(self, req: TransactionRequest) -> tuple[float, list[str]]:
        """Fast feature evaluation and calibrated risk scoring.

        Uses mathematical logit formulation for low latency and deterministic outputs:
        logit = beta_0 + sum(beta_i * feature_i)
        probability = 1 / (1 + exp(-logit))
        """
        cfg = self.config
        risk_factors: list[str] = []

        # Base intercept calibrated for low baseline fraud (~2%)
        logit: float = -3.5

        # 1. Critical Hard Boundary: Extreme velocity (attack burst)
        if req.velocity_1h >= cfg.critical_velocity_threshold:
            risk_factors.append(f"CRITICAL_VELOCITY_BURST: {req.velocity_1h}/hr")
            return 0.99, risk_factors

        # 2. Velocity Risk
        if req.velocity_1h >= cfg.high_velocity_threshold:
            logit += 1.8 + 0.15 * (req.velocity_1h - cfg.high_velocity_threshold)
            risk_factors.append(f"HIGH_VELOCITY: {req.velocity_1h}/hr")
        elif req.velocity_1h > 1:
            logit += 0.25 * (req.velocity_1h - 1)

        # 3. Location & Billing Country Mismatch
        if req.location_country != req.billing_country:
            logit += 2.0
            risk_factors.append(
                f"COUNTRY_MISMATCH: {req.location_country} != {req.billing_country}"
            )

        # 4. Device Trust Score Boundary
        if req.device_trust_score < cfg.untrusted_device_threshold:
            penalty = (cfg.untrusted_device_threshold - req.device_trust_score) * 4.0
            logit += 1.5 + penalty
            risk_factors.append(f"UNTRUSTED_DEVICE: score={req.device_trust_score:.2f}")
        else:
            # Positive device trust provides modest risk deduction
            logit -= 0.5 * req.device_trust_score

        # 5. Merchant Category Risk
        category_clean = req.merchant_category.lower().strip()
        if category_clean in cfg.high_risk_categories:
            logit += 1.6
            risk_factors.append(f"HIGH_RISK_MERCHANT_CATEGORY: {category_clean}")

        # 6. Transaction Amount Risk (log-scaled with extreme threshold boundary)
        if req.amount >= cfg.extreme_amount_threshold:
            logit += 2.2
            risk_factors.append(f"EXTREME_AMOUNT: ${req.amount:,.2f}")
        elif req.amount > 500.0:
            # Smooth logarithmic scale for amounts above $500
            amount_factor = math.log10(req.amount / 500.0)
            logit += 0.8 * amount_factor

        # Standard Sigmoid Activation for calibrated probability [0.0, 1.0]
        # Clamped logit to avoid overflow
        logit_clamped = max(-15.0, min(15.0, logit))
        raw_score = 1.0 / (1.0 + math.exp(-logit_clamped))
        normalized_score = round(raw_score, 4)

        return normalized_score, risk_factors

    def predict(self, req: TransactionRequest) -> TransactionResponse:
        """Score a single transaction with strict sub-millisecond execution."""
        start_ns = time.perf_counter_ns()

        score, risk_factors = self._extract_and_score(req)

        # Decision Boundary Mapping
        if score >= self.config.reject_threshold:
            decision = RiskDecision.REJECT
            is_fraud = True
        elif score >= self.config.review_threshold:
            decision = RiskDecision.REVIEW
            is_fraud = False
        else:
            decision = RiskDecision.APPROVE
            is_fraud = False

        elapsed_ms = (time.perf_counter_ns() - start_ns) / 1_000_000.0

        return TransactionResponse(
            transaction_id=req.transaction_id,
            is_fraud=is_fraud,
            fraud_score=score,
            decision=decision,
            risk_factors=risk_factors,
            inference_latency_ms=round(elapsed_ms, 4),
        )

    def predict_batch(self, batch: BatchTransactionRequest) -> BatchTransactionResponse:
        """Perform batch inference and aggregate latency & throughput metrics."""
        start_ns = time.perf_counter_ns()
        results: list[TransactionResponse] = [
            self.predict(tx) for tx in batch.transactions
        ]
        total_ms = (time.perf_counter_ns() - start_ns) / 1_000_000.0

        total_processed = len(results)
        throughput_rps = (
            (total_processed / (total_ms / 1000.0)) if total_ms > 0 else 0.0
        )

        return BatchTransactionResponse(
            results=results,
            total_latency_ms=round(total_ms, 3),
            throughput_rps=round(throughput_rps, 1),
            total_processed=total_processed,
        )
