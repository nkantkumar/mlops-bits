"""Multi-model client supporting both microservice HTTP dispatch and in-process fallback."""

from __future__ import annotations

import os
import time
from typing import Any

import httpx

from mlops_bits.config import FraudDetectorConfig
from mlops_bits.detector import LowLatencyFraudDetector
from mlops_bits.schemas import TransactionRequest, TransactionResponse


class MultiModelClient:
    """Client coordinating predictions across Champion and Challenger models."""

    def __init__(
        self,
        champion_url: str | None = None,
        challenger_url: str | None = None,
    ) -> None:
        self.champion_url = champion_url or os.getenv("CHAMPION_SERVICE_URL")
        self.challenger_url = challenger_url or os.getenv("CHALLENGER_SERVICE_URL")

        # In-process fallbacks
        self._in_process_champion = LowLatencyFraudDetector(FraudDetectorConfig())
        # Challenger model has stricter boundary thresholds (candidate for promotion)
        challenger_cfg = FraudDetectorConfig(
            reject_threshold=0.70,
            review_threshold=0.35,
            high_velocity_threshold=4,
            untrusted_device_threshold=0.35,
        )
        self._in_process_challenger = LowLatencyFraudDetector(challenger_cfg)

    def predict_champion(self, req: TransactionRequest) -> TransactionResponse:
        """Score transaction using Champion model."""
        if self.champion_url:
            return self._predict_remote(self.champion_url, req)
        return self._in_process_champion.predict(req)

    def predict_challenger(self, req: TransactionRequest) -> TransactionResponse:
        """Score transaction using Challenger (Canary / Variant B) model."""
        if self.challenger_url:
            return self._predict_remote(self.challenger_url, req)
        return self._in_process_challenger.predict(req)

    def _predict_remote(
        self, base_url: str, req: TransactionRequest
    ) -> TransactionResponse:
        """Forward transaction to a remote model service endpoint."""
        url = f"{base_url.rstrip('/')}/v1/predict"
        t0 = time.perf_counter_ns()
        with httpx.Client(timeout=2.0) as client:
            resp = client.post(url, json=req.model_dump(mode="json"))
            resp.raise_for_status()
            elapsed_ms = (time.perf_counter_ns() - t0) / 1_000_000.0
            data = resp.json()
            data["inference_latency_ms"] = round(elapsed_ms, 4)
            return TransactionResponse.model_validate(data)

    def compare(self, req: TransactionRequest) -> dict[str, Any]:
        """Shadow-score transaction across both models and return comparison."""
        t0 = time.perf_counter_ns()
        res_a = self.predict_champion(req)
        res_b = self.predict_challenger(req)
        total_ms = (time.perf_counter_ns() - t0) / 1_000_000.0

        agreement = res_a.decision == res_b.decision
        score_diff = round(abs(res_a.fraud_score - res_b.fraud_score), 4)

        return {
            "transaction_id": req.transaction_id,
            "agreement": agreement,
            "decision_match": f"{res_a.decision.value} == {res_b.decision.value}"
            if agreement
            else f"{res_a.decision.value} != {res_b.decision.value}",
            "score_delta": score_diff,
            "total_latency_ms": round(total_ms, 4),
            "champion": {
                "decision": res_a.decision.value,
                "fraud_score": res_a.fraud_score,
                "is_fraud": res_a.is_fraud,
                "latency_ms": res_a.inference_latency_ms,
            },
            "challenger": {
                "decision": res_b.decision.value,
                "fraud_score": res_b.fraud_score,
                "is_fraud": res_b.is_fraud,
                "latency_ms": res_b.inference_latency_ms,
            },
        }
