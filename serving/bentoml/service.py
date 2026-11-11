"""BentoML production service definition for LowLatencyFraudDetector."""

from typing import Any

import bentoml

from mlops_bits.config import FraudDetectorConfig
from mlops_bits.detector import LowLatencyFraudDetector
from mlops_bits.schemas import (
    BatchTransactionRequest,
    BatchTransactionResponse,
    TransactionRequest,
    TransactionResponse,
)


@bentoml.service(
    name="fraud_detection_service",
    resources={"cpu": "1"},
    traffic={"timeout": 10},
)
class FraudDetectionService:
    """Production BentoML service for sub-millisecond fraud scoring."""

    def __init__(self) -> None:
        self.config = FraudDetectorConfig()
        self.detector = LowLatencyFraudDetector(self.config)

    @bentoml.api
    def health(self) -> dict[str, Any]:
        """Health and readiness check endpoint."""
        return {
            "status": "healthy",
            "service": "fraud_detection_service",
            "framework": "bentoml",
            "target_sla_p99_ms": self.config.target_sla_p99_ms,
        }

    @bentoml.api
    def predict(self, transaction: dict[str, Any]) -> dict[str, Any]:
        """Score a single incoming financial transaction."""
        req = TransactionRequest.model_validate(transaction)
        response: TransactionResponse = self.detector.predict(req)
        return response.model_dump()

    @bentoml.api
    def predict_batch(self, batch: dict[str, Any]) -> dict[str, Any]:
        """High-throughput batch transaction scoring endpoint."""
        batch_req = BatchTransactionRequest.model_validate(batch)
        response: BatchTransactionResponse = self.detector.predict_batch(batch_req)
        return response.model_dump()
