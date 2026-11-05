"""FastAPI enterprise production service for fraud detection and drift monitoring."""

from typing import Any

from fastapi import FastAPI, HTTPException, status

from mlops_bits.detector import LowLatencyFraudDetector
from mlops_bits.drift import DistributionDriftMonitor
from mlops_bits.schemas import (
    BatchTransactionRequest,
    BatchTransactionResponse,
    DriftAnalysisRequest,
    DriftAnalysisResponse,
    TransactionRequest,
    TransactionResponse,
)

app = FastAPI(
    title="Enterprise Fraud Detection Service",
    description="Production-grade low-latency fraud detection module and drift monitoring for MLOps.",
    version="0.1.0",
)

# In-memory singletons pre-initialized for zero-warmup latency
detector = LowLatencyFraudDetector()
drift_monitor = DistributionDriftMonitor()


@app.get("/health", status_code=status.HTTP_200_OK)
def health_check() -> dict[str, Any]:
    """Health and readiness probe."""
    return {
        "status": "healthy",
        "service": "mlops-bits-fraud-detector",
        "version": "0.1.0",
        "sla_p99_target_ms": detector.config.target_sla_p99_ms,
    }


@app.post(
    "/v1/predict", response_model=TransactionResponse, status_code=status.HTTP_200_OK
)
def score_transaction(request: TransactionRequest) -> TransactionResponse:
    """Low-latency inference endpoint to score an incoming financial transaction."""
    try:
        return detector.predict(request)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference error: {exc}",
        ) from exc


@app.post(
    "/v1/predict/batch",
    response_model=BatchTransactionResponse,
    status_code=status.HTTP_200_OK,
)
def score_batch(batch_request: BatchTransactionRequest) -> BatchTransactionResponse:
    """High-throughput batch scoring endpoint."""
    return detector.predict_batch(batch_request)


@app.post(
    "/v1/monitor/drift",
    response_model=DriftAnalysisResponse,
    status_code=status.HTTP_200_OK,
)
def evaluate_drift(request: DriftAnalysisRequest) -> DriftAnalysisResponse:
    """Data drift analysis endpoint calculating Population Stability Index (PSI)."""
    try:
        return drift_monitor.analyze(request)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
