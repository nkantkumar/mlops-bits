"""Data schemas and type definitions for enterprise fraud detection."""

from datetime import UTC, datetime
from enum import Enum

from pydantic import BaseModel, Field, field_validator


class RiskDecision(str, Enum):
    """Actionable decision output from the fraud detection module."""

    APPROVE = "APPROVE"
    REVIEW = "REVIEW"
    REJECT = "REJECT"


class TransactionRequest(BaseModel):
    """Inference input payload for a single financial transaction."""

    transaction_id: str = Field(
        ..., description="Unique transaction UUID or identifier"
    )
    user_id: str = Field(..., description="Unique user or account identifier")
    amount: float = Field(
        ..., gt=0.0, description="Transaction amount (strictly positive)"
    )
    currency: str = Field(
        default="USD", min_length=3, max_length=3, description="ISO 4217 currency code"
    )
    merchant_category: str = Field(
        default="general", description="Merchant category or MCC code"
    )
    device_trust_score: float = Field(
        default=0.8, ge=0.0, le=1.0, description="Device reputation score [0.0, 1.0]"
    )
    location_country: str = Field(
        default="US",
        min_length=2,
        max_length=2,
        description="Two-letter ISO country code where transaction originated",
    )
    billing_country: str = Field(
        default="US",
        min_length=2,
        max_length=2,
        description="Two-letter ISO billing address country code",
    )
    velocity_1h: int = Field(
        default=1,
        ge=0,
        description="Number of transactions initiated by user in last 60 minutes",
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(UTC), description="Transaction timestamp"
    )

    @field_validator("currency", "location_country", "billing_country")
    @classmethod
    def normalize_uppercase(cls, v: str) -> str:
        return v.strip().upper()


class TransactionResponse(BaseModel):
    """Inference output response for a scored transaction."""

    transaction_id: str
    is_fraud: bool
    fraud_score: float = Field(
        ..., ge=0.0, le=1.0, description="Normalized risk score [0.0, 1.0]"
    )
    decision: RiskDecision
    risk_factors: list[str] = Field(
        default_factory=list, description="Triggered risk rules or high-weight signals"
    )
    inference_latency_ms: float = Field(
        ..., description="Time taken to score in milliseconds"
    )


class BatchTransactionRequest(BaseModel):
    """Batch inference request payload."""

    transactions: list[TransactionRequest] = Field(..., min_length=1, max_length=10000)


class BatchTransactionResponse(BaseModel):
    """Batch inference output with aggregated performance metrics."""

    results: list[TransactionResponse]
    total_latency_ms: float
    throughput_rps: float
    total_processed: int


class DriftStatus(str, Enum):
    """Standardized PSI drift severity levels."""

    NO_DRIFT = "NO_DRIFT"  # PSI < 0.1
    MODERATE_DRIFT = "MODERATE_DRIFT"  # 0.1 <= PSI < 0.25
    SIGNIFICANT_DRIFT = "SIGNIFICANT_DRIFT"  # PSI >= 0.25


class DriftAnalysisRequest(BaseModel):
    """Payload to trigger data drift analysis against a baseline reference."""

    feature_name: str
    baseline_values: list[float] = Field(..., min_length=10)
    current_values: list[float] = Field(..., min_length=10)
    num_bins: int = Field(default=10, ge=2, le=50)


class DriftAnalysisResponse(BaseModel):
    """Population Stability Index (PSI) drift report."""

    feature_name: str
    psi: float
    drift_status: DriftStatus
    alert_triggered: bool
    mean_baseline: float
    mean_current: float
    min_current: float
    max_current: float
