"""Pydantic schemas for Evidently monitoring requests, summaries, and telemetry."""

from __future__ import annotations

from pydantic import BaseModel, Field


class FeatureDriftDetail(BaseModel):
    """Detailed drift status for an individual feature."""

    feature_name: str
    drift_detected: bool
    drift_score: float = Field(
        ..., description="Calculated statistical drift score or p-value"
    )
    stat_test: str = Field(
        ..., description="Statistical test method used (e.g. K-S, Chi-Square)"
    )
    threshold: float = Field(default=0.05, description="Significance threshold applied")


class EvidentlyDriftSummary(BaseModel):
    """High-level summary of an Evidently data drift evaluation."""

    drift_detected: bool = Field(
        ..., description="True if share of drifted columns exceeds threshold"
    )
    drift_share: float = Field(
        ..., description="Fraction of evaluated features exhibiting statistical drift"
    )
    drifted_features_count: int = Field(..., description="Number of drifted features")
    total_features: int = Field(..., description="Total features analyzed")
    drifted_features: list[str] = Field(
        default_factory=list, description="List of drifted feature names"
    )
    feature_details: list[FeatureDriftDetail] = Field(
        default_factory=list, description="Per-feature drift analysis metrics"
    )
    report_html_path: str | None = None
    report_json_path: str | None = None
    snapshot_id: str | None = None
    timestamp: str | None = None


class EvidentlyQualitySummary(BaseModel):
    """Summary of data quality and integrity report."""

    total_rows: int
    total_columns: int
    missing_values_count: int = 0
    missing_values_share: float = 0.0
    report_html_path: str | None = None
    report_json_path: str | None = None


class EvidentlyTestDetail(BaseModel):
    """Result of an individual test case in an Evidently TestSuite."""

    name: str
    status: str = Field(..., description="'SUCCESS', 'FAIL', 'WARNING', or 'ERROR'")
    description: str


class EvidentlyTestSuiteSummary(BaseModel):
    """Summary of automated test suite execution (for CI/CD and deployment gating)."""

    all_passed: bool = Field(..., description="Whether all executed tests passed")
    total_tests: int = Field(
        ..., description="Total number of evaluated test assertions"
    )
    passed_tests: int = Field(..., description="Number of passed tests")
    failed_tests: int = Field(..., description="Number of failed tests")
    test_details: list[EvidentlyTestDetail] = Field(
        default_factory=list, description="List of individual test results"
    )
    report_html_path: str | None = None


class EvidentlyReportRequest(BaseModel):
    """Request payload for triggering on-demand Evidently monitoring reports."""

    reference_path: str | None = Field(
        default=None,
        description="Optional path to reference baseline CSV (e.g. data/raw/transactions.csv)",
    )
    current_path: str | None = Field(
        default=None, description="Optional path to current production CSV dataset"
    )
    include_quality: bool = Field(
        default=True,
        description="Whether to include data quality analysis alongside data drift",
    )
    save_html: bool = Field(
        default=True,
        description="Whether to generate and save a standalone HTML dashboard report",
    )
    sync_workspace: bool = Field(
        default=True, description="Whether to sync snapshot into Evidently UI workspace"
    )


class InferenceTelemetryRecord(BaseModel):
    """Telemetry payload captured from live production inference scoring."""

    transaction_id: str
    amount: float
    currency: str = "USD"
    merchant_category: str
    device_trust_score: float
    location_country: str
    billing_country: str
    velocity_1h: int
    fraud_score: float
    is_fraud: bool
    decision: str
    latency_ms: float
    timestamp: float
