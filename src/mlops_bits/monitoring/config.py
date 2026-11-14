"""Configuration for Evidently monitoring and observability."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field


class MonitoringConfig(BaseModel):
    """Configuration parameters for Evidently data drift and quality monitoring."""

    workspace_dir: str = Field(
        default="workspace",
        description="Directory path for Evidently UI workspace and snapshots",
    )
    project_name: str = Field(
        default="Fraud Detection Observability",
        description="Evidently project name in the workspace",
    )
    reports_dir: str = Field(
        default="reports/evidently",
        description="Directory to save standalone HTML and JSON reports",
    )
    drift_share_threshold: float = Field(
        default=0.30,
        ge=0.0,
        le=1.0,
        description="Share of drifted features that triggers a dataset-level drift alert",
    )
    stat_test_threshold: float = Field(
        default=0.05,
        ge=0.0,
        le=1.0,
        description="Significance level (p-value threshold) for statistical tests",
    )
    num_stat_test: str = Field(
        default="ks",
        description="Statistical test for numerical features: 'ks', 'wasserstein', 'psi'",
    )
    cat_stat_test: str = Field(
        default="chisquare",
        description="Statistical test for categorical features: 'chisquare', 'psi'",
    )
    window_size: int = Field(
        default=1000,
        ge=1,
        description="Maximum capacity of the sliding live inference telemetry buffer",
    )

    def ensure_directories(self) -> None:
        """Create necessary directories for workspace and reports."""
        Path(self.workspace_dir).mkdir(parents=True, exist_ok=True)
        Path(self.reports_dir).mkdir(parents=True, exist_ok=True)
