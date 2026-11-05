"""Configuration and threshold definitions for fraud detection and MLOps monitoring."""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class FraudDetectorConfig:
    """Production decision thresholds and inference weights for fraud scoring."""

    # Decision boundaries
    reject_threshold: float = 0.75
    review_threshold: float = 0.40

    # Boundary amounts
    extreme_amount_threshold: float = 10_000.0  # Instant review/risk flag
    high_velocity_threshold: int = 5  # Transactions per hour triggering velocity alert
    critical_velocity_threshold: int = (
        10  # Extreme velocity triggering immediate rejection
    )

    # Device trust boundaries
    untrusted_device_threshold: float = 0.30

    # High risk merchant categories (MCC / string tokens)
    high_risk_categories: set[str] = field(
        default_factory=lambda: {
            "crypto",
            "gambling",
            "wire_transfer",
            "cash_advance",
            "prepaid_cards",
        }
    )

    # Inference latency SLA (milliseconds)
    target_sla_p99_ms: float = 5.0
    target_sla_mean_ms: float = 1.0


@dataclass(frozen=True)
class DriftConfig:
    """Thresholds for Population Stability Index (PSI) and statistical monitoring."""

    psi_warning_threshold: float = 0.10
    psi_critical_threshold: float = 0.25
    epsilon: float = 1e-4  # Zero-division smoothing factor for bin probabilities
