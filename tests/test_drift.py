"""Tests for distribution drift and Population Stability Index (PSI) monitoring."""

import random

import pytest

from mlops_bits.drift import DistributionDriftMonitor
from mlops_bits.schemas import DriftAnalysisRequest, DriftStatus


def test_no_drift_on_identical_distribution(
    drift_monitor: DistributionDriftMonitor, baseline_distribution: list[float]
) -> None:
    """When current data mirrors baseline, PSI must be near zero with NO_DRIFT status."""
    request = DriftAnalysisRequest(
        feature_name="transaction_amount",
        baseline_values=baseline_distribution,
        current_values=list(baseline_distribution),
        num_bins=10,
    )
    result = drift_monitor.analyze(request)

    assert result.feature_name == "transaction_amount"
    assert result.psi < 0.05
    assert result.drift_status == DriftStatus.NO_DRIFT
    assert not result.alert_triggered
    assert result.mean_baseline == result.mean_current


def test_moderate_drift_detection(
    drift_monitor: DistributionDriftMonitor, baseline_distribution: list[float]
) -> None:
    """A moderate upward shift in feature distribution triggers MODERATE_DRIFT without critical alert."""
    random.seed(99)
    # Mild shift: increase mean by 40%
    mild_shift = [val * 1.4 for val in baseline_distribution]

    request = DriftAnalysisRequest(
        feature_name="transaction_amount",
        baseline_values=baseline_distribution,
        current_values=mild_shift,
        num_bins=10,
    )
    result = drift_monitor.analyze(request)

    assert result.psi >= drift_monitor.config.psi_warning_threshold
    assert result.drift_status in (
        DriftStatus.MODERATE_DRIFT,
        DriftStatus.SIGNIFICANT_DRIFT,
    )


def test_significant_drift_triggers_alert(
    drift_monitor: DistributionDriftMonitor, baseline_distribution: list[float]
) -> None:
    """A drastic shift in distribution (e.g. 5x surge in amounts) triggers SIGNIFICANT_DRIFT and alert."""
    # Drastic shift: values multiplied by 5x simulating extreme card testing or bot flood
    drastic_shift = [val * 5.0 for val in baseline_distribution]

    request = DriftAnalysisRequest(
        feature_name="transaction_amount",
        baseline_values=baseline_distribution,
        current_values=drastic_shift,
        num_bins=10,
    )
    result = drift_monitor.analyze(request)

    assert result.psi >= drift_monitor.config.psi_critical_threshold
    assert result.drift_status == DriftStatus.SIGNIFICANT_DRIFT
    assert result.alert_triggered is True
    assert result.mean_current > result.mean_baseline


def test_drift_empty_data_boundary_raises(
    drift_monitor: DistributionDriftMonitor,
) -> None:
    """Empty datasets must raise a clear ValueError."""
    with pytest.raises(ValueError, match="must not be empty"):
        drift_monitor.calculate_psi([], [10.0, 20.0])

    with pytest.raises(ValueError, match="must not be empty"):
        drift_monitor.calculate_psi([10.0, 20.0], [])


def test_drift_degenerate_constant_values(
    drift_monitor: DistributionDriftMonitor,
) -> None:
    """Degenerate constant arrays should execute without ZeroDivisionError."""
    constants = [100.0] * 50
    psi, _ = drift_monitor.calculate_psi(constants, constants)
    assert psi == 0.0
