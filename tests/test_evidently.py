"""Unit and integration tests for Evidently AI monitoring and observability."""

import tempfile
from pathlib import Path

import pandas as pd
import pytest
from starlette.testclient import TestClient

from mlops_bits.api import app
from mlops_bits.cli import main as cli_main
from mlops_bits.monitoring import (
    EvidentlyMonitor,
    InferenceTelemetryCollector,
    MonitoringConfig,
)
from mlops_bits.schemas import (
    RiskDecision,
    TransactionRequest,
    TransactionResponse,
)


@pytest.fixture
def sample_reference_df() -> pd.DataFrame:
    """Fixture providing baseline reference transactions."""
    return pd.DataFrame(
        {
            "amount": [50.0, 75.0, 100.0, 125.0, 150.0, 175.0, 200.0, 225.0],
            "currency": ["USD"] * 8,
            "merchant_category": [
                "grocery",
                "retail",
                "grocery",
                "retail",
                "restaurant",
                "grocery",
                "retail",
                "restaurant",
            ],
            "device_trust_score": [0.90, 0.85, 0.95, 0.88, 0.92, 0.89, 0.94, 0.91],
            "location_country": ["US"] * 8,
            "billing_country": ["US"] * 8,
            "velocity_1h": [1, 2, 1, 1, 2, 1, 2, 1],
        }
    )


@pytest.fixture
def sample_drifted_df() -> pd.DataFrame:
    """Fixture providing current transactions with massive distribution drift."""
    return pd.DataFrame(
        {
            "amount": [
                5000.0,
                7500.0,
                10000.0,
                12500.0,
                15000.0,
                17500.0,
                20000.0,
                25000.0,
            ],
            "currency": ["EUR"] * 8,
            "merchant_category": [
                "crypto",
                "gambling",
                "wire_transfer",
                "crypto",
                "gambling",
                "wire_transfer",
                "crypto",
                "gambling",
            ],
            "device_trust_score": [0.10, 0.15, 0.05, 0.12, 0.08, 0.11, 0.07, 0.09],
            "location_country": ["NG", "RU", "NG", "RU", "NG", "RU", "NG", "RU"],
            "billing_country": ["US"] * 8,
            "velocity_1h": [8, 12, 10, 14, 9, 11, 15, 13],
        }
    )


def test_evidently_monitor_clean_data(sample_reference_df: pd.DataFrame) -> None:
    """Evidently reports no drift when comparing identical or similar distributions."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config = MonitoringConfig(
            reports_dir=str(Path(tmpdir) / "reports"),
            workspace_dir=str(Path(tmpdir) / "workspace"),
            drift_share_threshold=0.30,
        )
        monitor = EvidentlyMonitor(config)

        # Current is a copy with slight variation
        curr = sample_reference_df.copy()
        curr["amount"] = curr["amount"] + 1.0

        summary = monitor.generate_data_drift_report(
            reference_data=sample_reference_df,
            current_data=curr,
            save_html=True,
            save_json=True,
        )

        assert summary.drift_detected is False
        assert summary.drift_share < 0.30
        assert summary.total_features > 0
        assert summary.report_html_path is not None
        assert Path(summary.report_html_path).exists()
        assert summary.report_json_path is not None
        assert Path(summary.report_json_path).exists()


def test_evidently_monitor_drifted_data(
    sample_reference_df: pd.DataFrame, sample_drifted_df: pd.DataFrame
) -> None:
    """Evidently detects statistically significant drift across drifted features."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config = MonitoringConfig(
            reports_dir=str(Path(tmpdir) / "reports"),
            workspace_dir=str(Path(tmpdir) / "workspace"),
            drift_share_threshold=0.30,
        )
        monitor = EvidentlyMonitor(config)

        summary = monitor.generate_data_drift_report(
            reference_data=sample_reference_df,
            current_data=sample_drifted_df,
            save_html=True,
        )

        assert summary.drift_detected is True
        assert summary.drift_share >= 0.30
        assert summary.drifted_features_count >= 2
        assert len(summary.drifted_features) >= 2
        assert any(d.feature_name == "amount" for d in summary.feature_details)


def test_evidently_monitor_data_quality(sample_reference_df: pd.DataFrame) -> None:
    """Evidently generates valid data quality metrics."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config = MonitoringConfig(
            reports_dir=str(Path(tmpdir) / "reports"),
            workspace_dir=str(Path(tmpdir) / "workspace"),
        )
        monitor = EvidentlyMonitor(config)

        quality = monitor.generate_data_quality_report(
            reference_data=sample_reference_df,
            current_data=sample_reference_df,
            save_html=True,
        )

        assert quality.total_rows == len(sample_reference_df)
        assert quality.total_columns == len(sample_reference_df.columns)
        assert quality.missing_values_count == 0
        assert quality.report_html_path is not None
        assert Path(quality.report_html_path).exists()


def test_evidently_automated_test_suite(
    sample_reference_df: pd.DataFrame, sample_drifted_df: pd.DataFrame
) -> None:
    """Evidently TestSuite executes assertions and produces pass/fail summary."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config = MonitoringConfig(
            reports_dir=str(Path(tmpdir) / "reports"),
            workspace_dir=str(Path(tmpdir) / "workspace"),
        )
        monitor = EvidentlyMonitor(config)

        suite_summary = monitor.run_drift_test_suite(
            reference_data=sample_reference_df,
            current_data=sample_drifted_df,
            save_html=True,
        )

        assert suite_summary.total_tests > 0
        assert len(suite_summary.test_details) > 0
        assert suite_summary.report_html_path is not None
        assert Path(suite_summary.report_html_path).exists()
        # With massive drift, all_passed should be False
        assert suite_summary.all_passed is False
        assert suite_summary.failed_tests > 0


def test_evidently_workspace_run_sync(sample_reference_df: pd.DataFrame) -> None:
    """Evidently snapshots are synced and queryable in workspace projects."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config = MonitoringConfig(
            reports_dir=str(Path(tmpdir) / "reports"),
            workspace_dir=str(Path(tmpdir) / "workspace"),
            project_name="Test Workspace Sync",
        )
        monitor = EvidentlyMonitor(config)

        summary = monitor.generate_data_drift_report(
            reference_data=sample_reference_df,
            current_data=sample_reference_df,
            sync_workspace=True,
        )

        assert summary.snapshot_id is not None
        runs = monitor.workspace.list_runs(monitor.project.id)
        assert len(runs) >= 1


def test_inference_telemetry_collector() -> None:
    """InferenceTelemetryCollector stores and exports live telemetry events."""
    collector = InferenceTelemetryCollector(MonitoringConfig(window_size=5))
    assert collector.count() == 0

    tx = TransactionRequest(
        transaction_id="tx-test-1",
        user_id="usr-1",
        amount=99.95,
        currency="USD",
        merchant_category="retail",
        device_trust_score=0.92,
        location_country="US",
        billing_country="US",
        velocity_1h=1,
    )
    res = TransactionResponse(
        transaction_id="tx-test-1",
        is_fraud=False,
        fraud_score=0.02,
        decision=RiskDecision.APPROVE,
        risk_factors=[],
        inference_latency_ms=0.003,
    )

    collector.record(tx, res)
    assert collector.count() == 1

    df = collector.get_current_dataframe()
    assert len(df) == 1
    assert df.loc[0, "transaction_id"] == "tx-test-1"
    assert df.loc[0, "amount"] == 99.95
    assert df.loc[0, "fraud_score"] == 0.02

    # Ring buffer overflow test
    for i in range(10):
        tx_i = tx.model_copy(update={"transaction_id": f"tx-over-{i}"})
        collector.record(tx_i, res)

    assert collector.count() == 5  # Capped at window_size
    collector.clear()
    assert collector.count() == 0


def test_api_evidently_endpoints() -> None:
    """FastAPI Evidently monitoring endpoints respond correctly."""
    client = TestClient(app)

    # 1. Prediction logs telemetry
    tx_payload = {
        "transaction_id": "api-tx-001",
        "user_id": "user-123",
        "amount": 125.0,
        "currency": "USD",
        "merchant_category": "electronics",
        "device_trust_score": 0.90,
        "location_country": "US",
        "billing_country": "US",
        "velocity_1h": 2,
    }
    pred_res = client.post("/v1/predict", json=tx_payload)
    assert pred_res.status_code == 200

    # 2. Monitoring summary
    sum_res = client.get("/v1/monitor/evidently/summary")
    assert sum_res.status_code == 200
    sum_data = sum_res.json()
    assert sum_data["status"] == "ready"
    assert sum_data["live_telemetry_buffered"] >= 1

    # 3. Generate drift report
    report_res = client.post(
        "/v1/monitor/evidently/report", json={"include_quality": False}
    )
    assert report_res.status_code == 200
    report_data = report_res.json()
    assert "drift_detected" in report_data
    assert "drift_share" in report_data
    assert "total_features" in report_data

    # 4. Run test suite
    suite_res = client.post("/v1/monitor/evidently/test-suite", json={})
    assert suite_res.status_code == 200
    suite_data = suite_res.json()
    assert "all_passed" in suite_data
    assert "total_tests" in suite_data

    # 5. Serve HTML dashboard
    html_res = client.get("/v1/monitor/evidently/report.html")
    assert html_res.status_code == 200
    assert "text/html" in html_res.headers["content-type"]
    assert len(html_res.content) > 1000


def test_cli_monitor_commands() -> None:
    """CLI monitor report and test subcommands execute properly."""
    # Test report command
    with tempfile.TemporaryDirectory() as tmpdir:
        cli_main(
            [
                "monitor",
                "report",
                "--output-dir",
                str(Path(tmpdir) / "reports"),
                "--workspace",
                str(Path(tmpdir) / "workspace"),
            ]
        )
        assert (Path(tmpdir) / "reports" / "data_drift_latest.html").exists()

    # Test test command
    with tempfile.TemporaryDirectory() as tmpdir:
        # Should not raise exception
        cli_main(["monitor", "test"])
