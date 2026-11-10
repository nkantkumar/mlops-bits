"""Tests for MLflow experiment tracking and model benchmarking."""

import tempfile
from pathlib import Path

import mlflow
import pandas as pd

from experiments.run_experiment import (
    engineer_features,
    load_or_generate_dataset,
    run_all_experiments,
)


def test_engineer_features() -> None:
    """Feature engineering creates required numerical and categorical features."""
    raw_data = [
        {
            "transaction_id": "tx-001",
            "user_id": "u-1",
            "amount": 10500.0,
            "currency": "USD",
            "merchant_category": "crypto",
            "device_trust_score": 0.15,
            "location_country": "NG",
            "billing_country": "US",
            "velocity_1h": 12,
        },
        {
            "transaction_id": "tx-002",
            "user_id": "u-2",
            "amount": 45.0,
            "currency": "USD",
            "merchant_category": "grocery",
            "device_trust_score": 0.95,
            "location_country": "US",
            "billing_country": "US",
            "velocity_1h": 2,
        },
    ]
    df = pd.DataFrame(raw_data)
    feats = engineer_features(df)

    assert feats.loc[0, "country_mismatch"] == 1
    assert feats.loc[1, "country_mismatch"] == 0
    assert feats.loc[0, "is_high_risk_category"] == 1
    assert feats.loc[1, "is_high_risk_category"] == 0
    assert feats.loc[0, "is_extreme_amount"] == 1
    assert feats.loc[1, "is_extreme_amount"] == 0
    assert feats.loc[0, "is_untrusted_device"] == 1
    assert feats.loc[1, "is_untrusted_device"] == 0
    assert feats.loc[0, "is_critical_velocity"] == 1
    assert feats.loc[1, "is_critical_velocity"] == 0


def test_dataset_generation_and_labels() -> None:
    """Dataset generator deterministically produces balanced/stratified labels."""
    df = load_or_generate_dataset(
        data_path=None, num_samples=100, fraud_ratio=0.10, seed=42
    )
    assert len(df) == 100
    assert "is_fraud" in df.columns
    fraud_count = df["is_fraud"].sum()
    assert 5 <= fraud_count <= 15


def test_mlflow_experiment_logging() -> None:
    """run_all_experiments creates MLflow experiment and logs runs, metrics, and models."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tracking_db = Path(tmpdir) / "test_mlflow.db"
        tracking_uri = f"sqlite:///{tracking_db}"
        exp_name = "test-fraud-experiment"

        summaries = run_all_experiments(
            experiment_name=exp_name,
            tracking_uri=tracking_uri,
            data_path=None,  # Generates deterministic test set
            num_samples=120,
            fraud_ratio=0.15,
            seed=42,
            test_size=0.30,
            models=["detector", "logistic_regression", "random_forest"],
        )

        assert len(summaries) == 3

        # Verify each summary has valid metrics
        for s in summaries:
            assert s.run_id is not None
            assert s.roc_auc >= 0.70
            assert s.f1_score >= 0.50
            assert s.mean_latency_ms >= 0.0
            assert s.p99_latency_ms > 0.0
            if s.model_family == "expert_calibrated_engine":
                assert s.sla_passed is True
                assert s.p99_latency_ms < 1.0  # Sub-millisecond SLA

        # Verify against MLflow client
        client = mlflow.tracking.MlflowClient(tracking_uri=tracking_uri)
        experiment = client.get_experiment_by_name(exp_name)
        assert experiment is not None

        runs = client.search_runs(experiment_ids=[experiment.experiment_id])
        assert len(runs) == 3

        # Verify artifacts exist in run
        run0 = runs[0]
        artifacts = client.list_artifacts(run0.info.run_id)
        artifact_paths = [a.path for a in artifacts]
        assert (
            "diagnostic_plots" in artifact_paths
            or "reports" in artifact_paths
            or "model" in artifact_paths
        )
