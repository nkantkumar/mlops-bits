"""Production Evidently monitoring service for data drift, quality, and automated test suites."""

from __future__ import annotations

import datetime
import logging
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

import pandas as pd
from evidently import Report
from evidently.legacy.test_preset import DataDriftTestPreset
from evidently.legacy.test_suite import TestSuite
from evidently.presets import DataDriftPreset, DataSummaryPreset
from evidently.ui.workspace import Workspace

from mlops_bits.monitoring.collector import InferenceTelemetryCollector
from mlops_bits.monitoring.config import MonitoringConfig
from mlops_bits.monitoring.schemas import (
    EvidentlyDriftSummary,
    EvidentlyQualitySummary,
    EvidentlyTestDetail,
    EvidentlyTestSuiteSummary,
    FeatureDriftDetail,
)
from mlops_bits.schemas import TransactionRequest

# Default features evaluated for fraud drift
FEATURE_COLUMNS = [
    "amount",
    "currency",
    "merchant_category",
    "device_trust_score",
    "location_country",
    "billing_country",
    "velocity_1h",
]


class EvidentlyMonitor:
    """Full-fledged enterprise monitoring engine powered by Evidently AI."""

    def __init__(self, config: MonitoringConfig | None = None) -> None:
        self.config = config or MonitoringConfig()
        self.config.ensure_directories()
        self._workspace: Workspace | None = None
        self._project_id: str | None = None

    @property
    def workspace(self) -> Workspace:
        """Lazily initialize and return the Evidently UI Workspace."""
        if self._workspace is None:
            self._workspace = Workspace.create(self.config.workspace_dir)
        return self._workspace

    @property
    def project(self) -> Any:
        """Get or create the Evidently workspace project."""
        ws = self.workspace
        projects = ws.list_projects()
        for p in projects:
            if p.name == self.config.project_name:
                self._project_id = str(p.id)
                return p
        new_project = ws.create_project(self.config.project_name)
        self._project_id = str(new_project.id)
        return new_project

    def _prepare_dataframe(
        self,
        data: pd.DataFrame
        | Sequence[TransactionRequest]
        | Sequence[dict[str, Any]]
        | str
        | Path,
    ) -> pd.DataFrame:
        """Convert heterogeneous input sources into a cleaned pandas DataFrame."""
        if isinstance(data, (str, Path)):
            df = pd.read_csv(data)
        elif isinstance(data, pd.DataFrame):
            df = data.copy()
        elif isinstance(data, Sequence):
            if not data:
                return pd.DataFrame()
            if isinstance(data[0], TransactionRequest):
                df = pd.DataFrame([tx.model_dump() for tx in data])  # type: ignore[union-attr]
            elif isinstance(data[0], dict):
                df = pd.DataFrame(data)
            else:
                df = pd.DataFrame(list(data))
        else:
            raise TypeError(
                f"Unsupported data type for Evidently monitoring: {type(data)}"
            )

        # Keep feature columns if present, otherwise keep all numeric/categorical columns
        cols_to_keep = [col for col in FEATURE_COLUMNS if col in df.columns]
        if cols_to_keep:
            # Also keep target/prediction columns if they exist
            extra_cols = [
                c for c in ["fraud_score", "is_fraud", "decision"] if c in df.columns
            ]
            df = df[cols_to_keep + extra_cols]
        else:
            # Fallback: drop purely unique identifiers if present
            drop_cols = [
                c for c in ["transaction_id", "user_id", "timestamp"] if c in df.columns
            ]
            if drop_cols:
                df = df.drop(columns=drop_cols)

        # Explicitly cast categorical columns to 'category' dtype to prevent NLP vectorizer parsing
        cat_cols = [
            "currency",
            "merchant_category",
            "location_country",
            "billing_country",
            "decision",
        ]
        for c in cat_cols:
            if c in df.columns:
                df[c] = df[c].astype("category")

        return df

    def _align_dataframes(
        self,
        ref_df: pd.DataFrame,
        curr_df: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Align columns to only those common to both datasets."""
        common_cols = [c for c in ref_df.columns if c in curr_df.columns]
        if not common_cols:
            raise ValueError(
                "No common columns found between reference and current datasets."
            )
        return ref_df[common_cols].copy(), curr_df[common_cols].copy()

    def generate_data_drift_report(
        self,
        reference_data: pd.DataFrame | Sequence[Any] | str | Path,
        current_data: pd.DataFrame | Sequence[Any] | str | Path,
        save_html: bool = True,
        save_json: bool = True,
        sync_workspace: bool = True,
    ) -> EvidentlyDriftSummary:
        """Generate an Evidently Data Drift Report comparing reference and current datasets."""
        ref_df = self._prepare_dataframe(reference_data)
        curr_df = self._prepare_dataframe(current_data)

        if ref_df.empty or curr_df.empty:
            raise ValueError(
                "Reference and current datasets must not be empty for drift analysis."
            )

        ref_df, curr_df = self._align_dataframes(ref_df, curr_df)

        # Build Evidently report with configured drift parameters
        report = Report(
            [
                DataDriftPreset(
                    drift_share=self.config.drift_share_threshold,
                    threshold=self.config.stat_test_threshold,
                )
            ]
        )
        snapshot = report.run(current_data=curr_df, reference_data=ref_df)

        # Parse metrics from snapshot
        metrics_dict = snapshot.dict().get("metrics", [])
        drift_share = 0.0
        drifted_count = 0
        total_features = 0
        feature_details: list[FeatureDriftDetail] = []
        drifted_features: list[str] = []

        for m in metrics_dict:
            metric_name = m.get("metric_name", "")
            val = m.get("value")

            if "DriftedColumnsCount" in metric_name and isinstance(val, dict):
                drift_share = float(val.get("share", 0.0))
                drifted_count = int(val.get("count", 0))

            elif "ValueDrift" in metric_name:
                # Parse column name and method from string like 'ValueDrift(column=amount,method=K-S p_value,threshold=0.05)'
                col_match = re.search(r"column=([a-zA-Z0-9_]+)", metric_name)
                method_match = re.search(r"method=([^,]+)", metric_name)
                col_name = col_match.group(1) if col_match else "unknown"
                method_name = (
                    method_match.group(1) if method_match else "statistical_test"
                )

                drift_score = float(val) if val is not None else 0.0
                # A p-value <= threshold means reject null hypothesis -> drift detected
                is_drifted = drift_score <= self.config.stat_test_threshold
                if is_drifted:
                    drifted_features.append(col_name)

                feature_details.append(
                    FeatureDriftDetail(
                        feature_name=col_name,
                        drift_detected=is_drifted,
                        drift_score=round(drift_score, 6),
                        stat_test=method_name,
                        threshold=self.config.stat_test_threshold,
                    )
                )

        total_features = len(feature_details)
        if total_features > 0 and drifted_count == 0:
            drifted_count = len(drifted_features)
            drift_share = round(drifted_count / total_features, 4)

        drift_detected = drift_share >= self.config.drift_share_threshold

        timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        reports_dir = Path(self.config.reports_dir)
        reports_dir.mkdir(parents=True, exist_ok=True)

        html_path: str | None = None
        json_path: str | None = None

        if save_html:
            report_file = reports_dir / f"data_drift_{timestamp_str}.html"
            snapshot.save_html(str(report_file))
            # Also save latest alias
            latest_file = reports_dir / "data_drift_latest.html"
            snapshot.save_html(str(latest_file))
            html_path = str(report_file)

        if save_json:
            json_file = reports_dir / f"data_drift_{timestamp_str}.json"
            snapshot.save_json(str(json_file))
            latest_json = reports_dir / "data_drift_latest.json"
            snapshot.save_json(str(latest_json))
            json_path = str(json_file)

        snapshot_id: str | None = None
        if sync_workspace:
            try:
                proj = self.project
                ref = self.workspace.add_run(proj.id, snapshot)
                snapshot_id = str(ref.id)
            except (OSError, RuntimeError, ValueError) as exc:
                logger.warning("Could not sync snapshot to workspace: %s", exc)

        return EvidentlyDriftSummary(
            drift_detected=drift_detected,
            drift_share=round(drift_share, 4),
            drifted_features_count=drifted_count,
            total_features=total_features,
            drifted_features=drifted_features,
            feature_details=feature_details,
            report_html_path=html_path,
            report_json_path=json_path,
            snapshot_id=snapshot_id,
            timestamp=timestamp_str,
        )

    def generate_data_quality_report(
        self,
        reference_data: pd.DataFrame | Sequence[Any] | str | Path,
        current_data: pd.DataFrame | Sequence[Any] | str | Path,
        save_html: bool = True,
        save_json: bool = True,
        sync_workspace: bool = True,
    ) -> EvidentlyQualitySummary:
        """Generate an Evidently Data Quality and Data Summary Report."""
        ref_df = self._prepare_dataframe(reference_data)
        curr_df = self._prepare_dataframe(current_data)
        ref_df, curr_df = self._align_dataframes(ref_df, curr_df)

        report = Report([DataSummaryPreset()])
        snapshot = report.run(current_data=curr_df, reference_data=ref_df)

        metrics_dict = snapshot.dict().get("metrics", [])
        total_rows = len(curr_df)
        total_cols = len(curr_df.columns)
        missing_count = 0
        missing_share = 0.0

        for m in metrics_dict:
            metric_name = m.get("metric_name", "")
            val = m.get("value")
            if "DatasetMissingValueCount" in metric_name and isinstance(val, dict):
                missing_count = int(val.get("count", 0))
                missing_share = float(val.get("share", 0.0))

        timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        reports_dir = Path(self.config.reports_dir)
        reports_dir.mkdir(parents=True, exist_ok=True)

        html_path: str | None = None
        json_path: str | None = None

        if save_html:
            report_file = reports_dir / f"data_quality_{timestamp_str}.html"
            snapshot.save_html(str(report_file))
            latest_file = reports_dir / "data_quality_latest.html"
            snapshot.save_html(str(latest_file))
            html_path = str(report_file)

        if save_json:
            json_file = reports_dir / f"data_quality_{timestamp_str}.json"
            snapshot.save_json(str(json_file))
            latest_json = reports_dir / "data_quality_latest.json"
            snapshot.save_json(str(latest_json))
            json_path = str(json_file)

        if sync_workspace:
            try:
                proj = self.project
                self.workspace.add_run(proj.id, snapshot)
            except (OSError, RuntimeError, ValueError) as exc:
                logger.warning("Could not sync quality snapshot to workspace: %s", exc)

        return EvidentlyQualitySummary(
            total_rows=total_rows,
            total_columns=total_cols,
            missing_values_count=missing_count,
            missing_values_share=round(missing_share, 4),
            report_html_path=html_path,
            report_json_path=json_path,
        )

    def run_drift_test_suite(
        self,
        reference_data: pd.DataFrame | Sequence[Any] | str | Path,
        current_data: pd.DataFrame | Sequence[Any] | str | Path,
        save_html: bool = True,
    ) -> EvidentlyTestSuiteSummary:
        """Run an automated Evidently Test Suite for CI/CD gates and deployment checks."""
        ref_df = self._prepare_dataframe(reference_data)
        curr_df = self._prepare_dataframe(current_data)
        ref_df, curr_df = self._align_dataframes(ref_df, curr_df)

        suite = TestSuite(
            tests=[
                DataDriftTestPreset(drift_share=self.config.drift_share_threshold),
            ]
        )
        suite.run(current_data=curr_df, reference_data=ref_df)

        result_dict = suite.as_dict()
        summary = result_dict.get("summary", {})
        tests_list = result_dict.get("tests", [])

        all_passed = bool(summary.get("all_passed", False))
        total_tests = int(summary.get("total_tests", len(tests_list)))
        passed_tests = int(summary.get("success_tests", 0))
        failed_tests = int(summary.get("failed_tests", 0))

        test_details: list[EvidentlyTestDetail] = []
        for t in tests_list:
            test_details.append(
                EvidentlyTestDetail(
                    name=t.get("name", "Test"),
                    status=t.get("status", "UNKNOWN"),
                    description=t.get("description", ""),
                )
            )

        timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        reports_dir = Path(self.config.reports_dir)
        reports_dir.mkdir(parents=True, exist_ok=True)

        html_path: str | None = None
        if save_html:
            report_file = reports_dir / f"test_suite_{timestamp_str}.html"
            suite.save_html(str(report_file))
            latest_file = reports_dir / "test_suite_latest.html"
            suite.save_html(str(latest_file))
            html_path = str(report_file)

        return EvidentlyTestSuiteSummary(
            all_passed=all_passed,
            total_tests=total_tests,
            passed_tests=passed_tests,
            failed_tests=failed_tests,
            test_details=test_details,
            report_html_path=html_path,
        )

    def generate_live_drift_report(
        self,
        collector: InferenceTelemetryCollector,
        reference_path: str | Path = "data/raw/transactions.csv",
        min_samples: int = 10,
    ) -> EvidentlyDriftSummary:
        """Run drift report against live inference telemetry collected in the ring buffer."""
        curr_df = collector.get_current_dataframe()
        if len(curr_df) < min_samples:
            raise ValueError(
                f"Insufficient live telemetry samples ({len(curr_df)}). Need at least {min_samples}."
            )

        ref_df = collector.load_reference_dataset(reference_path)
        return self.generate_data_drift_report(
            reference_data=ref_df,
            current_data=curr_df,
            save_html=True,
            save_json=True,
            sync_workspace=True,
        )
