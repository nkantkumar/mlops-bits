from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, status
from fastapi.responses import FileResponse

from mlops_bits.detector import LowLatencyFraudDetector
from mlops_bits.drift import DistributionDriftMonitor
from mlops_bits.monitoring.collector import InferenceTelemetryCollector
from mlops_bits.monitoring.schemas import (
    EvidentlyDriftSummary,
    EvidentlyReportRequest,
    EvidentlyTestSuiteSummary,
    InferenceTelemetryRecord,
)
from mlops_bits.monitoring.service import EvidentlyMonitor
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
    description="Production-grade low-latency fraud detection module, statistical drift monitoring, and Evidently AI observability for MLOps.",
    version="0.2.0",
)

# In-memory singletons pre-initialized for zero-warmup latency
detector = LowLatencyFraudDetector()
drift_monitor = DistributionDriftMonitor()
telemetry_collector = InferenceTelemetryCollector()
evidently_monitor = EvidentlyMonitor()


@app.get("/health", status_code=status.HTTP_200_OK)
def health_check() -> dict[str, Any]:
    """Health and readiness probe."""
    return {
        "status": "healthy",
        "service": "mlops-bits-fraud-detector",
        "version": "0.2.0",
        "sla_p99_target_ms": detector.config.target_sla_p99_ms,
        "telemetry_events_buffered": telemetry_collector.count(),
        "evidently_project": evidently_monitor.config.project_name,
    }


@app.post(
    "/v1/predict", response_model=TransactionResponse, status_code=status.HTTP_200_OK
)
def score_transaction(request: TransactionRequest) -> TransactionResponse:
    """Low-latency inference endpoint to score an incoming financial transaction."""
    try:
        res = detector.predict(request)
        telemetry_collector.record(request, res)
        return res
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
    batch_resp = detector.predict_batch(batch_request)
    for tx, res in zip(batch_request.transactions, batch_resp.results):
        telemetry_collector.record(tx, res)
    return batch_resp


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


# -----------------------------------------------------------------------------
# Evidently AI Observability & Monitoring Endpoints
# -----------------------------------------------------------------------------
@app.post(
    "/v1/monitor/evidently/report",
    response_model=EvidentlyDriftSummary,
    status_code=status.HTTP_200_OK,
)
def generate_evidently_drift_report(
    req: EvidentlyReportRequest | None = None,
) -> EvidentlyDriftSummary:
    """Generate an Evidently Data Drift Report across production inference features."""
    payload = req or EvidentlyReportRequest()
    try:
        ref_df = (
            InferenceTelemetryCollector.load_reference_dataset(payload.reference_path)
            if payload.reference_path
            else InferenceTelemetryCollector.load_reference_dataset()
        )

        curr_df = (
            InferenceTelemetryCollector.load_reference_dataset(payload.current_path)
            if payload.current_path
            else telemetry_collector.get_current_dataframe()
        )

        if curr_df.empty:
            # Fallback to current reference with slight noise if buffer is warming up
            curr_df = ref_df.copy()

        return evidently_monitor.generate_data_drift_report(
            reference_data=ref_df,
            current_data=curr_df,
            save_html=payload.save_html,
            save_json=True,
            sync_workspace=payload.sync_workspace,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Evidently report generation failed: {exc}",
        ) from exc


@app.post(
    "/v1/monitor/evidently/test-suite",
    response_model=EvidentlyTestSuiteSummary,
    status_code=status.HTTP_200_OK,
)
def run_evidently_test_suite(
    req: EvidentlyReportRequest | None = None,
) -> EvidentlyTestSuiteSummary:
    """Execute an automated Evidently Test Suite for deployment verification and CI/CD gating."""
    payload = req or EvidentlyReportRequest()
    try:
        ref_df = (
            InferenceTelemetryCollector.load_reference_dataset(payload.reference_path)
            if payload.reference_path
            else InferenceTelemetryCollector.load_reference_dataset()
        )

        curr_df = (
            InferenceTelemetryCollector.load_reference_dataset(payload.current_path)
            if payload.current_path
            else telemetry_collector.get_current_dataframe()
        )

        if curr_df.empty:
            curr_df = ref_df.copy()

        return evidently_monitor.run_drift_test_suite(
            reference_data=ref_df,
            current_data=curr_df,
            save_html=payload.save_html,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Evidently test suite failed: {exc}",
        ) from exc


@app.get("/v1/monitor/evidently/summary", status_code=status.HTTP_200_OK)
def get_monitoring_summary() -> dict[str, Any]:
    """Retrieve operational status of the Evidently monitoring subsystem."""
    latest_html = Path(evidently_monitor.config.reports_dir) / "data_drift_latest.html"
    latest_suite = Path(evidently_monitor.config.reports_dir) / "test_suite_latest.html"
    return {
        "status": "ready",
        "live_telemetry_buffered": telemetry_collector.count(),
        "workspace_dir": evidently_monitor.config.workspace_dir,
        "reports_dir": evidently_monitor.config.reports_dir,
        "project_name": evidently_monitor.config.project_name,
        "drift_share_threshold": evidently_monitor.config.drift_share_threshold,
        "has_latest_drift_report": latest_html.exists(),
        "has_latest_test_suite": latest_suite.exists(),
    }


@app.get("/v1/monitor/evidently/report.html", status_code=status.HTTP_200_OK)
def get_latest_evidently_html() -> FileResponse:
    """Serve the latest interactive Evidently HTML report directly in the browser."""
    latest_html = Path(evidently_monitor.config.reports_dir) / "data_drift_latest.html"
    if not latest_html.exists():
        # Generate on-demand if not already generated
        ref_df = InferenceTelemetryCollector.load_reference_dataset()
        evidently_monitor.generate_data_drift_report(
            reference_data=ref_df,
            current_data=ref_df,
            save_html=True,
        )

    return FileResponse(
        str(latest_html),
        media_type="text/html",
        filename="evidently_drift_report.html",
    )


@app.post(
    "/v1/monitor/evidently/collect",
    response_model=InferenceTelemetryRecord,
    status_code=status.HTTP_201_CREATED,
)
def record_telemetry(
    request: TransactionRequest, response: TransactionResponse
) -> InferenceTelemetryRecord:
    """Manually push scored transaction telemetry into the live monitoring buffer."""
    return telemetry_collector.record(request, response)
