"""FastAPI Unified API Gateway for Multi-Model Serving, A/B Testing, and Async Redis Queue."""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException, Response, status
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from mlops_bits.monitoring import EvidentlyMonitor, InferenceTelemetryCollector
from mlops_bits.monitoring.schemas import EvidentlyDriftSummary, EvidentlyReportRequest
from mlops_bits.schemas import (
    TransactionRequest,
    TransactionResponse,
)
from serving.async_worker.queue import JobQueueManager
from serving.gateway.router import GatewayRouter, RoutingMode

app = FastAPI(
    title="Enterprise ML Model Serving Gateway",
    description="Unified API Gateway with A/B Testing, Canary Rollouts, Multi-Model Routing, Async Redis Predictions, and Evidently AI Observability.",
    version="0.2.0",
)

# Core singletons
router = GatewayRouter()
queue_mgr = JobQueueManager()
telemetry_collector = InferenceTelemetryCollector()
evidently_monitor = EvidentlyMonitor()


class RoutingModeUpdate(BaseModel):
    mode: str = Field(..., description="One of: 'default', 'ab_test', 'canary'")


class CanaryWeightUpdate(BaseModel):
    weight: float = Field(
        ..., ge=0.0, le=1.0, description="Canary traffic fraction [0.0, 1.0]"
    )


class ABTestSplitUpdate(BaseModel):
    champion_ratio: float = Field(
        ..., ge=0.0, le=1.0, description="Champion fraction [0.0, 1.0]"
    )
    experiment_name: str | None = None


class AsyncJobResponse(BaseModel):
    job_id: str
    status: str
    poll_url: str


@app.get("/health", status_code=status.HTTP_200_OK)
def health_check() -> dict[str, Any]:
    """Health check for gateway and backend connectivity."""
    return {
        "status": "healthy",
        "service": "mlops-bits-api-gateway",
        "version": "0.2.0",
        "routing_mode": router.mode.value,
        "redis_connected": queue_mgr.is_redis_connected,
    }


# -----------------------------------------------------------------------------
# 1. Smart Dynamic Prediction Endpoint (A/B Test / Canary / Default)
# -----------------------------------------------------------------------------
@app.post("/v1/predict", response_model=TransactionResponse)
def predict_transaction(
    req: TransactionRequest,
    response: Response,
) -> TransactionResponse:
    """Smart routing inference endpoint with telemetry headers."""
    decision = router.route_and_predict(req)
    if decision.response is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Model prediction failed.",
        )

    # Attach operational headers for client transparency and tracing
    response.headers["X-Model-Variant"] = decision.selected_model
    response.headers["X-Routing-Mode"] = decision.routing_mode.value
    if decision.experiment_group:
        response.headers["X-Experiment-Group"] = decision.experiment_group
    if decision.canary_weight is not None:
        response.headers["X-Canary-Weight"] = str(decision.canary_weight)
    response.headers["X-Inference-Latency-Ms"] = str(
        decision.response.inference_latency_ms
    )

    # Record inference event in live telemetry buffer for Evidently monitoring
    telemetry_collector.record(req, decision.response)

    return decision.response


# -----------------------------------------------------------------------------
# 2. Asynchronous Predictions (Redis-Backed Queue)
# -----------------------------------------------------------------------------
@app.post(
    "/v1/predict/async",
    response_model=AsyncJobResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
def predict_async(req: TransactionRequest) -> AsyncJobResponse:
    """Submit transaction for asynchronous background execution via Redis."""
    job_id = queue_mgr.submit_job(req.model_dump(mode="json"))
    return AsyncJobResponse(
        job_id=job_id,
        status="PENDING",
        poll_url=f"/v1/predict/job/{job_id}",
    )


@app.get("/v1/predict/job/{job_id}", status_code=status.HTTP_200_OK)
def get_prediction_job(job_id: str) -> dict[str, Any]:
    """Poll asynchronous prediction job status and result."""
    job_info = queue_mgr.get_job_status(job_id)
    if not job_info:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job '{job_id}' not found.",
        )
    return job_info


# -----------------------------------------------------------------------------
# 3. Direct Multi-Model Query & Comparison
# -----------------------------------------------------------------------------
@app.post("/v1/models/champion/predict", response_model=TransactionResponse)
def query_champion(req: TransactionRequest) -> TransactionResponse:
    """Directly query the production Champion model."""
    return router.client.predict_champion(req)


@app.post("/v1/models/challenger/predict", response_model=TransactionResponse)
def query_challenger(req: TransactionRequest) -> TransactionResponse:
    """Directly query the candidate Challenger model."""
    return router.client.predict_challenger(req)


@app.post("/v1/models/compare", status_code=status.HTTP_200_OK)
def compare_models(req: TransactionRequest) -> dict[str, Any]:
    """Shadow-score transaction across both models and return comparison."""
    return router.client.compare(req)


# -----------------------------------------------------------------------------
# 4. Traffic Routing & Administration APIs
# -----------------------------------------------------------------------------
@app.get("/v1/admin/routing", status_code=status.HTTP_200_OK)
def get_routing_status() -> dict[str, Any]:
    """Retrieve current routing configuration and traffic telemetry."""
    return router.get_status_report()


@app.post("/v1/admin/routing/mode", status_code=status.HTTP_200_OK)
def set_routing_mode(payload: RoutingModeUpdate) -> dict[str, Any]:
    """Set the routing strategy: 'default', 'ab_test', or 'canary'."""
    try:
        mode_enum = RoutingMode(payload.mode.lower())
        router.set_mode(mode_enum)
        return {"status": "success", "active_mode": mode_enum.value}
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid mode '{payload.mode}'. Valid options: {[m.value for m in RoutingMode]}",
        ) from exc


@app.post("/v1/admin/canary/weight", status_code=status.HTTP_200_OK)
def set_canary_weight(payload: CanaryWeightUpdate) -> dict[str, Any]:
    """Set the fraction of traffic routed to the canary model [0.0 - 1.0]."""
    router.update_canary_weight(payload.weight)
    router.set_mode(RoutingMode.CANARY)
    return {
        "status": "success",
        "active_mode": "canary",
        "canary_weight": payload.weight,
        "champion_weight": round(1.0 - payload.weight, 4),
    }


@app.post("/v1/admin/canary/rollback", status_code=status.HTTP_200_OK)
def trigger_canary_rollback(reason: str = "Manual admin rollback") -> dict[str, Any]:
    """Immediately reset canary weight to 0.0 and restore 100% Champion traffic."""
    router.trigger_rollback(reason=reason)
    return {
        "status": "success",
        "action": "rolled_back_to_champion",
        "active_mode": "default",
        "canary_weight": 0.0,
        "reason": reason,
    }


@app.post("/v1/admin/ab-test/split", status_code=status.HTTP_200_OK)
def set_ab_test_split(payload: ABTestSplitUpdate) -> dict[str, Any]:
    """Set the A/B test split ratio between Champion (A) and Challenger (B)."""
    router.update_ab_split(
        champion_ratio=payload.champion_ratio,
        experiment_name=payload.experiment_name,
    )
    router.set_mode(RoutingMode.AB_TEST)
    return {
        "status": "success",
        "active_mode": "ab_test",
        "champion_ratio_a": payload.champion_ratio,
        "challenger_ratio_b": round(1.0 - payload.champion_ratio, 4),
        "experiment_name": router.ab_config.experiment_name,
    }


# -----------------------------------------------------------------------------
# 5. Evidently AI Observability & Monitoring Endpoints
# -----------------------------------------------------------------------------
@app.post(
    "/v1/monitor/evidently/report",
    response_model=EvidentlyDriftSummary,
    status_code=status.HTTP_200_OK,
)
def generate_evidently_drift_report(
    req: EvidentlyReportRequest | None = None,
) -> EvidentlyDriftSummary:
    """Generate an Evidently Data Drift Report across live Gateway production traffic."""
    from pathlib import Path

    payload = req or EvidentlyReportRequest()
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

    return evidently_monitor.generate_data_drift_report(
        reference_data=ref_df,
        current_data=curr_df,
        save_html=payload.save_html,
        save_json=True,
        sync_workspace=payload.sync_workspace,
    )


@app.get("/v1/monitor/evidently/summary", status_code=status.HTTP_200_OK)
def get_gateway_monitoring_summary() -> dict[str, Any]:
    """Retrieve operational status of Evidently monitoring on the API Gateway."""
    from pathlib import Path

    latest_html = Path(evidently_monitor.config.reports_dir) / "data_drift_latest.html"
    return {
        "status": "ready",
        "live_telemetry_buffered": telemetry_collector.count(),
        "workspace_dir": evidently_monitor.config.workspace_dir,
        "reports_dir": evidently_monitor.config.reports_dir,
        "project_name": evidently_monitor.config.project_name,
        "has_latest_drift_report": latest_html.exists(),
    }


@app.get("/v1/monitor/evidently/report.html", status_code=status.HTTP_200_OK)
def get_latest_gateway_evidently_html() -> FileResponse:
    """Serve the latest interactive Evidently HTML report directly in the browser."""
    from pathlib import Path

    latest_html = Path(evidently_monitor.config.reports_dir) / "data_drift_latest.html"
    if not latest_html.exists():
        ref_df = InferenceTelemetryCollector.load_reference_dataset()
        evidently_monitor.generate_data_drift_report(
            reference_data=ref_df,
            current_data=ref_df,
            save_html=True,
        )

    return FileResponse(
        str(latest_html),
        media_type="text/html",
        filename="evidently_gateway_drift_report.html",
    )
