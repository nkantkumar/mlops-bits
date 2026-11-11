"""Tests for advanced model serving, API Gateway, A/B testing, Canary rollout, and Async Redis worker."""

import pytest
from starlette.testclient import TestClient

from serving.async_worker.queue import JobQueueManager
from serving.async_worker.worker import AsyncPredictionWorker
from serving.bentoml.service import FraudDetectionService
from serving.gateway.app import app, queue_mgr, router
from serving.gateway.router import RoutingMode


@pytest.fixture
def sample_tx() -> dict:
    return {
        "transaction_id": "tx-test-100",
        "user_id": "usr-test-999",
        "amount": 125.0,
        "currency": "USD",
        "merchant_category": "retail",
        "device_trust_score": 0.95,
        "location_country": "US",
        "billing_country": "US",
        "velocity_1h": 1,
    }


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


# -----------------------------------------------------------------------------
# 1. BentoML Service Tests
# -----------------------------------------------------------------------------
def test_bentoml_service_execution(sample_tx: dict) -> None:
    """BentoML service initializes and scores transactions cleanly."""
    svc = FraudDetectionService()

    # Health
    health = svc.health()
    assert health["status"] == "healthy"
    assert health["framework"] == "bentoml"

    # Single prediction
    res = svc.predict(sample_tx)
    assert res["transaction_id"] == "tx-test-100"
    assert "decision" in res
    assert "fraud_score" in res
    assert res["inference_latency_ms"] >= 0.0

    # Batch prediction
    batch_res = svc.predict_batch({"transactions": [sample_tx, sample_tx]})
    assert batch_res["total_processed"] == 2
    assert len(batch_res["results"]) == 2


# -----------------------------------------------------------------------------
# 2. Async Queue & Worker Tests
# -----------------------------------------------------------------------------
def test_async_prediction_worker_flow(sample_tx: dict) -> None:
    """Async worker pulls job from queue, scores it, and updates job result."""
    qm = JobQueueManager()
    worker = AsyncPredictionWorker(queue_manager=qm)

    job_id = qm.submit_job(sample_tx)
    status_pending = qm.get_job_status(job_id)
    assert status_pending is not None
    assert status_pending["status"] == "PENDING"

    # Worker consumes and processes the job
    processed = worker.process_one_job()
    assert processed is True

    status_completed = qm.get_job_status(job_id)
    assert status_completed is not None
    assert status_completed["status"] == "COMPLETED"
    assert status_completed["result"]["transaction_id"] == "tx-test-100"
    assert "decision" in status_completed["result"]


# -----------------------------------------------------------------------------
# 3. Gateway Smart Routing & A/B Testing Tests
# -----------------------------------------------------------------------------
def test_gateway_default_routing(client: TestClient, sample_tx: dict) -> None:
    """Default gateway routing directs 100% of traffic to Champion."""
    router.set_mode(RoutingMode.DEFAULT)
    resp = client.post("/v1/predict", json=sample_tx)
    assert resp.status_code == 200
    assert resp.headers["X-Model-Variant"] == "champion"
    assert resp.headers["X-Routing-Mode"] == "default"


def test_gateway_ab_testing_deterministic_cohort(client: TestClient) -> None:
    """A/B testing deterministically routes the same user to the same cohort."""
    router.set_mode(RoutingMode.AB_TEST)
    router.update_ab_split(champion_ratio=0.50)

    user_a = "usr-alpha-123"
    payload = {
        "transaction_id": "tx-ab-1",
        "user_id": user_a,
        "amount": 50.0,
        "currency": "USD",
        "merchant_category": "grocery",
        "device_trust_score": 0.9,
        "location_country": "US",
        "billing_country": "US",
        "velocity_1h": 1,
    }

    # First request
    resp1 = client.post("/v1/predict", json=payload)
    group1 = resp1.headers["X-Experiment-Group"]
    variant1 = resp1.headers["X-Model-Variant"]

    # Second request with same user_id must match exactly
    resp2 = client.post("/v1/predict", json={**payload, "transaction_id": "tx-ab-2"})
    assert resp2.headers["X-Experiment-Group"] == group1
    assert resp2.headers["X-Model-Variant"] == variant1


# -----------------------------------------------------------------------------
# 4. Canary Rollout & Auto-Rollback Tests
# -----------------------------------------------------------------------------
def test_gateway_canary_routing_and_rollback(
    client: TestClient, sample_tx: dict
) -> None:
    """Canary routing shifts traffic and triggers auto-rollback on SLA breach."""
    router.set_mode(RoutingMode.CANARY)
    router.update_canary_weight(0.50)

    resp = client.post("/v1/predict", json=sample_tx)
    assert resp.status_code == 200
    assert resp.headers["X-Routing-Mode"] == "canary"
    assert resp.headers["X-Canary-Weight"] == "0.5"

    # Simulate SLA breach in canary
    for _ in range(25):
        router._record_canary_telemetry(latency_ms=10.0, is_error=False)

    # Router should have auto-rolled back to Champion
    status_report = router.get_status_report()
    assert status_report["canary"]["rollback_triggered"] is True
    assert status_report["canary"]["weight"] == 0.0
    assert status_report["active_mode"] == "default"


# -----------------------------------------------------------------------------
# 5. Gateway Multi-Model Compare & Async API Tests
# -----------------------------------------------------------------------------
def test_gateway_multi_model_compare(client: TestClient, sample_tx: dict) -> None:
    """Compare endpoint evaluates transaction across both models simultaneously."""
    resp = client.post("/v1/models/compare", json=sample_tx)
    assert resp.status_code == 200
    data = resp.json()
    assert "agreement" in data
    assert "champion" in data
    assert "challenger" in data
    assert data["champion"]["decision"] is not None
    assert data["challenger"]["decision"] is not None


def test_gateway_async_endpoint_flow(client: TestClient, sample_tx: dict) -> None:
    """Gateway async endpoint queues job and serves status polling."""
    # Submit async prediction
    submit_resp = client.post("/v1/predict/async", json=sample_tx)
    assert submit_resp.status_code == 202
    job_id = submit_resp.json()["job_id"]

    # Poll status (should be PENDING)
    poll_resp1 = client.get(f"/v1/predict/job/{job_id}")
    assert poll_resp1.status_code == 200
    assert poll_resp1.json()["status"] == "PENDING"

    # Execute worker step
    worker = AsyncPredictionWorker(queue_manager=queue_mgr)
    worker.process_one_job()

    # Poll status again (should be COMPLETED)
    poll_resp2 = client.get(f"/v1/predict/job/{job_id}")
    assert poll_resp2.status_code == 200
    assert poll_resp2.json()["status"] == "COMPLETED"
    assert poll_resp2.json()["result"]["transaction_id"] == "tx-test-100"
