"""Integration tests for the FastAPI production serving endpoints."""

from starlette.testclient import TestClient

from mlops_bits.api import app

client = TestClient(app)


def test_health_endpoint() -> None:
    """GET /health must return 200 with service metadata."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["service"] == "mlops-bits-fraud-detector"
    assert "version" in data


def test_predict_endpoint_success() -> None:
    """POST /v1/predict with valid payload returns 200 and schema."""
    payload = {
        "transaction_id": "api-tx-001",
        "user_id": "user-api",
        "amount": 89.99,
        "currency": "USD",
        "merchant_category": "retail",
        "device_trust_score": 0.9,
        "location_country": "US",
        "billing_country": "US",
        "velocity_1h": 1,
    }
    response = client.post("/v1/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["transaction_id"] == "api-tx-001"
    assert data["decision"] in ("APPROVE", "REVIEW", "REJECT")
    assert 0.0 <= data["fraud_score"] <= 1.0
    assert "inference_latency_ms" in data


def test_predict_endpoint_validation_error() -> None:
    """POST /v1/predict with invalid amount or missing fields returns 422."""
    invalid_payload = {
        "transaction_id": "api-err-1",
        "user_id": "user-api",
        "amount": -50.0,  # Negative amount rejected by schema
        "location_country": "US",
        "billing_country": "US",
    }
    response = client.post("/v1/predict", json=invalid_payload)
    assert response.status_code == 422


def test_batch_predict_endpoint() -> None:
    """POST /v1/predict/batch processes list and returns throughput."""
    payload = {
        "transactions": [
            {
                "transaction_id": f"batch-{i}",
                "user_id": f"user-{i}",
                "amount": 25.0 * (i + 1),
                "location_country": "US",
                "billing_country": "US",
            }
            for i in range(5)
        ]
    }
    response = client.post("/v1/predict/batch", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["total_processed"] == 5
    assert len(data["results"]) == 5
    assert data["throughput_rps"] > 0.0


def test_drift_monitor_endpoint() -> None:
    """POST /v1/monitor/drift returns statistical drift evaluation."""
    payload = {
        "feature_name": "transaction_amount",
        "baseline_values": [
            10.0,
            20.0,
            30.0,
            40.0,
            50.0,
            60.0,
            70.0,
            80.0,
            90.0,
            100.0,
        ],
        "current_values": [12.0, 22.0, 32.0, 42.0, 52.0, 62.0, 72.0, 82.0, 92.0, 102.0],
        "num_bins": 5,
    }
    response = client.post("/v1/monitor/drift", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["feature_name"] == "transaction_amount"
    assert "psi" in data
    assert data["drift_status"] in ("NO_DRIFT", "MODERATE_DRIFT", "SIGNIFICANT_DRIFT")
