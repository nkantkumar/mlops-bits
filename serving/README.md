# Enterprise Model Serving & Deployment Stack

Production-grade serving architecture for `mlops-bits` covering:
1. **BentoML Model Packaging**
2. **Container Management via Portainer**
3. **Multi-Model Serving Behind Unified API Gateway**
4. **Deterministic A/B Testing**
5. **Canary Rollouts with Automatic SLA Rollback**
6. **Async Predictions with Redis-Backed Worker**
7. **Production Docker Compose Deployment**

---

## Architecture Overview

```text
                               +-------------------------------------------------+
                               |             Client Applications / Web           |
                               +-------------------------------------------------+
                                                        |
                                                        v
                                       +---------------------------------+
                                       |   Unified API Gateway (:8000)   |
                                       |  - Dynamic Smart Router         |
                                       |  - A/B Testing Cohort Hasher    |
                                       |  - Canary Traffic Shifter       |
                                       |  - Auto-Rollback Circuit        |
                                       +---------------------------------+
                                         /              |              \
               +------------------------+               |               +-----------------------+
               | (Champion Traffic)                     | (Canary/Variant B)                    | (Async Queue)
               v                                        v                                       v
+-----------------------------+          +-----------------------------+          +-----------------------------+
| Model Champion (:8001)      |          | Model Challenger (:8002)    |          | Redis Message Broker (:6379)|
| Production Calibrated Engine|          | Tuned Boundary Candidate    |          | - Job Queue & Result Store  |
+-----------------------------+          +-----------------------------+          +-----------------------------+
                                                                                                |
                                                                                                v
                                                                                  +-----------------------------+
                                                                                  | Async Prediction Worker     |
                                                                                  | Background Job Consumer     |
                                                                                  +-----------------------------+
```

---

## 1. BentoML Model Service

The service is packaged using BentoML 1.4 in [`serving/bentoml/`](file:///Users/nishi/Documents/GitHub/mlops-bits/serving/bentoml/):

```bash
# Test service locally
uv run python -c "from serving.bentoml.service import FraudDetectionService; s = FraudDetectionService(); print(s.health())"

# Build Bento package
uv run bentoml build -f serving/bentoml/bentofile.yaml
```

---

## 2. Docker Compose & Portainer Stack

The stack spins up the Unified Gateway, Model Microservices, Redis, the Async Worker, and Portainer CE:

```bash
# Start all containers in background
docker compose up -d --build

# View container status
docker compose ps
```

### Accessing Portainer
Open your browser to: **http://localhost:9000** (or `https://localhost:9443`)
- Create your initial admin password on first launch.
- Select the local Docker environment to visually inspect container logs, CPU/memory usage, network bridges, and deploy rolling updates.

---

## 3. Dynamic A/B Testing

Deterministic user-cohort hashing guarantees that repeated transactions from the same `user_id` consistently hit the exact same variant (Variant A or Variant B):

```bash
# Run A/B test simulation
uv run python -m serving.simulate_ab_test --samples 300 --champion-ratio 0.50
```

To configure via the Gateway REST API:
```bash
curl -X POST http://localhost:8000/v1/admin/ab-test/split \
  -H "Content-Type: application/json" \
  -d '{"champion_ratio": 0.50, "experiment_name": "fraud_detector_v1_vs_v2"}'
```

Every response includes tracing headers:
- `X-Model-Variant`: `champion` or `challenger`
- `X-Routing-Mode`: `ab_test`
- `X-Experiment-Group`: `A` or `B`

---

## 4. Canary Rollouts & Safety Circuit Rollback

Gradually shift live production traffic to a new candidate model version, with automated circuit-breaker rollback if p95 latency or error rates exceed target thresholds:

```bash
# Run Canary Rollout simulation
uv run python -m serving.simulate_canary --samples 60
```

To adjust canary traffic via the Gateway REST API:
```bash
# Set canary weight to 25%
curl -X POST http://localhost:8000/v1/admin/canary/weight \
  -H "Content-Type: application/json" \
  -d '{"weight": 0.25}'

# Trigger emergency rollback
curl -X POST http://localhost:8000/v1/admin/canary/rollback
```

---

## 5. Async Predictions with Redis

For high-volume transaction ingestion where callers cannot wait on network hops:

```bash
# Submit transaction asynchronously (< 2ms response)
curl -X POST http://localhost:8000/v1/predict/async \
  -H "Content-Type: application/json" \
  -d '{
    "transaction_id": "tx-async-01",
    "user_id": "usr-881",
    "amount": 450.0,
    "currency": "USD",
    "merchant_category": "retail",
    "device_trust_score": 0.95,
    "location_country": "US",
    "billing_country": "US",
    "velocity_1h": 1
  }'
```
Response:
```json
{
  "job_id": "job-a1b2c3d4e5f6",
  "status": "PENDING",
  "poll_url": "/v1/predict/job/job-a1b2c3d4e5f6"
}
```

Poll job result:
```bash
curl http://localhost:8000/v1/predict/job/job-a1b2c3d4e5f6
```

---

## 6. Multi-Model Direct Comparison Endpoint

Shadow-evaluate transactions through both Champion and Challenger models simultaneously:

```bash
curl -X POST http://localhost:8000/v1/models/compare \
  -H "Content-Type: application/json" \
  -d '{
    "transaction_id": "tx-shadow-01",
    "user_id": "usr-554",
    "amount": 8500.0,
    "currency": "USD",
    "merchant_category": "crypto",
    "device_trust_score": 0.20,
    "location_country": "NG",
    "billing_country": "US",
    "velocity_1h": 5
  }'
```
Response returns side-by-side decisions, scores, and agreement status.
