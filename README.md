# mlops-bits: Enterprise Fraud Detection Module

Production-grade, low-latency fraud detection module and statistical drift monitoring engineered for enterprise MLOps. Packaged deterministically using [`uv`](https://docs.astral.sh/uv/) and versioned with [DVC](https://dvc.org/).

---

## Architecture & Features

- **Strict Low-Latency Inference**: Sub-millisecond single transaction scoring (< 0.005 ms mean latency, < 0.01 ms p99, exceeding the < 5.0 ms SLA target) processing > 200,000 req/s.
- **Calibrated Risk Engine**: Sigmoid-calibrated logit risk scorer with deterministic boundary constraints:
  - Critical velocity burst hard boundary ($v \ge 10/\text{hr} \rightarrow \text{REJECT}$)
  - Location and billing country mismatch detection
  - Device trust score boundary penalties
  - High-risk merchant category evaluation
  - Extreme transaction amount flags ($\ge \$10,000$)
- **Data Drift & Distribution Monitoring**: Population Stability Index (PSI) monitor detecting statistical shifts across inference features:
  - $\text{PSI} < 0.10$: `NO_DRIFT`
  - $0.10 \le \text{PSI} < 0.25$: `MODERATE_DRIFT`
  - $\text{PSI} \ge 0.25$: `SIGNIFICANT_DRIFT` (triggers automated alert)
- **Evidently AI Observability & Monitoring**: Full-fledged production monitoring with automated Data Drift reports, Data Quality integrity metrics, CI/CD automated Test Suites, live inference streaming telemetry collector, interactive HTML visual dashboards, and Evidently UI Workspace on port `:8003`.
- **Data Version Control (DVC)**: Reproducible MLOps pipeline tracking data generation, model evaluation, metrics, and Google Drive remote storage.
- **Deterministic Packaging & Build**: Modern PEP 517/621 packaging with `pyproject.toml`, pinned lockfile `uv.lock`, and distribution builds (`uv build`).
- **Rigorous Testing**: 43 comprehensive pytest tests covering unit logic, boundary validations, drift thresholds, pipeline stages, sub-millisecond SLA performance, and Evidently reports/test-suites.

---

## Prerequisites

- **Python**: `>= 3.14`
- **uv**: `>= 0.12`
- **DVC**: `>= 3.67` (included via dev dependencies `dvc` and `dvc-gdrive`)

To install `uv`:

```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# macOS via Homebrew
brew install uv
```

---

## Installation & Setup

```bash
git clone https://github.com/nkantkumar/mlops-bits.git
cd mlops-bits

# Deterministically install exact dependencies from uv.lock
uv sync
```

---

## Data Version Control (DVC) Pipeline

The repository integrates DVC for reproducible data preparation, experiment tracking, and remote storage.

### 1. DVC Pipeline Stages (`dvc.yaml`)

```text
+---------+  
| prepare |  
+---------+  
      *      
      *      
      *      
+----------+ 
| evaluate | 
+----------+ 
```

- **`prepare`**: Deterministically generates raw transaction datasets into `data/raw/transactions.csv` based on `params.yaml`.
- **`evaluate`**: Evaluates dataset through `LowLatencyFraudDetector` and `DistributionDriftMonitor`, recording metrics to `metrics.json`.

### 2. Run DVC Pipeline

```bash
# Execute the full pipeline (or only changed stages)
uv run dvc repro

# View pipeline DAG
uv run dvc dag

# Inspect performance and drift metrics
uv run dvc metrics show

# Inspect parameter differences
uv run dvc params diff
```

Sample metrics output:
```text
Path          amount_psi    approval_rate    drift_alert    drift_status    fraud_count    mean_latency_ms    p95_latency_ms    p99_latency_ms    reject_rate    review_rate    sla_passed    throughput_req_per_s    total_transactions
metrics.json  0.1599        0.95             False          MODERATE_DRIFT  50             0.0027             0.0030            0.0043            0.05           0.0            True          326962.2                1000
```

### 3. Remote Storage (Google Drive)

Remote storage is configured in `.dvc/config`:

```ini
[core]
    remote = myremote
['remote "myremote"']
    url = gdrive://dvc-workspace
```

To sync data with remote storage:

```bash
# Push tracked data to Google Drive
uv run dvc push

# Pull tracked data from Google Drive
uv run dvc pull
```

---

## Execution & Serving

### 1. Run Built-In Low-Latency Benchmark

Verify strict latency SLA directly from the CLI:

```bash
uv run mlops-bits benchmark --samples 1000
```

### 2. Start FastAPI Production Server

```bash
uv run mlops-bits serve --host 0.0.0.0 --port 8000
```

### 3. API Endpoints

- `GET /health`: Health and readiness probe.
- `POST /v1/predict`: Low-latency transaction scoring.
- `POST /v1/predict/batch`: High-throughput batch scoring.
- `POST /v1/monitor/drift`: Population Stability Index (PSI) drift report.

Example predict payload:
```json
{
  "transaction_id": "tx-1001",
  "user_id": "usr-882",
  "amount": 250.00,
  "currency": "USD",
  "merchant_category": "electronics",
  "device_trust_score": 0.95,
  "location_country": "US",
  "billing_country": "US",
  "velocity_1h": 1
}
```

Response:
```json
{
  "transaction_id": "tx-1001",
  "is_fraud": false,
  "fraud_score": 0.0195,
  "decision": "APPROVE",
  "risk_factors": [],
  "inference_latency_ms": 0.0024
}
```

---

## Testing & Validation

```bash
# Run full pytest suite (unit logic, drift, latency SLA, API, pipeline)
uv run pytest -v

# Run only latency benchmark tests
uv run pytest tests/test_latency.py -v

# Run only drift & distribution tests
uv run pytest tests/test_drift.py -v

# Run only DVC pipeline stage tests
uv run pytest tests/test_pipeline.py -v
```

---

## Code Quality & Packaging

```bash
# Lint checks
uv run ruff check .

# Code formatting
uv run ruff format --check .

# Static type verification
uv run mypy src tests

# Build deterministic wheel and source distribution
uv build
```

---

## MLflow Experiment Tracking

The repository includes a dedicated `experiments/` module that benchmarks and logs comparative machine learning models to [MLflow](https://mlflow.org/).

### 1. Evaluated Models

- **`LowLatencyFraudDetector-CalibratedEngine`**: Sub-millisecond rule and calibrated sigmoid logit engine.
- **`LogisticRegression-L2-Balanced`**: Scaled linear baseline model with L2 regularization.
- **`RandomForest-Ensemble-Depth6`**: Non-linear ensemble model with tree-based importance and probability calibration.

### 2. Run Experiments

```bash
# Run comparative experiment suite across all models
uv run python -m experiments.run_experiment

# Or via CLI subcommand
uv run mlops-bits experiment --experiment-name fraud-detection

# Run specific models with custom parameters
uv run python experiments/run_experiment.py \
  --experiment-name fraud-detection \
  --models detector logistic_regression \
  --test-size 0.30
```

### 3. Launch MLflow Tracking UI

```bash
uv run mlflow server \
  --backend-store-uri sqlite:///mlflow.db \
  --default-artifact-root ./mlartifacts \
  --host 127.0.0.1 \
  --port 5000
```

Open **http://127.0.0.1:5000** to compare runs, inspect parameters, metrics (ROC-AUC, PR-AUC, F1, latency p99), diagnostic plots (`confusion_matrix.png`, `roc_curve.png`), and packaged model artifacts.

---

## Production Model Serving & Deployment Patterns

`mlops-bits` provides an enterprise-ready model serving architecture featuring BentoML packaging, Portainer container management, A/B testing, Redis async prediction workers, a Unified API Gateway, and Canary rollout with automated circuit-breaker rollback.

```
                           ┌──────────────────────────────┐
                           │   Unified API Gateway        │
                           │   (FastAPI on Port 8000)     │
                           └──────────────┬───────────────┘
                                          │
            ┌─────────────────────────────┼─────────────────────────────┐
            │ [Dynamic Router]            │                             │
            ▼                             ▼                             ▼
   ┌──────────────────┐          ┌──────────────────┐          ┌──────────────────┐
   │ Model Champion   │          │ Model Challenger │          │ Redis Queue      │
   │ (Port 8001)      │          │ (Port 8002)      │          │ (Port 6379)      │
   └──────────────────┘          └──────────────────┘          └────────┬─────────┘
                                                                        │
                                                                        ▼
                                                               ┌──────────────────┐
                                                               │ Async Worker     │
                                                               │ (Low-Latency)    │
                                                               └──────────────────┘
```

### 1. Package a Model as a BentoML Service

Model serving is packaged using BentoML 1.4 (`serving/bentoml/`):
- `serving/bentoml/service.py`: Self-contained `@bentoml.service` exposing `/predict`, `/predict_batch`, and `/health` APIs.
- `serving/bentoml/bentofile.yaml`: Bento packaging recipe bundling models and dependencies into a standard OCI container.

```bash
# Serve BentoML model service locally
uv run bentoml serve serving.bentoml.service:FraudDetectionService --port 3000

# Build production Bento bundle
uv run bentoml build -f serving/bentoml/bentofile.yaml serving/bentoml
```

### 2. Unified API Gateway & Multi-Model Serving

The API Gateway (`serving/gateway/app.py`) provides a single, unified entry point for all client scoring traffic:
- **`POST /v1/predict`**: Dynamic smart router routing to Champion or Challenger with zero downtime.
- **`POST /v1/predict/async`**: Non-blocking asynchronous job submission.
- **`GET /v1/predict/job/{job_id}`**: Polling endpoint for job status and completed prediction payload.
- **`POST /v1/models/compare`**: Side-by-side comparative inference benchmarking across both models simultaneously.
- **`POST /v1/admin/routing`**: Live router mode switching (`default`, `ab_test`, `canary`).
- **`POST /v1/admin/canary/weight`**: Live canary traffic weight adjustment (`0.0` to `1.0`).
- **`POST /v1/admin/ab-test/split`**: Live A/B traffic split adjustment.

```bash
# Start Gateway locally
uv run mlops-bits gateway --port 8000
```

### 3. Implement A/B Testing for Model Deployment

Deterministic cohort assignment via `MD5(user_id + salt) % 100` ensures sticky sessions:
- Any given `user_id` consistently routes to the exact same model variant across repeated transactions.
- Traffic split ratios can be configured on the fly via admin endpoints.

```bash
# Run automated A/B test simulation (verifies 50/50 split and 100% user stickiness)
uv run mlops-bits simulate ab-test
```

### 4. Simulate a Canary Rollout with Auto-Rollback

Automated canary rollout progressively routes live traffic to the candidate model while monitoring error rates and latency SLA (< 5.0ms p95):
1. **Stage 0**: Baseline (100% Champion / 0% Canary)
2. **Stage 1**: Initial Canary (10% Traffic)
3. **Stage 2**: Expanded Canary (25% Traffic)
4. **Stage 3**: Balanced Canary (50% Traffic)
5. **Stage 4**: Full Promotion (100% Canary)
- **Automated Circuit Breaker**: If canary latency degrades or error rate spikes, the gateway automatically rolls traffic back to 100% Champion with zero dropped requests.

```bash
# Run Canary rollout simulation with visual ASCII progress bars and auto-rollback
uv run mlops-bits simulate canary
```

### 5. Async Predictions with Redis-Backed Worker

For non-blocking high-volume ingestion:
- Transactions are enqueued onto Redis (`mlops:queue:predictions`) using `LPUSH` / `BLPOP`.
- Background worker (`serving/async_worker/worker.py`) consumes and scores transactions with sub-millisecond throughput.
- Includes automatic fallback to an in-memory queue if Redis is temporarily unreachable.

```bash
# Start the async worker daemon
uv run mlops-bits worker
```

### 6. Production Model Serving with Docker Compose & Portainer

A complete multi-container production stack is defined in [`docker-compose.yml`](file:///Users/nishi/Documents/GitHub/mlops-bits/docker-compose.yml):

| Service | Port | Image / Context | Purpose |
| :--- | :--- | :--- | :--- |
| **`portainer`** | `:9000` | `portainer/portainer-ce:latest` | Container Management GUI |
| **`gateway`** | `:8000` | `Dockerfile.gateway` | Unified API Gateway & Smart Router |
| **`model-champion`** | `:8001` | `Dockerfile.model` | Baseline Production Scorer |
| **`model-challenger`**| `:8002` | `Dockerfile.model` | Candidate Model Variant |
| **`redis`** | `:6379` | `redis:7-alpine` | Async Prediction Job Queue |
| **`async-worker`** | N/A | `Dockerfile.worker` | Background Processing Engine |
| **`evidently-ui`** | `:8003` | `Dockerfile.evidently` | Evidently AI Observability Dashboard & Workspace |

#### Deploying with Docker Compose:

```bash
# Build and start all services in detached mode
docker compose up -d --build

# Inspect running services
docker compose ps

# View gateway logs
docker compose logs -f gateway

# View Evidently UI logs
docker compose logs -f evidently-ui
```

#### Managing via Portainer:
1. Open your browser to **http://localhost:9000**.
2. Set your admin credentials upon initial launch.
3. Access the **local** Docker environment to monitor container health, restart services, inspect logs, and manage CPU/memory allocations visually.

#### Monitoring via Evidently UI Dashboard:
1. Open your browser to **http://localhost:8003**.
2. View historical drift runs, test suite executions, dataset distributions, and project dashboards.

---

## Evidently AI Observability & Monitoring

`mlops-bits` includes a full-fledged monitoring and observability subsystem powered by [Evidently AI](https://evidentlyai.com/). It provides statistical data drift detection, data quality profiling, automated test suites for deployment gates, live inference telemetry capture, standalone interactive HTML visual reports, and an interactive Evidently UI workspace.

```text
                           ┌──────────────────────────────┐
                           │   Unified API Gateway        │
                           │   (Port 8000 / Live Scoring) │
                           └──────────────┬───────────────┘
                                          │ [Telemetry Ingestion]
                                          ▼
                           ┌──────────────────────────────┐
                           │ InferenceTelemetryCollector  │
                           │ (Thread-Safe Ring Buffer)    │
                           └──────────────┬───────────────┘
                                          │
                     ┌────────────────────┴────────────────────┐
                     ▼                                         ▼
         ┌─────────────────────────┐               ┌─────────────────────────┐
         │ Evidently Data Drift    │               │ Automated Test Suite    │
         │ (K-S, Chi-Square, PSI)  │               │ (CI/CD Deployment Gate) │
         └───────────┬─────────────┘               └───────────┬─────────────┘
                     │                                         │
                     └────────────────────┬────────────────────┘
                                          │
                                          ▼
                           ┌──────────────────────────────┐
                           │ Evidently UI Workspace &     │
                           │ Interactive HTML Reports     │
                           │ (Dashboard on Port 8003)     │
                           └──────────────────────────────┘
```

### 1. Key Capabilities
- **Feature-Level Data Drift**: Evaluates statistical drift across all inference inputs (`amount`, `velocity_1h`, `device_trust_score`, `merchant_category`, `location_country`, `billing_country`) using calibrated hypothesis tests (Kolmogorov-Smirnov for numerical, Chi-Square/Z-test for categorical).
- **Data Quality & Integrity Profiling**: Analyzes row counts, column counts, missing values, duplicates, and statistical quantiles.
- **Automated Test Suites for CI/CD Gates**: Evaluates pre-configured assertions (`DataDriftTestPreset`, `DataQualityTestPreset`) with granular pass/fail verdicts and reason descriptions. Exits with code `1` on failure to block degraded canary deployments.
- **Live Inference Telemetry Buffer**: Automatically records incoming live production predictions in an in-memory sliding ring buffer (`InferenceTelemetryCollector`) without adding latency to inference calls.
- **Multiple Output Formats**:
  - Interactive standalone HTML reports (`reports/evidently/data_drift_latest.html`)
  - Structured JSON metrics (`reports/evidently/data_drift_latest.json`)
  - Continuous snapshot runs registered into Evidently Workspace (`workspace/`)
  - Integration with DVC (`metrics.json`) and MLflow (`mlflow.log_artifact`)

### 2. CLI Monitoring Commands
Run monitoring and observability workflows directly from the command line:

```bash
# Generate Data Drift & Quality HTML and JSON reports
uv run mlops-bits monitor report

# Run report against custom baseline and current CSVs with custom drift threshold
uv run mlops-bits monitor report \
  --baseline data/raw/transactions.csv \
  --current data/raw/drifted_transactions.csv \
  --drift-share 0.25

# Run automated Evidently Test Suite (ideal for CI/CD pipelines & deployment gates)
uv run mlops-bits monitor test

# Run test suite and fail with exit code 1 if any assertions fail
uv run mlops-bits monitor test --fail-on-error

# Launch the Evidently UI Monitoring Dashboard server
uv run mlops-bits monitor ui --port 8003 --workspace workspace
```

### 3. API Monitoring Endpoints
Both the FastAPI Model Server (`mlops_bits.api:app`) and Unified API Gateway (`serving.gateway.app:app`) expose Evidently monitoring endpoints:

- **`POST /v1/monitor/evidently/report`**: Computes data drift between baseline reference data and recent live inference telemetry.
- **`POST /v1/monitor/evidently/test-suite`**: Runs automated assertion tests for deployment gating.
- **`GET /v1/monitor/evidently/summary`**: Operational health and buffer status of the monitoring engine.
- **`GET /v1/monitor/evidently/report.html`**: Serves the latest interactive HTML report directly to the web browser.
- **`POST /v1/monitor/evidently/collect`**: Manually push externally scored transactions into the telemetry buffer.

### 4. DVC Pipeline Integration
Evidently drift share and test suite metrics are automatically tracked in the DVC pipeline:

```bash
# Reproduce pipeline evaluation with Evidently
uv run dvc repro evaluate

# View all evaluation and Evidently drift metrics
uv run dvc metrics show
```

### 5. MLflow Experiment Tracking
During model training and comparative benchmarking (`experiments/run_experiment.py`), Evidently HTML reports and drift metrics are logged as first-class MLflow artifacts under `artifact_path="evidently"` and tracked across models in the MLflow UI.

### 6. Docker Compose Service
The Evidently UI service runs as a containerized dashboard in the stack:

```bash
# Start all services including Evidently UI
docker compose up -d

# Open Evidently UI Dashboard in your browser:
# http://localhost:8003
```

---

## Project Structure

```text
mlops-bits/
├── .dvc/
│   ├── config                  # DVC remote configuration (Google Drive)
│   └── .gitignore              # Ignores local DVC cache and credentials
├── data/
│   └── raw/
│       ├── .gitignore          # DVC auto-generated (ignores transactions.csv in Git)
│       └── transactions.csv    # Versioned data artifact
├── experiments/
│   ├── __init__.py             # Experiment module exports
│   ├── run_experiment.py       # MLflow experiment runner, feature pipeline & models
│   └── README.md               # Experiment tracking guide and documentation
├── serving/
│   ├── bentoml/
│   │   ├── bentofile.yaml      # BentoML 1.4 build specification
│   │   └── service.py          # BentoML FraudDetectionService definition
│   ├── async_worker/
│   │   ├── __init__.py
│   │   ├── queue.py            # Redis queue manager with in-memory fallback
│   │   └── worker.py           # Background prediction consumer daemon
│   ├── gateway/
│   │   ├── __init__.py
│   │   ├── app.py              # FastAPI Unified API Gateway application
│   │   ├── models_client.py    # Multi-model client (Champion / Challenger)
│   │   └── router.py           # Dynamic traffic router (A/B, Canary, Default)
│   ├── simulate_ab_test.py     # Controlled A/B test simulation script
│   ├── simulate_canary.py      # Multi-stage Canary rollout simulation with rollback
│   └── README.md               # Serving architecture and deployment operations guide
├── src/
│   └── mlops_bits/
│       ├── __init__.py         # Public exports and entry points
│       ├── api.py              # Core FastAPI application
│       ├── cli.py              # CLI runner (benchmark, serve, experiment, gateway, worker, simulate, monitor)
│       ├── config.py           # Decision thresholds and SLA parameters
│       ├── detector.py         # Sub-millisecond fraud scoring engine
│       ├── drift.py            # PSI distribution drift monitor
│       ├── schemas.py          # Pydantic v2 schemas and boundary validations
│       ├── monitoring/         # Evidently AI observability and drift package
│       │   ├── __init__.py
│       │   ├── collector.py    # Thread-safe sliding ring buffer for live inference
│       │   ├── config.py       # Monitoring thresholds and directories
│       │   ├── schemas.py      # Evidently drift and test suite schemas
│       │   └── service.py      # EvidentlyMonitor core engine & workspace sync
│       └── pipeline/
│           ├── __init__.py
│           ├── prepare.py      # DVC stage: reproducible transaction generation
│           └── evaluate.py     # DVC stage: model evaluation, PSI & Evidently metrics
├── tests/
│   ├── __init__.py
│   ├── conftest.py             # Shared fixtures (transactions & distributions)
│   ├── test_api.py             # FastAPI endpoint integration tests
│   ├── test_detector.py        # Unit logic and boundary constraint tests
│   ├── test_drift.py           # Statistical drift and PSI tests
│   ├── test_evidently.py       # Evidently AI drift, quality, test suite & API tests
│   ├── test_experiments.py     # MLflow experiment tracking & logging tests
│   ├── test_latency.py         # Low-latency SLA benchmarks (<5ms p99)
│   ├── test_pipeline.py        # DVC pipeline stage unit tests
│   ├── test_serving.py         # Serving, BentoML, Gateway, A/B, Canary, & Worker tests
│   └── test_main.py            # CLI sanity test
├── Dockerfile.evidently        # Production Evidently AI Observability UI container
├── Dockerfile.gateway          # Production Gateway container
├── Dockerfile.model            # Production Model serving container
├── Dockerfile.worker           # Production Async Worker container
├── docker-compose.yml          # Multi-container production stack (Gateway, Models, Redis, Portainer, Evidently UI)
├── .dockerignore               # Build context optimization
├── dvc.yaml                    # DVC pipeline definition (prepare -> evaluate)
├── dvc.lock                    # Cryptographic pipeline state lockfile
├── params.yaml                 # Hyperparameters and pipeline configuration
├── metrics.json                # DVC evaluation metrics artifact
├── mlflow.db                   # SQLite MLflow tracking backend
├── pyproject.toml              # Project packaging & dependencies
├── uv.lock                     # Deterministic lockfile
└── README.md
```
