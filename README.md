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
- **Data Version Control (DVC)**: Reproducible MLOps pipeline tracking data generation, model evaluation, metrics, and Google Drive remote storage.
- **Deterministic Packaging & Build**: Modern PEP 517/621 packaging with `pyproject.toml`, pinned lockfile `uv.lock`, and distribution builds (`uv build`).
- **Rigorous Testing**: 25 comprehensive pytest tests covering unit logic, boundary validations, drift thresholds, pipeline stages, and sub-millisecond SLA performance.

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

## Project Structure

```text
mlops-bits/
├── .dvc/
│   ├── config               # DVC remote configuration (Google Drive)
│   └── .gitignore           # Ignores local DVC cache and credentials
├── data/
│   └── raw/
│       ├── .gitignore       # DVC auto-generated (ignores transactions.csv in Git)
│       └── transactions.csv # Versioned data artifact
├── experiments/
│   ├── __init__.py          # Experiment module exports
│   ├── run_experiment.py    # MLflow experiment runner, feature pipeline & models
│   └── README.md            # Experiment tracking guide and documentation
├── src/
│   └── mlops_bits/
│       ├── __init__.py      # Public exports and entry points
│       ├── api.py           # FastAPI production application
│       ├── cli.py           # CLI runner (benchmark, serve, experiment)
│       ├── config.py        # Decision thresholds and SLA parameters
│       ├── detector.py      # Sub-millisecond fraud scoring engine
│       ├── drift.py         # PSI distribution drift monitor
│       ├── schemas.py       # Pydantic v2 schemas and boundary validations
│       └── pipeline/
│           ├── __init__.py
│           ├── prepare.py   # DVC stage: reproducible transaction generation
│           └── evaluate.py  # DVC stage: model evaluation & PSI metrics
├── tests/
│   ├── __init__.py
│   ├── conftest.py          # Shared fixtures (transactions & distributions)
│   ├── test_api.py          # FastAPI endpoint integration tests
│   ├── test_detector.py     # Unit logic and boundary constraint tests
│   ├── test_drift.py        # Statistical drift and PSI tests
│   ├── test_experiments.py  # MLflow experiment tracking & logging tests
│   ├── test_latency.py      # Low-latency SLA benchmarks (<5ms p99)
│   ├── test_pipeline.py     # DVC pipeline stage unit tests
│   └── test_main.py         # CLI sanity test
├── dvc.yaml                 # DVC pipeline definition (prepare -> evaluate)
├── dvc.lock                 # Cryptographic pipeline state lockfile
├── params.yaml              # Hyperparameters and pipeline configuration
├── metrics.json             # DVC evaluation metrics artifact
├── mlflow.db                # SQLite MLflow tracking backend
├── pyproject.toml           # Project packaging & dependencies
├── uv.lock                  # Deterministic lockfile
└── README.md
```
