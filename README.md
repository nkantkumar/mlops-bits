# mlops-bits: Enterprise Fraud Detection Module

Production-grade, low-latency fraud detection module and statistical drift monitoring engineered for enterprise MLOps. Packaged deterministically using [`uv`](https://docs.astral.sh/uv/) and served via FastAPI.

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
- **Deterministic Packaging & Build**: Modern PEP 517/621 packaging with `pyproject.toml`, pinned lockfile `uv.lock`, and distribution builds (`uv build`).
- **Rigorous Testing**: 22 comprehensive pytest tests covering unit logic, boundary validations, drift thresholds, and sub-millisecond SLA performance.

---

## Prerequisites

- **Python**: `>= 3.14`
- **uv**: `>= 0.12`

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

## Execution & Serving

### 1. Run Built-In Low-Latency Benchmark

Verify strict latency SLA directly from the CLI:

```bash
uv run mlops-bits benchmark --samples 1000
```

Sample output:
```text
🚀 Running benchmark with 1,000 synthetic transactions...
=======================================================
📊 BENCHMARK RESULTS (Low-Latency Inference SLA)
=======================================================
Total Transactions Scored : 1,000
Total Execution Time      : 0.005 s
Throughput                : 211,504.0 req/s
Mean Latency              : 0.0013 ms
p50 Median Latency        : 0.0011 ms
p95 Latency               : 0.0013 ms
p99 Latency               : 0.0022 ms
=======================================================
✅ Low-latency SLA PASSED (< 5.0ms p99)
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
# Run full pytest suite (unit logic, drift, latency SLA, API)
uv run pytest -v

# Run only latency benchmark tests
uv run pytest tests/test_latency.py -v

# Run only drift & distribution tests
uv run pytest tests/test_drift.py -v
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

## Project Structure

```text
mlops-bits/
├── src/
│   └── mlops_bits/
│       ├── __init__.py      # Public exports and entry points
│       ├── api.py           # FastAPI production application
│       ├── cli.py           # CLI runner (benchmark & serve)
│       ├── config.py        # Decision thresholds and SLA parameters
│       ├── detector.py      # Sub-millisecond fraud scoring engine
│       ├── drift.py         # PSI distribution drift monitor
│       └── schemas.py       # Pydantic v2 schemas and boundary validations
├── tests/
│   ├── __init__.py
│   ├── conftest.py          # Shared fixtures (transactions & distributions)
│   ├── test_api.py          # FastAPI endpoint integration tests
│   ├── test_detector.py     # Unit logic and boundary constraint tests
│   ├── test_drift.py        # Statistical drift and PSI tests
│   ├── test_latency.py      # Low-latency SLA benchmarks (<5ms p99)
│   └── test_main.py         # CLI sanity test
├── pyproject.toml           # Project packaging & pytest/ruff/mypy config
├── uv.lock                  # Deterministic lockfile
└── README.md
```
