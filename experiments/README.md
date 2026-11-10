# MLflow Experiment Tracking

Comprehensive experiment tracking, model evaluation, and diagnostic reporting for `mlops-bits` using [MLflow](https://mlflow.org/).

---

## Overview

The `experiments/` module evaluates and compares multiple modeling approaches for fraud detection:

1. **`LowLatencyFraudDetector-CalibratedEngine`**: Sub-millisecond rule and calibrated sigmoid logit engine.
2. **`LogisticRegression-L2-Balanced`**: Scaled linear baseline model with L2 regularization and balanced class weights.
3. **`RandomForest-Ensemble-Depth6`**: Non-linear ensemble model with tree-based importance and probability calibration.

---

## Logged Metadata & Artifacts

For each experiment run, the following metadata is logged to MLflow:

### 1. Parameters
- **Data Properties**: Raw sample size, train/test split size, fraud ratio, feature dimensions.
- **Model Hyperparameters**: Decision thresholds, regularization (`C`), tree estimators, maximum depth, SLA targets.

### 2. Metrics
- **Discrimination & Classification**: `roc_auc`, `pr_auc` (Average Precision), `f1_score`, `precision`, `recall`, `accuracy`, `log_loss`, `brier_score`.
- **Inference Latency & SLA**: `mean_latency_ms`, `p50_latency_ms`, `p95_latency_ms`, `p99_latency_ms`, `throughput_rps`, `sla_passed`.
- **Data Drift**: `amount_psi` (Population Stability Index), `drift_alert`.

### 3. Visual & Structured Artifacts
- **Diagnostic Plots**:
  - `confusion_matrix.png`: Heatmap with true vs. predicted counts and percentages.
  - `roc_curve.png`: Receiver Operating Characteristic curve with AUC.
  - `precision_recall_curve.png`: Precision-Recall curve with Average Precision (AP).
  - `feature_importance.png`: Top-10 feature coefficients or Gini importances.
- **Reports**:
  - `classification_report.json`: Per-class precision, recall, and F1 metrics.
  - `metrics_summary.json`: Complete dictionary of all logged metrics.

### 4. Deployable Model Artifacts
- Standardized MLflow Model package (`model/`):
  - Model binary and inference wrapper (`mlflow.sklearn` or `mlflow.pyfunc`).
  - Model signature (`inputs` and `outputs` schema).
  - Input example (`input_example.json`).
  - Python / conda dependencies (`MLmodel`, `requirements.txt`).

---

## Running the Experiments

### 1. Execute All Models

```bash
uv run python -m experiments.run_experiment
```

Or via direct script execution:

```bash
uv run python experiments/run_experiment.py
```

### 2. Custom Arguments

```bash
# Run with custom experiment name and only tree ensemble model
uv run python experiments/run_experiment.py \
  --experiment-name fraud-detection-v2 \
  --models random_forest logistic_regression \
  --test-size 0.30 \
  --seed 123
```

---

## Launching the MLflow UI

To view the experiment dashboard, compare runs, inspect metric curves, and view logged diagnostic plots:

```bash
uv run mlflow server \
  --backend-store-uri sqlite:///mlflow.db \
  --default-artifact-root ./mlartifacts \
  --host 127.0.0.1 \
  --port 5000
```

Open your browser to: **http://127.0.0.1:5000**
