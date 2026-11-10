"""MLflow Experiment Tracking and Model Evaluation for Enterprise Fraud Detection.

This module trains, evaluates, benchmarks, and logs multiple fraud detection models
to MLflow, including:
1. LowLatencyFraudDetector: Production rule & calibrated logit engine (Custom PyFunc).
2. LogisticRegression: Scaled linear baseline model with L2 regularization.
3. RandomForestClassifier: Non-linear ensemble model with tree-based importance.

All runs track hyperparameters, classification metrics, sub-millisecond inference
latencies, data drift, visual diagnostic plots, and deployable model artifacts.
"""

from __future__ import annotations

import argparse
import json
import random
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Configure headless matplotlib backend before importing pyplot
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow.models import infer_signature
from mlflow.pyfunc import PythonModel, PythonModelContext
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from mlops_bits.config import FraudDetectorConfig
from mlops_bits.detector import LowLatencyFraudDetector
from mlops_bits.drift import DistributionDriftMonitor
from mlops_bits.pipeline.prepare import generate_transactions
from mlops_bits.schemas import DriftAnalysisRequest, TransactionRequest

HIGH_RISK_CATEGORIES = {
    "crypto",
    "gambling",
    "wire_transfer",
    "cash_advance",
    "prepaid_cards",
    "electronics",
}


@dataclass
class RunSummary:
    """Summary of a completed MLflow experiment run."""

    run_id: str
    run_name: str
    model_family: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    roc_auc: float
    pr_auc: float
    mean_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    throughput_rps: float
    sla_passed: bool


class FraudDetectorPyFuncWrapper(PythonModel):
    """MLflow PyFunc wrapper for the production LowLatencyFraudDetector."""

    def __init__(self, config_dict: dict[str, Any] | None = None) -> None:
        self.config_dict = config_dict or {}
        self.detector: LowLatencyFraudDetector | None = None

    def load_context(self, context: PythonModelContext) -> None:
        cfg = (
            FraudDetectorConfig(**self.config_dict)
            if self.config_dict
            else FraudDetectorConfig()
        )
        self.detector = LowLatencyFraudDetector(cfg)

    def predict(
        self,
        context: Any,
        model_input: Any,
        params: dict[str, Any] | None = None,
    ) -> Any:
        if self.detector is None:
            self.detector = LowLatencyFraudDetector()

        if isinstance(model_input, list):
            df = pd.DataFrame(model_input)
        else:
            df = model_input

        predictions: list[int] = []
        for _, row in df.iterrows():
            tx = TransactionRequest(
                transaction_id=str(row.get("transaction_id", "eval-tx")),
                user_id=str(row.get("user_id", "eval-usr")),
                amount=float(row["amount"]),
                currency=str(row.get("currency", "USD")),
                merchant_category=str(row["merchant_category"]),
                device_trust_score=float(row["device_trust_score"]),
                location_country=str(row["location_country"]),
                billing_country=str(row["billing_country"]),
                velocity_1h=int(row["velocity_1h"]),
            )
            res = self.detector.predict(tx)
            predictions.append(1 if res.is_fraud else 0)
        return predictions


def load_or_generate_dataset(
    data_path: Path | None = None,
    num_samples: int = 1000,
    fraud_ratio: float = 0.05,
    seed: int = 42,
) -> pd.DataFrame:
    """Load transactions dataset from CSV or generate synthetic transactions."""
    if data_path and data_path.exists():
        df = pd.read_csv(data_path)
    else:
        records = generate_transactions(
            num_samples=num_samples, fraud_ratio=fraud_ratio, seed=seed
        )
        df = pd.DataFrame(records)

    # Establish ground-truth fraud label from enterprise risk definitions
    # Matches synthetic generation distribution and fraud injection criteria
    is_fraud = (
        (
            (df["device_trust_score"] < 0.30)
            & (df["location_country"] != df["billing_country"])
        )
        | (df["velocity_1h"] >= 6)
        | (
            (df["amount"] >= 5000.0)
            & df["merchant_category"].isin(["crypto", "gambling", "electronics"])
        )
    ).astype(int)

    df["is_fraud"] = is_fraud
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Generate high-signal engineered features for fraud modeling."""
    feats = df.copy()
    feats["country_mismatch"] = (
        feats["location_country"] != feats["billing_country"]
    ).astype(int)
    feats["is_high_risk_category"] = (
        feats["merchant_category"].str.lower().isin(HIGH_RISK_CATEGORIES).astype(int)
    )
    feats["log_amount"] = np.log1p(np.maximum(feats["amount"], 0.0))
    feats["is_extreme_amount"] = (feats["amount"] >= 10000.0).astype(int)
    feats["is_untrusted_device"] = (feats["device_trust_score"] < 0.30).astype(int)
    feats["is_critical_velocity"] = (feats["velocity_1h"] >= 10).astype(int)
    return feats


def compute_latency_metrics(
    predict_fn: Any,
    eval_data: list[Any] | pd.DataFrame,
    target_sla_p99_ms: float = 5.0,
) -> dict[str, Any]:
    """Benchmark prediction latency and throughput on evaluation samples."""
    latencies_ms: list[float] = []
    n = len(eval_data)
    if n == 0:
        return {
            "mean_latency_ms": 0.0,
            "p50_latency_ms": 0.0,
            "p95_latency_ms": 0.0,
            "p99_latency_ms": 0.0,
            "throughput_rps": 0.0,
            "sla_passed": True,
        }

    # Warmup
    warmup_n = min(10, n)
    if isinstance(eval_data, pd.DataFrame):
        for i in range(warmup_n):
            _ = predict_fn(eval_data.iloc[[i]])
    else:
        for i in range(warmup_n):
            _ = predict_fn(eval_data[i])

    start_total = time.perf_counter()
    if isinstance(eval_data, pd.DataFrame):
        for i in range(n):
            sample = eval_data.iloc[[i]]
            t0 = time.perf_counter_ns()
            _ = predict_fn(sample)
            t_ms = (time.perf_counter_ns() - t0) / 1_000_000.0
            latencies_ms.append(t_ms)
    else:
        for item in eval_data:
            t0 = time.perf_counter_ns()
            _ = predict_fn(item)
            t_ms = (time.perf_counter_ns() - t0) / 1_000_000.0
            latencies_ms.append(t_ms)
    total_time_s = time.perf_counter() - start_total

    latencies_ms.sort()
    mean_lat = float(np.mean(latencies_ms))
    p50_lat = float(np.percentile(latencies_ms, 50))
    p95_lat = float(np.percentile(latencies_ms, 95))
    p99_lat = float(np.percentile(latencies_ms, 99))
    throughput = n / total_time_s if total_time_s > 0 else 0.0

    return {
        "mean_latency_ms": round(mean_lat, 4),
        "p50_latency_ms": round(p50_lat, 4),
        "p95_latency_ms": round(p95_lat, 4),
        "p99_latency_ms": round(p99_lat, 4),
        "throughput_rps": round(throughput, 1),
        "sla_passed": p99_lat <= target_sla_p99_ms,
    }


def save_diagnostic_plots(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    y_pred: np.ndarray,
    feature_names: list[str] | None,
    feature_importances: np.ndarray | None,
    output_dir: Path,
) -> dict[str, Path]:
    """Generate and save publication-quality diagnostic plots."""
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_paths: dict[str, Path] = {}

    # 1. Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.set_title("Confusion Matrix", fontsize=14, pad=12)
    fig.colorbar(im, ax=ax)
    tick_marks = np.arange(2)
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(["Legitimate (0)", "Fraud (1)"], fontsize=10)
    ax.set_yticklabels(["Legitimate (0)", "Fraud (1)"], fontsize=10)
    ax.set_ylabel("True Label", fontsize=11)
    ax.set_xlabel("Predicted Label", fontsize=11)
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                f"{cm[i, j]:,}",
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=12,
                weight="bold",
            )
    fig.tight_layout()
    cm_path = output_dir / "confusion_matrix.png"
    fig.savefig(cm_path, dpi=150)
    plt.close(fig)
    plot_paths["confusion_matrix"] = cm_path

    # 2. ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc_val = roc_auc_score(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(
        fpr,
        tpr,
        color="#1f77b4",
        lw=2.5,
        label=f"Model ROC (AUC = {auc_val:.4f})",
    )
    ax.plot(
        [0, 1],
        [0, 1],
        color="gray",
        lw=1.5,
        linestyle="--",
        label="Random Baseline",
    )
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.05)
    ax.set_xlabel("False Positive Rate (FPR)", fontsize=11)
    ax.set_ylabel("True Positive Rate (TPR / Recall)", fontsize=11)
    ax.set_title("Receiver Operating Characteristic (ROC)", fontsize=14, pad=12)
    ax.legend(loc="lower right", frameon=True)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    roc_path = output_dir / "roc_curve.png"
    fig.savefig(roc_path, dpi=150)
    plt.close(fig)
    plot_paths["roc_curve"] = roc_path

    # 3. Precision-Recall Curve
    precision_vals, recall_vals, _ = precision_recall_curve(y_true, y_prob)
    ap_val = average_precision_score(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(
        recall_vals,
        precision_vals,
        color="#2ca02c",
        lw=2.5,
        label=f"PR Curve (AP = {ap_val:.4f})",
    )
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.05)
    ax.set_xlabel("Recall", fontsize=11)
    ax.set_ylabel("Precision", fontsize=11)
    ax.set_title("Precision-Recall Curve", fontsize=14, pad=12)
    ax.legend(loc="lower left", frameon=True)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    pr_path = output_dir / "precision_recall_curve.png"
    fig.savefig(pr_path, dpi=150)
    plt.close(fig)
    plot_paths["precision_recall_curve"] = pr_path

    # 4. Feature Importance / Coefficients (if supplied)
    if feature_names and feature_importances is not None:
        fig, ax = plt.subplots(figsize=(8, 5))
        indices = np.argsort(np.abs(feature_importances))[-10:]
        top_names = [feature_names[i] for i in indices]
        top_vals = [feature_importances[i] for i in indices]
        y_pos = np.arange(len(top_names))
        ax.barh(y_pos, top_vals, color="#4c72b0", align="center")
        ax.set_yticks(y_pos)
        ax.set_yticklabels(top_names, fontsize=9)
        ax.set_xlabel("Importance / Coefficient Weight", fontsize=11)
        ax.set_title("Top 10 Feature Importances", fontsize=14, pad=12)
        ax.grid(alpha=0.3, axis="x")
        fig.tight_layout()
        fi_path = output_dir / "feature_importance.png"
        fig.savefig(fi_path, dpi=150)
        plt.close(fig)
        plot_paths["feature_importance"] = fi_path

    return plot_paths


def build_preprocessor(
    cat_features: list[str], num_features: list[str]
) -> ColumnTransformer:
    """Construct deterministic Scikit-Learn preprocessing column transformer."""
    return ColumnTransformer(
        transformers=[
            (
                "num",
                StandardScaler(),
                num_features,
            ),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                cat_features,
            ),
        ]
    )


def run_detector_experiment(
    df_test: pd.DataFrame,
    temp_dir: Path,
) -> RunSummary:
    """Evaluate and log the production LowLatencyFraudDetector rule & logit engine."""
    run_name = "LowLatencyFraudDetector-CalibratedEngine"
    print(f"\n⚡ Running Experiment Run: {run_name}...")

    detector_cfg = FraudDetectorConfig()
    detector = LowLatencyFraudDetector(detector_cfg)

    # Convert test df rows to TransactionRequest objects
    test_transactions: list[TransactionRequest] = []
    for _, row in df_test.iterrows():
        tx = TransactionRequest(
            transaction_id=str(row["transaction_id"]),
            user_id=str(row["user_id"]),
            amount=float(row["amount"]),
            currency=str(row["currency"]),
            merchant_category=str(row["merchant_category"]),
            device_trust_score=float(row["device_trust_score"]),
            location_country=str(row["location_country"]),
            billing_country=str(row["billing_country"]),
            velocity_1h=int(row["velocity_1h"]),
        )
        test_transactions.append(tx)

    # Score test transactions
    decisions = [detector.predict(tx) for tx in test_transactions]
    y_true = df_test["is_fraud"].to_numpy()
    y_prob = np.array([d.fraud_score for d in decisions])
    y_pred = np.array([1 if d.is_fraud else 0 for d in decisions])

    # Compute classification metrics
    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    roc_auc = float(roc_auc_score(y_true, y_prob))
    pr_auc = float(average_precision_score(y_true, y_prob))
    brier = float(brier_score_loss(y_true, y_prob))

    # Latency benchmarking
    lat_metrics = compute_latency_metrics(
        detector.predict, test_transactions, detector_cfg.target_sla_p99_ms
    )

    # Distribution drift analysis on scored amounts
    drift_monitor = DistributionDriftMonitor()
    random.seed(42)
    baseline_amounts = [
        round(random.lognormvariate(3.9, 0.6), 2) for _ in range(len(df_test))
    ]
    drift_report = drift_monitor.analyze(
        DriftAnalysisRequest(
            feature_name="transaction_amount",
            baseline_values=baseline_amounts,
            current_values=df_test["amount"].tolist(),
            num_bins=10,
        )
    )

    with mlflow.start_run(run_name=run_name) as run:
        run_id = run.info.run_id

        # Tags
        mlflow.set_tag("model_family", "expert_calibrated_engine")
        mlflow.set_tag("task", "fraud_detection")
        mlflow.set_tag("inference_type", "sub_millisecond_realtime")
        mlflow.set_tag("framework", "custom_python_engine")

        # Parameters
        mlflow.log_params(
            {
                "reject_threshold": detector_cfg.reject_threshold,
                "review_threshold": detector_cfg.review_threshold,
                "extreme_amount_threshold": detector_cfg.extreme_amount_threshold,
                "high_velocity_threshold": detector_cfg.high_velocity_threshold,
                "critical_velocity_threshold": detector_cfg.critical_velocity_threshold,
                "untrusted_device_threshold": detector_cfg.untrusted_device_threshold,
                "target_sla_p99_ms": detector_cfg.target_sla_p99_ms,
                "target_sla_mean_ms": detector_cfg.target_sla_mean_ms,
                "test_samples": len(df_test),
            }
        )

        # Metrics
        all_metrics = {
            "accuracy": round(acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1_score": round(f1, 4),
            "roc_auc": round(roc_auc, 4),
            "pr_auc": round(pr_auc, 4),
            "brier_score": round(brier, 4),
            "mean_latency_ms": lat_metrics["mean_latency_ms"],
            "p50_latency_ms": lat_metrics["p50_latency_ms"],
            "p95_latency_ms": lat_metrics["p95_latency_ms"],
            "p99_latency_ms": lat_metrics["p99_latency_ms"],
            "throughput_rps": lat_metrics["throughput_rps"],
            "sla_passed": 1 if lat_metrics["sla_passed"] else 0,
            "amount_psi": drift_report.psi,
            "drift_alert": 1 if drift_report.alert_triggered else 0,
        }
        mlflow.log_metrics(all_metrics)

        # Artifacts: Diagnostic plots
        plots_dir = temp_dir / "detector_plots"
        saved_plots = save_diagnostic_plots(
            y_true=y_true,
            y_prob=y_prob,
            y_pred=y_pred,
            feature_names=None,
            feature_importances=None,
            output_dir=plots_dir,
        )
        for plot_path in saved_plots.values():
            mlflow.log_artifact(str(plot_path), artifact_path="diagnostic_plots")

        # Artifacts: Reports & JSON
        report_dict = classification_report(
            y_true, y_pred, output_dict=True, zero_division=0
        )
        report_path = plots_dir / "classification_report.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report_dict, f, indent=2)
        mlflow.log_artifact(str(report_path), artifact_path="reports")

        metrics_path = plots_dir / "metrics_summary.json"
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(all_metrics, f, indent=2)
        mlflow.log_artifact(str(metrics_path), artifact_path="reports")

        # Log Model as MLflow PyFunc
        sample_input = df_test[
            [
                "transaction_id",
                "user_id",
                "amount",
                "currency",
                "merchant_category",
                "device_trust_score",
                "location_country",
                "billing_country",
                "velocity_1h",
            ]
        ].head(5)
        pyfunc_model = FraudDetectorPyFuncWrapper(
            config_dict={
                "reject_threshold": detector_cfg.reject_threshold,
                "review_threshold": detector_cfg.review_threshold,
                "extreme_amount_threshold": detector_cfg.extreme_amount_threshold,
                "high_velocity_threshold": detector_cfg.high_velocity_threshold,
                "critical_velocity_threshold": detector_cfg.critical_velocity_threshold,
                "untrusted_device_threshold": detector_cfg.untrusted_device_threshold,
                "target_sla_p99_ms": detector_cfg.target_sla_p99_ms,
                "target_sla_mean_ms": detector_cfg.target_sla_mean_ms,
            }
        )
        sig = infer_signature(sample_input, np.array([0, 0, 0, 0, 0]))
        mlflow.pyfunc.log_model(
            name="model",
            python_model=pyfunc_model,
            signature=sig,
            input_example=sample_input,
        )

        print(
            f"✅ Logged {run_name} | Run ID: {run_id} | ROC-AUC: {roc_auc:.4f} | F1: {f1:.4f} | p99: {lat_metrics['p99_latency_ms']:.4f} ms"
        )

        return RunSummary(
            run_id=run_id,
            run_name=run_name,
            model_family="expert_calibrated_engine",
            accuracy=acc,
            precision=prec,
            recall=rec,
            f1_score=f1,
            roc_auc=roc_auc,
            pr_auc=pr_auc,
            mean_latency_ms=lat_metrics["mean_latency_ms"],
            p95_latency_ms=lat_metrics["p95_latency_ms"],
            p99_latency_ms=lat_metrics["p99_latency_ms"],
            throughput_rps=lat_metrics["throughput_rps"],
            sla_passed=lat_metrics["sla_passed"],
        )


def run_sklearn_experiment(
    model_name: str,
    pipeline: Pipeline,
    hyperparams: dict[str, Any],
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    temp_dir: Path,
) -> RunSummary:
    """Train, evaluate, benchmark, and log a Scikit-Learn pipeline to MLflow."""
    print(f"\n🤖 Running Experiment Run: {model_name}...")

    # Fit pipeline
    pipeline.fit(X_train, y_train)

    # Predictions and probabilities
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    # Metrics
    acc = float(accuracy_score(y_test, y_pred))
    prec = float(precision_score(y_test, y_pred, zero_division=0))
    rec = float(recall_score(y_test, y_pred, zero_division=0))
    f1 = float(f1_score(y_test, y_pred, zero_division=0))
    roc_auc = float(roc_auc_score(y_test, y_prob))
    pr_auc = float(average_precision_score(y_test, y_prob))
    brier = float(brier_score_loss(y_test, y_prob))
    logloss = float(log_loss(y_test, y_prob))

    # Single-sample inference latency benchmarking
    lat_metrics = compute_latency_metrics(
        pipeline.predict, X_test, target_sla_p99_ms=5.0
    )

    # Extract feature names & importances/coefficients
    preprocessor: ColumnTransformer = pipeline.named_steps["preprocessor"]
    feature_names = list(preprocessor.get_feature_names_out())
    estimator = pipeline.named_steps["classifier"]

    feature_importances: np.ndarray | None = None
    if hasattr(estimator, "coef_"):
        feature_importances = estimator.coef_[0]
    elif hasattr(estimator, "feature_importances_"):
        feature_importances = estimator.feature_importances_

    with mlflow.start_run(run_name=model_name) as run:
        run_id = run.info.run_id

        # Tags
        model_family = (
            "linear_model"
            if isinstance(estimator, LogisticRegression)
            else "tree_ensemble"
        )
        mlflow.set_tag("model_family", model_family)
        mlflow.set_tag("task", "fraud_detection")
        mlflow.set_tag("framework", "scikit-learn")

        # Hyperparameters and data properties
        log_params_dict = {
            "model_type": estimator.__class__.__name__,
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "n_features_raw": X_train.shape[1],
            "n_features_encoded": len(feature_names),
        }
        for k, v in hyperparams.items():
            log_params_dict[f"param_{k}"] = str(v)
        mlflow.log_params(log_params_dict)

        # Metrics
        all_metrics = {
            "accuracy": round(acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1_score": round(f1, 4),
            "roc_auc": round(roc_auc, 4),
            "pr_auc": round(pr_auc, 4),
            "brier_score": round(brier, 4),
            "log_loss": round(logloss, 4),
            "mean_latency_ms": lat_metrics["mean_latency_ms"],
            "p50_latency_ms": lat_metrics["p50_latency_ms"],
            "p95_latency_ms": lat_metrics["p95_latency_ms"],
            "p99_latency_ms": lat_metrics["p99_latency_ms"],
            "throughput_rps": lat_metrics["throughput_rps"],
            "sla_passed": 1 if lat_metrics["sla_passed"] else 0,
        }
        mlflow.log_metrics(all_metrics)

        # Artifacts: Diagnostic plots
        plots_dir = temp_dir / f"{model_name}_plots"
        saved_plots = save_diagnostic_plots(
            y_true=y_test,
            y_prob=y_prob,
            y_pred=y_pred,
            feature_names=feature_names,
            feature_importances=feature_importances,
            output_dir=plots_dir,
        )
        for plot_path in saved_plots.values():
            mlflow.log_artifact(str(plot_path), artifact_path="diagnostic_plots")

        # Artifacts: Reports & JSON
        report_dict = classification_report(
            y_test, y_pred, output_dict=True, zero_division=0
        )
        report_path = plots_dir / "classification_report.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report_dict, f, indent=2)
        mlflow.log_artifact(str(report_path), artifact_path="reports")

        metrics_path = plots_dir / "metrics_summary.json"
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(all_metrics, f, indent=2)
        mlflow.log_artifact(str(metrics_path), artifact_path="reports")

        # Log Model artifact with signature and input example
        sample_input = X_test.head(5)
        sig = infer_signature(sample_input, y_pred[:5])
        mlflow.sklearn.log_model(
            name="model",
            sk_model=pipeline,
            signature=sig,
            input_example=sample_input,
            serialization_format="cloudpickle",
        )

        print(
            f"✅ Logged {model_name} | Run ID: {run_id} | ROC-AUC: {roc_auc:.4f} | F1: {f1:.4f} | p99: {lat_metrics['p99_latency_ms']:.4f} ms"
        )

        return RunSummary(
            run_id=run_id,
            run_name=model_name,
            model_family=model_family,
            accuracy=acc,
            precision=prec,
            recall=rec,
            f1_score=f1,
            roc_auc=roc_auc,
            pr_auc=pr_auc,
            mean_latency_ms=lat_metrics["mean_latency_ms"],
            p95_latency_ms=lat_metrics["p95_latency_ms"],
            p99_latency_ms=lat_metrics["p99_latency_ms"],
            throughput_rps=lat_metrics["throughput_rps"],
            sla_passed=lat_metrics["sla_passed"],
        )


def print_comparison_table(summaries: list[RunSummary]) -> None:
    """Print an aligned terminal comparison table of all logged MLflow runs."""
    print("\n" + "=" * 105)
    print("🏆 MLFLOW EXPERIMENT COMPARISON SUMMARY")
    print("=" * 105)
    header = (
        f"{'Model / Run Name':<38} | {'Family':<15} | {'ROC-AUC':<8} | "
        f"{'F1':<6} | {'Recall':<6} | {'p99 Latency':<12} | {'SLA (<5ms)':<10}"
    )
    print(header)
    print("-" * 105)
    for s in summaries:
        sla_str = "✅ PASS" if s.sla_passed else "❌ FAIL"
        row = (
            f"{s.run_name:<38} | {s.model_family:<15} | {s.roc_auc:>8.4f} | "
            f"{s.f1_score:>6.4f} | {s.recall:>6.4f} | {f'{s.p99_latency_ms:.4f} ms':>12} | {sla_str:<10}"
        )
        print(row)
    print("=" * 105)

    best_roc = max(summaries, key=lambda x: x.roc_auc)
    best_latency = min(summaries, key=lambda x: x.p99_latency_ms)
    print(
        f"🥇 Highest Discrimination (ROC-AUC) : {best_roc.run_name} ({best_roc.roc_auc:.4f})"
    )
    print(
        f"⚡ Fastest Inference (p99 Latency)  : {best_latency.run_name} ({best_latency.p99_latency_ms:.4f} ms)"
    )
    print("=" * 105)


def run_all_experiments(
    experiment_name: str = "fraud-detection",
    tracking_uri: str = "sqlite:///mlflow.db",
    artifact_location: str | None = None,
    data_path: Path | None = Path("data/raw/transactions.csv"),
    num_samples: int = 1000,
    fraud_ratio: float = 0.05,
    seed: int = 42,
    test_size: float = 0.25,
    models: list[str] | None = None,
) -> list[RunSummary]:
    """Execute complete comparative ML experiment suite and log all runs to MLflow."""
    print("=" * 80)
    print(f"🔬 INITIALIZING MLFLOW EXPERIMENT: '{experiment_name}'")
    print(f"📍 Tracking URI   : {tracking_uri}")
    print("=" * 80)

    mlflow.set_tracking_uri(tracking_uri)

    # Ensure experiment exists with proper artifact location
    exp = mlflow.get_experiment_by_name(experiment_name)
    if exp is None:
        kwargs: dict[str, Any] = {}
        if artifact_location:
            kwargs["artifact_location"] = artifact_location
        exp_id = mlflow.create_experiment(experiment_name, **kwargs)
        mlflow.set_experiment(experiment_name)
        print(f"📁 Created new MLflow experiment '{experiment_name}' (ID: {exp_id})")
    else:
        mlflow.set_experiment(experiment_name)
        print(
            f"📁 Connected to existing MLflow experiment '{experiment_name}' (ID: {exp.experiment_id})"
        )

    # Load and prepare dataset
    df = load_or_generate_dataset(
        data_path=data_path,
        num_samples=num_samples,
        fraud_ratio=fraud_ratio,
        seed=seed,
    )
    df_feat = engineer_features(df)

    num_cols = [
        "amount",
        "log_amount",
        "device_trust_score",
        "velocity_1h",
        "country_mismatch",
        "is_high_risk_category",
        "is_extreme_amount",
        "is_untrusted_device",
        "is_critical_velocity",
    ]
    cat_cols = ["merchant_category"]
    feature_cols = num_cols + cat_cols

    X = df_feat[feature_cols]
    y = df_feat["is_fraud"].to_numpy()

    # Train / Test Split
    train_idx, test_idx = train_test_split(
        np.arange(len(df_feat)),
        test_size=test_size,
        random_state=seed,
        stratify=y,
    )
    X_train = X.iloc[train_idx].copy()
    y_train = y[train_idx]
    X_test = X.iloc[test_idx].copy()
    y_test = y[test_idx]
    df_test_full = df.iloc[test_idx].copy()

    print(
        f"📊 Dataset split: {len(X_train)} train samples, {len(X_test)} test samples (Fraud ratio: {y.mean():.1%})"
    )

    selected_models = set(
        models or ["detector", "logistic_regression", "random_forest"]
    )
    summaries: list[RunSummary] = []

    with tempfile.TemporaryDirectory() as tmp_str:
        temp_dir = Path(tmp_str)

        # 1. LowLatencyFraudDetector (Production Rule & Calibrated Engine)
        if "detector" in selected_models:
            det_summary = run_detector_experiment(
                df_test=df_test_full,
                temp_dir=temp_dir,
            )
            summaries.append(det_summary)

        # 2. Logistic Regression Baseline
        if "logistic_regression" in selected_models:
            lr_hyperparams = {
                "C": 1.0,
                "max_iter": 1000,
                "class_weight": "balanced",
                "random_state": seed,
            }
            lr_pipe = Pipeline(
                steps=[
                    ("preprocessor", build_preprocessor(cat_cols, num_cols)),
                    ("classifier", LogisticRegression(**lr_hyperparams)),
                ]
            )
            lr_summary = run_sklearn_experiment(
                model_name="LogisticRegression-L2-Balanced",
                pipeline=lr_pipe,
                hyperparams=lr_hyperparams,
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                temp_dir=temp_dir,
            )
            summaries.append(lr_summary)

        # 3. Random Forest Classifier
        if "random_forest" in selected_models:
            rf_hyperparams = {
                "n_estimators": 100,
                "max_depth": 6,
                "min_samples_split": 5,
                "class_weight": "balanced",
                "random_state": seed,
            }
            rf_pipe = Pipeline(
                steps=[
                    ("preprocessor", build_preprocessor(cat_cols, num_cols)),
                    ("classifier", RandomForestClassifier(**rf_hyperparams)),
                ]
            )
            rf_summary = run_sklearn_experiment(
                model_name="RandomForest-Ensemble-Depth6",
                pipeline=rf_pipe,
                hyperparams=rf_hyperparams,
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                temp_dir=temp_dir,
            )
            summaries.append(rf_summary)

    print_comparison_table(summaries)
    print("\n🔍 TO VIEW EXPERIMENT RESULTS IN MLFLOW UI:")
    print("   uv run mlflow server \\")
    print("     --backend-store-uri sqlite:///mlflow.db \\")
    print("     --default-artifact-root ./mlartifacts \\")
    print("     --host 127.0.0.1 --port 5000\n")

    return summaries


def main() -> None:
    """CLI runner for MLflow experiments."""
    parser = argparse.ArgumentParser(
        description="Run and log fraud detection experiments to MLflow."
    )
    parser.add_argument(
        "--experiment-name",
        default="fraud-detection",
        help="MLflow experiment name (default: fraud-detection)",
    )
    parser.add_argument(
        "--tracking-uri",
        default="sqlite:///mlflow.db",
        help="MLflow tracking URI (default: sqlite:///mlflow.db)",
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path("data/raw/transactions.csv"),
        help="Path to raw transactions CSV",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=1000,
        help="Number of samples if synthetic data generated (default: 1000)",
    )
    parser.add_argument(
        "--fraud-ratio",
        type=float,
        default=0.05,
        help="Fraud ratio if synthetic data generated (default: 0.05)",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.25,
        help="Test split fraction (default: 0.25)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["detector", "logistic_regression", "random_forest"],
        choices=["detector", "logistic_regression", "random_forest"],
        help="Models to evaluate and log",
    )

    args = parser.parse_args()

    run_all_experiments(
        experiment_name=args.experiment_name,
        tracking_uri=args.tracking_uri,
        data_path=args.data_path,
        num_samples=args.num_samples,
        fraud_ratio=args.fraud_ratio,
        seed=args.seed,
        test_size=args.test_size,
        models=args.models,
    )


if __name__ == "__main__":
    main()
