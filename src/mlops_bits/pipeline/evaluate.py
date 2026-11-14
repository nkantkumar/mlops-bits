"""DVC Pipeline Stage 2: Evaluate fraud detector performance and distribution drift."""

import csv
import json
import random
import time
from pathlib import Path
from typing import Any

import yaml

from mlops_bits.config import FraudDetectorConfig
from mlops_bits.detector import LowLatencyFraudDetector
from mlops_bits.drift import DistributionDriftMonitor
from mlops_bits.monitoring import EvidentlyMonitor, MonitoringConfig
from mlops_bits.schemas import DriftAnalysisRequest, RiskDecision, TransactionRequest


def load_params() -> tuple[dict[str, Any], dict[str, Any]]:
    """Load evaluation and monitoring parameters from params.yaml."""
    params_file = Path("params.yaml")
    if not params_file.exists():
        raise FileNotFoundError(f"Missing {params_file}")
    with open(params_file, "r", encoding="utf-8") as f:
        params: dict[str, Any] = yaml.safe_load(f)
    return params.get("evaluate", {}), params.get("monitoring", {})


def load_transactions_from_csv(csv_path: Path) -> list[TransactionRequest]:
    """Parse transactions from CSV into validated TransactionRequest schemas."""
    transactions: list[TransactionRequest] = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            tx = TransactionRequest(
                transaction_id=row["transaction_id"],
                user_id=row["user_id"],
                amount=float(row["amount"]),
                currency=row["currency"],
                merchant_category=row["merchant_category"],
                device_trust_score=float(row["device_trust_score"]),
                location_country=row["location_country"],
                billing_country=row["billing_country"],
                velocity_1h=int(row["velocity_1h"]),
            )
            transactions.append(tx)
    return transactions


def main() -> None:
    """Execute evaluation and drift analysis stage."""
    eval_params, mon_params = load_params()
    input_path = Path(eval_params.get("input_path", "data/raw/transactions.csv"))
    metrics_path = Path(eval_params.get("metrics_output", "metrics.json"))
    drift_bins = int(eval_params.get("drift_num_bins", 10))

    if not input_path.exists():
        raise FileNotFoundError(f"Data input file not found: {input_path}")

    transactions = load_transactions_from_csv(input_path)
    total_tx = len(transactions)
    print(f"🔬 Evaluating {total_tx} transactions from {input_path}...")

    detector = LowLatencyFraudDetector(FraudDetectorConfig())
    drift_monitor = DistributionDriftMonitor()
    mon_config = MonitoringConfig(
        drift_share_threshold=float(mon_params.get("drift_share_threshold", 0.30)),
        stat_test_threshold=float(mon_params.get("stat_test_threshold", 0.05)),
        reports_dir=str(mon_params.get("reports_dir", "reports/evidently")),
        workspace_dir=str(mon_params.get("workspace_dir", "workspace")),
    )
    evidently_monitor = EvidentlyMonitor(mon_config)

    # 1. Warm-up
    for _ in range(25):
        if transactions:
            detector.predict(transactions[0])

    # 2. Performance & Decision Scoring
    latencies_ms: list[float] = []
    decisions = {
        RiskDecision.APPROVE: 0,
        RiskDecision.REVIEW: 0,
        RiskDecision.REJECT: 0,
    }
    fraud_count = 0
    amounts: list[float] = []

    start_total = time.perf_counter()
    for tx in transactions:
        start_ns = time.perf_counter_ns()
        res = detector.predict(tx)
        elapsed_ms = (time.perf_counter_ns() - start_ns) / 1_000_000.0

        latencies_ms.append(elapsed_ms)
        decisions[res.decision] += 1
        if res.is_fraud:
            fraud_count += 1
        amounts.append(tx.amount)
    total_eval_time_s = time.perf_counter() - start_total

    latencies_ms.sort()
    mean_lat = sum(latencies_ms) / total_tx if total_tx > 0 else 0.0
    p95_lat = latencies_ms[int(total_tx * 0.95)] if total_tx > 0 else 0.0
    p99_lat = latencies_ms[int(total_tx * 0.99)] if total_tx > 0 else 0.0
    throughput = total_tx / total_eval_time_s if total_eval_time_s > 0 else 0.0

    # 3. Distribution Drift Evaluation against expected consumer baseline
    random.seed(42)
    expected_baseline_amounts = [
        round(random.lognormvariate(3.9, 0.6), 2) for _ in range(total_tx)
    ]
    drift_report = drift_monitor.analyze(
        DriftAnalysisRequest(
            feature_name="transaction_amount",
            baseline_values=expected_baseline_amounts,
            current_values=amounts,
            num_bins=drift_bins,
        )
    )

    # 4. Evidently AI Full Feature Data Drift & Test Suite
    print("📈 Running Evidently AI Data Drift & Quality Evaluation...")
    ref_tx = transactions[: max(1, total_tx // 2)]
    curr_tx = transactions[max(1, total_tx // 2) :]
    evidently_drift = evidently_monitor.generate_data_drift_report(
        reference_data=ref_tx,
        current_data=curr_tx,
        save_html=True,
        save_json=True,
        sync_workspace=True,
    )
    evidently_suite = evidently_monitor.run_drift_test_suite(
        reference_data=ref_tx,
        current_data=curr_tx,
        save_html=True,
    )

    metrics_data = {
        "total_transactions": total_tx,
        "fraud_count": fraud_count,
        "approval_rate": round(decisions[RiskDecision.APPROVE] / total_tx, 4)
        if total_tx > 0
        else 0.0,
        "review_rate": round(decisions[RiskDecision.REVIEW] / total_tx, 4)
        if total_tx > 0
        else 0.0,
        "reject_rate": round(decisions[RiskDecision.REJECT] / total_tx, 4)
        if total_tx > 0
        else 0.0,
        "mean_latency_ms": round(mean_lat, 4),
        "p95_latency_ms": round(p95_lat, 4),
        "p99_latency_ms": round(p99_lat, 4),
        "throughput_req_per_s": round(throughput, 1),
        "sla_passed": p99_lat <= detector.config.target_sla_p99_ms,
        "amount_psi": drift_report.psi,
        "drift_status": drift_report.drift_status.value,
        "drift_alert": drift_report.alert_triggered,
        "evidently_drift_detected": evidently_drift.drift_detected,
        "evidently_drift_share": evidently_drift.drift_share,
        "evidently_drifted_features_count": evidently_drift.drifted_features_count,
        "evidently_test_suite_passed": evidently_suite.all_passed,
        "evidently_passed_tests": evidently_suite.passed_tests,
        "evidently_total_tests": evidently_suite.total_tests,
    }

    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_data, f, indent=2)

    print(f"📊 Metrics written to {metrics_path}")
    print(json.dumps(metrics_data, indent=2))


if __name__ == "__main__":
    main()
