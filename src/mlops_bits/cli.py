"""Command-line interface for mlops-bits production service and benchmarking."""

import argparse
import sys
import time

from mlops_bits.config import FraudDetectorConfig
from mlops_bits.detector import LowLatencyFraudDetector
from mlops_bits.schemas import TransactionRequest


def run_benchmark(num_samples: int = 1000) -> None:
    """Run an in-memory latency and throughput benchmark."""
    print(f"🚀 Running benchmark with {num_samples:,} synthetic transactions...")
    detector = LowLatencyFraudDetector()

    tx = TransactionRequest(
        transaction_id="bench-001",
        user_id="user-999",
        amount=149.50,
        currency="USD",
        merchant_category="electronics",
        device_trust_score=0.95,
        location_country="US",
        billing_country="US",
        velocity_1h=2,
    )

    # Warmup
    for _ in range(50):
        detector.predict(tx)

    latencies_ms: list[float] = []
    start_total = time.perf_counter()
    for i in range(num_samples):
        # Vary transaction slightly
        tx_i = tx.model_copy(update={"transaction_id": f"bench-{i}"})
        res = detector.predict(tx_i)
        latencies_ms.append(res.inference_latency_ms)
    elapsed_total = time.perf_counter() - start_total

    latencies_ms.sort()
    mean_lat = sum(latencies_ms) / len(latencies_ms)
    p50_lat = latencies_ms[int(num_samples * 0.50)]
    p95_lat = latencies_ms[int(num_samples * 0.95)]
    p99_lat = latencies_ms[int(num_samples * 0.99)]
    rps = num_samples / elapsed_total

    print("=" * 55)
    print("📊 BENCHMARK RESULTS (Low-Latency Inference SLA)")
    print("=" * 55)
    print(f"Total Transactions Scored : {num_samples:,}")
    print(f"Total Execution Time      : {elapsed_total:.3f} s")
    print(f"Throughput                : {rps:,.1f} req/s")
    print(f"Mean Latency              : {mean_lat:.4f} ms")
    print(f"p50 Median Latency        : {p50_lat:.4f} ms")
    print(f"p95 Latency               : {p95_lat:.4f} ms")
    print(f"p99 Latency               : {p99_lat:.4f} ms")
    print("=" * 55)

    if p99_lat <= FraudDetectorConfig().target_sla_p99_ms:
        print("✅ Low-latency SLA PASSED (< 5.0ms p99)")
    else:
        print("❌ Low-latency SLA VIOLATION")
        sys.exit(1)


def main(argv: list[str] | None = None) -> None:
    """CLI entry point for mlops-bits."""
    parser = argparse.ArgumentParser(
        prog="mlops-bits",
        description="Enterprise Fraud Detection MLOps Production Service",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # Serve command
    serve_parser = subparsers.add_parser(
        "serve", help="Start the FastAPI production server"
    )
    serve_parser.add_argument(
        "--host", default="0.0.0.0", help="Bind host (default: 0.0.0.0)"
    )
    serve_parser.add_argument(
        "--port", type=int, default=8000, help="Bind port (default: 8000)"
    )
    serve_parser.add_argument(
        "--reload", action="store_true", help="Enable dev auto-reload"
    )

    # Benchmark command
    bench_parser = subparsers.add_parser(
        "benchmark", help="Run low-latency inference benchmark"
    )
    bench_parser.add_argument(
        "--samples",
        type=int,
        default=1000,
        help="Number of benchmark iterations (default: 1000)",
    )

    # Experiment command
    exp_parser = subparsers.add_parser(
        "experiment", help="Run and log fraud detection experiments to MLflow"
    )
    exp_parser.add_argument(
        "--experiment-name",
        default="fraud-detection",
        help="MLflow experiment name (default: fraud-detection)",
    )
    exp_parser.add_argument(
        "--tracking-uri",
        default="sqlite:///mlflow.db",
        help="MLflow tracking URI (default: sqlite:///mlflow.db)",
    )
    exp_parser.add_argument(
        "--models",
        nargs="+",
        default=["detector", "logistic_regression", "random_forest"],
        choices=["detector", "logistic_regression", "random_forest"],
        help="Models to evaluate and log",
    )
    exp_parser.add_argument(
        "--test-size",
        type=float,
        default=0.25,
        help="Test split fraction (default: 0.25)",
    )
    exp_parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )

    # Gateway command
    gw_parser = subparsers.add_parser("gateway", help="Start the Unified API Gateway")
    gw_parser.add_argument(
        "--host", default="0.0.0.0", help="Bind host (default: 0.0.0.0)"
    )
    gw_parser.add_argument(
        "--port", type=int, default=8000, help="Bind port (default: 8000)"
    )
    gw_parser.add_argument(
        "--reload", action="store_true", help="Enable dev auto-reload"
    )

    # Worker command
    worker_parser = subparsers.add_parser(
        "worker", help="Start the Redis-backed async prediction worker"
    )
    worker_parser.add_argument("--redis-url", default=None, help="Redis connection URL")

    # Simulate command
    sim_parser = subparsers.add_parser(
        "simulate", help="Simulate Canary rollout or A/B testing"
    )
    sim_parser.add_argument(
        "scenario",
        choices=["canary", "ab-test"],
        help="Simulation scenario to execute",
    )

    # Monitor command (Evidently AI)
    mon_parser = subparsers.add_parser(
        "monitor", help="Evidently AI data drift, quality monitoring, and UI"
    )
    mon_subparsers = mon_parser.add_subparsers(
        dest="monitor_action", help="Monitoring action (report, test, ui)"
    )

    # monitor report
    rep_parser = mon_subparsers.add_parser(
        "report", help="Generate Evidently Data Drift & Quality report"
    )
    rep_parser.add_argument(
        "--baseline", default="data/raw/transactions.csv", help="Baseline dataset CSV"
    )
    rep_parser.add_argument("--current", default=None, help="Current dataset CSV")
    rep_parser.add_argument(
        "--drift-share",
        type=float,
        default=0.30,
        help="Drift share alert threshold (default: 0.30)",
    )
    rep_parser.add_argument(
        "--workspace",
        default="workspace",
        help="Evidently workspace path (default: workspace)",
    )
    rep_parser.add_argument(
        "--output-dir", default="reports/evidently", help="Report output directory"
    )

    # monitor test
    test_parser = mon_subparsers.add_parser(
        "test", help="Run Evidently automated Test Suite for CI/CD gating"
    )
    test_parser.add_argument(
        "--baseline", default="data/raw/transactions.csv", help="Baseline dataset CSV"
    )
    test_parser.add_argument("--current", default=None, help="Current dataset CSV")
    test_parser.add_argument(
        "--fail-on-error", action="store_true", help="Exit with code 1 if tests fail"
    )

    # monitor ui
    ui_parser = mon_subparsers.add_parser(
        "ui", help="Launch Evidently UI dashboard server"
    )
    ui_parser.add_argument(
        "--host", default="0.0.0.0", help="Bind host (default: 0.0.0.0)"
    )
    ui_parser.add_argument(
        "--port", type=int, default=8003, help="Bind port (default: 8003)"
    )
    ui_parser.add_argument(
        "--workspace", default="workspace", help="Workspace path (default: workspace)"
    )

    args = parser.parse_args(argv)

    if args.command == "benchmark":
        run_benchmark(num_samples=args.samples)
    elif args.command == "serve":
        import uvicorn

        print(f"Starting server on {args.host}:{args.port}...")
        uvicorn.run(
            "mlops_bits.api:app", host=args.host, port=args.port, reload=args.reload
        )
    elif args.command == "gateway":
        from pathlib import Path

        repo_root = str(Path(__file__).resolve().parent.parent.parent)
        if repo_root not in sys.path:
            sys.path.insert(0, repo_root)

        import uvicorn

        print(f"Starting Unified API Gateway on {args.host}:{args.port}...")
        uvicorn.run(
            "serving.gateway.app:app",
            host=args.host,
            port=args.port,
            reload=args.reload,
        )
    elif args.command == "worker":
        from pathlib import Path

        repo_root = str(Path(__file__).resolve().parent.parent.parent)
        if repo_root not in sys.path:
            sys.path.insert(0, repo_root)

        from serving.async_worker.queue import JobQueueManager
        from serving.async_worker.worker import AsyncPredictionWorker

        print("Starting Async Prediction Worker...")
        queue_mgr = JobQueueManager(redis_url=args.redis_url)
        worker = AsyncPredictionWorker(queue_manager=queue_mgr)
        worker.run()
    elif args.command == "simulate":
        from pathlib import Path

        repo_root = str(Path(__file__).resolve().parent.parent.parent)
        if repo_root not in sys.path:
            sys.path.insert(0, repo_root)

        if args.scenario == "canary":
            from serving.simulate_canary import simulate_canary_rollout

            simulate_canary_rollout()
        elif args.scenario == "ab-test":
            from serving.simulate_ab_test import run_ab_test_simulation

            run_ab_test_simulation()
    elif args.command == "experiment":
        from pathlib import Path

        repo_root = str(Path(__file__).resolve().parent.parent.parent)
        if repo_root not in sys.path:
            sys.path.insert(0, repo_root)

        from experiments.run_experiment import run_all_experiments

        run_all_experiments(
            experiment_name=args.experiment_name,
            tracking_uri=args.tracking_uri,
            models=args.models,
            test_size=args.test_size,
            seed=args.seed,
        )
    elif args.command == "monitor":
        from pathlib import Path

        repo_root = str(Path(__file__).resolve().parent.parent.parent)
        if repo_root not in sys.path:
            sys.path.insert(0, repo_root)

        from mlops_bits.monitoring import (
            EvidentlyMonitor,
            InferenceTelemetryCollector,
            MonitoringConfig,
        )

        if args.monitor_action == "ui":
            print(
                f"🌐 Starting Evidently UI Dashboard on http://{args.host}:{args.port} (Workspace: {args.workspace})..."
            )
            import subprocess

            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "evidently",
                    "ui",
                    "--host",
                    args.host,
                    "--port",
                    str(args.port),
                    "--workspace",
                    args.workspace,
                ],
                check=False,
            )
        elif args.monitor_action == "test":
            print(
                f"🧪 Running Evidently Test Suite against baseline {args.baseline}..."
            )
            ref_df = InferenceTelemetryCollector.load_reference_dataset(args.baseline)
            if args.current:
                curr_df = InferenceTelemetryCollector.load_reference_dataset(
                    args.current
                )
            else:
                curr_df = ref_df.iloc[len(ref_df) // 2 :].copy()
                ref_df = ref_df.iloc[: len(ref_df) // 2].copy()

            monitor = EvidentlyMonitor()
            res = monitor.run_drift_test_suite(
                reference_data=ref_df, current_data=curr_df, save_html=True
            )
            print("=" * 60)
            print("🧪 EVIDENTLY AUTOMATED TEST SUITE SUMMARY")
            print("=" * 60)
            print(f"All Passed    : {'✅ YES' if res.all_passed else '❌ NO'}")
            print(f"Total Tests   : {res.total_tests}")
            print(f"Passed Tests  : {res.passed_tests}")
            print(f"Failed Tests  : {res.failed_tests}")
            print(f"HTML Report   : {res.report_html_path}")
            print("=" * 60)
            for t in res.test_details:
                icon = "✅" if t.status == "SUCCESS" else "❌"
                print(f"  {icon} [{t.status}] {t.name}: {t.description}")
            if getattr(args, "fail_on_error", False) and not res.all_passed:
                sys.exit(1)
        else:
            # Default to report
            baseline = getattr(args, "baseline", "data/raw/transactions.csv")
            current = getattr(args, "current", None)
            drift_share = getattr(args, "drift_share", 0.30)
            out_dir = getattr(args, "output_dir", "reports/evidently")
            ws_dir = getattr(args, "workspace", "workspace")

            print(
                f"📊 Running Evidently Data Drift Analysis (Threshold: {drift_share:.1%})..."
            )
            ref_df = InferenceTelemetryCollector.load_reference_dataset(baseline)
            if current:
                curr_df = InferenceTelemetryCollector.load_reference_dataset(current)
            else:
                curr_df = ref_df.iloc[len(ref_df) // 2 :].copy()
                ref_df = ref_df.iloc[: len(ref_df) // 2].copy()

            monitor = EvidentlyMonitor(
                MonitoringConfig(
                    drift_share_threshold=drift_share,
                    reports_dir=out_dir,
                    workspace_dir=ws_dir,
                )
            )
            summary = monitor.generate_data_drift_report(
                reference_data=ref_df,
                current_data=curr_df,
                save_html=True,
                save_json=True,
                sync_workspace=True,
            )
            print("=" * 60)
            print("📈 EVIDENTLY DATA DRIFT REPORT SUMMARY")
            print("=" * 60)
            print(
                f"Drift Status       : {'🚨 DRIFT DETECTED' if summary.drift_detected else '✅ NO DRIFT'}"
            )
            print(f"Drift Share        : {summary.drift_share:.1%}")
            print(
                f"Drifted Features   : {summary.drifted_features_count} / {summary.total_features}"
            )
            if summary.drifted_features:
                print(f"Drifted Columns    : {', '.join(summary.drifted_features)}")
            print(f"HTML Report        : {summary.report_html_path}")
            print(f"JSON Report        : {summary.report_json_path}")
            print(f"Workspace Snapshot : {summary.snapshot_id}")
            print("=" * 60)
    else:
        # Default behavior when run with no arguments: print greeting and quick benchmark
        print("Hello from mlops-bits! Enterprise Fraud Detection Module.")
        run_benchmark(num_samples=500)


if __name__ == "__main__":
    main()
