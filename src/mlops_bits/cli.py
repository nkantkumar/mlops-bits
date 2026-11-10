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

    args = parser.parse_args(argv)

    if args.command == "benchmark":
        run_benchmark(num_samples=args.samples)
    elif args.command == "serve":
        import uvicorn

        print(f"Starting server on {args.host}:{args.port}...")
        uvicorn.run(
            "mlops_bits.api:app", host=args.host, port=args.port, reload=args.reload
        )
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
    else:
        # Default behavior when run with no arguments: print greeting and quick benchmark
        print("Hello from mlops-bits! Enterprise Fraud Detection Module.")
        run_benchmark(num_samples=500)


if __name__ == "__main__":
    main()
