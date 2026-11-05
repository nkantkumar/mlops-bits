"""Performance and low-latency SLA benchmark tests for the fraud detection module."""

import time

from mlops_bits.detector import LowLatencyFraudDetector
from mlops_bits.schemas import TransactionRequest


def test_inference_latency_sla_under_5ms(
    detector: LowLatencyFraudDetector, legitimate_transaction: TransactionRequest
) -> None:
    """Rigorous performance verification asserting mean < 1ms and p99 < 5ms over 1,000 transactions."""
    num_iterations = 1_000

    # 1. Warm-up
    for _ in range(50):
        detector.predict(legitimate_transaction)

    # 2. Benchmark run
    latencies_ms: list[float] = []
    for i in range(num_iterations):
        sample_tx = legitimate_transaction.model_copy(
            update={"transaction_id": f"lat-{i}", "amount": 10.0 + (i % 200)}
        )
        start_ns = time.perf_counter_ns()
        res = detector.predict(sample_tx)
        elapsed_ms = (time.perf_counter_ns() - start_ns) / 1_000_000.0

        latencies_ms.append(elapsed_ms)
        assert res.decision is not None

    latencies_ms.sort()
    mean_latency = sum(latencies_ms) / num_iterations
    p95_latency = latencies_ms[int(num_iterations * 0.95)]
    p99_latency = latencies_ms[int(num_iterations * 0.99)]

    # Assert SLA constraints
    assert mean_latency <= detector.config.target_sla_mean_ms, (
        f"Mean latency {mean_latency:.4f}ms exceeded SLA ({detector.config.target_sla_mean_ms}ms)"
    )
    assert p99_latency <= detector.config.target_sla_p99_ms, (
        f"p99 latency {p99_latency:.4f}ms exceeded SLA ({detector.config.target_sla_p99_ms}ms)"
    )
    assert p95_latency <= p99_latency


def test_batch_latency_scaling(
    detector: LowLatencyFraudDetector, legitimate_transaction: TransactionRequest
) -> None:
    """Verify that batch processing 500 items achieves high throughput (> 5,000 req/s)."""
    batch_size = 500
    txs = [
        legitimate_transaction.model_copy(update={"transaction_id": f"batch-{i}"})
        for i in range(batch_size)
    ]

    start = time.perf_counter()
    for tx in txs:
        detector.predict(tx)
    total_time_s = time.perf_counter() - start

    throughput = batch_size / total_time_s
    # In pure Python, our optimized logic easily exceeds 5,000 ops/sec
    assert throughput >= 5_000, (
        f"Throughput {throughput:.1f} rps fell below 5,000 rps target"
    )
