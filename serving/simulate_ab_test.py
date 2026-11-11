"""Simulate and evaluate an A/B testing experiment between Champion and Challenger models."""

from __future__ import annotations

import argparse
from typing import Any, TypedDict

from mlops_bits.pipeline.prepare import generate_transactions
from mlops_bits.schemas import RiskDecision, TransactionRequest
from serving.gateway.router import GatewayRouter, RoutingMode


class VariantStat(TypedDict):
    name: str
    count: int
    latencies: list[float]
    decisions: dict[str, int]


def run_ab_test_simulation(
    num_samples: int = 300,
    champion_ratio: float = 0.50,
) -> dict[str, Any]:
    """Execute controlled A/B test simulation across Champion and Challenger models."""
    print("=" * 80)
    print("🧪 STARTING MODEL DEPLOYMENT A/B TEST SIMULATION")
    print(
        f"   Target Split: {champion_ratio:.0%} Variant A (Champion) / {1 - champion_ratio:.0%} Variant B (Challenger)"
    )
    print("=" * 80)

    router = GatewayRouter()
    router.set_mode(RoutingMode.AB_TEST)
    router.update_ab_split(
        champion_ratio=champion_ratio, experiment_name="fraud_model_v1_vs_v2"
    )

    # 1. Verify User Stickiness (Deterministic Hashing)
    print("\n1. Verifying User Stickiness (Deterministic Cohort Mapping)...")
    sticky_user = "usr-persistent-42"
    user_variants = set()
    for i in range(10):
        sample_req = TransactionRequest(
            transaction_id=f"tx-sticky-{i}",
            user_id=sticky_user,
            amount=150.0 + i,
            currency="USD",
            merchant_category="retail",
            device_trust_score=0.9,
            location_country="US",
            billing_country="US",
            velocity_1h=1,
        )
        dec = router.route_and_predict(sample_req)
        user_variants.add(dec.experiment_group)

    is_sticky = len(user_variants) == 1
    variant_assigned = next(iter(user_variants))
    print(f"   User '{sticky_user}' tested 10 times across transactions.")
    print(f"   Assigned Variant Cohort: Variant {variant_assigned}")
    print(
        f"   Sticky Cohort Consistency: {'✅ PASS (100% Consistent)' if is_sticky else '❌ FAIL'}"
    )

    # 2. Run Main Experiment Traffic Stream
    print(f"\n2. Streaming {num_samples} Real-Time Transactions...")
    transactions = generate_transactions(
        num_samples=num_samples, fraud_ratio=0.07, seed=99
    )

    stats: dict[str, VariantStat] = {
        "A": {
            "name": "Champion",
            "count": 0,
            "latencies": [],
            "decisions": {d.value: 0 for d in RiskDecision},
        },
        "B": {
            "name": "Challenger",
            "count": 0,
            "latencies": [],
            "decisions": {d.value: 0 for d in RiskDecision},
        },
    }

    for raw in transactions:
        req = TransactionRequest(
            transaction_id=raw["transaction_id"],
            user_id=raw["user_id"],
            amount=raw["amount"],
            currency=raw["currency"],
            merchant_category=raw["merchant_category"],
            device_trust_score=raw["device_trust_score"],
            location_country=raw["location_country"],
            billing_country=raw["billing_country"],
            velocity_1h=raw["velocity_1h"],
        )
        decision = router.route_and_predict(req)
        grp = decision.experiment_group or "A"
        res = decision.response
        if res:
            stats[grp]["count"] += 1
            stats[grp]["latencies"].append(res.inference_latency_ms)
            stats[grp]["decisions"][res.decision.value] += 1

    # 3. Print Statistical Comparison Table
    print("\n" + "=" * 80)
    print("📊 A/B TEST EXPERIMENT RESULTS SUMMARY")
    print("=" * 80)
    header = f"{'Metric':<28} | {'Variant A (Champion)':<22} | {'Variant B (Challenger)':<22}"
    print(header)
    print("-" * 80)

    count_a = stats["A"]["count"]
    count_b = stats["B"]["count"]
    total = count_a + count_b

    print(
        f"{'Sample Count (Traffic Split)':<28} | {f'{count_a} ({count_a / total:.1%})':<22} | {f'{count_b} ({count_b / total:.1%})':<22}"
    )

    lat_a = stats["A"]["latencies"]
    lat_b = stats["B"]["latencies"]
    p95_a = sorted(lat_a)[int(len(lat_a) * 0.95)] if lat_a else 0.0
    p95_b = sorted(lat_b)[int(len(lat_b) * 0.95)] if lat_b else 0.0
    p99_a = sorted(lat_a)[int(len(lat_a) * 0.99)] if lat_a else 0.0
    p99_b = sorted(lat_b)[int(len(lat_b) * 0.99)] if lat_b else 0.0

    print(f"{'p95 Latency':<28} | {f'{p95_a:.4f} ms':<22} | {f'{p95_b:.4f} ms':<22}")
    print(f"{'p99 Latency':<28} | {f'{p99_a:.4f} ms':<22} | {f'{p99_b:.4f} ms':<22}")

    for dec_name in ["APPROVE", "REVIEW", "REJECT"]:
        cnt_a = stats["A"]["decisions"][dec_name]
        cnt_b = stats["B"]["decisions"][dec_name]
        rate_a = cnt_a / count_a if count_a > 0 else 0.0
        rate_b = cnt_b / count_b if count_b > 0 else 0.0
        label = f"{dec_name} Rate"
        val_a = f"{rate_a:.1%} ({cnt_a})"
        val_b = f"{rate_b:.1%} ({cnt_b})"
        print(f"{label:<28} | {val_a:<22} | {val_b:<22}")

    print("=" * 80)
    print("✅ A/B TEST SIMULATION COMPLETE\n")

    return {
        "variant_a": stats["A"],
        "variant_b": stats["B"],
        "is_sticky": is_sticky,
    }


def main() -> None:
    """CLI runner for A/B testing simulation."""
    parser = argparse.ArgumentParser(
        description="Simulate model A/B testing experiment."
    )
    parser.add_argument(
        "--samples", type=int, default=300, help="Total transactions to stream"
    )
    parser.add_argument(
        "--champion-ratio", type=float, default=0.50, help="Traffic ratio for Variant A"
    )
    args = parser.parse_args()

    run_ab_test_simulation(
        num_samples=args.samples,
        champion_ratio=args.champion_ratio,
    )


if __name__ == "__main__":
    main()
