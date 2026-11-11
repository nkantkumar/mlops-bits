"""Simulate and verify automated Canary Rollout and rollback for model deployment."""

from __future__ import annotations

import argparse
import time
from typing import Any

from mlops_bits.pipeline.prepare import generate_transactions
from mlops_bits.schemas import TransactionRequest
from serving.gateway.router import GatewayRouter, RoutingMode


def render_ascii_bar(ratio: float, width: int = 30) -> str:
    """Render an ASCII traffic distribution bar."""
    filled = round(ratio * width)
    empty = width - filled
    return f"[{'█' * filled}{'░' * empty}] {ratio * 100:>5.1f}%"


def simulate_canary_rollout(
    total_steps: int = 5,
    samples_per_step: int = 60,
    simulate_failure: bool = True,
) -> None:
    """Run an automated multi-stage canary rollout simulation."""
    print("=" * 80)
    print("🚀 STARTING CANARY ROLLOUT SIMULATION (Target SLA: < 5.0ms)")
    print("=" * 80)

    router = GatewayRouter()
    transactions = generate_transactions(
        num_samples=samples_per_step * total_steps, fraud_ratio=0.08, seed=42
    )

    stages: list[dict[str, Any]] = [
        {"name": "Stage 0: Baseline (100% Champion)", "weight": 0.0},
        {"name": "Stage 1: Initial Canary (10% Traffic)", "weight": 0.10},
        {"name": "Stage 2: Expanded Canary (25% Traffic)", "weight": 0.25},
        {"name": "Stage 3: Balanced Canary (50% Traffic)", "weight": 0.50},
        {"name": "Stage 4: Full Promotion (100% Canary)", "weight": 1.00},
    ]

    tx_idx = 0
    for stage in stages:
        stage_name = stage["name"]
        weight = float(stage["weight"])

        router.set_mode(RoutingMode.CANARY)
        router.update_canary_weight(weight)

        champ_count = 0
        canary_count = 0
        latencies: list[float] = []

        for _ in range(samples_per_step):
            raw = transactions[tx_idx]
            tx_idx += 1
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
            if decision.response:
                latencies.append(decision.response.inference_latency_ms)
            if "challenger" in decision.selected_model:
                canary_count += 1
            else:
                champ_count += 1

        total = champ_count + canary_count
        canary_ratio = canary_count / total if total > 0 else 0.0
        p99_lat = sorted(latencies)[int(len(latencies) * 0.99)] if latencies else 0.0

        print(f"\n📍 {stage_name}")
        print(f"   Target Weight       : {weight:.0%}")
        print(f"   Observed Canary     : {render_ascii_bar(canary_ratio)}")
        print(f"   Counts (Champ/Canary: {champ_count} / {canary_count} reqs)")
        print(
            f"   Observed p99 Latency: {p99_lat:.4f} ms (SLA: {'✅ PASS' if p99_lat < 5.0 else '❌ BREACH'})"
        )
        time.sleep(0.05)

    if simulate_failure:
        print("\n" + "=" * 80)
        print("⚠️ SIMULATING ANOMALOUS CANARY BREACH & AUTO-ROLLBACK TRIGGER")
        print("=" * 80)

        # Set canary weight to 30% and inject high latencies into telemetry
        router.set_mode(RoutingMode.CANARY)
        router.update_canary_weight(0.30)
        print(
            "   Injecting synthetic latency degradation into Canary model (> 10ms)..."
        )

        for i in range(25):
            # Artificially record high latency to test auto-rollback safety circuit
            router._record_canary_telemetry(latency_ms=12.5 + i * 0.2, is_error=False)

        status = router.get_status_report()
        canary_status = status["canary"]
        print(f"   Auto-Rollback Triggered : {canary_status['rollback_triggered']}")
        print(f"   Current Active Mode     : {status['active_mode']}")
        print(f"   Current Canary Weight   : {canary_status['weight']}")
        print(f"   Rollback Reason         : {canary_status['rollback_reason']}")

        if canary_status["rollback_triggered"] and canary_status["weight"] == 0.0:
            print(
                "   🛡️ SAFETY CIRCUIT PASSED: Traffic immediately and automatically reverted to 100% Champion!"
            )
        else:
            print("   ❌ Rollback circuit failed.")

    print("\n" + "=" * 80)
    print("✅ CANARY ROLLOUT & AUTO-ROLLBACK SIMULATION COMPLETE")
    print("=" * 80 + "\n")


def main() -> None:
    """CLI runner for canary simulation."""
    parser = argparse.ArgumentParser(
        description="Simulate canary rollout and automatic rollback."
    )
    parser.add_argument(
        "--samples", type=int, default=60, help="Samples per rollout stage"
    )
    parser.add_argument(
        "--no-failure", action="store_true", help="Skip failure/rollback test"
    )
    args = parser.parse_args()

    simulate_canary_rollout(
        samples_per_step=args.samples,
        simulate_failure=not args.no_failure,
    )


if __name__ == "__main__":
    main()
