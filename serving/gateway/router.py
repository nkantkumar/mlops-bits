"""Dynamic traffic routing engine supporting A/B testing, Canary rollout, and auto-rollback."""

from __future__ import annotations

import enum
import hashlib
import random
import threading
import time
from dataclasses import dataclass
from typing import Any

from mlops_bits.schemas import TransactionRequest, TransactionResponse
from serving.gateway.models_client import MultiModelClient


class RoutingMode(str, enum.Enum):
    """Operational mode for the Gateway router."""

    DEFAULT = "default"  # 100% Champion
    AB_TEST = "ab_test"  # Controlled A/B experiment
    CANARY = "canary"  # Canary gradual rollout


@dataclass
class ABTestConfig:
    """Configuration for controlled A/B experiment."""

    experiment_name: str = "fraud_scorer_v1_vs_v2"
    champion_ratio: float = 0.50  # 50% A (Champion), 50% B (Challenger)
    hash_salt: str = "mlops_bits_exp_salt_2026"
    active: bool = True


@dataclass
class CanaryConfig:
    """Configuration for canary deployment rollout."""

    canary_weight: float = 0.10  # 10% to canary initially
    target_sla_p99_ms: float = 5.0
    max_error_rate: float = 0.05
    auto_rollback_enabled: bool = True
    rollback_triggered: bool = False
    rollback_reason: str | None = None


@dataclass
class RoutingDecision:
    """Metadata describing how a request was routed."""

    selected_model: str  # "champion" or "challenger"
    routing_mode: RoutingMode
    experiment_group: str | None = None  # "A" or "B" for A/B testing
    canary_weight: float | None = None
    response: TransactionResponse | None = None


class GatewayRouter:
    """Thread-safe dynamic traffic router for model serving."""

    def __init__(self, models_client: MultiModelClient | None = None) -> None:
        self.client = models_client or MultiModelClient()
        self.mode = RoutingMode.DEFAULT
        self.ab_config = ABTestConfig()
        self.canary_config = CanaryConfig()
        self._lock = threading.Lock()

        # Telemetry & metrics
        self.metrics: dict[str, Any] = {
            "total_requests": 0,
            "champion_requests": 0,
            "challenger_requests": 0,
            "canary_requests": 0,
            "canary_latencies_ms": [],
            "canary_errors": 0,
            "decisions_divergence": 0,
            "started_at": time.time(),
        }

    def set_mode(self, mode: RoutingMode | str) -> None:
        """Update active routing mode."""
        with self._lock:
            self.mode = RoutingMode(mode)

    def update_canary_weight(self, weight: float) -> None:
        """Update canary traffic fraction [0.0, 1.0]."""
        clamped = max(0.0, min(1.0, float(weight)))
        with self._lock:
            self.canary_config.canary_weight = clamped
            if clamped > 0.0:
                self.canary_config.rollback_triggered = False
                self.canary_config.rollback_reason = None

    def trigger_rollback(self, reason: str = "Manual emergency rollback") -> None:
        """Immediately revert all traffic to Champion."""
        with self._lock:
            self.canary_config.canary_weight = 0.0
            self.canary_config.rollback_triggered = True
            self.canary_config.rollback_reason = reason
            self.mode = RoutingMode.DEFAULT

    def update_ab_split(
        self, champion_ratio: float, experiment_name: str | None = None
    ) -> None:
        """Update A/B testing ratio and experiment name."""
        with self._lock:
            self.ab_config.champion_ratio = max(0.0, min(1.0, float(champion_ratio)))
            if experiment_name:
                self.ab_config.experiment_name = experiment_name

    def route_and_predict(self, req: TransactionRequest) -> RoutingDecision:
        """Route request according to active strategy and execute prediction."""
        with self._lock:
            current_mode = self.mode

        if current_mode == RoutingMode.AB_TEST:
            return self._route_ab_test(req)
        elif current_mode == RoutingMode.CANARY:
            return self._route_canary(req)
        else:
            return self._route_default(req)

    def _route_default(self, req: TransactionRequest) -> RoutingDecision:
        """Route 100% of traffic to production Champion model."""
        res = self.client.predict_champion(req)
        with self._lock:
            self.metrics["total_requests"] += 1
            self.metrics["champion_requests"] += 1
        return RoutingDecision(
            selected_model="champion",
            routing_mode=RoutingMode.DEFAULT,
            response=res,
        )

    def _route_ab_test(self, req: TransactionRequest) -> RoutingDecision:
        """Deterministically route based on user_id hash for consistent user experience."""
        salt = self.ab_config.hash_salt
        token = f"{req.user_id}:{salt}".encode()
        hash_val = int(hashlib.md5(token).hexdigest()[:8], 16) % 100
        threshold = int(self.ab_config.champion_ratio * 100)

        if hash_val < threshold:
            group = "A"
            model_name = "champion"
            res = self.client.predict_champion(req)
        else:
            group = "B"
            model_name = "challenger"
            res = self.client.predict_challenger(req)

        with self._lock:
            self.metrics["total_requests"] += 1
            if model_name == "champion":
                self.metrics["champion_requests"] += 1
            else:
                self.metrics["challenger_requests"] += 1

        return RoutingDecision(
            selected_model=model_name,
            routing_mode=RoutingMode.AB_TEST,
            experiment_group=group,
            response=res,
        )

    def _route_canary(self, req: TransactionRequest) -> RoutingDecision:
        """Stochastically route fraction of traffic to Canary (Challenger) model."""
        weight = self.canary_config.canary_weight
        use_canary = (
            random.random() < weight
        ) and not self.canary_config.rollback_triggered

        if use_canary:
            model_name = "challenger"
            try:
                res = self.client.predict_challenger(req)
                lat = res.inference_latency_ms
                self._record_canary_telemetry(latency_ms=lat, is_error=False)
            except Exception:  # noqa: BLE001
                self._record_canary_telemetry(latency_ms=999.0, is_error=True)
                # Fallback to champion on canary exception
                res = self.client.predict_champion(req)
                model_name = "champion (canary-fallback)"
        else:
            model_name = "champion"
            res = self.client.predict_champion(req)

        with self._lock:
            self.metrics["total_requests"] += 1
            if "challenger" in model_name:
                self.metrics["canary_requests"] += 1
            else:
                self.metrics["champion_requests"] += 1

        return RoutingDecision(
            selected_model=model_name,
            routing_mode=RoutingMode.CANARY,
            canary_weight=weight,
            response=res,
        )

    def _record_canary_telemetry(self, latency_ms: float, is_error: bool) -> None:
        """Monitor canary health and auto-trigger rollback if SLA is violated."""
        with self._lock:
            if is_error:
                self.metrics["canary_errors"] += 1

            history: list[float] = self.metrics["canary_latencies_ms"]
            history.append(latency_ms)
            if len(history) > 200:
                history.pop(0)

            # Auto-rollback evaluation
            if self.canary_config.auto_rollback_enabled and len(history) >= 20:
                p95 = sorted(history)[int(len(history) * 0.95)]
                if p95 > self.canary_config.target_sla_p99_ms:
                    reason = f"Canary p95 latency ({p95:.2f}ms) breached target SLA ({self.canary_config.target_sla_p99_ms}ms)"
                    self.canary_config.canary_weight = 0.0
                    self.canary_config.rollback_triggered = True
                    self.canary_config.rollback_reason = reason
                    self.mode = RoutingMode.DEFAULT

    def get_status_report(self) -> dict[str, Any]:
        """Generate comprehensive operational and routing report."""
        with self._lock:
            total = int(self.metrics["total_requests"])
            champ = int(self.metrics["champion_requests"])
            chall = int(self.metrics["challenger_requests"])
            canary = int(self.metrics["canary_requests"])
            canary_lats: list[float] = list(self.metrics["canary_latencies_ms"])
            p95_canary = (
                sorted(canary_lats)[int(len(canary_lats) * 0.95)]
                if canary_lats
                else 0.0
            )

            return {
                "active_mode": self.mode.value,
                "ab_test": {
                    "active": self.ab_config.active,
                    "experiment_name": self.ab_config.experiment_name,
                    "champion_ratio": self.ab_config.champion_ratio,
                    "challenger_ratio": round(1.0 - self.ab_config.champion_ratio, 2),
                },
                "canary": {
                    "weight": self.canary_config.canary_weight,
                    "target_sla_p99_ms": self.canary_config.target_sla_p99_ms,
                    "p95_observed_ms": round(p95_canary, 4),
                    "rollback_triggered": self.canary_config.rollback_triggered,
                    "rollback_reason": self.canary_config.rollback_reason,
                },
                "traffic_stats": {
                    "total_requests": total,
                    "champion_requests": champ,
                    "challenger_requests": chall,
                    "canary_requests": canary,
                    "champion_share": round(champ / total, 3) if total > 0 else 0.0,
                    "challenger_share": round(chall / total, 3) if total > 0 else 0.0,
                },
            }
