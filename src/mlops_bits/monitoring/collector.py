"""Thread-safe inference telemetry collector for live production drift monitoring."""

from __future__ import annotations

import collections
import threading
import time
from pathlib import Path
from typing import Any

import pandas as pd

from mlops_bits.monitoring.config import MonitoringConfig
from mlops_bits.monitoring.schemas import InferenceTelemetryRecord
from mlops_bits.schemas import TransactionRequest, TransactionResponse


class InferenceTelemetryCollector:
    """Collects and stores inference inputs and outputs in a sliding ring buffer."""

    def __init__(self, config: MonitoringConfig | None = None) -> None:
        self.config = config or MonitoringConfig()
        self._lock = threading.Lock()
        self._buffer: collections.deque[dict[str, Any]] = collections.deque(
            maxlen=self.config.window_size
        )

    def record(
        self,
        request: TransactionRequest,
        response: TransactionResponse,
    ) -> InferenceTelemetryRecord:
        """Record a single prediction transaction into the live buffer."""
        record = InferenceTelemetryRecord(
            transaction_id=request.transaction_id,
            amount=request.amount,
            currency=request.currency,
            merchant_category=request.merchant_category,
            device_trust_score=request.device_trust_score,
            location_country=request.location_country,
            billing_country=request.billing_country,
            velocity_1h=request.velocity_1h,
            fraud_score=response.fraud_score,
            is_fraud=response.is_fraud,
            decision=response.decision.value,
            latency_ms=response.inference_latency_ms,
            timestamp=time.time(),
        )

        with self._lock:
            self._buffer.append(record.model_dump())

        return record

    def get_current_dataframe(self) -> pd.DataFrame:
        """Retrieve recent inference events as a pandas DataFrame."""
        with self._lock:
            if not self._buffer:
                return pd.DataFrame()
            return pd.DataFrame(list(self._buffer))

    def count(self) -> int:
        """Return number of events currently in the buffer."""
        with self._lock:
            return len(self._buffer)

    def clear(self) -> None:
        """Clear all events from the buffer."""
        with self._lock:
            self._buffer.clear()

    @staticmethod
    def load_reference_dataset(
        path: str | Path = "data/raw/transactions.csv",
    ) -> pd.DataFrame:
        """Load reference/baseline transactions from a CSV file.

        If the file does not exist, generates synthetic transactions for fallback.
        """
        csv_path = Path(path)
        if csv_path.exists():
            return pd.read_csv(csv_path)

        # Fallback to generating reference transactions
        from mlops_bits.pipeline.prepare import generate_transactions

        records = generate_transactions(num_samples=500, fraud_ratio=0.03, seed=42)
        return pd.DataFrame(records)
