"""Statistical data drift and boundary constraint monitor."""

import math
from collections.abc import Sequence

from mlops_bits.config import DriftConfig
from mlops_bits.schemas import DriftAnalysisRequest, DriftAnalysisResponse, DriftStatus


class DistributionDriftMonitor:
    """Calculates Population Stability Index (PSI) and monitors distribution shifts."""

    def __init__(self, config: DriftConfig | None = None) -> None:
        self.config = config or DriftConfig()

    def _compute_bin_edges(self, values: Sequence[float], num_bins: int) -> list[float]:
        """Compute equal-width bin boundaries based on baseline reference."""
        min_val = min(values)
        max_val = max(values)
        if math.isclose(min_val, max_val):
            # Degenerate case: single value distribution
            return [min_val - 1.0, max_val + 1.0]

        step = (max_val - min_val) / num_bins
        return [min_val + i * step for i in range(num_bins + 1)]

    def _bin_frequencies(
        self, values: Sequence[float], edges: Sequence[float]
    ) -> list[float]:
        """Calculate normalized fraction of values falling in each bin."""
        num_bins = len(edges) - 1
        counts = [0] * num_bins
        total = len(values)
        if total == 0:
            return [1.0 / num_bins] * num_bins

        for val in values:
            if val <= edges[0]:
                counts[0] += 1
            elif val >= edges[-1]:
                counts[-1] += 1
            else:
                for b in range(num_bins):
                    if edges[b] <= val < edges[b + 1]:
                        counts[b] += 1
                        break

        # Apply smoothing epsilon and normalize
        eps = self.config.epsilon
        smoothed = [c + eps for c in counts]
        smoothed_total = sum(smoothed)
        return [c / smoothed_total for c in smoothed]

    def calculate_psi(
        self, baseline: Sequence[float], current: Sequence[float], num_bins: int = 10
    ) -> tuple[float, list[float]]:
        """Calculate Population Stability Index (PSI) between baseline and current data."""
        if len(baseline) == 0 or len(current) == 0:
            raise ValueError("Baseline and current datasets must not be empty.")

        edges = self._compute_bin_edges(baseline, num_bins)
        p_expected = self._bin_frequencies(baseline, edges)
        q_actual = self._bin_frequencies(current, edges)

        psi = 0.0
        for actual, expected in zip(q_actual, p_expected):
            psi += (actual - expected) * math.log(actual / expected)

        return max(0.0, round(psi, 4)), edges

    def analyze(self, request: DriftAnalysisRequest) -> DriftAnalysisResponse:
        """Run full drift analysis and classify severity."""
        psi, _ = self.calculate_psi(
            request.baseline_values, request.current_values, request.num_bins
        )

        cfg = self.config
        if psi >= cfg.psi_critical_threshold:
            status = DriftStatus.SIGNIFICANT_DRIFT
            alert = True
        elif psi >= cfg.psi_warning_threshold:
            status = DriftStatus.MODERATE_DRIFT
            alert = False
        else:
            status = DriftStatus.NO_DRIFT
            alert = False

        mean_base = sum(request.baseline_values) / len(request.baseline_values)
        mean_curr = sum(request.current_values) / len(request.current_values)

        return DriftAnalysisResponse(
            feature_name=request.feature_name,
            psi=psi,
            drift_status=status,
            alert_triggered=alert,
            mean_baseline=round(mean_base, 3),
            mean_current=round(mean_curr, 3),
            min_current=min(request.current_values),
            max_current=max(request.current_values),
        )
