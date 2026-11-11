"""Background worker processing asynchronous transaction scoring jobs."""

from __future__ import annotations

import argparse
import logging
import signal
import time
from typing import Any

from mlops_bits.detector import LowLatencyFraudDetector
from mlops_bits.schemas import TransactionRequest
from serving.async_worker.queue import JobQueueManager

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] [AsyncWorker] %(message)s",
)
logger = logging.getLogger(__name__)


class AsyncPredictionWorker:
    """Consumes and scores fraud prediction jobs from Redis queue."""

    def __init__(
        self,
        queue_manager: JobQueueManager | None = None,
        detector: LowLatencyFraudDetector | None = None,
    ) -> None:
        self.queue_mgr = queue_manager or JobQueueManager()
        self.detector = detector or LowLatencyFraudDetector()
        self._running: bool = False

    def process_one_job(self, timeout: float = 1.0) -> bool:
        """Fetch and process a single prediction job. Returns True if a job was processed."""
        job_data = self.queue_mgr.get_job(timeout=timeout)
        if not job_data:
            return False

        job_id, payload = job_data
        t0 = time.perf_counter_ns()
        try:
            req = TransactionRequest.model_validate(payload)
            res = self.detector.predict(req)
            elapsed_ms = (time.perf_counter_ns() - t0) / 1_000_000.0

            result_dict: dict[str, Any] = res.model_dump()
            self.queue_mgr.complete_job(
                job_id=job_id,
                result=result_dict,
                processing_time_ms=elapsed_ms,
            )
            logger.info(
                "Processed job %s in %.3fms (Decision: %s, Score: %.4f)",
                job_id,
                elapsed_ms,
                res.decision.value,
                res.fraud_score,
            )
            return True
        except Exception as exc:
            logger.exception("Failed to process job %s", job_id)
            self.queue_mgr.fail_job(job_id=job_id, error_message=str(exc))
            return True

    def run(self, max_jobs: int | None = None, poll_interval: float = 0.1) -> None:
        """Run worker loop until interrupted or max_jobs processed."""
        self._running = True
        logger.info(
            "Starting Async Prediction Worker (Backend: %s)...",
            "Redis" if self.queue_mgr.is_redis_connected else "In-Memory Queue",
        )

        def handle_signal(sig: int, frame: Any) -> None:
            logger.info("Termination signal received. Shutting down worker...")
            self._running = False

        signal.signal(signal.SIGINT, handle_signal)
        signal.signal(signal.SIGTERM, handle_signal)

        jobs_completed = 0
        while self._running:
            processed = self.process_one_job(timeout=poll_interval)
            if processed:
                jobs_completed += 1
                if max_jobs and jobs_completed >= max_jobs:
                    logger.info(
                        "Reached maximum requested jobs (%d). Stopping.", max_jobs
                    )
                    break
            else:
                time.sleep(poll_interval)

        logger.info(
            "Worker stopped gracefully. Total jobs completed: %d", jobs_completed
        )


def main() -> None:
    """CLI entrypoint for standalone async worker."""
    parser = argparse.ArgumentParser(description="Run Redis async prediction worker.")
    parser.add_argument("--redis-url", default=None, help="Redis connection URL")
    parser.add_argument(
        "--max-jobs", type=int, default=None, help="Max jobs to process before exit"
    )
    args = parser.parse_args()

    qm = JobQueueManager(redis_url=args.redis_url)
    worker = AsyncPredictionWorker(queue_manager=qm)
    worker.run(max_jobs=args.max_jobs)


if __name__ == "__main__":
    main()
