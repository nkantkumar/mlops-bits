"""Redis-backed queue manager with thread-safe in-memory fallback."""

from __future__ import annotations

import json
import logging
import os
import queue
import threading
import time
import uuid
from typing import Any

import redis

logger = logging.getLogger(__name__)

QUEUE_KEY = "mlops_bits:fraud_prediction_queue"
JOB_PREFIX = "mlops_bits:fraud_job:"
DEFAULT_JOB_TTL = 3600  # 1 hour expiration for completed results


class JobQueueManager:
    """Manages asynchronous prediction job queue and result storage."""

    def __init__(self, redis_url: str | None = None) -> None:
        self.redis_url = redis_url or os.getenv("REDIS_URL", "redis://localhost:6379/0")
        self._redis_client: redis.Redis | None = None
        self._use_in_memory: bool = False

        # In-memory fallback structures
        self._mem_queue: queue.Queue[tuple[str, dict[str, Any]]] = queue.Queue()
        self._mem_storage: dict[str, dict[str, Any]] = {}
        self._lock = threading.Lock()

        self._init_connection()

    def _init_connection(self) -> None:
        """Attempt connection to Redis; fall back to in-memory if unreachable."""
        try:
            url = self.redis_url or "redis://localhost:6379/0"
            client = redis.Redis.from_url(
                url,
                socket_connect_timeout=0.5,
                socket_timeout=1.0,
                decode_responses=True,
            )
            client.ping()
            self._redis_client = client
            self._use_in_memory = False
            logger.info("Connected to Redis backend at %s", self.redis_url)
        except (redis.RedisError, OSError) as exc:
            self._use_in_memory = True
            self._redis_client = None
            logger.warning(
                "Redis connection failed (%s); operating with thread-safe in-memory queue fallback.",
                exc,
            )

    @property
    def is_redis_connected(self) -> bool:
        """Return True if currently backed by live Redis server."""
        return not self._use_in_memory and self._redis_client is not None

    def submit_job(self, transaction_data: dict[str, Any]) -> str:
        """Enqueue an incoming transaction scoring job."""
        job_id = f"job-{uuid.uuid4().hex[:12]}"
        now = time.time()
        initial_payload = {
            "job_id": job_id,
            "status": "PENDING",
            "created_at": now,
            "updated_at": now,
            "transaction_data": transaction_data,
            "result": None,
            "error": None,
        }

        if self._use_in_memory or self._redis_client is None:
            with self._lock:
                self._mem_storage[job_id] = initial_payload
            self._mem_queue.put((job_id, transaction_data))
        else:
            try:
                pipe = self._redis_client.pipeline()
                pipe.set(
                    f"{JOB_PREFIX}{job_id}",
                    json.dumps(initial_payload, default=str),
                    ex=DEFAULT_JOB_TTL,
                )
                queue_item = json.dumps(
                    {"job_id": job_id, "data": transaction_data}, default=str
                )
                pipe.rpush(QUEUE_KEY, queue_item)
                pipe.execute()
            except (redis.RedisError, OSError) as exc:
                logger.error("Redis write error, switching to in-memory: %s", exc)
                self._use_in_memory = True
                with self._lock:
                    self._mem_storage[job_id] = initial_payload
                self._mem_queue.put((job_id, transaction_data))

        return job_id

    def get_job(self, timeout: float = 1.0) -> tuple[str, dict[str, Any]] | None:
        """Dequeue the next prediction job (blocking up to timeout seconds)."""
        if self._use_in_memory or self._redis_client is None:
            try:
                return self._mem_queue.get(timeout=timeout)
            except queue.Empty:
                return None
        else:
            try:
                # BLPOP returns (key, item)
                raw_item = self._redis_client.blpop(
                    QUEUE_KEY, timeout=int(max(1, timeout))
                )
                if not raw_item:
                    return None
                parsed = json.loads(raw_item[1])
                return parsed["job_id"], parsed["data"]
            except (redis.RedisError, OSError) as exc:
                logger.error("Redis read error: %s", exc)
                return None

    def complete_job(
        self,
        job_id: str,
        result: dict[str, Any],
        processing_time_ms: float,
    ) -> None:
        """Mark a job as completed with prediction results."""
        now = time.time()
        update_data = {
            "status": "COMPLETED",
            "result": result,
            "processing_time_ms": round(processing_time_ms, 4),
            "updated_at": now,
        }

        if self._use_in_memory or self._redis_client is None:
            with self._lock:
                if job_id in self._mem_storage:
                    self._mem_storage[job_id].update(update_data)
        else:
            try:
                key = f"{JOB_PREFIX}{job_id}"
                raw = self._redis_client.get(key)
                if raw:
                    payload = json.loads(raw)
                    payload.update(update_data)
                    self._redis_client.set(
                        key, json.dumps(payload, default=str), ex=DEFAULT_JOB_TTL
                    )
            except (redis.RedisError, OSError) as exc:
                logger.error("Failed to complete job in Redis: %s", exc)

    def fail_job(self, job_id: str, error_message: str) -> None:
        """Mark a job as failed with an error message."""
        now = time.time()
        update_data = {
            "status": "FAILED",
            "error": error_message,
            "updated_at": now,
        }

        if self._use_in_memory or self._redis_client is None:
            with self._lock:
                if job_id in self._mem_storage:
                    self._mem_storage[job_id].update(update_data)
        else:
            try:
                key = f"{JOB_PREFIX}{job_id}"
                raw = self._redis_client.get(key)
                if raw:
                    payload = json.loads(raw)
                    payload.update(update_data)
                    self._redis_client.set(
                        key, json.dumps(payload, default=str), ex=DEFAULT_JOB_TTL
                    )
            except (redis.RedisError, OSError) as exc:
                logger.error("Failed to fail job in Redis: %s", exc)

    def get_job_status(self, job_id: str) -> dict[str, Any] | None:
        """Retrieve status and results for a job_id."""
        if self._use_in_memory or self._redis_client is None:
            with self._lock:
                item = self._mem_storage.get(job_id)
                return item.copy() if item else None
        else:
            try:
                key = f"{JOB_PREFIX}{job_id}"
                raw = self._redis_client.get(key)
                return json.loads(raw) if raw else None
            except (redis.RedisError, OSError) as exc:
                logger.error("Failed to get job status from Redis: %s", exc)
                return None
