"""Job registry for dynamic specimen progress tracking.

Jobs are ephemeral — stored in-memory, not persisted. Each job tracks
status, a bounded deque of progress messages, and the underlying
asyncio.Task running the specimen function.
"""
from __future__ import annotations

import asyncio
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Literal, Optional

from ascribe_link.progress import ProgressMessage


Status = Literal["running", "done", "error"]


@dataclass
class Job:
    """A single specimen-load job.

    Fields are mutated from the job's own asyncio.Task (single-appender for
    `messages`) and read from poll handlers. `collections.deque` is safe
    for single-appender / multi-reader without explicit locking; poll
    handlers snapshot `(next_seq, list(messages))` before serializing.
    """

    id: str
    specimen_id: str
    params: dict[str, Any]
    room_id: str
    status: Status = "running"
    messages: deque[ProgressMessage] = field(
        default_factory=lambda: deque(maxlen=50)
    )
    next_seq: int = 0
    result: Any = None
    error: Optional[str] = None
    task: Optional[asyncio.Task] = None
    created_at: float = field(default_factory=time.monotonic)
    finished_at: Optional[float] = None
    # For federated jobs: the (worker_id, worker_job_id) to proxy to.
    federated_to: Optional[tuple[str, str]] = None

    def append_message(self, text: str) -> ProgressMessage:
        """Append a new progress message and bump next_seq."""
        msg = ProgressMessage(seq=self.next_seq, text=text, ts=time.time())
        self.messages.append(msg)
        self.next_seq += 1
        return msg

    def messages_since(self, since: int) -> list[ProgressMessage]:
        """Return only messages with seq > since (kept in the deque)."""
        return [m for m in self.messages if m.seq > since]


class JobRegistry:
    """In-memory job store with TTL sweeping for completed jobs."""

    def __init__(self, ttl_seconds: float = 300.0) -> None:
        self._jobs: dict[str, Job] = {}
        self._lock = asyncio.Lock()
        self._ttl = ttl_seconds

    async def create(
        self,
        specimen_id: str,
        params: dict[str, Any],
        room_id: str,
    ) -> Job:
        """Create a new running job and register it atomically."""
        job_id = uuid.uuid4().hex
        job = Job(
            id=job_id,
            specimen_id=specimen_id,
            params=params,
            room_id=room_id,
        )
        async with self._lock:
            self._jobs[job_id] = job
        return job

    async def get(self, job_id: str) -> Optional[Job]:
        async with self._lock:
            return self._jobs.get(job_id)

    async def delete(self, job_id: str) -> None:
        async with self._lock:
            self._jobs.pop(job_id, None)

    async def sweep_expired(self) -> int:
        """Remove jobs whose finished_at is older than ttl_seconds.

        Running jobs (status == "running") are never swept even if old.
        Returns the number of jobs removed.
        """
        now = time.monotonic()
        removed = 0
        async with self._lock:
            for job_id in list(self._jobs.keys()):
                job = self._jobs[job_id]
                if job.finished_at is None:
                    continue
                if now - job.finished_at > self._ttl:
                    del self._jobs[job_id]
                    removed += 1
        return removed

    async def run_sweeper(self, interval: float = 30.0) -> None:
        """Background task — sweep expired jobs every `interval` seconds."""
        while True:
            try:
                await asyncio.sleep(interval)
                await self.sweep_expired()
            except asyncio.CancelledError:
                raise
            except Exception:
                # Never let a sweep failure kill the sweeper.
                pass
