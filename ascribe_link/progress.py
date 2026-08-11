"""Progress reporting primitives for dynamic specimen functions.

ProgressReporter is the interface that specimen functions depend on.
The no-op default lets functions be called directly (in tests, REPL)
without a job context. JobReporter (defined below) binds to a Job and
appends messages to the job's bounded message deque.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ProgressMessage:
    """A single progress message in a job's history."""

    seq: int
    text: str
    ts: float  # epoch seconds


class ProgressReporter:
    """Base reporter — no-op by default.

    Functions declare a parameter of this type; FunctionRegistry.invoke_async
    will inject a real JobReporter if invoked under a Job, else this no-op.
    """

    def report(self, text: str) -> None:
        """Append a progress message. No-op in the base class."""
        return None


class JobReporter(ProgressReporter):
    """Reporter that appends messages to a specific Job's deque."""

    def __init__(self, job: Job) -> None:  # noqa: F821 — forward ref
        self._job = job

    def report(self, text: str) -> None:
        self._job.append_message(text)
