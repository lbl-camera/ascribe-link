"""Tests for the agent subprocess isolation plumbing."""
from __future__ import annotations

import queue as queue_mod
import threading
import time

import pytest

from ascribe_link.agent_generator import (
    _agent_process_worker,
    _run_agent_in_subprocess,
    _QueueReporter,
)
from ascribe_link.progress import ProgressReporter


class _RecordingReporter(ProgressReporter):
    def __init__(self) -> None:
        self.calls: list[str] = []

    def report(self, text: str) -> None:
        self.calls.append(text)


class _FakeProc:
    """Stands in for multiprocessing.Process in parent-loop tests."""

    def __init__(self, alive=True, exitcode=None):
        self._alive = alive
        self.exitcode = exitcode
        self.terminated = False
        self.pid = 12345

    def start(self):
        pass

    def is_alive(self):
        return self._alive

    def terminate(self):
        self.terminated = True
        self._alive = False

    def join(self, timeout=None):
        pass


def _run_parent_loop(q, proc, reporter, kwargs=None, grace=60.0):
    """Drive the queue-consuming loop against a plain queue + fake proc."""
    import ascribe_link.agent_generator as ag
    import multiprocessing

    class _FakeCtx:
        def Queue(self):
            return q

        def Process(self, *a, **k):
            return proc

    real_get_context = multiprocessing.get_context
    multiprocessing.get_context = lambda *a, **k: _FakeCtx()
    try:
        return ag._run_agent_in_subprocess(
            kwargs or {"timeout": 5.0}, reporter, grace=grace
        )
    finally:
        multiprocessing.get_context = real_get_context


def test_progress_relayed_then_result_returned():
    q = queue_mod.Queue()
    q.put(("progress", "step 1"))
    q.put(("progress", "step 2"))
    q.put(("result", {"type": "mesh", "vertices": [], "indices": []}))
    reporter = _RecordingReporter()
    result = _run_parent_loop(q, _FakeProc(), reporter)
    assert result["type"] == "mesh"
    assert reporter.calls == ["step 1", "step 2"]


def test_child_error_raises_valueerror():
    q = queue_mod.Queue()
    q.put(("error", ("RuntimeError", "boom")))
    with pytest.raises(ValueError, match="RuntimeError.*boom"):
        _run_parent_loop(q, _FakeProc(), _RecordingReporter())


def test_child_timeout_raises_timeouterror():
    q = queue_mod.Queue()
    q.put(("error", ("TimeoutError", "Agent timed out after 5s")))
    with pytest.raises(TimeoutError):
        _run_parent_loop(q, _FakeProc(), _RecordingReporter())


def test_dead_child_with_empty_queue_raises():
    q = queue_mod.Queue()
    proc = _FakeProc(alive=False, exitcode=1)
    with pytest.raises(ValueError, match="exited unexpectedly"):
        _run_parent_loop(q, proc, _RecordingReporter())


def test_parent_deadline_backstop():
    q = queue_mod.Queue()
    proc = _FakeProc(alive=True)
    with pytest.raises(TimeoutError, match="grace"):
        _run_parent_loop(
            q, proc, _RecordingReporter(), kwargs={"timeout": 0.0}, grace=0.0
        )
    assert proc.terminated


def test_worker_sends_result(monkeypatch):
    """_agent_process_worker's queue protocol, exercised in-process."""
    import ascribe_link.agent_generator as ag

    async def fake_generate(reporter=None, **kwargs):
        reporter.report("working")
        return {"type": "volume", "shape": [2, 2, 2]}

    monkeypatch.setattr(ag, "generate_with_agent", fake_generate)
    q = queue_mod.Queue()
    ag._agent_process_worker(q, {"prompt": "x"})
    assert q.get_nowait() == ("progress", "working")
    kind, payload = q.get_nowait()
    assert kind == "result"
    assert payload["type"] == "volume"


def test_worker_sends_error(monkeypatch):
    import ascribe_link.agent_generator as ag

    async def fake_generate(reporter=None, **kwargs):
        raise RuntimeError("agent exploded")

    monkeypatch.setattr(ag, "generate_with_agent", fake_generate)
    q = queue_mod.Queue()
    ag._agent_process_worker(q, {})
    assert q.get_nowait() == ("error", ("RuntimeError", "agent exploded"))


def test_queue_reporter_never_raises():
    class BadQueue:
        def put(self, item):
            raise OSError("pipe closed")

    _QueueReporter(BadQueue()).report("hello")
