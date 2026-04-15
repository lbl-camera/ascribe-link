# Specimen Loading Progress Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the blocking `POST /api/specimens/{id}/data` + `POST /api/processing/invoke` path for dynamic specimens with a job-based API (`/start` → `/progress` → `/result`) that streams text progress messages from the server to ascribe-xr, with multiplayer-aware RPC broadcast to peers in a room.

**Architecture:** Server-side in-memory job registry tracks per-job status + bounded message history (50 messages). Functions opt in to progress reporting via a `ProgressReporter` parameter detected by signature inspection; the AI agent wires its SDK message stream into the reporter. Godot authority polls `/progress` and rebroadcasts messages to peers via RPC so only one client hits ascribe-link regardless of room size. Spec: `docs/superpowers/specs/2026-04-15-specimen-progress-design.md`.

**Tech Stack:** Python 3.11+, Litestar (async), pytest + httpx for tests. Godot 4 (GDScript). No new server dependencies.

---

## File Structure

### New files (ascribe-link)

| Path | Responsibility |
|---|---|
| `ascribe_link/progress.py` | `ProgressReporter` base class (no-op), `ProgressMessage` dataclass, `JobReporter` concrete impl |
| `ascribe_link/job_registry.py` | `Job` dataclass, `JobRegistry` with TTL sweeper |
| `ascribe_link/routes/jobs.py` | `JobController` with `GET /api/jobs/{id}/progress`, `GET /api/jobs/{id}/result`, `DELETE /api/jobs/{id}` |
| `tests/__init__.py` | Empty |
| `tests/conftest.py` | Shared pytest fixtures (app factory, httpx test client) |
| `tests/test_progress.py` | `ProgressReporter`, `JobReporter` unit tests |
| `tests/test_job_registry.py` | `Job` / `JobRegistry` lifecycle and TTL tests |
| `tests/test_reporter_injection.py` | Signature-driven injection + schema filter tests |
| `tests/test_jobs_api.py` | HTTP endpoint integration tests |
| `tests/test_agent_progress.py` | Agent integration tests with mocked SDK |
| `tests/test_federation_jobs.py` | Relay proxying with mocked worker |
| `test_jobs_e2e.py` (repo root, alongside existing `test_dynamic_specimen.py`) | Live server smoke test for AI agent specimen |

### Modified files (ascribe-link)

| Path | Change |
|---|---|
| `ascribe_link/processing.py` | `FunctionRegistry.invoke_async` gains `reporter` kwarg with signature-driven injection; `create_schema` skips `ProgressReporter`-annotated params |
| `ascribe_link/routes/specimens.py` | Add `POST /{specimen_id}/start`; leave existing `GET/POST /data` untouched for backwards compatibility |
| `ascribe_link/agent_generator.py` | `generate_with_agent` + `create_agent_function` accept `reporter`; wire SDK messages into `.report()` calls |
| `ascribe_link/app.py` | DI provider for `JobRegistry`, register `JobController`, start TTL sweeper task |
| `ascribe_link/federation.py` | Add proxy helper for federated job start/progress/result (called from routes) |
| `pyproject.toml` | Add `pytest-asyncio` to dev deps |

### New / Modified files (ascribe-xr, Godot)

| Path | Change |
|---|---|
| `scripts/DataSources/ascribe_link_client.gd` | Add `run_job()` method + `job_progress`/`job_complete`/`job_error` signals |
| `scripts/DataSources/http_source.gd` | Route through `run_job()` |
| `scripts/Specimen/dynamic_mesh_specimen.gd` | Wire progress signal → RPC broadcast → UI; authority-only polling |
| `scenes/UI/LoadingLayer.tscn` (or equivalent) | Add `RichTextLabel` named `MessageLog` and `ProgressBar` named `DownloadBar` |

---

## Phase 1 — Server Foundation (ascribe-link)

All server-side paths are relative to `~/PycharmProjects/ascribe-link/`. Use Windows-native paths via the `.venv` inside that directory for test runs.

### Task 1: Progress types — `ProgressReporter`, `ProgressMessage`

**Files:**
- Create: `ascribe_link/progress.py`
- Create: `tests/__init__.py` (empty)
- Create: `tests/conftest.py`
- Create: `tests/test_progress.py`
- Modify: `pyproject.toml` (add `pytest-asyncio`)

- [ ] **Step 1: Add `pytest-asyncio` to dev dependencies**

Edit `pyproject.toml`:

```toml
[project.optional-dependencies]
agent = [
    "claude-agent-sdk>=0.1.0",
]
dev = [
    "pytest>=6.2",
    "pytest-asyncio>=0.21",
    "httpx>=0.24",
]
```

Install it:

```bash
cd ~/PycharmProjects/ascribe-link && .venv/Scripts/pip install -e ".[dev]"
```

Expected: pytest-asyncio installed.

- [ ] **Step 2: Create empty `tests/__init__.py` and `tests/conftest.py`**

Create `tests/__init__.py` with contents:

```python
```

Create `tests/conftest.py`:

```python
"""Shared pytest fixtures for ascribe-link tests."""
import pytest


@pytest.fixture
def anyio_backend():
    return "asyncio"
```

Also add to `pyproject.toml` a pytest section:

```toml
[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]
```

- [ ] **Step 3: Write failing tests for `ProgressReporter` and `ProgressMessage`**

Create `tests/test_progress.py`:

```python
"""Tests for ProgressReporter and ProgressMessage."""
from ascribe_link.progress import ProgressReporter, ProgressMessage


def test_progress_message_is_frozen_dataclass():
    msg = ProgressMessage(seq=0, text="hello", ts=1.0)
    assert msg.seq == 0
    assert msg.text == "hello"
    assert msg.ts == 1.0


def test_progress_reporter_noop_report_does_not_raise():
    reporter = ProgressReporter()
    # Should silently succeed with no job bound
    reporter.report("anything")
    reporter.report("")


def test_progress_reporter_is_base_class():
    # JobReporter will inherit from this
    assert hasattr(ProgressReporter, "report")
```

- [ ] **Step 4: Run tests to verify they fail**

```bash
cd ~/PycharmProjects/ascribe-link && .venv/Scripts/pytest tests/test_progress.py -v
```

Expected: `ModuleNotFoundError: No module named 'ascribe_link.progress'`

- [ ] **Step 5: Create `ascribe_link/progress.py` with the types**

```python
"""Progress reporting primitives for dynamic specimen functions.

ProgressReporter is the interface that specimen functions depend on.
The no-op default lets functions be called directly (in tests, REPL)
without a job context. JobReporter (defined below) binds to a Job and
appends messages to the job's bounded message deque.
"""
from __future__ import annotations

import time
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
```

- [ ] **Step 6: Run tests to verify they pass**

```bash
cd ~/PycharmProjects/ascribe-link && .venv/Scripts/pytest tests/test_progress.py -v
```

Expected: all three tests PASS.

- [ ] **Step 7: Commit**

```bash
cd ~/PycharmProjects/ascribe-link && git add pyproject.toml tests/__init__.py tests/conftest.py tests/test_progress.py ascribe_link/progress.py && git commit -m "$(cat <<'EOF'
Add ProgressReporter and ProgressMessage base types

Adds the no-op ProgressReporter base class and frozen ProgressMessage
dataclass. Wires up pytest-asyncio and tests/ for future TDD work.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 2: `Job` dataclass and `JobRegistry`

**Files:**
- Create: `ascribe_link/job_registry.py`
- Create: `tests/test_job_registry.py`

- [ ] **Step 1: Write failing tests for `Job` and `JobRegistry`**

Create `tests/test_job_registry.py`:

```python
"""Tests for Job and JobRegistry."""
from __future__ import annotations

import asyncio
import time

import pytest

from ascribe_link.job_registry import Job, JobRegistry


@pytest.fixture
def registry():
    return JobRegistry(ttl_seconds=5.0)


async def test_create_returns_job_with_uuid(registry):
    job = await registry.create(specimen_id="sphere", params={"r": 1}, room_id="ascribe")
    assert job.id
    assert len(job.id) >= 32  # UUID-ish
    assert job.specimen_id == "sphere"
    assert job.status == "running"


async def test_get_returns_same_instance(registry):
    job = await registry.create(specimen_id="sphere", params={}, room_id="ascribe")
    assert await registry.get(job.id) is job


async def test_get_unknown_returns_none(registry):
    assert await registry.get("no-such-id") is None


async def test_messages_start_empty_and_seq_zero(registry):
    job = await registry.create(specimen_id="sphere", params={}, room_id="ascribe")
    assert len(job.messages) == 0
    assert job.next_seq == 0


async def test_append_message_increments_seq(registry):
    job = await registry.create(specimen_id="sphere", params={}, room_id="ascribe")
    job.append_message("first")
    job.append_message("second")
    assert [m.seq for m in job.messages] == [0, 1]
    assert [m.text for m in job.messages] == ["first", "second"]
    assert job.next_seq == 2


async def test_message_deque_is_bounded(registry):
    job = await registry.create(specimen_id="sphere", params={}, room_id="ascribe")
    for i in range(60):
        job.append_message(f"msg {i}")
    # Only last 50 retained, but next_seq keeps counting
    assert len(job.messages) == 50
    assert job.next_seq == 60
    assert job.messages[0].text == "msg 10"
    assert job.messages[-1].text == "msg 59"


async def test_messages_since_returns_only_new(registry):
    job = await registry.create(specimen_id="sphere", params={}, room_id="ascribe")
    job.append_message("a")
    job.append_message("b")
    job.append_message("c")
    new = job.messages_since(0)
    assert [m.text for m in new] == ["b", "c"]  # seq > 0
    assert job.messages_since(2) == []
    assert [m.text for m in job.messages_since(-1)] == ["a", "b", "c"]


async def test_delete_removes_job(registry):
    job = await registry.create(specimen_id="sphere", params={}, room_id="ascribe")
    await registry.delete(job.id)
    assert await registry.get(job.id) is None


async def test_expired_jobs_are_swept(registry):
    reg = JobRegistry(ttl_seconds=0.05)
    job = await reg.create(specimen_id="sphere", params={}, room_id="ascribe")
    job.status = "done"
    job.finished_at = time.monotonic()
    await asyncio.sleep(0.1)
    await reg.sweep_expired()
    assert await reg.get(job.id) is None


async def test_running_jobs_not_swept_even_if_old(registry):
    reg = JobRegistry(ttl_seconds=0.05)
    job = await reg.create(specimen_id="sphere", params={}, room_id="ascribe")
    # status stays "running"; no finished_at
    await asyncio.sleep(0.1)
    await reg.sweep_expired()
    assert await reg.get(job.id) is job
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd ~/PycharmProjects/ascribe-link && .venv/Scripts/pytest tests/test_job_registry.py -v
```

Expected: `ModuleNotFoundError: No module named 'ascribe_link.job_registry'`

- [ ] **Step 3: Create `ascribe_link/job_registry.py`**

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd ~/PycharmProjects/ascribe-link && .venv/Scripts/pytest tests/test_job_registry.py -v
```

Expected: all 10 tests PASS.

- [ ] **Step 5: Commit**

```bash
cd ~/PycharmProjects/ascribe-link && git add ascribe_link/job_registry.py tests/test_job_registry.py && git commit -m "$(cat <<'EOF'
Add Job dataclass and JobRegistry with TTL sweeper

In-memory registry with UUID-keyed jobs, bounded (maxlen=50) message
deque, and a sweeper coroutine that drops completed jobs after a TTL
while leaving running jobs alone.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 3: `JobReporter` binds `ProgressReporter` to a `Job`

**Files:**
- Modify: `ascribe_link/progress.py`
- Modify: `tests/test_progress.py`

- [ ] **Step 1: Add failing test for `JobReporter`**

Append to `tests/test_progress.py`:

```python
from ascribe_link.progress import JobReporter
from ascribe_link.job_registry import Job


def _make_job() -> Job:
    return Job(id="j1", specimen_id="sphere", params={}, room_id="ascribe")


def test_job_reporter_appends_to_job():
    job = _make_job()
    reporter = JobReporter(job)
    reporter.report("hello")
    reporter.report("world")
    assert [m.text for m in job.messages] == ["hello", "world"]
    assert job.next_seq == 2


def test_job_reporter_is_progress_reporter():
    job = _make_job()
    reporter = JobReporter(job)
    assert isinstance(reporter, ProgressReporter)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd ~/PycharmProjects/ascribe-link && .venv/Scripts/pytest tests/test_progress.py -v
```

Expected: `ImportError: cannot import name 'JobReporter' from 'ascribe_link.progress'`

- [ ] **Step 3: Add `JobReporter` to `ascribe_link/progress.py`**

Append to `ascribe_link/progress.py`:

```python
class JobReporter(ProgressReporter):
    """Reporter that appends messages to a specific Job's deque."""

    def __init__(self, job: "Job") -> None:  # noqa: F821 — forward ref
        self._job = job

    def report(self, text: str) -> None:
        self._job.append_message(text)
```

Note: we use a string forward reference for `Job` to avoid an import cycle
(`job_registry.py` imports `ProgressMessage` from `progress.py`).

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd ~/PycharmProjects/ascribe-link && .venv/Scripts/pytest tests/test_progress.py -v
```

Expected: all 5 tests PASS.

- [ ] **Step 5: Commit**

```bash
cd ~/PycharmProjects/ascribe-link && git add ascribe_link/progress.py tests/test_progress.py && git commit -m "$(cat <<'EOF'
Add JobReporter that binds ProgressReporter to a Job

JobReporter subclasses ProgressReporter and appends each report() call
to the bound Job's message deque. Uses a forward reference on Job to
avoid an import cycle with job_registry.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 4: Reporter injection in `FunctionRegistry.invoke_async` + schema filter

**Files:**
- Modify: `ascribe_link/processing.py`
- Create: `tests/test_reporter_injection.py`

- [ ] **Step 1: Write failing tests for injection and schema filter**

Create `tests/test_reporter_injection.py`:

```python
"""Tests for ProgressReporter injection and schema filtering."""
from __future__ import annotations

from ascribe_link.processing import FunctionRegistry, create_schema
from ascribe_link.progress import ProgressReporter


class _RecordingReporter(ProgressReporter):
    def __init__(self) -> None:
        self.calls: list[str] = []

    def report(self, text: str) -> None:
        self.calls.append(text)


async def test_reporter_injected_when_function_declares_it():
    registry = FunctionRegistry()

    async def fn(radius: float = 1.0, reporter: ProgressReporter = None):
        reporter.report(f"radius={radius}")
        return ([0.0, 0.0, 0.0], [0, 0, 0])

    registry.register_function(fn, "fn", return_type="mesh")
    rec = _RecordingReporter()
    await registry.invoke_async("fn", [], {"radius": 2.0}, reporter=rec)
    assert rec.calls == ["radius=2.0"]


async def test_reporter_not_injected_when_function_omits_it():
    registry = FunctionRegistry()

    async def fn(radius: float = 1.0):
        return ([0.0, 0.0, 0.0], [0, 0, 0])

    registry.register_function(fn, "fn", return_type="mesh")
    rec = _RecordingReporter()
    # Should not raise even though rec is passed
    await registry.invoke_async("fn", [], {"radius": 2.0}, reporter=rec)
    assert rec.calls == []


async def test_noop_reporter_injected_when_none_provided():
    registry = FunctionRegistry()

    async def fn(reporter: ProgressReporter = None):
        # Should not be None — should be a no-op reporter
        reporter.report("called")
        return ([0.0, 0.0, 0.0], [0, 0, 0])

    registry.register_function(fn, "fn", return_type="mesh")
    # No reporter kwarg at all
    await registry.invoke_async("fn", [], {})


def test_create_schema_skips_progress_reporter_param():
    def fn(radius: float = 1.0, reporter: ProgressReporter = None):
        pass

    schema = create_schema(fn)
    assert "radius" in schema["properties"]
    assert "reporter" not in schema["properties"]


def test_create_schema_keeps_other_params():
    def fn(radius: float = 1.0, resolution: int = 32):
        pass

    schema = create_schema(fn)
    assert set(schema["properties"].keys()) == {"radius", "resolution"}
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd ~/PycharmProjects/ascribe-link && .venv/Scripts/pytest tests/test_reporter_injection.py -v
```

Expected: several failures — schema filter not implemented, reporter kwarg not accepted by `invoke_async`.

- [ ] **Step 3: Modify `FunctionRegistry.invoke_async` and `create_schema` in `processing.py`**

In `ascribe_link/processing.py`:

1. Add an import at the top (after existing imports):

```python
from ascribe_link.progress import ProgressReporter
```

2. Add a helper method inside `FunctionRegistry` (just above `invoke_async`):

```python
    def _inject_reporter(
        self,
        func: Callable,
        kwargs: dict[str, Any],
        reporter: ProgressReporter | None,
    ) -> dict[str, Any]:
        """If the function declares a ProgressReporter parameter, inject it."""
        try:
            hints = get_type_hints(func)
        except Exception:
            return kwargs
        effective = reporter if reporter is not None else ProgressReporter()
        for param_name, annotation in hints.items():
            if annotation is ProgressReporter:
                kwargs = {**kwargs, param_name: effective}
                break
        return kwargs
```

3. Change the signature and body of `invoke_async`:

Find:

```python
    async def invoke_async(
        self,
        name: str,
        args: list | None = None,
        kwargs: dict | None = None,
    ) -> ProcessingResult:
```

Replace with:

```python
    async def invoke_async(
        self,
        name: str,
        args: list | None = None,
        kwargs: dict | None = None,
        reporter: ProgressReporter | None = None,
    ) -> ProcessingResult:
```

Then, inside the body, after the `_coerce_kwargs` line, add:

```python
        kwargs = self._inject_reporter(func, kwargs, reporter)
```

So the method body becomes:

```python
        func = self._functions.get(name)
        if func is None:
            raise KeyError(f"Unknown function: {name}")

        kwargs = self._coerce_kwargs(func, kwargs or {})
        kwargs = self._inject_reporter(func, kwargs, reporter)

        if asyncio.iscoroutinefunction(func):
            result = await func(*(args or []), **kwargs)
        else:
            result = func(*(args or []), **kwargs)

        return self._convert_result(result, self._return_types.get(name))
```

4. Modify `create_schema` at the bottom of the file to skip `ProgressReporter` params:

Find:

```python
    for param_name, param in sig.parameters.items():
        annotation = resolved.get(param_name, param.annotation)
        prop_schema = _type_to_schema(annotation)
```

Replace with:

```python
    for param_name, param in sig.parameters.items():
        annotation = resolved.get(param_name, param.annotation)
        if annotation is ProgressReporter:
            continue  # Injected by registry; not a user-facing parameter.
        prop_schema = _type_to_schema(annotation)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd ~/PycharmProjects/ascribe-link && .venv/Scripts/pytest tests/test_reporter_injection.py -v
```

Expected: all 5 tests PASS.

- [ ] **Step 5: Run the full suite to catch regressions**

```bash
cd ~/PycharmProjects/ascribe-link && .venv/Scripts/pytest -v
```

Expected: all tests PASS (progress, job_registry, reporter_injection).

- [ ] **Step 6: Commit**

```bash
cd ~/PycharmProjects/ascribe-link && git add ascribe_link/processing.py tests/test_reporter_injection.py && git commit -m "$(cat <<'EOF'
Inject ProgressReporter into functions via signature detection

FunctionRegistry.invoke_async accepts an optional reporter kwarg and
injects it into any function parameter annotated as ProgressReporter.
create_schema skips these parameters so they don't appear in the
specimen UI. Functions without a ProgressReporter param are unchanged.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 5: Job routes — `POST /start`, `GET /progress`, `GET /result`, `DELETE /jobs/{id}`

**Files:**
- Create: `ascribe_link/routes/jobs.py`
- Modify: `ascribe_link/routes/specimens.py`
- Modify: `ascribe_link/app.py`
- Create: `tests/test_jobs_api.py`

This task wires the new HTTP surface. It's the biggest task — split mentally into three sub-pieces: `/start` on specimens, `/progress` + `/result` + `DELETE` on the new JobController, and app-level DI wiring.

- [ ] **Step 1: Write failing HTTP integration tests**

Create `tests/test_jobs_api.py`:

```python
"""End-to-end tests for the job-based specimen API."""
from __future__ import annotations

import asyncio

import pytest
from httpx import AsyncClient

from ascribe_link.app import create_app
from ascribe_link.processing import FunctionRegistry
from ascribe_link.progress import ProgressReporter


@pytest.fixture
async def client():
    """Spin up the real Litestar app in-process with a fast specimen."""
    from litestar.testing import AsyncTestClient

    mesh_functions = {
        "fast_sphere": _fast_sphere,
        "slow_sphere": _slow_sphere,
        "failing": _failing,
    }

    # Register them as specimens by hand via a small app hook.
    # We piggy-back on create_app, then post-register specimens on the registry.
    app = create_app(mesh_functions=mesh_functions)

    # Register as specimens through the DI provider state
    registry: FunctionRegistry = app.dependencies["function_registry"].dependency()
    registry.register_specimen(
        _fast_sphere, display_name="Fast", name="fast_sphere", return_type="mesh"
    )
    registry.register_specimen(
        _slow_sphere, display_name="Slow", name="slow_sphere", return_type="mesh"
    )
    registry.register_specimen(
        _failing, display_name="Failing", name="failing", return_type="mesh"
    )

    async with AsyncTestClient(app=app) as c:
        yield c


async def _fast_sphere(reporter: ProgressReporter = None) -> tuple[list, list]:
    reporter.report("computing fast sphere")
    return ([0.0, 0.0, 0.0] * 3, [0, 1, 2])


async def _slow_sphere(reporter: ProgressReporter = None) -> tuple[list, list]:
    reporter.report("step 1")
    await asyncio.sleep(0.05)
    reporter.report("step 2")
    await asyncio.sleep(0.05)
    reporter.report("step 3")
    return ([0.0, 0.0, 0.0] * 3, [0, 1, 2])


async def _failing(reporter: ProgressReporter = None) -> tuple[list, list]:
    reporter.report("about to fail")
    raise RuntimeError("boom")


async def test_start_returns_job_id_and_running_status(client):
    resp = await client.post(
        "/api/specimens/fast_sphere/start", json={"params": {}, "room_id": "ascribe"}
    )
    assert resp.status_code == 200
    body = resp.json()
    assert "job_id" in body
    assert body["status"] in ("running", "done")


async def test_progress_returns_messages_in_order(client):
    start = (await client.post(
        "/api/specimens/slow_sphere/start",
        json={"params": {}, "room_id": "ascribe"},
    )).json()
    job_id = start["job_id"]

    # Poll until done
    seen_texts: list[str] = []
    last_seq = -1
    for _ in range(100):
        prog = (await client.get(
            f"/api/jobs/{job_id}/progress?since={last_seq}"
        )).json()
        for m in prog["messages"]:
            seen_texts.append(m["text"])
            last_seq = max(last_seq, m["seq"])
        if prog["status"] in ("done", "error"):
            break
        await asyncio.sleep(0.02)

    assert "step 1" in seen_texts
    assert "step 2" in seen_texts
    assert "step 3" in seen_texts
    # Bracket start/end messages are also present
    assert any("Starting" in t for t in seen_texts)
    assert any("Finished" in t for t in seen_texts)


async def test_progress_since_returns_only_new(client):
    start = (await client.post(
        "/api/specimens/slow_sphere/start",
        json={"params": {}, "room_id": "ascribe"},
    )).json()
    job_id = start["job_id"]

    # Wait for completion
    for _ in range(100):
        p = (await client.get(f"/api/jobs/{job_id}/progress")).json()
        if p["status"] == "done":
            break
        await asyncio.sleep(0.02)
    else:
        pytest.fail("job did not finish")

    full = (await client.get(f"/api/jobs/{job_id}/progress")).json()
    tail = (await client.get(
        f"/api/jobs/{job_id}/progress?since={full['messages'][1]['seq']}"
    )).json()
    assert len(tail["messages"]) == len(full["messages"]) - 2


async def test_result_returns_409_while_running(client):
    start = (await client.post(
        "/api/specimens/slow_sphere/start",
        json={"params": {}, "room_id": "ascribe"},
    )).json()
    job_id = start["job_id"]

    # Immediately — should still be running
    r = await client.get(f"/api/jobs/{job_id}/result")
    assert r.status_code == 409


async def test_result_returns_data_when_done(client):
    start = (await client.post(
        "/api/specimens/fast_sphere/start",
        json={"params": {}, "room_id": "ascribe"},
    )).json()
    job_id = start["job_id"]

    for _ in range(100):
        p = (await client.get(f"/api/jobs/{job_id}/progress")).json()
        if p["status"] == "done":
            break
        await asyncio.sleep(0.02)
    else:
        pytest.fail("job did not finish")

    r = await client.get(f"/api/jobs/{job_id}/result")
    assert r.status_code == 200
    body = r.json()
    assert body.get("type") == "mesh"
    assert "vertices" in body


async def test_result_returns_410_on_error(client):
    start = (await client.post(
        "/api/specimens/failing/start",
        json={"params": {}, "room_id": "ascribe"},
    )).json()
    job_id = start["job_id"]

    for _ in range(100):
        p = (await client.get(f"/api/jobs/{job_id}/progress")).json()
        if p["status"] == "error":
            assert "boom" in (p.get("error") or "")
            break
        await asyncio.sleep(0.02)
    else:
        pytest.fail("job did not error")

    r = await client.get(f"/api/jobs/{job_id}/result")
    assert r.status_code == 410


async def test_progress_404_for_unknown_job(client):
    r = await client.get("/api/jobs/does-not-exist/progress")
    assert r.status_code == 404


async def test_delete_cancels_running_job(client):
    start = (await client.post(
        "/api/specimens/slow_sphere/start",
        json={"params": {}, "room_id": "ascribe"},
    )).json()
    job_id = start["job_id"]

    d = await client.delete(f"/api/jobs/{job_id}")
    assert d.status_code == 204

    # Wait for it to show as errored
    for _ in range(100):
        p = (await client.get(f"/api/jobs/{job_id}/progress")).json()
        if p["status"] == "error":
            assert "cancel" in (p.get("error") or "").lower()
            break
        await asyncio.sleep(0.02)
    else:
        pytest.fail("job did not cancel")
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd ~/PycharmProjects/ascribe-link && .venv/Scripts/pytest tests/test_jobs_api.py -v
```

Expected: all fail (endpoints don't exist, app has no `JobRegistry` DI).

- [ ] **Step 3: Create `ascribe_link/routes/jobs.py`**

```python
"""Job endpoints: poll progress, fetch result, cancel."""
from __future__ import annotations

from typing import Any

from litestar import Controller, Response, get, delete
from litestar.exceptions import HTTPException, NotFoundException

from ascribe_link.job_registry import JobRegistry
from ascribe_link.models import result_to_dict


class JobController(Controller):
    path = "/api/jobs"

    @get("/{job_id:str}/progress")
    async def get_progress(
        self,
        job_registry: JobRegistry,
        job_id: str,
        since: int = -1,
    ) -> dict[str, Any]:
        """Return new progress messages for this job."""
        job = await job_registry.get(job_id)
        if job is None:
            raise NotFoundException(detail=f"Unknown job: {job_id}")

        messages = [
            {"seq": m.seq, "text": m.text, "ts": m.ts}
            for m in job.messages_since(since)
        ]
        return {
            "status": job.status,
            "messages": messages,
            "error": job.error,
        }

    @get("/{job_id:str}/result")
    async def get_result(
        self,
        job_registry: JobRegistry,
        job_id: str,
    ) -> dict[str, Any]:
        job = await job_registry.get(job_id)
        if job is None:
            raise NotFoundException(detail=f"Unknown job: {job_id}")
        if job.status == "running":
            raise HTTPException(status_code=409, detail="Job still running")
        if job.status == "error":
            raise HTTPException(
                status_code=410, detail=f"Job failed: {job.error}"
            )
        # status == "done"
        if isinstance(job.result, dict):
            return job.result
        return result_to_dict(job.result)

    @delete("/{job_id:str}", status_code=204)
    async def delete_job(
        self,
        job_registry: JobRegistry,
        job_id: str,
    ) -> None:
        job = await job_registry.get(job_id)
        if job is None:
            raise NotFoundException(detail=f"Unknown job: {job_id}")
        if job.task is not None and not job.task.done():
            job.task.cancel()
```

- [ ] **Step 4: Add `POST /{specimen_id}/start` to `ascribe_link/routes/specimens.py`**

Add this import at the top of the file:

```python
import asyncio
import time

from ascribe_link.job_registry import Job, JobRegistry
from ascribe_link.progress import JobReporter
```

Add this method inside `SpecimenController` (after `get_data_post`, before `reload_specimens`):

```python
    @post("/{specimen_id:str}/start")
    async def start_job(
        self,
        specimen_store: SpecimenStore,
        specimen_id: str,
        function_registry: FunctionRegistry,
        result_cache: RoomResultCache,
        job_registry: JobRegistry,
        federation_hub: FederationHub | None = None,
        data: dict[str, Any] | None = None,
    ) -> dict[str, str]:
        """Start a dynamic specimen load as a background job.

        Returns `{job_id, status}`. Status is "done" on cache hit; else "running".
        """
        data = data or {}
        params: dict[str, Any] = data.get("params", {}) or {}
        room_id: str = data.get("room_id", "ascribe")

        # Resolve the specimen and ensure it's dynamic.
        if ":" in specimen_id and federation_hub:
            # Federated — proxy to worker (handled in Task 8).
            worker_id, actual_id = specimen_id.split(":", 1)
            return await _proxy_federated_start(
                federation_hub,
                worker_id,
                actual_id,
                params,
                room_id,
                job_registry,
                specimen_id,
            )

        meta = function_registry.get_specimen(specimen_id)
        if meta is None:
            meta = specimen_store.get(specimen_id)
        if meta is None:
            raise NotFoundException(detail=f"Specimen not found: {specimen_id}")
        if not meta.function_name:
            raise HTTPException(
                status_code=400,
                detail=f"Specimen {specimen_id} is static; use GET /data instead",
            )

        # Extract defaults if params not provided.
        if not params and meta.schema:
            params = _extract_schema_defaults(meta.schema)

        job = await job_registry.create(
            specimen_id=specimen_id, params=params, room_id=room_id
        )

        # Cache hit shortcut — no task needed.
        cached = result_cache.get(room_id, meta.function_name, params)
        if cached is not None:
            job.append_message("cache hit")
            job.result = cached
            job.status = "done"
            job.finished_at = time.monotonic()
            return {"job_id": job.id, "status": "done"}

        # Spawn the runner task; register it so DELETE can cancel.
        job.task = asyncio.create_task(
            _run_job(
                job=job,
                function_registry=function_registry,
                result_cache=result_cache,
                func_name=meta.function_name,
            )
        )
        return {"job_id": job.id, "status": "running"}
```

Add the imports `HTTPException` from litestar.exceptions if not already imported. Then add these helper functions at the bottom of `specimens.py`, after `_extract_schema_defaults`:

```python
async def _run_job(
    *,
    job: Job,
    function_registry: FunctionRegistry,
    result_cache: RoomResultCache,
    func_name: str,
) -> None:
    """Execute the specimen function, populating the job's result/status."""
    reporter = JobReporter(job)
    job.append_message(f"Starting {job.specimen_id}")
    t0 = time.monotonic()
    try:
        result = await function_registry.invoke_async(
            func_name, [], job.params, reporter=reporter
        )
        result_dict = result_to_dict(result)
        result_cache.put(job.room_id, func_name, job.params, result_dict)
        job.result = result_dict
        job.status = "done"
        job.append_message(f"Finished in {time.monotonic() - t0:.2f}s")
    except asyncio.CancelledError:
        job.status = "error"
        job.error = "cancelled"
        job.append_message("Cancelled")
        raise
    except Exception as e:
        job.status = "error"
        job.error = str(e)
        job.append_message(f"Error: {e}")
    finally:
        job.finished_at = time.monotonic()


async def _proxy_federated_start(
    federation_hub: FederationHub,
    worker_id: str,
    actual_id: str,
    params: dict,
    room_id: str,
    job_registry: JobRegistry,
    original_specimen_id: str,
) -> dict[str, str]:
    """Placeholder — fully implemented in Task 8."""
    raise HTTPException(
        status_code=501,
        detail="Federated jobs not yet implemented in this task",
    )
```

Add `from litestar.exceptions import HTTPException` to the imports at the top.

- [ ] **Step 5: Wire `JobRegistry` DI and register `JobController` in `app.py`**

Modify `ascribe_link/app.py`:

Add imports:

```python
from ascribe_link.job_registry import JobRegistry
from ascribe_link.routes.jobs import JobController
```

Inside `create_app`, after the `result_cache = RoomResultCache(...)` line, add:

```python
    # --- Job registry for progress-tracked dynamic loads ---
    job_registry = JobRegistry(ttl_seconds=300.0)
    logger.info("Job registry enabled (TTL=300s)")
```

Add the DI provider alongside the others:

```python
    def provide_job_registry() -> JobRegistry:
        return job_registry
```

Add it to the `dependencies` dict:

```python
        dependencies={
            "specimen_store": Provide(provide_specimen_store, sync_to_thread=False),
            "function_registry": Provide(provide_function_registry, sync_to_thread=False),
            "federation_hub": Provide(provide_federation_hub, sync_to_thread=False),
            "result_cache": Provide(provide_result_cache, sync_to_thread=False),
            "job_registry": Provide(provide_job_registry, sync_to_thread=False),
        },
```

Add `JobController` to the route handler list:

```python
    route_handlers = [SpecimenController, ProcessingController, JobController]
```

- [ ] **Step 6: Run the tests**

```bash
cd ~/PycharmProjects/ascribe-link && .venv/Scripts/pytest tests/test_jobs_api.py -v
```

Expected: all 8 tests PASS.

- [ ] **Step 7: Run the full suite**

```bash
cd ~/PycharmProjects/ascribe-link && .venv/Scripts/pytest -v
```

Expected: all tests PASS.

- [ ] **Step 8: Commit**

```bash
cd ~/PycharmProjects/ascribe-link && git add ascribe_link/routes/jobs.py ascribe_link/routes/specimens.py ascribe_link/app.py tests/test_jobs_api.py && git commit -m "$(cat <<'EOF'
Add job-based specimen API (/start, /progress, /result, DELETE)

POST /api/specimens/{id}/start creates a Job, spawns the function task
with a JobReporter, and returns {job_id, status}. GET /api/jobs/{id}/
progress returns new messages since a seq cursor. GET /api/jobs/{id}/
result returns the computed data (409 while running, 410 on error).
DELETE /api/jobs/{id} cancels a running task.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 6: Wire AI agent SDK messages into the reporter

**Files:**
- Modify: `ascribe_link/agent_generator.py`
- Create: `tests/test_agent_progress.py`

- [ ] **Step 1: Write failing tests against a yet-to-exist `_emit_agent_events`**

Create `tests/test_agent_progress.py` with the real assertions up front. The tests construct lightweight stand-in classes for the SDK types and inject them as `claude_agent_sdk` via `sys.modules`, so `_emit_agent_events` can be unit-tested without the real SDK:

```python
"""Tests for agent progress message emission."""
from __future__ import annotations

import sys
from unittest.mock import MagicMock

import pytest

from ascribe_link.progress import ProgressReporter


class _RecordingReporter(ProgressReporter):
    def __init__(self) -> None:
        self.calls: list[str] = []

    def report(self, text: str) -> None:
        self.calls.append(text)


@pytest.fixture
def reporter():
    return _RecordingReporter()


def _install_fake_sdk(monkeypatch):
    class AssistantMessage:
        def __init__(self, content):
            self.content = content

    class TextBlock:
        def __init__(self, text):
            self.text = text

    class ToolUseBlock:
        def __init__(self, name):
            self.name = name

    class ToolResultBlock:
        def __init__(self, is_error=False, content=""):
            self.is_error = is_error
            self.content = content

    fake_sdk = MagicMock()
    fake_sdk.AssistantMessage = AssistantMessage
    fake_sdk.TextBlock = TextBlock
    fake_sdk.ToolResultBlock = ToolResultBlock
    monkeypatch.setitem(sys.modules, "claude_agent_sdk", fake_sdk)
    return AssistantMessage, TextBlock, ToolUseBlock, ToolResultBlock


def test_text_block_reports_first_line(reporter, monkeypatch):
    AssistantMessage, TextBlock, _, _ = _install_fake_sdk(monkeypatch)
    from ascribe_link import agent_generator

    msg = AssistantMessage(content=[TextBlock("Let me plan.\nMore detail")])
    agent_generator._emit_agent_events(msg, reporter)
    assert reporter.calls == ["Let me plan."]


def test_tool_use_block_reports_tool_name(reporter, monkeypatch):
    AssistantMessage, _, ToolUseBlock, _ = _install_fake_sdk(monkeypatch)
    from ascribe_link import agent_generator

    msg = AssistantMessage(content=[ToolUseBlock(name="Bash")])
    agent_generator._emit_agent_events(msg, reporter)
    assert reporter.calls == ["Tool: Bash"]


def test_tool_result_error_is_reported(reporter, monkeypatch):
    _, _, _, ToolResultBlock = _install_fake_sdk(monkeypatch)
    from ascribe_link import agent_generator

    msg = ToolResultBlock(is_error=True, content="command not found: xyz")
    agent_generator._emit_agent_events(msg, reporter)
    assert len(reporter.calls) == 1
    assert "Tool error" in reporter.calls[0]


def test_tool_result_success_is_ignored(reporter, monkeypatch):
    _, _, _, ToolResultBlock = _install_fake_sdk(monkeypatch)
    from ascribe_link import agent_generator

    msg = ToolResultBlock(is_error=False, content="lots of output")
    agent_generator._emit_agent_events(msg, reporter)
    assert reporter.calls == []


def test_long_text_block_is_truncated_to_200_chars(reporter, monkeypatch):
    AssistantMessage, TextBlock, _, _ = _install_fake_sdk(monkeypatch)
    from ascribe_link import agent_generator

    long_line = "x" * 500
    msg = AssistantMessage(content=[TextBlock(long_line)])
    agent_generator._emit_agent_events(msg, reporter)
    assert len(reporter.calls[0]) == 200
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
cd ~/PycharmProjects/ascribe-link && .venv/Scripts/pytest tests/test_agent_progress.py -v
```

Expected: `AttributeError: module 'ascribe_link.agent_generator' has no attribute '_emit_agent_events'`

- [ ] **Step 3: Extract event emission into a testable function and wire into `generate_with_agent`**

In `ascribe_link/agent_generator.py`, add an import at the top:

```python
from ascribe_link.progress import ProgressReporter
```

Add the helper function after the existing imports and before `MESH_GENERATION_SKILL`:

```python
def _emit_agent_events(msg: Any, reporter: ProgressReporter) -> None:
    """Translate an SDK message to one or more reporter.report() calls.

    Called from the receive_response() loop. Kept as a module-level function
    so it can be unit-tested with mocked SDK types without spinning up a
    real ClaudeSDKClient.
    """
    # Import lazily so the module still imports when claude_agent_sdk
    # isn't installed (agent is an optional extra).
    from claude_agent_sdk import AssistantMessage, TextBlock, ToolResultBlock

    if isinstance(msg, AssistantMessage):
        for block in msg.content:
            if isinstance(block, TextBlock):
                first_line = (block.text or "").strip().splitlines()
                if first_line:
                    reporter.report(first_line[0][:200])
            elif hasattr(block, "name"):
                # ToolUseBlock — report just the tool name.
                reporter.report(f"Tool: {block.name}")
    elif isinstance(msg, ToolResultBlock):
        # Only surface errors; successful tool results are noisy.
        is_error = getattr(msg, "is_error", False)
        if is_error:
            content = getattr(msg, "content", None)
            summary = str(content)[:120] if content else "unknown"
            reporter.report(f"Tool error: {summary}")
```

Change the signature of `generate_with_agent` to accept a reporter:

Find:

```python
async def generate_with_agent(
    prompt: str,
    file_path: str | None = None,
    model: str = "claude-sonnet-4-20250514",
    timeout: float = 300.0,
    working_dir: str | None = None,
    sandbox: bool = True,
    sandbox_config: SandboxConfig | None = None,
) -> dict[str, Any]:
```

Replace with:

```python
async def generate_with_agent(
    prompt: str,
    file_path: str | None = None,
    model: str = "claude-sonnet-4-20250514",
    timeout: float = 300.0,
    working_dir: str | None = None,
    sandbox: bool = True,
    sandbox_config: SandboxConfig | None = None,
    reporter: ProgressReporter | None = None,
) -> dict[str, Any]:
```

Inside the function body, after the line `result = AgentResult()`:

```python
    reporter = reporter or ProgressReporter()
```

Inside `process_responses()`, after `msg_count += 1`:

```python
                try:
                    _emit_agent_events(msg, reporter)
                except Exception:
                    # Never let progress emission break the agent loop.
                    pass
```

And when the mesh/volume is submitted, add one more reporter line. Find:

```python
                    # Check if we got a result
                    if result.submitted:
                        logger.info("Result submitted, exiting response loop")
                        return
```

Replace with:

```python
                    # Check if we got a result
                    if result.submitted:
                        reporter.report(
                            f"{result.result_type.capitalize()} submitted"
                            if result.result_type
                            else "Result submitted"
                        )
                        logger.info("Result submitted, exiting response loop")
                        return
```

Finally, update the `create_agent_function` wrapper to accept a reporter (so registry injection wires it automatically):

Find the inner function definition:

```python
    async def agent_generate(
        prompt: str = r"...",
        file_path: str = "",
    ) -> dict[str, Any]:
```

Replace with:

```python
    async def agent_generate(
        prompt: str = r"Load the CT head volume from PNG stack at C:\Users\rp\Documents\vr-start\specimen_data\cthead-8bit\ (files named cthead-8bit001.png through the last one). Stack them into a 3D array, then extract an isosurface using marching cubes at threshold 100. Submit the resulting mesh.",
        file_path: str = "",
        reporter: ProgressReporter = None,
    ) -> dict[str, Any]:
```

And in its body, pass `reporter` through:

```python
        return await generate_with_agent(
            prompt=prompt,
            file_path=file_path,
            model=model,
            timeout=timeout,
            sandbox=sandbox,
            sandbox_config=sandbox_config,
            reporter=reporter,
        )
```

- [ ] **Step 4: Run the agent tests**

```bash
cd ~/PycharmProjects/ascribe-link && .venv/Scripts/pytest tests/test_agent_progress.py -v
```

Expected: all 5 tests PASS.

- [ ] **Step 5: Run the full suite**

```bash
cd ~/PycharmProjects/ascribe-link && .venv/Scripts/pytest -v
```

Expected: all tests PASS.

- [ ] **Step 6: Commit**

```bash
cd ~/PycharmProjects/ascribe-link && git add ascribe_link/agent_generator.py tests/test_agent_progress.py && git commit -m "$(cat <<'EOF'
Wire AI agent SDK message stream into ProgressReporter

Extracts _emit_agent_events for unit testing, routes assistant text
blocks (first line, 200-char cap), tool-use blocks (Tool: <name>), and
tool-result errors into reporter.report(). Mesh/volume submission is
also reported. Adds reporter parameter to create_agent_function so it
is injected by the registry.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 7: TTL sweeper background task wired into app lifecycle

**Files:**
- Modify: `ascribe_link/app.py`
- Modify: `tests/test_job_registry.py` (add startup-hook test — optional)

- [ ] **Step 1: Add `on_startup`/`on_shutdown` hooks to `create_app`**

Modify `ascribe_link/app.py` — change the `Litestar(...)` construction at the bottom. Above it, define lifespan helpers:

```python
    # --- Lifecycle hooks for the job TTL sweeper ---
    sweeper_task_holder: dict[str, asyncio.Task] = {}

    async def _start_sweeper(app_: Litestar) -> None:
        sweeper_task_holder["task"] = asyncio.create_task(
            job_registry.run_sweeper(interval=30.0)
        )

    async def _stop_sweeper(app_: Litestar) -> None:
        task = sweeper_task_holder.get("task")
        if task and not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
```

Add `import asyncio` at the top of `app.py` if not already present.

Then pass these to `Litestar(...)`:

```python
    app = Litestar(
        route_handlers=route_handlers,
        dependencies={ ... },
        cors_config=CORSConfig(allow_origins=["*"]),
        exception_handlers={Exception: log_exception_handler},
        debug=True,
        openapi_config=OpenAPIConfig( ... ),
        on_startup=[_start_sweeper],
        on_shutdown=[_stop_sweeper],
    )
```

- [ ] **Step 2: Run the full test suite to confirm nothing regressed**

```bash
cd ~/PycharmProjects/ascribe-link && .venv/Scripts/pytest -v
```

Expected: all tests PASS. The sweeper runs at 30-second intervals in production and will be cancelled cleanly when the app shuts down.

- [ ] **Step 3: Commit**

```bash
cd ~/PycharmProjects/ascribe-link && git add ascribe_link/app.py && git commit -m "$(cat <<'EOF'
Start JobRegistry TTL sweeper as a background task

on_startup spawns job_registry.run_sweeper; on_shutdown cancels it
cleanly. Sweeper runs every 30s and drops jobs whose finished_at is
older than the registry TTL.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 8: Federation — proxy job endpoints to workers

**Files:**
- Modify: `ascribe_link/routes/specimens.py`
- Modify: `ascribe_link/routes/jobs.py`
- Modify: `ascribe_link/federation.py` (add proxy request types if needed)
- Create: `tests/test_federation_jobs.py`

The relay stores `{relay_job_id → (worker_id, worker_job_id)}` on each federated Job and proxies `/progress` and `/result` calls.

- [ ] **Step 1: Write failing tests with a mocked federation hub**

Create `tests/test_federation_jobs.py`:

```python
"""Tests for federated job proxying in relay mode."""
from __future__ import annotations

from unittest.mock import AsyncMock

import pytest
from litestar.testing import AsyncTestClient

from ascribe_link.app import create_app
from ascribe_link.federation import FederationHub


@pytest.fixture
async def relay_client():
    app = create_app(relay_mode=True)

    # Inject a fake worker into the hub.
    hub: FederationHub = app.dependencies["federation_hub"].dependency()

    # Simulate a registered worker with one specimen.
    worker_id = "worker_a"
    hub._workers[worker_id] = _FakeWorker(
        worker_id=worker_id,
        specimens=[{"id": "remote_sphere", "display_name": "Remote", "type": "mesh"}],
    )
    # Mock proxy_request for federated calls.
    hub.proxy_request = AsyncMock(side_effect=_fake_proxy)

    async with AsyncTestClient(app=app) as c:
        yield c, hub


class _FakeWorker:
    def __init__(self, worker_id, specimens):
        self.worker_id = worker_id
        self.specimens = specimens


async def _fake_proxy(worker_id, method, payload):
    if method == "start_job":
        return {"job_id": "remote-job-42", "status": "running"}
    if method == "get_progress":
        return {
            "status": "done",
            "messages": [{"seq": 0, "text": "remote msg", "ts": 1.0}],
            "error": None,
        }
    if method == "get_result":
        return {"type": "mesh", "vertices": [0.0] * 9, "indices": [0, 1, 2]}
    raise NotImplementedError(method)


async def test_relay_start_proxies_to_worker(relay_client):
    c, hub = relay_client
    r = await c.post(
        "/api/specimens/worker_a:remote_sphere/start",
        json={"params": {}, "room_id": "ascribe"},
    )
    assert r.status_code == 200
    body = r.json()
    assert "job_id" in body
    assert body["status"] == "running"
    # Proxy was called with start_job
    hub.proxy_request.assert_any_call(
        "worker_a",
        "start_job",
        {"specimen_id": "remote_sphere", "params": {}, "room_id": "ascribe"},
    )


async def test_relay_progress_proxies_to_worker(relay_client):
    c, hub = relay_client
    start = (await c.post(
        "/api/specimens/worker_a:remote_sphere/start",
        json={"params": {}, "room_id": "ascribe"},
    )).json()
    job_id = start["job_id"]

    r = await c.get(f"/api/jobs/{job_id}/progress")
    assert r.status_code == 200
    body = r.json()
    assert body["messages"][0]["text"] == "remote msg"


async def test_relay_result_proxies_to_worker(relay_client):
    c, hub = relay_client
    start = (await c.post(
        "/api/specimens/worker_a:remote_sphere/start",
        json={"params": {}, "room_id": "ascribe"},
    )).json()
    job_id = start["job_id"]

    r = await c.get(f"/api/jobs/{job_id}/result")
    assert r.status_code == 200
    body = r.json()
    assert body["type"] == "mesh"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd ~/PycharmProjects/ascribe-link && .venv/Scripts/pytest tests/test_federation_jobs.py -v
```

Expected: `_proxy_federated_start` raises 501 — test failures.

- [ ] **Step 3: Implement `_proxy_federated_start` in `routes/specimens.py`**

Replace the stub at the bottom of `ascribe_link/routes/specimens.py`:

```python
async def _proxy_federated_start(
    federation_hub: FederationHub,
    worker_id: str,
    actual_id: str,
    params: dict,
    room_id: str,
    job_registry: JobRegistry,
    original_specimen_id: str,
) -> dict[str, str]:
    """Start a job on a federated worker and proxy via a local relay-side Job."""
    worker_response = await federation_hub.proxy_request(
        worker_id,
        "start_job",
        {"specimen_id": actual_id, "params": params, "room_id": room_id},
    )
    if "error" in worker_response:
        raise HTTPException(
            status_code=502, detail=f"Worker error: {worker_response['error']}"
        )

    worker_job_id = worker_response["job_id"]
    relay_job = await job_registry.create(
        specimen_id=original_specimen_id, params=params, room_id=room_id
    )
    relay_job.federated_to = (worker_id, worker_job_id)
    # Inherit status from the worker — if the worker said "done" (cache hit),
    # we record that locally so /result is served via a direct proxy fetch.
    if worker_response.get("status") == "done":
        relay_job.status = "done"
    return {"job_id": relay_job.id, "status": relay_job.status}
```

- [ ] **Step 4: Teach `JobController` to proxy progress/result for federated jobs**

Modify `ascribe_link/routes/jobs.py`. Change the dependency list of `JobController` methods to accept `federation_hub`:

```python
from ascribe_link.federation import FederationHub
```

And add `federation_hub: FederationHub | None = None` to `get_progress`, `get_result`, `delete_job`:

```python
    @get("/{job_id:str}/progress")
    async def get_progress(
        self,
        job_registry: JobRegistry,
        job_id: str,
        federation_hub: FederationHub | None = None,
        since: int = -1,
    ) -> dict[str, Any]:
        job = await job_registry.get(job_id)
        if job is None:
            raise NotFoundException(detail=f"Unknown job: {job_id}")

        # Federated — proxy to worker.
        if job.federated_to is not None and federation_hub is not None:
            worker_id, worker_job_id = job.federated_to
            response = await federation_hub.proxy_request(
                worker_id,
                "get_progress",
                {"job_id": worker_job_id, "since": since},
            )
            # Mirror the terminal status locally so /result can serve.
            if response.get("status") in ("done", "error"):
                job.status = response["status"]
                if response.get("error"):
                    job.error = response["error"]
            return response

        messages = [
            {"seq": m.seq, "text": m.text, "ts": m.ts}
            for m in job.messages_since(since)
        ]
        return {
            "status": job.status,
            "messages": messages,
            "error": job.error,
        }

    @get("/{job_id:str}/result")
    async def get_result(
        self,
        job_registry: JobRegistry,
        job_id: str,
        federation_hub: FederationHub | None = None,
    ) -> dict[str, Any]:
        job = await job_registry.get(job_id)
        if job is None:
            raise NotFoundException(detail=f"Unknown job: {job_id}")

        if job.federated_to is not None and federation_hub is not None:
            worker_id, worker_job_id = job.federated_to
            response = await federation_hub.proxy_request(
                worker_id, "get_result", {"job_id": worker_job_id}
            )
            if "error" in response:
                raise HTTPException(status_code=410, detail=response["error"])
            return response

        if job.status == "running":
            raise HTTPException(status_code=409, detail="Job still running")
        if job.status == "error":
            raise HTTPException(
                status_code=410, detail=f"Job failed: {job.error}"
            )
        if isinstance(job.result, dict):
            return job.result
        return result_to_dict(job.result)

    @delete("/{job_id:str}", status_code=204)
    async def delete_job(
        self,
        job_registry: JobRegistry,
        job_id: str,
        federation_hub: FederationHub | None = None,
    ) -> None:
        job = await job_registry.get(job_id)
        if job is None:
            raise NotFoundException(detail=f"Unknown job: {job_id}")
        if job.federated_to is not None and federation_hub is not None:
            worker_id, worker_job_id = job.federated_to
            await federation_hub.proxy_request(
                worker_id, "cancel_job", {"job_id": worker_job_id}
            )
        if job.task is not None and not job.task.done():
            job.task.cancel()
```

- [ ] **Step 5: Run federation tests**

```bash
cd ~/PycharmProjects/ascribe-link && .venv/Scripts/pytest tests/test_federation_jobs.py -v
```

Expected: all 3 tests PASS.

- [ ] **Step 6: Run the full suite**

```bash
cd ~/PycharmProjects/ascribe-link && .venv/Scripts/pytest -v
```

Expected: all tests PASS.

- [ ] **Step 7: Commit**

```bash
cd ~/PycharmProjects/ascribe-link && git add ascribe_link/routes/specimens.py ascribe_link/routes/jobs.py tests/test_federation_jobs.py && git commit -m "$(cat <<'EOF'
Proxy job endpoints across federation relay

Relay-side Jobs that target federated specimens record
(worker_id, worker_job_id) and proxy /progress, /result, and DELETE to
the worker. Worker-reported 'done' status on /start is mirrored
locally for cache-hit parity.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
EOF
)"
```

**Note on worker-side handlers:** the worker must implement the same `POST /start`, `GET /progress`, `GET /result`, `DELETE /jobs/{id}` endpoints. Since workers run the same `create_app(...)` (in non-relay mode), they already will after Tasks 1–7. The only worker-specific piece is mapping incoming `"start_job"` / `"get_progress"` / `"get_result"` / `"cancel_job"` method calls in their federation-protocol dispatcher to the real HTTP handlers — this already exists for `"get_data"` and `"get_thumbnail"` (see `routes/federation.py` for the pattern; mirror it for the new methods if tests fail on a real relay).

---

### Task 9: End-to-end smoke test for the AI agent specimen

**Files:**
- Create: `test_jobs_e2e.py` (repo root, next to existing `test_dynamic_specimen.py`)

- [ ] **Step 1: Create the smoke test**

```python
#!/usr/bin/env python3
"""End-to-end smoke test for job-based AI agent specimen loading.

Run against a live ascribe-link server with the agent enabled:

    ascribe-link --enable-agent

Then:

    python test_jobs_e2e.py
"""
import asyncio

import httpx


async def main() -> None:
    base = "http://localhost:8000"
    async with httpx.AsyncClient(timeout=None) as c:
        print("1. POST /api/specimens/ai_generate/start ...")
        r = await c.post(
            f"{base}/api/specimens/ai_generate/start",
            json={
                "params": {"prompt": "make a unit sphere"},
                "room_id": "ascribe",
            },
        )
        assert r.status_code == 200, r.text
        start = r.json()
        job_id = start["job_id"]
        print(f"   job_id={job_id} status={start['status']}")

        print("\n2. Poll /progress ...")
        last_seq = -1
        while True:
            p = (await c.get(
                f"{base}/api/jobs/{job_id}/progress?since={last_seq}"
            )).json()
            for m in p["messages"]:
                print(f"   [{m['seq']}] {m['text']}")
                last_seq = max(last_seq, m["seq"])
            if p["status"] in ("done", "error"):
                print(f"   final status: {p['status']}")
                if p["status"] == "error":
                    print(f"   error: {p['error']}")
                    return
                break
            await asyncio.sleep(0.5)

        print("\n3. GET /result ...")
        r = await c.get(f"{base}/api/jobs/{job_id}/result")
        assert r.status_code == 200, r.text
        result = r.json()
        print(f"   type: {result.get('type')}")
        assert result.get("type") == "mesh"
        verts = result.get("vertices", [])
        idx = result.get("indices", [])
        assert len(verts) > 0 and len(verts) % 3 == 0
        assert len(idx) > 0 and len(idx) % 3 == 0
        print(f"   vertices: {len(verts) // 3}, triangles: {len(idx) // 3}")

        print("\nE2E OK.")


if __name__ == "__main__":
    asyncio.run(main())
```

- [ ] **Step 2: Manual verification (run against a live server)**

In one terminal:

```bash
cd ~/PycharmProjects/ascribe-link && .venv/Scripts/ascribe-link --enable-agent
```

In another:

```bash
cd ~/PycharmProjects/ascribe-link && .venv/Scripts/python test_jobs_e2e.py
```

Expected: progress messages stream to the terminal; final `E2E OK.` prints.

- [ ] **Step 3: Commit**

```bash
cd ~/PycharmProjects/ascribe-link && git add test_jobs_e2e.py && git commit -m "$(cat <<'EOF'
Add end-to-end smoke test for AI agent job

Drives the full /start → /progress → /result flow against a live server
with the agent enabled. Prints streaming progress and asserts the final
mesh has valid vertex/index counts.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Phase 2 — Godot Client (ascribe-xr)

All client paths are relative to `C:\Users\rp\Documents\vr-start\`. Godot 4. No gdUnit4 setup — verification is via running the game locally against a live ascribe-link.

### Task 10: `AscribeLinkClient.run_job` — start/poll/result state machine

**Files:**
- Modify: `scripts/DataSources/ascribe_link_client.gd`

- [ ] **Step 1: Add signals and `run_job` to `AscribeLinkClient`**

Add these signals near the top of `scripts/DataSources/ascribe_link_client.gd`, after the existing `request_error`:

```gdscript
signal job_progress(text: String)
signal job_complete(result: Dictionary)
signal job_error(error: String)
```

Add this method. Paste it above the existing `_get_result_string` helper:

```gdscript
## Run a dynamic specimen as a job: POST /start, poll /progress, GET /result.
## Authority should be the only caller. Results are emitted via signals.
func run_job(specimen_id: String, params: Dictionary, room_id: String = "ascribe") -> void:
    if _parent == null:
        job_error.emit("Client not set up")
        return

    # --- 1. POST /start ---
    var start_http := HTTPRequest.new()
    _parent.add_child(start_http)
    var start_url := _base_url + "/api/specimens/" + specimen_id + "/start"
    var start_body := JSON.stringify({"params": params, "room_id": room_id})
    var err := start_http.request(
        start_url,
        ["Content-Type: application/json"],
        HTTPClient.METHOD_POST,
        start_body,
    )
    if err != OK:
        start_http.queue_free()
        job_error.emit("Failed to POST /start: %s" % error_string(err))
        return

    var start_response = await start_http.request_completed
    start_http.queue_free()

    var start_result: int = start_response[0]
    var start_code: int = start_response[1]
    var start_payload: PackedByteArray = start_response[3]
    if start_result != HTTPRequest.RESULT_SUCCESS or start_code != 200:
        job_error.emit("POST /start failed: HTTP %d" % start_code)
        return

    var start_json: Variant = JSON.parse_string(start_payload.get_string_from_utf8())
    if not (start_json is Dictionary):
        job_error.emit("Invalid /start response")
        return
    var job_id: String = start_json.get("job_id", "")
    var start_status: String = start_json.get("status", "")
    if job_id.is_empty():
        job_error.emit("Missing job_id in /start response")
        return

    # --- 2. Poll /progress until status is terminal ---
    if start_status != "done":
        var last_seq := -1
        while true:
            var prog_http := HTTPRequest.new()
            _parent.add_child(prog_http)
            var prog_url := "%s/api/jobs/%s/progress?since=%d" % [_base_url, job_id, last_seq]
            err = prog_http.request(prog_url)
            if err != OK:
                prog_http.queue_free()
                job_error.emit("Failed to GET /progress: %s" % error_string(err))
                return
            var prog_response = await prog_http.request_completed
            prog_http.queue_free()

            var prog_result: int = prog_response[0]
            var prog_code: int = prog_response[1]
            var prog_payload: PackedByteArray = prog_response[3]
            if prog_result != HTTPRequest.RESULT_SUCCESS or prog_code != 200:
                # One retry path: wait and try again, then give up.
                await _parent.get_tree().create_timer(0.5).timeout
                continue

            var prog_json: Variant = JSON.parse_string(prog_payload.get_string_from_utf8())
            if not (prog_json is Dictionary):
                job_error.emit("Invalid /progress response")
                return

            for m in prog_json.get("messages", []):
                if m is Dictionary:
                    job_progress.emit(str(m.get("text", "")))
                    last_seq = max(last_seq, int(m.get("seq", last_seq)))

            var st: String = prog_json.get("status", "running")
            if st == "error":
                job_error.emit(str(prog_json.get("error", "unknown error")))
                return
            if st == "done":
                break

            await _parent.get_tree().create_timer(0.5).timeout

    # --- 3. GET /result ---
    var result_http := HTTPRequest.new()
    _parent.add_child(result_http)
    var result_url := "%s/api/jobs/%s/result" % [_base_url, job_id]
    err = result_http.request(result_url)
    if err != OK:
        result_http.queue_free()
        job_error.emit("Failed to GET /result: %s" % error_string(err))
        return

    var result_response = await result_http.request_completed
    result_http.queue_free()

    var r_result: int = result_response[0]
    var r_code: int = result_response[1]
    var r_payload: PackedByteArray = result_response[3]
    if r_result != HTTPRequest.RESULT_SUCCESS or r_code != 200:
        job_error.emit("GET /result failed: HTTP %d" % r_code)
        return

    var result_json: Variant = JSON.parse_string(r_payload.get_string_from_utf8())
    if not (result_json is Dictionary):
        job_error.emit("Invalid /result response")
        return
    job_complete.emit(result_json)
```

- [ ] **Step 2: Manual verification — call `run_job` from a debug command**

In the Godot editor, open a scene that hosts a `DynamicMeshSpecimen` or write a one-off test script that calls `_link_client.run_job("generate_sphere", {"radius": 1.5, "resolution": 32}, "ascribe")` and connects the three signals to print statements.

Expected:
- Console prints "Starting generate_sphere", "Finished in 0.0…s".
- `job_complete` fires with a mesh dict.

- [ ] **Step 3: Commit**

```bash
cd ~/Documents/vr-start && git add scripts/DataSources/ascribe_link_client.gd && git commit -m "$(cat <<'EOF'
Add run_job to AscribeLinkClient

Implements the /start → poll /progress → GET /result state machine
using Godot's HTTPRequest. Emits job_progress, job_complete, and
job_error signals. Authority should be the only caller.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 11: `DynamicMeshSpecimen` progress wiring + RPC broadcast

**Files:**
- Modify: `scripts/Specimen/dynamic_mesh_specimen.gd`

- [ ] **Step 1: Replace the `_http_source.fetch()` call with `_link_client.run_job()`**

In `scripts/Specimen/dynamic_mesh_specimen.gd`, find where the current HTTPSource is driven (around the `_http_source.set_request(...)` / `_http_source.fetch()` calls) and route through the new client method. Add a room state field near the top of the class:

```gdscript
var _room_id: String = "ascribe"
var _active_job_id: String = ""
var _message_log: Array[String] = []
const MESSAGE_LOG_CAP := 50
```

Replace the invocation block (the set_request + fetch path) with:

```gdscript
func _load_dynamic_specimen(specimen_id: String, params: Dictionary) -> void:
    if not is_multiplayer_authority():
        return
    _message_log.clear()
    _link_client.job_progress.connect(_on_job_progress)
    _link_client.job_complete.connect(_on_job_complete)
    _link_client.job_error.connect(_on_job_error)
    _link_client.run_job(specimen_id, params, _room_id)


func _on_job_progress(text: String) -> void:
    _append_message(text)
    _rpc_progress.rpc(text)


func _on_job_complete(result: Dictionary) -> void:
    _on_http_data(result)  # existing mesh/volume dispatch
    _rpc_job_done.rpc()


func _on_job_error(error: String) -> void:
    push_error("Job failed: " + error)
    _append_message("Error: " + error)
    _rpc_job_error.rpc(error)


func _append_message(text: String) -> void:
    _message_log.append(text)
    if _message_log.size() > MESSAGE_LOG_CAP:
        _message_log = _message_log.slice(_message_log.size() - MESSAGE_LOG_CAP)
    _render_message(text)


func _render_message(text: String) -> void:
    if ui_instance == null:
        return
    var log := ui_instance.get_node_or_null("LoadingLayer/MessageLog")
    if log is RichTextLabel:
        log.append_text(text + "\n")


@rpc("authority", "call_remote", "reliable")
func _rpc_progress(text: String) -> void:
    _append_message(text)


@rpc("authority", "call_remote", "reliable")
func _rpc_job_done() -> void:
    if ui_instance:
        ui_instance.get_node("LoadingLayer").hide()


@rpc("authority", "call_remote", "reliable")
func _rpc_job_error(error: String) -> void:
    _append_message("Error: " + error)
```

Replace callers that used to do `_http_source.set_request(...) ; _http_source.fetch()` for dynamic invocation with `_load_dynamic_specimen(function_name, kwargs)`.

- [ ] **Step 2: Manual verification — launch the game, trigger a dynamic specimen, confirm progress appears in LoadingLayer**

In the Godot editor, run the scene. Select a dynamic specimen (e.g., `ai_generate` if available, else `generate_sphere`). The LoadingLayer should appear with the bracket and any intermediate messages, then hide when the mesh is set.

Expected:
- For `generate_sphere`: one `Starting …` message, then `Finished in …s`, then LoadingLayer hides.
- For `ai_generate`: a stream of `Tool: …` messages and agent text snippets.

- [ ] **Step 3: Commit**

```bash
cd ~/Documents/vr-start && git add scripts/Specimen/dynamic_mesh_specimen.gd && git commit -m "$(cat <<'EOF'
Wire job progress into DynamicMeshSpecimen with RPC broadcast

Authority calls AscribeLinkClient.run_job and mirrors each progress
message to peers via @rpc. Adds a 50-line local message log and renders
each new message into LoadingLayer/MessageLog if present. Existing
mesh/volume dispatch runs on job_complete.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 12: Add `MessageLog` and `DownloadBar` to `LoadingLayer`

**Files:**
- Modify: `scenes/UI/LoadingLayer.tscn` (or wherever the LoadingLayer node lives; search the repo for `LoadingLayer` if unsure)

- [ ] **Step 1: Open the scene containing `LoadingLayer` in the Godot editor**

In Godot, locate the scene that defines the `LoadingLayer` CanvasLayer or Control. Add:

1. A `RichTextLabel` node as a child of `LoadingLayer`, named `MessageLog`.
   - Set `bbcode_enabled = false` (plain text).
   - Set `scroll_following = true` so it auto-scrolls.
   - Anchor it to the bottom-right corner, sized about 400×200 px.
2. A `ProgressBar` node as a child of `LoadingLayer`, named `DownloadBar`.
   - Set `min_value = 0`, `max_value = 1`, `step = 0.01`.
   - Default `visible = false` (it will be shown only during `/result` download).

- [ ] **Step 2: Manual verification — run the game and trigger a slow dynamic specimen**

Expected: the LoadingLayer now shows a bottom-right text panel with live messages. The progress bar remains hidden.

- [ ] **Step 3: Commit**

```bash
cd ~/Documents/vr-start && git add scenes && git commit -m "$(cat <<'EOF'
Add MessageLog and DownloadBar to LoadingLayer scene

RichTextLabel auto-scrolls to show streaming job progress text. The
ProgressBar is reserved for result-download bytes progress and stays
hidden until the client switches to the /result fetch phase.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 13: Late-joiner state sync

**Files:**
- Modify: `scripts/Specimen/dynamic_mesh_specimen.gd`

- [ ] **Step 1: Track `_active_job_id` across run_job lifetime**

Set `_active_job_id` at the start of a load and clear it on completion/error. Inside `_load_dynamic_specimen`, wrap the call:

```gdscript
func _load_dynamic_specimen(specimen_id: String, params: Dictionary) -> void:
    if not is_multiplayer_authority():
        return
    _active_job_id = specimen_id  # use the specimen id as an in-progress marker
    _message_log.clear()
    _link_client.job_progress.connect(_on_job_progress)
    _link_client.job_complete.connect(_on_job_complete)
    _link_client.job_error.connect(_on_job_error)
    _link_client.run_job(specimen_id, params, _room_id)


func _on_job_complete(result: Dictionary) -> void:
    _on_http_data(result)
    _rpc_job_done.rpc()
    _active_job_id = ""


func _on_job_error(error: String) -> void:
    push_error("Job failed: " + error)
    _append_message("Error: " + error)
    _rpc_job_error.rpc(error)
    _active_job_id = ""
```

- [ ] **Step 2: On `peer_connected`, sync state to the new peer**

In `_ready()` of `DynamicMeshSpecimen`:

```gdscript
    multiplayer.peer_connected.connect(_on_peer_connected)
```

Add the handler:

```gdscript
func _on_peer_connected(peer_id: int) -> void:
    if not is_multiplayer_authority():
        return
    if _active_job_id.is_empty():
        return
    _rpc_sync_state.rpc_id(peer_id, _active_job_id, _message_log)


@rpc("authority", "call_remote", "reliable")
func _rpc_sync_state(job_specimen_id: String, backlog: Array) -> void:
    # Render the backlog so the joiner sees the current state of the load.
    if ui_instance:
        ui_instance.get_node("LoadingLayer").show()
    var log := ui_instance.get_node_or_null("LoadingLayer/MessageLog") if ui_instance else null
    if log is RichTextLabel:
        log.clear()
    for text in backlog:
        _append_message(str(text))
```

- [ ] **Step 3: Manual verification — two-client test**

1. Launch two game instances (one authority, one peer). Start a slow dynamic specimen load on the authority. Confirm peer sees messages streaming in.
2. Launch a third instance and have it join mid-load. Confirm it receives the backlog immediately.

- [ ] **Step 4: Commit**

```bash
cd ~/Documents/vr-start && git add scripts/Specimen/dynamic_mesh_specimen.gd && git commit -m "$(cat <<'EOF'
Sync in-progress job state to late-joining peers

Authority tracks _active_job_id and _message_log. When a new peer
connects during a load, authority replays the current message backlog
via _rpc_sync_state so the joiner sees the load in progress without
hitting ascribe-link.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 14: Optional — cancellation via `DELETE /api/jobs/{id}`

**Files:**
- Modify: `scripts/DataSources/ascribe_link_client.gd`
- Modify: `scripts/Specimen/dynamic_mesh_specimen.gd`

- [ ] **Step 1: Add a `cancel_job` helper to `AscribeLinkClient`**

Add a field to track the current `job_id` and a cancellation method:

```gdscript
var _current_job_id: String = ""

func cancel_current_job() -> void:
    if _parent == null or _current_job_id.is_empty():
        return
    var http := HTTPRequest.new()
    _parent.add_child(http)
    var url := "%s/api/jobs/%s" % [_base_url, _current_job_id]
    var err := http.request(url, [], HTTPClient.METHOD_DELETE)
    if err != OK:
        http.queue_free()
        return
    await http.request_completed
    http.queue_free()
```

Inside `run_job`, set `_current_job_id = job_id` after parsing the `/start` response, and clear it in all exit paths (complete, error).

- [ ] **Step 2: Hook cancellation to the LoadingLayer's close/cancel button**

In `dynamic_mesh_specimen.gd`, if the `LoadingLayer` has a cancel button (add one in the scene if needed):

```gdscript
func _on_loading_cancel_pressed() -> void:
    if not is_multiplayer_authority():
        return
    _link_client.cancel_current_job()
```

Wire the button's `pressed` signal to this handler in the scene, or connect it in `_ready()` if the button exists:

```gdscript
    var cancel_btn := ui_instance.get_node_or_null("LoadingLayer/CancelButton") if ui_instance else null
    if cancel_btn is Button:
        cancel_btn.pressed.connect(_on_loading_cancel_pressed)
```

- [ ] **Step 3: Manual verification**

Start a slow dynamic specimen load. Press cancel. Confirm the server-side task is cancelled (watch ascribe-link logs) and the LoadingLayer closes on all peers via `_rpc_job_error("cancelled")`.

- [ ] **Step 4: Commit**

```bash
cd ~/Documents/vr-start && git add scripts/DataSources/ascribe_link_client.gd scripts/Specimen/dynamic_mesh_specimen.gd && git commit -m "$(cat <<'EOF'
Add cancel button support via DELETE /api/jobs/{id}

Authority sends DELETE to ascribe-link, which cancels the underlying
task. Peers receive the resulting error via the existing rpc_job_error
path so they tear down their LoadingLayer consistently.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Coverage check against the spec

| Spec section | Covered by task |
|---|---|
| API surface (POST /start, GET /progress, GET /result, DELETE) | Task 5 (+ Task 8 for federation) |
| Job registry + lifecycle | Task 2 |
| TTL sweeper | Task 7 |
| ProgressReporter plumbing | Tasks 1, 3, 4 |
| AI agent integration | Task 6 |
| Federation | Task 8 |
| Godot authority/peer flow | Tasks 10, 11 |
| Late-joiner handling | Task 13 |
| LoadingLayer UI | Task 12 |
| Cancellation | Task 14 |
| E2E smoke test | Task 9 |
| Unit tests (5 server files) | Tasks 1, 2, 4, 5, 6, 8 |
