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
