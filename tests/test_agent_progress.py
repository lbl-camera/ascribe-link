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

    class TaskProgressMessage:
        def __init__(self, description="", last_tool_name=""):
            self.description = description
            self.last_tool_name = last_tool_name

    class TaskNotificationMessage:
        def __init__(self, status="", summary=""):
            self.status = status
            self.summary = summary

    class SessionMessage:
        def __init__(self, message=None):
            self.message = message

    fake_sdk = MagicMock()
    fake_sdk.AssistantMessage = AssistantMessage
    fake_sdk.TextBlock = TextBlock
    fake_sdk.ToolResultBlock = ToolResultBlock
    fake_sdk.TaskProgressMessage = TaskProgressMessage
    fake_sdk.TaskNotificationMessage = TaskNotificationMessage
    fake_sdk.SessionMessage = SessionMessage
    monkeypatch.setitem(sys.modules, "claude_agent_sdk", fake_sdk)
    return (AssistantMessage, TextBlock, ToolUseBlock, ToolResultBlock,
            TaskProgressMessage, TaskNotificationMessage, SessionMessage)


def test_text_block_reports_first_line(reporter, monkeypatch):
    AssistantMessage, TextBlock, *_ = _install_fake_sdk(monkeypatch)
    from ascribe_link import agent_generator

    msg = AssistantMessage(content=[TextBlock("Let me plan.\nMore detail")])
    agent_generator._emit_agent_events(msg, reporter)
    assert reporter.calls == ["Let me plan."]


def test_tool_use_block_reports_tool_name(reporter, monkeypatch):
    AssistantMessage, _, ToolUseBlock, *_ = _install_fake_sdk(monkeypatch)
    from ascribe_link import agent_generator

    msg = AssistantMessage(content=[ToolUseBlock(name="Bash")])
    agent_generator._emit_agent_events(msg, reporter)
    assert reporter.calls == ["Tool: Bash"]


def test_tool_result_error_is_reported(reporter, monkeypatch):
    _, _, _, ToolResultBlock, *_ = _install_fake_sdk(monkeypatch)
    from ascribe_link import agent_generator

    msg = ToolResultBlock(is_error=True, content="command not found: xyz")
    agent_generator._emit_agent_events(msg, reporter)
    assert len(reporter.calls) == 1
    assert "Tool error" in reporter.calls[0]


def test_tool_result_success_is_ignored(reporter, monkeypatch):
    _, _, _, ToolResultBlock, *_ = _install_fake_sdk(monkeypatch)
    from ascribe_link import agent_generator

    msg = ToolResultBlock(is_error=False, content="lots of output")
    agent_generator._emit_agent_events(msg, reporter)
    assert reporter.calls == []


def test_long_text_block_is_truncated_to_200_chars(reporter, monkeypatch):
    AssistantMessage, TextBlock, *_ = _install_fake_sdk(monkeypatch)
    from ascribe_link import agent_generator

    long_line = "x" * 500
    msg = AssistantMessage(content=[TextBlock(long_line)])
    agent_generator._emit_agent_events(msg, reporter)
    assert len(reporter.calls[0]) == 200


def test_task_progress_reports_description(reporter, monkeypatch):
    *_, TaskProgressMessage, _, _ = _install_fake_sdk(monkeypatch)
    from ascribe_link import agent_generator

    msg = TaskProgressMessage(description="Running marching cubes")
    agent_generator._emit_agent_events(msg, reporter)
    assert reporter.calls == ["Running marching cubes"]


def test_task_progress_falls_back_to_tool_name(reporter, monkeypatch):
    *_, TaskProgressMessage, _, _ = _install_fake_sdk(monkeypatch)
    from ascribe_link import agent_generator

    msg = TaskProgressMessage(description="", last_tool_name="Bash")
    agent_generator._emit_agent_events(msg, reporter)
    assert reporter.calls == ["Tool: Bash"]


def test_task_notification_reports_summary(reporter, monkeypatch):
    *_, TaskNotificationMessage, _ = _install_fake_sdk(monkeypatch)
    from ascribe_link import agent_generator

    msg = TaskNotificationMessage(status="completed", summary="Generated sphere mesh")
    agent_generator._emit_agent_events(msg, reporter)
    assert reporter.calls == ["Generated sphere mesh"]


def test_session_message_extracts_text_blocks(reporter, monkeypatch):
    AssistantMessage, TextBlock, _, _, _, _, SessionMessage = _install_fake_sdk(monkeypatch)
    from ascribe_link import agent_generator

    inner = AssistantMessage(content=[TextBlock("I'll generate the mesh now.")])
    msg = SessionMessage(message=inner)
    agent_generator._emit_agent_events(msg, reporter)
    assert reporter.calls == ["I'll generate the mesh now."]
