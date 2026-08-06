"""Tests for TranscriptWriter (agent conversation transcript)."""
from __future__ import annotations

import pytest

from ascribe_link.agent_generator import TranscriptWriter


class AssistantMessage:
    def __init__(self, content):
        self.content = content


class UserMessage:
    def __init__(self, content):
        self.content = content


class ResultMessage:
    def __init__(self, result=None, is_error=False):
        self.result = result
        self.is_error = is_error


class TextBlock:
    def __init__(self, text):
        self.text = text


class ThinkingBlock:
    def __init__(self, thinking):
        self.thinking = thinking


class ToolUseBlock:
    def __init__(self, name, input):
        self.name = name
        self.input = input


class ToolResultBlock:
    def __init__(self, content, is_error=False, tool_use_id="t1"):
        self.content = content
        self.is_error = is_error
        self.tool_use_id = tool_use_id


@pytest.fixture
def transcript(tmp_path):
    path = tmp_path / "transcript.md"
    return TranscriptWriter(path, "make a sphere", model="claude-sonnet-5"), path


def test_header_and_user_prompt(transcript):
    tw, path = transcript
    text = path.read_text(encoding="utf-8")
    assert "# Agent Transcript" in text
    assert "claude-sonnet-5" in text
    assert "## User" in text
    assert "make a sphere" in text


def test_assistant_text_block(transcript):
    tw, path = transcript
    tw.record(AssistantMessage([TextBlock("I'll build the sphere now.")]))
    text = path.read_text(encoding="utf-8")
    assert "## Assistant" in text
    assert "I'll build the sphere now." in text


def test_thinking_block_included(transcript):
    tw, path = transcript
    tw.record(AssistantMessage([ThinkingBlock("consider radius")]))
    text = path.read_text(encoding="utf-8")
    assert "*Thinking:* consider radius" in text


def test_tool_use_block_shows_name_and_args(transcript):
    tw, path = transcript
    tw.record(
        AssistantMessage(
            [ToolUseBlock("Bash", {"command": "ls", "timeout": 5})]
        )
    )
    text = path.read_text(encoding="utf-8")
    assert "### Tool: Bash" in text
    assert '"command": "ls"' in text
    assert '"timeout": 5' in text


def test_tool_args_truncated(transcript):
    tw, path = transcript
    big = "x" * 10_000
    tw.record(AssistantMessage([ToolUseBlock("Write", {"content": big})]))
    text = path.read_text(encoding="utf-8")
    assert "truncated" in text
    assert big not in text


def test_tool_result_and_error(transcript):
    tw, path = transcript
    tw.record(UserMessage([ToolResultBlock("file written")]))
    tw.record(UserMessage([ToolResultBlock("boom", is_error=True)]))
    text = path.read_text(encoding="utf-8")
    assert "**Tool result:** file written" in text
    assert "**Tool result (error):** boom" in text


def test_result_message(transcript):
    tw, path = transcript
    tw.record(ResultMessage(result="Mesh submitted"))
    text = path.read_text(encoding="utf-8")
    assert "## Result (success)" in text
    assert "Mesh submitted" in text


def test_malformed_message_does_not_raise(transcript):
    tw, path = transcript
    tw.record(object())
    tw.record(AssistantMessage(None))
    tw.record(AssistantMessage([ToolUseBlock("X", object())]))


def test_unwritable_path_does_not_raise(tmp_path):
    tw = TranscriptWriter(tmp_path / "no_dir" / "t.md", "hi")
    tw.record(AssistantMessage([TextBlock("hello")]))
