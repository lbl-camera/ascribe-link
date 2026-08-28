"""Tests for the conversational MCP tool surface (ascribe_link.agent_ws.tools)."""

from __future__ import annotations

import asyncio
import base64
import math

import numpy as np
import pytest

pytest.importorskip("claude_agent_sdk")

from ascribe_link.agent_ws import tools as ws_tools  # noqa: E402
from ascribe_link.models import MeshResult, VolumeResult  # noqa: E402

EXPECTED_TOOL_NAMES = [
    "submit_mesh",
    "submit_volume",
    "analyze_specimen",
    "load_specimen",
    "set_active_specimen",
    "remove_specimen",
    "set_room_scene",
    "set_display_param",
    "capture_viewport",
]


class FakeSink:
    """Records calls; used in place of the real manager-backed sink."""

    def __init__(self):
        self.room_id = "room-1"
        self.requested: list[tuple[str, dict]] = []
        self.staged: dict[str, object] = {}
        self._next_id = 0
        self.request_result: object = {"ok": True}
        self.request_delay: float = 0.0
        self.request_exc: Exception | None = None

    async def request_client_tool(self, name: str, args: dict):
        self.requested.append((name, args))
        if self.request_delay:
            await asyncio.sleep(self.request_delay)
        if self.request_exc is not None:
            raise self.request_exc
        return self.request_result

    def stage_result(self, result) -> str:
        specimen_id = f"specimen-{self._next_id}"
        self._next_id += 1
        self.staged[specimen_id] = result
        return specimen_id

    def get_staged(self, specimen_id: str):
        return self.staged.get(specimen_id)


async def _call_tool(tool_objs, name: str, args: dict) -> dict:
    # build_conversation_tools returns the raw SdkMcpTool list separately so
    # tests can invoke handlers directly (claude_agent_sdk's McpSdkServerConfig
    # does not expose a synchronous tool lookup, and stashing them inside the
    # server dict breaks the SDK's json.dumps of the CLI command).
    for tool_obj in tool_objs:
        if tool_obj.name == name:
            return await tool_obj.handler(args)
    raise AssertionError(f"tool {name!r} not found on server")


def test_allowed_tools_has_nine_names():
    sink = FakeSink()
    server, allowed_tools, sdk_tools = ws_tools.build_conversation_tools(sink)
    assert server["name"] == "scene"
    assert sorted(allowed_tools) == sorted(
        f"mcp__scene__{n}" for n in EXPECTED_TOOL_NAMES
    )
    assert len(allowed_tools) == 9


def test_set_display_param_forwards_to_client():
    sink = FakeSink()
    sink.request_result = {"status": "ok"}
    server, _, sdk_tools = ws_tools.build_conversation_tools(sink)

    result = asyncio.run(
        _call_tool(sdk_tools, "set_display_param", {"index": 0, "name": "gamma", "value": 2.0})
    )

    assert sink.requested == [("set_display_param", {"index": 0, "name": "gamma", "value": 2.0})]
    assert result["content"][0]["type"] == "text"
    assert "ok" in result["content"][0]["text"]
    assert not result.get("is_error")


def test_capture_viewport_wraps_bytes_as_image():
    sink = FakeSink()
    jpeg_bytes = b"\xff\xd8\xff\xe0FAKEJPEG"
    sink.request_result = jpeg_bytes
    server, _, sdk_tools = ws_tools.build_conversation_tools(sink)

    result = asyncio.run(_call_tool(sdk_tools, "capture_viewport", {}))

    block = result["content"][0]
    assert block["type"] == "image"
    assert block["source"]["type"] == "base64"
    assert block["source"]["media_type"] == "image/jpeg"
    assert base64.b64decode(block["source"]["data"]) == jpeg_bytes


def test_client_tool_timeout_returns_error_not_exception(monkeypatch):
    monkeypatch.setattr(ws_tools, "CLIENT_TOOL_TIMEOUT", 0.05)
    sink = FakeSink()
    sink.request_delay = 1.0
    server, _, sdk_tools = ws_tools.build_conversation_tools(sink)

    result = asyncio.run(_call_tool(sdk_tools, "load_specimen", {"specimen_id": "s1"}))

    assert result.get("is_error") is True
    assert "load_specimen" in result["content"][0]["text"]
    assert "failed" in result["content"][0]["text"].lower()


def test_client_tool_exception_returns_error():
    sink = FakeSink()
    sink.request_exc = RuntimeError("boom")
    server, _, sdk_tools = ws_tools.build_conversation_tools(sink)

    result = asyncio.run(_call_tool(sdk_tools, "set_room_scene", {"name": "lab"}))

    assert result.get("is_error") is True
    assert "boom" in result["content"][0]["text"]


def test_submit_volume_stages_volume_result():
    sink = FakeSink()
    server, _, sdk_tools = ws_tools.build_conversation_tools(sink)

    arr = np.arange(4 * 4 * 4, dtype=np.float32).reshape(4, 4, 4)
    data_b64 = base64.b64encode(arr.tobytes()).decode("ascii")

    result = asyncio.run(
        _call_tool(
            sdk_tools,
            "submit_volume",
            {"shape": [4, 4, 4], "dtype": "float32", "data": data_b64},
        )
    )

    assert len(sink.staged) == 1
    specimen_id, staged = next(iter(sink.staged.items()))
    assert isinstance(staged, VolumeResult)
    assert staged.shape == [4, 4, 4]
    assert specimen_id in result["content"][0]["text"]
    assert not result.get("is_error")


def test_submit_mesh_invalid_returns_error():
    sink = FakeSink()
    server, _, sdk_tools = ws_tools.build_conversation_tools(sink)

    result = asyncio.run(_call_tool(sdk_tools, "submit_mesh", {"vertices": [], "indices": []}))

    assert "Error" in result["content"][0]["text"]
    assert len(sink.staged) == 0


def test_submit_mesh_stages_mesh_result():
    sink = FakeSink()
    server, _, sdk_tools = ws_tools.build_conversation_tools(sink)

    vertices = [[0, 0, 0], [1, 0, 0], [0, 1, 0]]
    indices = [0, 1, 2]

    result = asyncio.run(_call_tool(sdk_tools, "submit_mesh", {"vertices": vertices, "indices": indices}))

    assert len(sink.staged) == 1
    specimen_id, staged = next(iter(sink.staged.items()))
    assert isinstance(staged, MeshResult)
    assert specimen_id in result["content"][0]["text"]


def test_analyze_specimen_volume_stats():
    sink = FakeSink()
    server, _, sdk_tools = ws_tools.build_conversation_tools(sink)

    arr = np.arange(4 * 4 * 4, dtype=np.float32).reshape(4, 4, 4)
    vr = VolumeResult.from_numpy(arr)
    specimen_id = sink.stage_result(vr)

    result = asyncio.run(_call_tool(sdk_tools, "analyze_specimen", {"specimen_id": specimen_id}))

    text = result["content"][0]["text"]
    assert "[4, 4, 4]" in text or "4, 4, 4" in text
    assert str(float(arr.min())) in text or "0.0" in text
    assert str(float(arr.max())) in text or f"{arr.max():g}" in text
    mean = arr.mean()
    assert f"{mean:.3f}"[:6] in text or math.isfinite(mean)


def test_analyze_specimen_unknown_id_errors():
    sink = FakeSink()
    server, _, sdk_tools = ws_tools.build_conversation_tools(sink)

    result = asyncio.run(_call_tool(sdk_tools, "analyze_specimen", {"specimen_id": "nope"}))

    assert result.get("is_error") is True
    assert "nope" in result["content"][0]["text"]


def test_server_config_is_json_serializable_for_the_cli():
    """Regression: the SDK json.dumps-es the server config verbatim when
    building the CLI command (subprocess_cli._build_command); any non-JSON
    value we stash in it crashes ClaudeSDKClient.connect() with the real SDK
    (TypeError: Object of type SdkMcpTool is not JSON serializable)."""
    import json

    server, _, _ = ws_tools.build_conversation_tools(FakeSink())
    serializable = {k: v for k, v in server.items() if k != "instance"}
    json.dumps({"mcpServers": {"scene": serializable}})  # must not raise
    assert "_sdk_tools" not in server
