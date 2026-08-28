"""End-to-end integration tests for the `/ws/agent/{room_id}` websocket.

Exercises the real Litestar wiring (route, DI, lifecycle) with a fake SDK
client factory threaded through `create_app(agent_client_factory=...)` --
no real `claude_agent_sdk` or API key needed.
"""

from __future__ import annotations

import asyncio

import pytest
from litestar.testing import AsyncTestClient

from ascribe_link.app import create_app

from .fake_sdk import FakeSDKFactory, ToolTrigger, text_msg


def _drain_until(ws, target_type, limit=20):
    """Receive frames until one of type `target_type` is seen; return all seen."""
    frames = []
    for _ in range(limit):
        frame = ws.receive_json()
        frames.append(frame)
        if frame["type"] == target_type:
            return frames
    raise AssertionError(f"never saw frame type {target_type!r}; got {frames}")


@pytest.fixture
def agent_factory():
    return FakeSDKFactory()


@pytest.fixture
async def agent_client(agent_factory):
    app = create_app(enable_agent=True, agent_client_factory=agent_factory)
    async with AsyncTestClient(app=app) as c:
        yield c


async def test_history_frame_carries_client_id(agent_client):
    ws = await agent_client.websocket_connect("/ws/agent/testroom")
    with ws:
        frame = ws.receive_json()
        assert frame["type"] == "history"
        assert frame["client_id"] == 0
        assert frame["entries"] == []


async def test_text_turn_yields_scripted_agent_text(agent_client, agent_factory):
    ws = await agent_client.websocket_connect("/ws/agent/testroom")
    with ws:
        ws.receive_json()  # history

        fake = agent_factory.clients[-1]
        fake.scripted_messages = [text_msg("hello from the fake agent")]

        ws.send_json({"type": "text", "text": "hello"})
        frames = _drain_until(ws, "agent_text_done")

        agent_texts = [f for f in frames if f["type"] == "agent_text"]
        assert any(f["text"] == "hello from the fake agent" for f in agent_texts)


async def test_tool_call_round_trip_completes_turn(agent_client, agent_factory):
    ws = await agent_client.websocket_connect("/ws/agent/testroom")
    with ws:
        ws.receive_json()  # history

        fake = agent_factory.clients[-1]
        manager = agent_client.app.state.agent_session_manager
        main_loop = manager._main_loop

        async def call_tool(name, args):
            # The fake's receive_response() runs on the conversation's own
            # worker-thread loop, but AgentSessionManager.request_client_tool
            # must run on the app's main loop (it touches the sockets) --
            # marshal across, mirroring manager.py's own
            # `_start_conversation.request_client_tool` closure.
            concurrent_future = asyncio.run_coroutine_threadsafe(
                manager.request_client_tool("testroom", name, args), main_loop
            )
            return await asyncio.wrap_future(concurrent_future)

        fake.request_client_tool = call_tool
        fake.scripted_messages = [
            ToolTrigger("load_specimen", {"specimen_id": "abc"}),
            text_msg("loaded it"),
        ]

        ws.send_json({"type": "text", "text": "load specimen abc"})

        tool_call_frame = _drain_until(ws, "tool_call")[-1]
        assert tool_call_frame["name"] == "load_specimen"
        assert tool_call_frame["args"] == {"specimen_id": "abc"}

        ws.send_json(
            {
                "type": "tool_result",
                "request_id": tool_call_frame["request_id"],
                "result": {"ok": True},
            }
        )

        frames = _drain_until(ws, "agent_text_done")
        agent_texts = [f for f in frames if f["type"] == "agent_text"]
        assert any(f["text"] == "loaded it" for f in agent_texts)
        assert fake.tool_results == [{"ok": True}]


async def test_second_client_on_same_room_receives_broadcasts(agent_client, agent_factory):
    ws1 = await agent_client.websocket_connect("/ws/agent/testroom")
    with ws1:
        ws1.receive_json()  # history for client 0

        ws2 = await agent_client.websocket_connect("/ws/agent/testroom")
        with ws2:
            hist2 = ws2.receive_json()
            assert hist2["client_id"] == 1

            fake = agent_factory.clients[-1]
            fake.scripted_messages = [text_msg("broadcast me")]

            ws1.send_json({"type": "text", "text": "hi"})

            frames1 = _drain_until(ws1, "agent_text_done")
            frames2 = _drain_until(ws2, "agent_text_done")

            assert any(
                f["type"] == "agent_text" and f["text"] == "broadcast me" for f in frames1
            )
            assert any(
                f["type"] == "agent_text" and f["text"] == "broadcast me" for f in frames2
            )


async def test_agent_disabled_rejects_websocket_connection():
    app = create_app(enable_agent=False)
    async with AsyncTestClient(app=app) as c:
        with pytest.raises(Exception):
            ws = await c.websocket_connect("/ws/agent/x")
            with ws:
                ws.receive_json()


async def test_specimens_route_unaffected_by_agent_ws(agent_client):
    resp = await agent_client.get("/api/specimens/")
    assert resp.status_code == 200


# ----------------------------------------------------------------------
# Agent-staged specimens are fetchable over the normal data route
# ----------------------------------------------------------------------


async def test_staged_specimen_is_served_by_the_data_route(agent_client):
    """The seam the client depends on: dispatcher -> specimen_job_done -> GET data.

    An agent-staged specimen has no catalog entry and no params, so the data
    route must consult the room's staged store first and serve the binary
    envelope.
    """
    import base64

    import numpy as np

    from ascribe_link.agent_ws.manager import _RoomSink
    from ascribe_link.envelope import ENVELOPE_MEDIA_TYPE, decode_envelope
    from ascribe_link.models import VolumeResult

    ws = await agent_client.websocket_connect("/ws/agent/testroom")
    with ws:
        ws.receive_json()  # history -- the room now exists

        manager = agent_client.app.state.agent_session_manager
        arr = np.arange(2 * 3 * 4, dtype=np.uint8).reshape(2, 3, 4)
        volume = VolumeResult(
            shape=[2, 3, 4],
            dtype="uint8",
            data=base64.b64encode(arr.tobytes()).decode("ascii"),
        )
        specimen_id = _RoomSink(manager, "testroom").stage_result(volume)

        resp = await agent_client.get(
            f"/api/specimens/{specimen_id}/data", params={"room_id": "testroom"}
        )

    assert resp.status_code == 200, resp.text
    assert resp.headers["content-type"].startswith(ENVELOPE_MEDIA_TYPE)
    decoded = decode_envelope(resp.content)
    assert list(decoded.shape) == [2, 3, 4]
    assert decoded.dtype == "uint8"


async def test_staged_specimen_via_real_submit_volume_tool(agent_client):
    """Same seam, driven through the actual `submit_volume` MCP tool handler."""
    import base64

    import numpy as np

    pytest.importorskip("claude_agent_sdk")
    from ascribe_link.agent_ws.manager import _RoomSink
    from ascribe_link.agent_ws.tools import build_conversation_tools
    from ascribe_link.envelope import decode_envelope

    ws = await agent_client.websocket_connect("/ws/agent/toolroom")
    with ws:
        ws.receive_json()  # history

        manager = agent_client.app.state.agent_session_manager
        sink = _RoomSink(manager, "toolroom")
        server, _allowed, sdk_tools = build_conversation_tools(sink)
        submit_volume = next(t for t in sdk_tools if t.name == "submit_volume")

        arr = np.linspace(0, 1, 8, dtype=np.float32).reshape(2, 2, 2)
        result = await submit_volume.handler(
            {
                "shape": [2, 2, 2],
                "dtype": "float32",
                "data": base64.b64encode(arr.tobytes()).decode("ascii"),
            }
        )
        text = result["content"][0]["text"]
        specimen_id = text.split("specimen '")[1].split("'")[0]

        resp = await agent_client.get(
            f"/api/specimens/{specimen_id}/data", params={"room_id": "toolroom"}
        )

    assert resp.status_code == 200, resp.text
    decoded = decode_envelope(resp.content)
    assert list(decoded.shape) == [2, 2, 2]


async def test_unknown_specimen_still_404s_for_a_room_with_no_staging(agent_client):
    resp = await agent_client.get(
        "/api/specimens/deadbeef/data", params={"room_id": "testroom"}
    )
    assert resp.status_code == 404
