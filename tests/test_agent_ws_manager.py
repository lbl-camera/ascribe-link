"""Tests for ascribe_link.agent_ws.manager.AgentSessionManager.

`manager.AgentConversation` is monkeypatched with a `FakeConversation` so no
real worker thread / SDK client is involved; the manager's own asyncio-level
behavior (fan-out, validation, tool correlation, binary routing, session
lifecycle) is exercised directly on the test's event loop.
"""

import asyncio

import pytest

from ascribe_link.agent_ws import manager as manager_module
from ascribe_link.agent_ws.manager import AgentSessionManager


class FakeSocket:
    def __init__(self, fail=False):
        self.sent = []
        self.fail = fail
        self.closed = False

    async def send_json(self, frame):
        if self.fail:
            self.closed = True
            raise RuntimeError("socket closed")
        self.sent.append(frame)


class FakeConversation:
    instances = []

    def __init__(self, room_id, *, client_factory, emit, request_client_tool, model, **kw):
        self.room_id = room_id
        self.client_factory = client_factory
        self.emit = emit
        self.request_client_tool = request_client_tool
        self.model = model
        self.started = False
        self.stopped = False
        self.submitted = []
        self.attached_images = []
        self.interrupted = False
        self._position = 0
        FakeConversation.instances.append(self)

    def start(self):
        self.started = True

    def submit_text(self, text):
        self.submitted.append(text)
        pos = self._position
        self._position += 1
        return pos

    def interrupt(self):
        self.interrupted = True

    def stop(self):
        self.stopped = True

    def history(self):
        return []

    def attach_image(self, payload):
        self.attached_images.append(payload)


@pytest.fixture(autouse=True)
def patch_conversation(monkeypatch):
    FakeConversation.instances = []
    monkeypatch.setattr(manager_module, "AgentConversation", FakeConversation)
    yield


def make_manager():
    return AgentSessionManager(model="claude-test", client_factory=lambda: object())


async def test_broadcast_reaches_both_sockets_and_prunes_dead():
    mgr = make_manager()
    s1 = FakeSocket()
    s2 = FakeSocket(fail=True)
    await mgr.connect("room1", s1)
    await mgr.connect("room1", s2)
    s1.sent.clear()
    s2.sent.clear()

    await mgr.broadcast("room1", {"type": "status", "text": "hi"})

    assert {"type": "status", "text": "hi"} in s1.sent
    room = mgr._rooms["room1"]
    assert s2 not in room.sockets
    assert s1 in room.sockets


async def test_text_frame_submits_and_queues_position():
    mgr = make_manager()
    s1 = FakeSocket()
    await mgr.connect("room1", s1)
    convo = FakeConversation.instances[-1]
    s1.sent.clear()

    await mgr.handle_frame("room1", s1, {"type": "text", "text": "hello"})
    assert convo.submitted == ["hello"]
    assert not any(f["type"] == "turn_queued" for f in s1.sent)

    await mgr.handle_frame("room1", s1, {"type": "text", "text": "second"})
    queued = [f for f in s1.sent if f["type"] == "turn_queued"]
    assert queued == [{"type": "turn_queued", "position": 1}]


async def test_invalid_frame_sends_error_to_sender_only():
    mgr = make_manager()
    s1 = FakeSocket()
    s2 = FakeSocket()
    await mgr.connect("room1", s1)
    await mgr.connect("room1", s2)
    s1.sent.clear()
    s2.sent.clear()

    await mgr.handle_frame("room1", s1, {"type": "bogus"})

    assert any(f["type"] == "error" for f in s1.sent)
    assert not any(f["type"] == "error" for f in s2.sent)


async def test_request_client_tool_broadcasts_and_resolves_on_tool_result():
    mgr = make_manager()
    s1 = FakeSocket()
    s2 = FakeSocket()
    await mgr.connect("room1", s1)
    await mgr.connect("room1", s2)
    s1.sent.clear()
    s2.sent.clear()

    task = asyncio.ensure_future(
        mgr.request_client_tool("room1", "load_specimen", {"specimen_id": "abc"})
    )
    await asyncio.sleep(0)  # let the broadcast happen

    calls = [f for f in s1.sent if f["type"] == "tool_call"]
    assert len(calls) == 1
    call = calls[0]
    assert call["executor"] == 0
    assert call["name"] == "load_specimen"
    assert s2.sent == s1.sent  # broadcast to all

    request_id = call["request_id"]
    await mgr.handle_frame(
        "room1", s2, {"type": "tool_result", "request_id": request_id, "result": {"ok": True}}
    )

    result = await asyncio.wait_for(task, 5.0)
    assert result == {"ok": True}


async def test_request_client_tool_times_out(monkeypatch):
    monkeypatch.setattr(manager_module, "TOOL_CALL_TIMEOUT", 0.05)
    mgr = make_manager()
    s1 = FakeSocket()
    await mgr.connect("room1", s1)

    with pytest.raises(asyncio.TimeoutError):
        await mgr.request_client_tool("room1", "load_specimen", {"specimen_id": "abc"})


async def test_capture_viewport_resolves_on_binary_screenshot_not_tool_result():
    mgr = make_manager()
    s1 = FakeSocket()
    await mgr.connect("room1", s1)

    task = asyncio.ensure_future(mgr.request_client_tool("room1", "capture_viewport", {}))
    await asyncio.sleep(0)
    call = [f for f in s1.sent if f["type"] == "tool_call"][0]
    request_id = call["request_id"]

    # tool_result for capture_viewport is an ack only -- must not resolve.
    await mgr.handle_frame(
        "room1", s1, {"type": "tool_result", "request_id": request_id, "result": "ack"}
    )
    await asyncio.sleep(0)
    assert not task.done()

    from ascribe_link.agent_ws import protocol

    binary = protocol.encode_binary({"kind": "screenshot"}, b"\xff\xd8jpegbytes")
    _header, payload = protocol.decode_binary(binary)
    await mgr.handle_binary("room1", s1, binary)

    result = await asyncio.wait_for(task, 5.0)
    assert result == payload


async def test_overlapping_capture_viewport_requests_both_resolve_in_order():
    mgr = make_manager()
    s1 = FakeSocket()
    await mgr.connect("room1", s1)

    task1 = asyncio.ensure_future(mgr.request_client_tool("room1", "capture_viewport", {}))
    await asyncio.sleep(0)
    task2 = asyncio.ensure_future(mgr.request_client_tool("room1", "capture_viewport", {}))
    await asyncio.sleep(0)

    calls = [f for f in s1.sent if f["type"] == "tool_call"]
    assert len(calls) == 2
    assert calls[0]["request_id"] != calls[1]["request_id"]

    from ascribe_link.agent_ws import protocol

    binary1 = protocol.encode_binary({"kind": "screenshot"}, b"first-frame")
    binary2 = protocol.encode_binary({"kind": "screenshot"}, b"second-frame")

    # First binary frame must resolve the FIRST (oldest) pending capture,
    # not whichever call happens to be most recently registered.
    await mgr.handle_binary("room1", s1, binary1)
    result1 = await asyncio.wait_for(task1, 5.0)
    assert result1 == b"first-frame"
    assert not task2.done()

    await mgr.handle_binary("room1", s1, binary2)
    result2 = await asyncio.wait_for(task2, 5.0)
    assert result2 == b"second-frame"

    assert mgr._capture_pending.get("room1") in (None, [])


async def test_binary_screenshot_without_pending_capture_attaches_image():
    mgr = make_manager()
    s1 = FakeSocket()
    await mgr.connect("room1", s1)
    convo = FakeConversation.instances[-1]

    from ascribe_link.agent_ws import protocol

    binary = protocol.encode_binary({"kind": "screenshot"}, b"rawjpegbytes")
    await mgr.handle_binary("room1", s1, binary)

    assert convo.attached_images == [b"rawjpegbytes"]


async def test_end_conversation_stops_and_next_text_creates_fresh_session():
    mgr = make_manager()
    s1 = FakeSocket()
    await mgr.connect("room1", s1)
    first = FakeConversation.instances[-1]
    assert len(FakeConversation.instances) == 1

    await mgr.handle_frame("room1", s1, {"type": "end_conversation"})
    assert first.stopped is True
    assert mgr._rooms["room1"].conversation is None

    await mgr.handle_frame("room1", s1, {"type": "text", "text": "hi again"})
    assert len(FakeConversation.instances) == 2
    second = FakeConversation.instances[-1]
    assert second is not first
    assert second.submitted == ["hi again"]
