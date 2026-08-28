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
        # Ordered (kind, payload) log covering BOTH send_json and
        # send_bytes -- needed to assert relative ordering between JSON
        # control frames and binary TTS frames.
        self.events = []
        self.fail = fail
        self.closed = False

    async def send_json(self, frame):
        if self.fail:
            self.closed = True
            raise RuntimeError("socket closed")
        self.sent.append(frame)
        self.events.append(("json", frame))

    async def send_bytes(self, data):
        if self.fail:
            self.closed = True
            raise RuntimeError("socket closed")
        self.events.append(("binary", data))


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


# ----------------------------------------------------------------------
# Cross-loop marshalling (the sink is called from the worker thread/loop)
# ----------------------------------------------------------------------


async def test_room_sink_request_client_tool_marshals_across_loops():
    """`_RoomSink.request_client_tool` must be safe to await on a worker loop.

    Regression: it used to `await manager.request_client_tool(...)` directly,
    which creates/awaits a main-loop future from a foreign loop and calls
    `socket.send_json` off-thread -- "attached to a different loop" against
    the real SDK. The fakes masked it because everything shared one loop.
    """
    import threading

    mgr = make_manager()
    s1 = FakeSocket()
    await mgr.connect("room1", s1)
    s1.sent.clear()

    sink = manager_module._RoomSink(mgr, "room1")
    box: dict = {}
    done = threading.Event()

    def worker() -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            box["result"] = loop.run_until_complete(
                sink.request_client_tool("load_specimen", {"specimen_id": "abc"})
            )
        except BaseException as err:  # noqa: BLE001 - recorded and re-asserted
            box["error"] = err
        finally:
            loop.close()
            done.set()

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()

    # Wait (on the main loop) for the tool_call broadcast to land.
    for _ in range(200):
        calls = [f for f in s1.sent if f["type"] == "tool_call"]
        if calls:
            break
        await asyncio.sleep(0.01)
    assert calls, "sink never produced a tool_call frame"

    await mgr.handle_frame(
        "room1",
        s1,
        {"type": "tool_result", "request_id": calls[0]["request_id"], "result": {"ok": True}},
    )

    for _ in range(200):
        if done.is_set():
            break
        await asyncio.sleep(0.01)
    assert done.is_set(), "worker thread never finished"
    assert "error" not in box, f"sink raised across loops: {box.get('error')!r}"
    assert box["result"] == {"ok": True}


async def test_conversation_receives_the_sink_marshalling_path():
    """There is exactly ONE marshalling path: the sink's own method."""
    mgr = make_manager()
    s1 = FakeSocket()
    await mgr.connect("room1", s1)
    convo = FakeConversation.instances[-1]
    assert isinstance(convo.request_client_tool.__self__, manager_module._RoomSink)


# ----------------------------------------------------------------------
# Executor identity
# ----------------------------------------------------------------------


async def test_executor_is_surviving_clients_id_after_first_disconnects():
    mgr = make_manager()
    s0 = FakeSocket()
    s1 = FakeSocket()
    await mgr.connect("room1", s0)
    await mgr.connect("room1", s1)
    assert s1.sent[0]["client_id"] == 1

    await mgr.disconnect("room1", s0)
    s1.sent.clear()

    task = asyncio.ensure_future(
        mgr.request_client_tool("room1", "load_specimen", {"specimen_id": "abc"})
    )
    await asyncio.sleep(0)

    call = [f for f in s1.sent if f["type"] == "tool_call"][0]
    assert call["executor"] == 1  # the survivor's persistent id, not index 0

    await mgr.handle_frame(
        "room1", s1, {"type": "tool_result", "request_id": call["request_id"], "result": "done"}
    )
    assert await asyncio.wait_for(task, 5.0) == "done"


async def test_pending_tool_fails_fast_when_executor_disconnects():
    mgr = make_manager()
    s0 = FakeSocket()
    s1 = FakeSocket()
    await mgr.connect("room1", s0)
    await mgr.connect("room1", s1)

    task = asyncio.ensure_future(
        mgr.request_client_tool("room1", "load_specimen", {"specimen_id": "abc"})
    )
    await asyncio.sleep(0)
    assert not task.done()

    await mgr.disconnect("room1", s0)  # the executor leaves

    with pytest.raises(RuntimeError, match="disconnected"):
        await asyncio.wait_for(task, 1.0)  # well under TOOL_CALL_TIMEOUT (30s)
    assert mgr._pending == {}


async def test_rejoining_client_gets_a_new_unique_id():
    mgr = make_manager()
    s0 = FakeSocket()
    s1 = FakeSocket()
    await mgr.connect("room1", s0)
    await mgr.connect("room1", s1)
    await mgr.disconnect("room1", s0)

    s2 = FakeSocket()
    await mgr.connect("room1", s2)

    ids = [s.sent[0]["client_id"] for s in (s0, s1, s2)]
    assert ids == [0, 1, 2]
    assert len(set(ids)) == 3


async def test_capture_pending_is_cleaned_up_on_executor_disconnect():
    mgr = make_manager()
    s0 = FakeSocket()
    await mgr.connect("room1", s0)

    task = asyncio.ensure_future(mgr.request_client_tool("room1", "capture_viewport", {}))
    await asyncio.sleep(0)
    assert mgr._capture_pending.get("room1")

    await mgr.disconnect("room1", s0)
    with pytest.raises(RuntimeError):
        await asyncio.wait_for(task, 1.0)
    assert mgr._capture_pending.get("room1") in (None, [])


# ----------------------------------------------------------------------
# Factory failure
# ----------------------------------------------------------------------


async def test_factory_failure_emits_error_frame_instead_of_dropping_socket(monkeypatch):
    def boom(*a, **kw):
        raise RuntimeError("no SDK installed")

    mgr = make_manager()
    s1 = FakeSocket()
    await mgr.connect("room1", s1)
    await mgr.handle_frame("room1", s1, {"type": "end_conversation"})
    monkeypatch.setattr(manager_module, "AgentConversation", boom)
    s1.sent.clear()

    await mgr.handle_frame("room1", s1, {"type": "text", "text": "hi"})

    errors = [f for f in s1.sent if f["type"] == "error"]
    assert errors and "no SDK installed" in errors[0]["message"]


# ----------------------------------------------------------------------
# Staged results
# ----------------------------------------------------------------------


async def test_get_staged_result_is_room_scoped():
    mgr = make_manager()
    sink = manager_module._RoomSink(mgr, "room1")
    specimen_id = sink.stage_result("payload")

    assert mgr.get_staged_result("room1", specimen_id) == "payload"
    assert mgr.get_staged_result("otherroom", specimen_id) is None
    assert mgr.get_staged_result("room1", "nope") is None


# ----------------------------------------------------------------------
# Voice review fixes: barge-in generation guard, widened speaking gate,
# reentrant socket-prune during finalize.
# ----------------------------------------------------------------------


import threading

import numpy as np

from .fake_voice import FakeSTT, FakeTTS


def make_voice_manager():
    mgr = make_manager()
    mgr.stt = FakeSTT()
    mgr.tts = FakeTTS()
    return mgr


class GatedTTS:
    """Blocks `synthesize` on a `threading.Event` until released."""

    def __init__(self):
        self.gate = threading.Event()

    def synthesize(self, text):
        self.gate.wait(timeout=5.0)
        return FakeTTS().synthesize(text)


def _tone_pcm16(n=8000, rate=16000, freq=440.0):
    t = np.arange(n, dtype=np.float32) / rate
    tone = (0.3 * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    return np.clip(tone * 32767, -32768, 32767).astype(np.int16).tobytes()


async def test_no_stale_tts_frames_after_barge_in(monkeypatch):
    """Finding 1: a delta racing the barge-in's cancellation must not spawn
    a fresh TTS task that outlives the barge-in's own agent_audio_end."""
    mgr = make_voice_manager()
    tts = GatedTTS()
    mgr.tts = tts

    s1 = FakeSocket()
    await mgr.connect("room1", s1)
    room = mgr._rooms["room1"]

    # Kick off a sentence -> spawns the drain task, which blocks in
    # synthesize() (simulating slow/streaming TTS mid-turn).
    await mgr._handle_text_delta("room1", "First sentence.")
    await asyncio.sleep(0)  # let the drain task start and enter synthesize()
    assert room.tts_task is not None
    assert room.speaking is True

    loop = asyncio.get_running_loop()
    # Simulate the worker thread delivering another delta exactly while
    # `_cancel_tts` is awaiting the cancelled drain task (before
    # conversation.interrupt() has run) -- this used to spawn a fresh,
    # un-invalidated task that broadcast audio AFTER agent_audio_end.
    loop.call_soon(
        lambda: asyncio.ensure_future(mgr._handle_text_delta("room1", "Second sentence."))
    )

    tts.gate.set()  # let any synthesize() call (correct or stale) resolve fast
    s1.events.clear()
    await mgr.handle_frame("room1", s1, {"type": "bind"})
    await asyncio.sleep(0.05)  # drain anything scheduled

    audio_end_positions = [
        i for i, (kind, f) in enumerate(s1.events) if kind == "json" and f["type"] == "agent_audio_end"
    ]
    assert audio_end_positions, s1.events
    after = s1.events[audio_end_positions[0] + 1 :]
    assert not any(kind == "binary" for kind, _ in after), s1.events


async def test_bind_interrupts_during_unterminated_stream():
    """Finding 2: an unterminated streaming delta must still set `speaking`
    so a bind mid-stream barges in (interrupts + ends audio) instead of
    letting the eventual flush talk over the new speaker."""
    mgr = make_voice_manager()
    s1 = FakeSocket()
    await mgr.connect("room1", s1)
    convo = FakeConversation.instances[-1]
    room = mgr._rooms["room1"]

    # No sentence terminator -> chunker.feed() yields no complete sentence.
    await mgr._handle_text_delta("room1", "Hello witho")
    assert room.speaking is True
    assert room.tts_task is None  # no full sentence yet, nothing to synthesize

    s1.sent.clear()
    await mgr.handle_frame("room1", s1, {"type": "bind"})

    assert convo.interrupted is True
    types = [f["type"] for f in s1.sent]
    assert "agent_audio_end" in types
    assert "speaker_bound" in types
    assert not any(kind == "binary" for kind, _ in s1.events)  # nothing was ever synthesized
    assert room.speaking is False


async def test_dead_speaker_finalize_does_not_reenter_on_broadcast_prune(monkeypatch):
    """Finding 3: finalizing a dead speaker's utterance must not reenter
    itself when its own broadcast fails against that same (already-dead)
    socket."""
    mgr = make_voice_manager()
    s1 = FakeSocket(fail=True)  # every send_json/send_bytes raises
    await mgr.connect("room1", s1)
    convo = FakeConversation.instances[-1]
    room = mgr._rooms["room1"]
    room.speaker = s1

    buf = manager_module.UtteranceBuffer()
    buf.add(_tone_pcm16(), 16000)
    room.utterance_buffer = buf

    drop_calls = []
    orig_drop = mgr._drop_socket

    async def counting_drop(room_, socket_, room_id_):
        drop_calls.append(socket_)
        await orig_drop(room_, socket_, room_id_)

    monkeypatch.setattr(mgr, "_drop_socket", counting_drop)

    await mgr._finalize_utterance(room, "room1", s1)

    assert drop_calls == [s1]  # exactly one prune, no reentrant double-drop
    assert len(convo.submitted) == 1  # not double-submitted
    assert s1 not in room.sockets
