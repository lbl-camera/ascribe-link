"""Tests for ascribe_link.agent_ws.session.AgentConversation.

Uses a FakeSDKClient (async context manager) so no real claude_agent_sdk
or API key is needed. Frames emitted via `emit` are collected into a
queue.Queue and asserted with timeouts (no sleeps).
"""

import asyncio
import queue
from types import SimpleNamespace

import pytest

from ascribe_link.agent_ws.session import AgentConversation


class FakeSDKClient:
    """Async context manager fake standing in for ClaudeSDKClient.

    `scripted_messages` is a list of message objects to yield from
    `receive_response()` for the *next* query() call. `gate` (an
    asyncio.Event), if set, is awaited before receive_response() yields
    anything, so tests can control interleaving without sleeps.
    """

    def __init__(self):
        self.enter_count = 0
        self.exit_count = 0
        self.queries = []
        self.scripted_messages = []
        self.gate = None

    async def __aenter__(self):
        self.enter_count += 1
        return self

    async def __aexit__(self, exc_type, exc, tb):
        self.exit_count += 1
        return False

    async def query(self, prompt_blocks):
        self.queries.append(prompt_blocks)

    async def receive_response(self):
        if self.gate is not None:
            await self.gate.wait()
        for msg in self.scripted_messages:
            yield msg


def text_msg(text):
    return SimpleNamespace(content=[SimpleNamespace(text=text, name=None)])


def tool_msg(name):
    return SimpleNamespace(content=[SimpleNamespace(text=None, name=name)])


async def fake_request_client_tool(name, args):
    return {"ok": True}


def make_conversation(fake_client, frame_q, request_client_tool=None):
    def factory():
        return fake_client

    def emit(frame):
        frame_q.put(frame)

    return AgentConversation(
        room_id="room1",
        client_factory=factory,
        emit=emit,
        request_client_tool=request_client_tool or fake_request_client_tool,
        model="claude-test",
    )


def drain(frame_q, count, timeout=5.0):
    frames = []
    for _ in range(count):
        frames.append(frame_q.get(timeout=timeout))
    return frames


@pytest.fixture
def frame_q():
    return queue.Queue()


def test_start_enters_client_exactly_once_across_two_turns(frame_q):
    fake = FakeSDKClient()
    fake.scripted_messages = [text_msg("Hello!")]
    convo = make_conversation(fake, frame_q)
    convo.start()
    try:
        convo.submit_text("hi")
        drain(frame_q, 3)  # status, agent_text, agent_text_done

        fake.scripted_messages = [text_msg("Again!")]
        convo.submit_text("hi again")
        drain(frame_q, 3)

        assert fake.enter_count == 1
    finally:
        convo.stop()


def test_emitted_frame_sequence_for_text(frame_q):
    fake = FakeSDKClient()
    fake.scripted_messages = [text_msg("Hello!")]
    convo = make_conversation(fake, frame_q)
    convo.start()
    try:
        convo.submit_text("hi")
        frames = drain(frame_q, 3)
        assert frames[0] == {"type": "status", "text": "thinking"}
        assert frames[1] == {"type": "agent_text", "text": "Hello!"}
        assert frames[2] == {"type": "agent_text_done"}
    finally:
        convo.stop()


def test_tool_block_emits_status(frame_q):
    fake = FakeSDKClient()
    fake.scripted_messages = [tool_msg("mcp__scene__load_specimen")]
    convo = make_conversation(fake, frame_q)
    convo.start()
    try:
        convo.submit_text("load it")
        frames = drain(frame_q, 3)
        assert frames[0] == {"type": "status", "text": "thinking"}
        assert frames[1] == {
            "type": "status",
            "text": "Using the mcp__scene__load_specimen tool...",
        }
        assert frames[2] == {"type": "agent_text_done"}
    finally:
        convo.stop()


def test_second_submit_while_busy_returns_queued_position(frame_q):
    fake = FakeSDKClient()
    fake.gate = asyncio.Event()
    fake.scripted_messages = [text_msg("Hello!")]
    convo = make_conversation(fake, frame_q)
    convo.start()
    try:
        pos1 = convo.submit_text("hi")
        assert pos1 == 0
        # First turn is blocked on the gate; second submit should queue.
        pos2 = convo.submit_text("second")
        assert pos2 == 1
    finally:
        fake.gate.set()
        convo.stop()


def test_interrupt_while_blocked_ends_turn_and_emits_interrupted(frame_q):
    fake = FakeSDKClient()
    fake.gate = asyncio.Event()
    fake.scripted_messages = [text_msg("Hello!")]
    convo = make_conversation(fake, frame_q)
    convo.start()
    try:
        convo.submit_text("hi")
        # status("thinking") should already be emitted before the gate blocks.
        first = frame_q.get(timeout=5.0)
        assert first == {"type": "status", "text": "thinking"}

        convo.interrupt()
        frame = frame_q.get(timeout=5.0)
        assert frame == {"type": "status", "text": "interrupted"}
    finally:
        fake.gate.set()
        convo.stop()


def test_stop_joins_thread_and_exits_client_context(frame_q):
    fake = FakeSDKClient()
    fake.scripted_messages = [text_msg("Hello!")]
    convo = make_conversation(fake, frame_q)
    convo.start()
    convo.submit_text("hi")
    drain(frame_q, 3)

    convo.stop()

    assert not convo._thread.is_alive()
    assert fake.exit_count == 1


def test_history_returns_user_and_agent_entries_in_order(frame_q):
    fake = FakeSDKClient()
    fake.scripted_messages = [text_msg("Hello!")]
    convo = make_conversation(fake, frame_q)
    convo.start()
    try:
        convo.submit_text("hi")
        drain(frame_q, 3)

        hist = convo.history()
        assert len(hist) == 2
        assert hist[0]["role"] == "user"
        assert hist[0]["text"] == "hi"
        assert hist[1]["role"] == "agent"
        assert hist[1]["text"] == "Hello!"
    finally:
        convo.stop()


def test_emit_failure_does_not_kill_worker(frame_q):
    """An emit() that raises must not permanently break the session."""
    fake = FakeSDKClient()
    fake.scripted_messages = [text_msg("Hello!")]

    def flaky_emit(frame):
        if frame.get("type") == "status" and frame.get("text") == "thinking":
            raise RuntimeError("boom: websocket send failed")
        frame_q.put(frame)

    def factory():
        return fake

    convo = AgentConversation(
        room_id="room1",
        client_factory=factory,
        emit=flaky_emit,
        request_client_tool=fake_request_client_tool,
        model="claude-test",
    )
    convo.start()
    try:
        # First turn's "thinking" status raises inside emit; the worker must
        # swallow it and keep going, still emitting agent_text/agent_text_done.
        convo.submit_text("hi")
        frames = drain(frame_q, 2)  # agent_text, agent_text_done ("thinking" was swallowed)
        assert frames[0] == {"type": "agent_text", "text": "Hello!"}
        assert frames[1] == {"type": "agent_text_done"}

        # A subsequent turn must still complete normally.
        fake.scripted_messages = [text_msg("Again!")]
        convo.submit_text("hi again")
        frames = drain(frame_q, 2)
        assert frames[0] == {"type": "agent_text", "text": "Again!"}
        assert frames[1] == {"type": "agent_text_done"}
    finally:
        convo.stop()

    assert not convo._thread.is_alive()


def test_history_storage_is_capped_at_200(frame_q):
    fake = FakeSDKClient()
    convo = make_conversation(fake, frame_q)
    convo.start()
    try:
        for i in range(250):
            fake.scripted_messages = [text_msg(f"reply {i}")]
            convo.submit_text(f"msg {i}")
            drain(frame_q, 3)

        hist = convo.history()
        assert len(hist) == 200
        # The oldest 50 pairs' worth of entries should be gone; the tail
        # should end with the very last turn's user+agent entries.
        assert hist[-1] == {"role": "agent", "text": "reply 249"}
        assert hist[-2] == {"role": "user", "text": "msg 249"}
        texts = [e["text"] for e in hist]
        assert "msg 0" not in texts
        assert "reply 0" not in texts
    finally:
        convo.stop()
