"""Integration tests for voice floor control, STT pipeline, TTS fan-out, barge-in.

Exercises the real Litestar wiring (`/ws/agent/{room_id}`) with fake STT/TTS
engines (`tests/fake_voice.py`) and a fake SDK client factory
(`tests/fake_sdk.py`) -- no real faster-whisper/kokoro-onnx/claude_agent_sdk
needed.
"""

from __future__ import annotations

import threading

import numpy as np
import pytest
from litestar.testing import AsyncTestClient

from ascribe_link.agent_ws import protocol
from ascribe_link.app import create_app

from .fake_sdk import FakeSDKFactory, text_msg
from .fake_voice import FakeSTT, FakeTTS


def _drain(ws, limit=40):
    """Receive raw messages (json or binary) until the queue looks drained.

    Returns a list of ("json", frame) / ("binary", (header, payload)) tuples.
    """
    frames = []
    for _ in range(limit):
        msg = ws.receive(timeout=5.0)
        if msg.get("bytes") is not None:
            header, payload = protocol.decode_binary(msg["bytes"])
            frames.append(("binary", (header, payload)))
        else:
            import json

            frames.append(("json", json.loads(msg["text"])))
    return frames


def _drain_until_json(ws, target_type, limit=60):
    """Receive frames until a JSON frame of type `target_type` is seen."""
    frames = []
    for _ in range(limit):
        msg = ws.receive(timeout=5.0)
        if msg.get("bytes") is not None:
            header, payload = protocol.decode_binary(msg["bytes"])
            frames.append(("binary", (header, payload)))
        else:
            import json

            frame = json.loads(msg["text"])
            frames.append(("json", frame))
            if frame["type"] == target_type:
                return frames
    raise AssertionError(f"never saw json frame type {target_type!r}; got {frames}")


def _make_utterance_bytes(rate=48000, tone_s=0.6, silence_s=2.1, freq=440.0):
    """0.6 s tone + 2.1 s silence, PCM16 mono @ `rate`, as a single binary frame."""
    n_tone = int(rate * tone_s)
    t = np.arange(n_tone, dtype=np.float32) / rate
    tone = (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    n_silence = int(rate * silence_s)
    silence = np.zeros(n_silence, dtype=np.float32)
    arr = np.concatenate([tone, silence])
    pcm16 = np.clip(arr * 32767, -32768, 32767).astype(np.int16).tobytes()
    return protocol.encode_binary(protocol.audio_header(rate), pcm16)


@pytest.fixture
def agent_factory():
    return FakeSDKFactory()


@pytest.fixture
async def voice_client(agent_factory):
    app = create_app(
        enable_agent=True,
        agent_client_factory=agent_factory,
        stt_engine=FakeSTT(),
        tts_engine=FakeTTS(),
    )
    async with AsyncTestClient(app=app) as c:
        yield c


@pytest.fixture
async def novoice_client(agent_factory):
    app = create_app(enable_agent=True, agent_client_factory=agent_factory)
    async with AsyncTestClient(app=app) as c:
        yield c


# ----------------------------------------------------------------------
# (a) bind floor control
# ----------------------------------------------------------------------


async def test_bind_grants_floor_and_blocks_second_client(voice_client):
    ws1 = await voice_client.websocket_connect("/ws/agent/voiceroom")
    with ws1:
        h1 = ws1.receive_json()  # history
        ws2 = await voice_client.websocket_connect("/ws/agent/voiceroom")
        with ws2:
            ws2.receive_json()  # history

            ws1.send_json({"type": "bind"})
            frame1 = ws1.receive_json()
            frame2 = ws2.receive_json()
            assert frame1 == {"type": "speaker_bound", "client_id": h1["client_id"]}
            assert frame2 == frame1

            ws2.send_json({"type": "bind"})
            err = ws2.receive_json()
            assert err["type"] == "error"
            assert "held" in err["message"]


# ----------------------------------------------------------------------
# (b) audio from a non-speaker is rejected
# ----------------------------------------------------------------------


async def test_audio_from_non_speaker_is_rejected(voice_client):
    ws1 = await voice_client.websocket_connect("/ws/agent/voiceroom2")
    with ws1:
        ws1.receive_json()  # history
        ws2 = await voice_client.websocket_connect("/ws/agent/voiceroom2")
        with ws2:
            ws2.receive_json()  # history

            ws1.send_json({"type": "bind"})
            ws1.receive_json()  # speaker_bound (self)
            ws2.receive_json()  # speaker_bound (broadcast)

            ws2.send_bytes(_make_utterance_bytes())
            err = ws2.receive_json()
            assert err["type"] == "error"


# ----------------------------------------------------------------------
# (c) full utterance pipeline: bind -> audio -> transcript -> submit_text
# ----------------------------------------------------------------------


async def test_utterance_pipeline_transcribes_and_submits(voice_client, agent_factory):
    ws = await voice_client.websocket_connect("/ws/agent/voiceroom3")
    with ws:
        ws.receive_json()  # history

        fake = agent_factory.clients[-1]
        fake.scripted_messages = [text_msg("agent reply here")]

        ws.send_json({"type": "bind"})
        ws.receive_json()  # speaker_bound

        ws.send_bytes(_make_utterance_bytes())

        frames = _drain_until_json(ws, "agent_text_done")
        json_frames = [f for kind, f in frames if kind == "json"]

        transcript_frames = [f for f in json_frames if f["type"] == "transcript"]
        assert transcript_frames, frames
        assert "FAKE" in transcript_frames[0]["text"]

        released = [f for f in json_frames if f["type"] == "speaker_released"]
        assert released

        # transcript arrives before release, which arrives before the reply.
        types_in_order = [f["type"] for f in json_frames]
        assert types_in_order.index("transcript") < types_in_order.index("speaker_released")
        assert "agent_text" in types_in_order


# ----------------------------------------------------------------------
# (d) TTS fan-out: ordered binary frames, agent_audio_end after agent_text_done
# ----------------------------------------------------------------------


async def test_tts_fanout_ordered_and_ends_after_agent_text_done(voice_client, agent_factory):
    ws = await voice_client.websocket_connect("/ws/agent/voiceroom4")
    with ws:
        ws.receive_json()  # history

        fake = agent_factory.clients[-1]
        fake.scripted_messages = [text_msg("Hello there. How are you?")]

        ws.send_json({"type": "text", "text": "hi"})

        frames = _drain_until_json(ws, "agent_audio_end")

        binary_frames = [payload for kind, payload in frames if kind == "binary"]
        assert len(binary_frames) >= 1
        for header, _payload in binary_frames:
            assert header["kind"] == "tts"
            assert header["rate"] == 24000
        seqs = [header["seq"] for header, _ in binary_frames]
        assert seqs == sorted(seqs)
        assert seqs == list(range(len(seqs)))

        json_types = [f for kind, f in frames if kind == "json"]
        type_order = [f["type"] for f in json_types]
        assert type_order.index("agent_text_done") < type_order.index("agent_audio_end")


# ----------------------------------------------------------------------
# (e) barge-in: bind mid-TTS cancels audio and grants the floor
# ----------------------------------------------------------------------


async def test_bind_during_tts_barges_in(agent_factory):
    gate = threading.Event()

    class GatedTTS:
        def synthesize(self, text):
            gate.wait(timeout=5.0)
            return FakeTTS().synthesize(text)

    app = create_app(
        enable_agent=True,
        agent_client_factory=agent_factory,
        stt_engine=FakeSTT(),
        tts_engine=GatedTTS(),
    )
    try:
        async with AsyncTestClient(app=app) as c:
            ws1 = await c.websocket_connect("/ws/agent/voiceroom5")
            with ws1:
                ws1.receive_json()  # history
                ws2 = await c.websocket_connect("/ws/agent/voiceroom5")
                with ws2:
                    h2 = ws2.receive_json()  # history

                    fake = agent_factory.clients[-1]
                    fake.scripted_messages = [text_msg("Hello there.")]
                    ws1.send_json({"type": "text", "text": "hi"})

                    # Wait for agent_text_done on ws2 -- the sentence is now
                    # queued and the (gated) synthesize() call is blocked.
                    seen = _drain_until_json(ws2, "agent_text_done")
                    assert any(
                        kind == "json" and f["type"] == "agent_text"
                        for kind, f in seen
                    )

                    ws2.send_json({"type": "bind"})
                    frames = _drain_until_json(ws2, "speaker_bound")
                    types_in_order = [f for kind, f in frames if kind == "json"]
                    type_names = [f["type"] for f in types_in_order]
                    assert "agent_audio_end" in type_names
                    assert type_names.index("agent_audio_end") < type_names.index(
                        "speaker_bound"
                    )
                    bound_frame = [f for f in types_in_order if f["type"] == "speaker_bound"][0]
                    assert bound_frame["client_id"] == h2["client_id"]
    finally:
        gate.set()


# ----------------------------------------------------------------------
# (f) voice disabled -> error
# ----------------------------------------------------------------------


async def test_bind_without_engines_returns_not_enabled_error(novoice_client):
    ws = await novoice_client.websocket_connect("/ws/agent/voiceroom6")
    with ws:
        ws.receive_json()  # history
        ws.send_json({"type": "bind"})
        frame = ws.receive_json()
        assert frame["type"] == "error"
        assert "voice is not enabled" in frame["message"]


# ----------------------------------------------------------------------
# (g) screenshot binary regression
# ----------------------------------------------------------------------


async def test_screenshot_binary_still_routes_to_attach_image(voice_client):
    ws = await voice_client.websocket_connect("/ws/agent/voiceroom7")
    with ws:
        ws.receive_json()  # history

        manager = voice_client.app.state.agent_session_manager
        room = manager._rooms["voiceroom7"]

        binary = protocol.encode_binary({"kind": "screenshot"}, b"\xff\xd8jpeg")
        ws.send_bytes(binary)

        # give the manager's task a moment to process
        import asyncio

        for _ in range(50):
            if room.conversation is not None and room.conversation._pending_image:
                break
            await asyncio.sleep(0.02)
        assert room.conversation is not None
        assert room.conversation._pending_image == b"\xff\xd8jpeg"
