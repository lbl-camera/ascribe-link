import struct

import pytest

from ascribe_link.agent_ws import protocol as p


def test_valid_text_frame():
    assert p.validate_client_frame({"type": "text", "text": "hi"}) == ""


def test_text_requires_nonempty_text():
    assert "text" in p.validate_client_frame({"type": "text", "text": ""})
    assert "text" in p.validate_client_frame({"type": "text"})


def test_tool_result_requires_request_id_and_result():
    ok = {"type": "tool_result", "request_id": "r1", "result": {"ok": True}}
    assert p.validate_client_frame(ok) == ""
    assert "request_id" in p.validate_client_frame({"type": "tool_result", "result": {}})


def test_reserved_type_rejected_with_reason():
    msg = p.validate_client_frame({"type": "audio"})
    assert "reserved" in msg and "audio" in msg


def test_unknown_type_rejected():
    assert "unknown" in p.validate_client_frame({"type": "bogus"})


def test_builders_have_type_key():
    assert p.tool_call("r1", "load_specimen", {"a": 1}) == {
        "type": "tool_call", "request_id": "r1", "name": "load_specimen", "args": {"a": 1}}
    assert p.status("thinking")["type"] == "status"
    assert p.turn_queued(2) == {"type": "turn_queued", "position": 2}


def test_binary_roundtrip():
    data = p.encode_binary({"kind": "screenshot", "mime": "image/jpeg"}, b"JPEGDATA")
    header, payload = p.decode_binary(data)
    assert header["kind"] == "screenshot"
    assert payload == b"JPEGDATA"
    n = struct.unpack("<I", data[:4])[0]
    assert data[4 + n:] == b"JPEGDATA"


def test_binary_truncated_raises():
    with pytest.raises(ValueError):
        p.decode_binary(b"\x00")


def test_bind_unbind_are_now_valid():
    assert p.validate_client_frame({"type": "bind"}) == ""
    assert p.validate_client_frame({"type": "unbind"}) == ""


def test_text_audio_frame_still_rejected():
    assert "reserved" in p.validate_client_frame({"type": "audio"})


def test_voice_builders():
    assert p.speaker_bound(3) == {"type": "speaker_bound", "client_id": 3}
    assert p.speaker_released() == {"type": "speaker_released"}
    assert p.transcript("hello", 2) == {
        "type": "transcript", "text": "hello", "client_id": 2, "final": True}
    assert p.agent_audio_end() == {"type": "agent_audio_end", "interrupted": False}
    assert p.agent_audio_end(interrupted=True) == {
        "type": "agent_audio_end", "interrupted": True}
    assert p.audio_header(44100) == {
        "kind": "audio", "rate": 44100, "format": "s16le", "channels": 1}
    assert p.tts_header(7)["seq"] == 7 and p.tts_header(7)["rate"] == 24000
