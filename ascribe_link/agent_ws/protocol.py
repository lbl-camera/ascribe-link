"""
Agent conversation wire protocol.

Defines frame schemas and binary framing for agent WebSocket communication.
TEXT frames are JSON objects with required 'type' key.
BINARY frames are <u32 LE header_len><UTF-8 JSON header><raw payload>.
"""

import json
import struct

CLIENT_TYPES = {"text", "tool_result", "screenshot_meta", "interrupt",
                "end_conversation"}
RESERVED_TYPES = {"bind", "unbind", "audio"}

SERVER_TYPES = {"agent_text", "agent_text_done", "tool_call", "status",
                "error", "history", "turn_queued"}
RESERVED_SERVER_TYPES = {"speaker_bound", "speaker_released", "agent_audio",
                         "transcript"}


def validate_client_frame(frame: dict) -> str:
    """
    Validate a client frame dict.

    Returns "" if valid, otherwise a human-readable error message.
    """
    if not isinstance(frame, dict):
        return "frame must be a dict"

    if "type" not in frame:
        return "frame must have a 'type' key"

    frame_type = frame.get("type")
    if not isinstance(frame_type, str):
        return "type must be a string"

    # Check reserved types first
    if frame_type in RESERVED_TYPES:
        return f"type '{frame_type}' is reserved for the voice phase"

    # Check if it's a valid client type
    if frame_type not in CLIENT_TYPES:
        return f"unknown frame type '{frame_type}'"

    # Type-specific validation
    if frame_type == "text":
        text = frame.get("text")
        if not isinstance(text, str) or not text:
            return "text frame requires non-empty 'text' field"

    elif frame_type == "tool_result":
        if "request_id" not in frame:
            return "tool_result frame requires 'request_id' field"
        if "result" not in frame:
            return "tool_result frame requires 'result' field"

    elif frame_type in ("interrupt", "end_conversation"):
        # No extra fields required
        pass

    return ""


def agent_text(text: str) -> dict:
    """Build an agent_text frame."""
    return {"type": "agent_text", "text": text}


def agent_text_done() -> dict:
    """Build an agent_text_done frame."""
    return {"type": "agent_text_done"}


def tool_call(request_id: str, name: str, args: dict) -> dict:
    """Build a tool_call frame."""
    return {
        "type": "tool_call",
        "request_id": request_id,
        "name": name,
        "args": args,
    }


def status(text: str) -> dict:
    """Build a status frame."""
    return {"type": "status", "text": text}


def error(message: str) -> dict:
    """Build an error frame."""
    return {"type": "error", "message": message}


def history(entries: list[dict]) -> dict:
    """Build a history frame."""
    return {"type": "history", "entries": entries}


def turn_queued(position: int) -> dict:
    """Build a turn_queued frame."""
    return {"type": "turn_queued", "position": position}


def encode_binary(header: dict, payload: bytes) -> bytes:
    """
    Encode a binary frame.

    Format: <u32 LE header_len><UTF-8 JSON header><raw payload>
    """
    header_json = json.dumps(header).encode("utf-8")
    header_len = len(header_json)
    return struct.pack("<I", header_len) + header_json + payload


def decode_binary(data: bytes) -> tuple[dict, bytes]:
    """
    Decode a binary frame.

    Returns (header_dict, payload_bytes).
    Raises ValueError on truncation or bad JSON.
    """
    if len(data) < 4:
        raise ValueError("truncated frame: less than 4 bytes")

    header_len = struct.unpack("<I", data[:4])[0]
    if len(data) < 4 + header_len:
        raise ValueError("truncated frame: header incomplete")

    try:
        header_json = data[4 : 4 + header_len].decode("utf-8")
        header = json.loads(header_json)
    except (UnicodeDecodeError, json.JSONDecodeError) as e:
        raise ValueError(f"invalid header JSON: {e}") from e

    payload = data[4 + header_len :]
    return header, payload
