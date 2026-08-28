"""Tiny manual smoke client for `/ws/agent/{room_id}` (pairs with fake_agent_server.py).

Text mode: connects, sends a text frame, and prints every frame received
until `agent_text_done`, asserting at least one `agent_text` frame arrived.

Voice mode (`--voice`): connects, binds the speaker floor, streams synthetic
PCM16 audio (1.0 s of a 440 Hz tone + 2.2 s of silence to trigger VAD
endpointing), and asserts the full voice pipeline responds: speaker_bound,
transcript, at least one binary "tts" frame, agent_audio_end, and
speaker_released.

Run (with fake_agent_server.py already running, e.g. with --voice):
    .venv\\Scripts\\python tools\\smoke_ws_client.py
    .venv\\Scripts\\python tools\\smoke_ws_client.py "make it darker"
    .venv\\Scripts\\python tools\\smoke_ws_client.py --voice
    .venv\\Scripts\\python tools\\smoke_ws_client.py --voice --port 8765
"""

from __future__ import annotations

import argparse
import asyncio
import json
import struct
import sys

import numpy as np
import websockets

DEFAULT_PORT = 8000


def _url(port: int) -> str:
    return f"ws://127.0.0.1:{port}/ws/agent/smoketest"


def _encode_binary(header: dict, payload: bytes) -> bytes:
    """Mirror ascribe_link.agent_ws.protocol.encode_binary without importing the package."""
    header_json = json.dumps(header).encode("utf-8")
    return struct.pack("<I", len(header_json)) + header_json + payload


def _decode_binary(data: bytes) -> tuple[dict, bytes]:
    header_len = struct.unpack("<I", data[:4])[0]
    header = json.loads(data[4 : 4 + header_len].decode("utf-8"))
    return header, data[4 + header_len :]


def _build_pcm() -> bytes:
    """1.0 s of 440 Hz tone + 2.2 s of silence, PCM16 mono @ 16000 Hz."""
    rate = 16000
    t_tone = np.arange(int(rate * 1.0), dtype=np.float32) / rate
    tone = (0.3 * np.sin(2 * np.pi * 440.0 * t_tone)).astype(np.float32)
    silence = np.zeros(int(rate * 2.2), dtype=np.float32)
    samples = np.concatenate([tone, silence])
    pcm16 = (samples * 32767.0).astype("<i2")
    return pcm16.tobytes()


async def run_voice(port: int) -> None:
    url = _url(port)
    seen: set[str] = set()
    expected = {
        "speaker_bound",
        "transcript",
        "tts_binary",
        "agent_audio_end",
        "speaker_released",
    }

    async def wait_for(sock, timeout: float = 30.0):
        return await asyncio.wait_for(sock.recv(), timeout=timeout)

    async with websockets.connect(url) as ws:
        history_raw = await wait_for(ws)
        history = json.loads(history_raw)
        print("recv:", history)
        assert history["type"] == "history"

        await ws.send(json.dumps({"type": "bind"}))
        print("sent: bind")

        pcm = _build_pcm()
        header = {"kind": "audio", "rate": 16000, "format": "s16le", "channels": 1}
        chunk_size = 8 * 1024
        for offset in range(0, len(pcm), chunk_size):
            chunk = pcm[offset : offset + chunk_size]
            await ws.send(_encode_binary(header, chunk))
        print(f"sent: {len(pcm)} bytes of PCM16 audio (1.0s tone + 2.2s silence) in "
              f"{(len(pcm) + chunk_size - 1) // chunk_size} chunks")

        deadline = asyncio.get_event_loop().time() + 30.0
        while expected - seen:
            remaining = deadline - asyncio.get_event_loop().time()
            if remaining <= 0:
                break
            try:
                raw = await wait_for(ws, timeout=remaining)
            except asyncio.TimeoutError:
                break

            if isinstance(raw, (bytes, bytearray)):
                bin_header, payload = _decode_binary(raw)
                print("recv binary:", bin_header, f"({len(payload)} bytes payload)")
                if bin_header.get("kind") == "tts":
                    seen.add("tts_binary")
                continue

            frame = json.loads(raw)
            print("recv:", frame)
            ftype = frame.get("type")
            if ftype in expected:
                seen.add(ftype)
            if ftype == "agent_text":
                seen.add("agent_text")
            if ftype == "tool_call":
                await ws.send(
                    json.dumps(
                        {
                            "type": "tool_result",
                            "request_id": frame["request_id"],
                            "result": {"ok": True},
                        }
                    )
                )
                print("sent: tool_result ack for", frame["name"])

        missing = expected - seen
        if missing:
            print(f"FAIL: never saw expected frame(s): {sorted(missing)}")
            sys.exit(1)

        print("OK: voice pipeline round-trip complete", sorted(seen))


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("text", nargs="?", default="hello", help="Text to send (text mode only)")
    parser.add_argument("--voice", action="store_true", help="Run the voice smoke instead of text")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="Server port")
    args = parser.parse_args()

    if args.voice:
        await run_voice(args.port)
        return

    text = args.text
    url = _url(args.port)

    async with websockets.connect(url) as ws:
        history = json.loads(await ws.recv())
        print("recv:", history)
        assert history["type"] == "history"

        await ws.send(json.dumps({"type": "text", "text": text}))
        print("sent:", text)

        saw_agent_text = False
        for _ in range(20):
            raw = await ws.recv()
            frame = json.loads(raw)
            print("recv:", frame)
            if frame["type"] == "tool_call":
                await ws.send(
                    json.dumps(
                        {
                            "type": "tool_result",
                            "request_id": frame["request_id"],
                            "result": {"ok": True},
                        }
                    )
                )
                print("sent: tool_result ack for", frame["name"])
            if frame["type"] == "agent_text":
                saw_agent_text = True
            if frame["type"] == "agent_text_done":
                break

        assert saw_agent_text, "never saw an agent_text frame"
        print("OK: agent_text frame received")


if __name__ == "__main__":
    asyncio.run(main())
