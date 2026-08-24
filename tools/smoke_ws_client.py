"""Tiny manual smoke client for `/ws/agent/{room_id}` (pairs with fake_agent_server.py).

Connects, sends a text frame, and prints every frame received until
`agent_text_done`, asserting at least one `agent_text` frame arrived.

Run (with fake_agent_server.py already running):
    .venv\\Scripts\\python tools\\smoke_ws_client.py
    .venv\\Scripts\\python tools\\smoke_ws_client.py "make it darker"
"""

from __future__ import annotations

import asyncio
import json
import sys

import websockets

URL = "ws://127.0.0.1:8000/ws/agent/smoketest"


async def main() -> None:
    text = sys.argv[1] if len(sys.argv) > 1 else "hello"

    async with websockets.connect(URL) as ws:
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
