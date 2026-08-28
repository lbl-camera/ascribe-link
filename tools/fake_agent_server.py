"""Manual smoke-test server for the `/ws/agent/{room_id}` conversation.

Runs the REAL Litestar app (`create_app(enable_agent=True, ...)`) with a
scripted fake SDK client factory instead of a real `claude_agent_sdk` client
-- no API key needed. Useful for exercising the full server-side wire
protocol (and, paired with a Godot client pointed at 127.0.0.1:8000, the
whole client/server stack) without burning real model calls.

Scripting rules (applied to the text of each submitted turn):

- Always: reply with a short canned streamed answer that echoes the prompt.
- If the text contains "darker": additionally issue a client-forwarded
  ``set_display_param`` tool call (index=0, name="gamma", value=2.0) before
  the reply.
- If the text contains "look": additionally issue a client-forwarded
  ``capture_viewport`` tool call before the reply.

Tool-call plumbing note: the real SDK invokes MCP tools (and thus
``sink.request_client_tool``) internally when the model decides to call one;
a scripted fake has no model, so it can't reproduce that internal dispatch.
Instead, this module reuses `tests/fake_sdk.py`'s `ToolTrigger` mechanism
(already built for exactly this): a scripted client calls
`self.request_client_tool(name, args)` directly during `receive_response()`,
which we wire (in `ScriptedFactory`) to `AgentSessionManager.request_client_tool`
for the room whose conversation is being started -- mirroring
`tests/test_agent_ws_integration.py::test_tool_call_round_trip_completes_turn`'s
`call_tool` helper. `AgentSessionManager` doesn't pass `room_id` into the
zero-arg `client_factory()` call, so `ScriptedFactory` learns it by wrapping
`manager._start_conversation` right after the app is built (see `main()`)
-- a private-attribute reach-in confined to this test/demo tool, not
production code.

Run: `.venv\\Scripts\\python tools\\fake_agent_server.py`
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tests"))

from fake_sdk import FakeSDKClient, ToolTrigger, text_msg  # noqa: E402
from fake_voice import FakeSTT, FakeTTS  # noqa: E402

from ascribe_link.app import create_app  # noqa: E402


class ScriptedFakeSTT(FakeSTT):
    """Fake STT that always transcribes to "make it darker".

    Drives the existing scripted tool flow (see `ScriptedClient.query` below)
    end-to-end from voice input, regardless of what audio was actually sent
    -- there's no real speech recognition here, just enough to exercise the
    bind -> audio -> transcript -> submit_text -> tool_call -> tts pipeline.
    """

    def transcribe(self, audio_16k) -> str:
        return "make it darker"


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("fake_agent_server")


class ScriptedClient(FakeSDKClient):
    """A `FakeSDKClient` whose scripted response depends on the prompt text."""

    async def query(self, prompt_blocks) -> None:
        await super().query(prompt_blocks)
        text = prompt_blocks if isinstance(prompt_blocks, str) else str(prompt_blocks)
        low = text.lower()

        messages: list = []
        if "darker" in low:
            messages.append(
                ToolTrigger("set_display_param", {"index": 0, "name": "gamma", "value": 2.0})
            )
        if "look" in low:
            messages.append(ToolTrigger("capture_viewport", {}))
        messages.append(text_msg(f"(fake agent) you said: {text!r}"))
        self.scripted_messages = messages


class ScriptedFactory:
    """`client_factory`-shaped callable minting `ScriptedClient`s, room-aware.

    `current_room` is set by `main()` (via a wrapper around
    `AgentSessionManager._start_conversation`) just before the factory is
    invoked for that room, so each minted client can wire its
    `request_client_tool` to that room's manager call.
    """

    def __init__(self) -> None:
        self.clients: list[ScriptedClient] = []
        self.current_room: str | None = None
        self.manager = None  # set by main() once app.state exists
        self.main_loop: asyncio.AbstractEventLoop | None = None

    def __call__(self) -> ScriptedClient:
        room_id = self.current_room
        client = ScriptedClient()

        async def call_tool(name: str, args: dict):
            assert self.manager is not None and self.main_loop is not None
            concurrent_future = asyncio.run_coroutine_threadsafe(
                self.manager.request_client_tool(room_id, name, args), self.main_loop
            )
            return await asyncio.wrap_future(concurrent_future)

        client.request_client_tool = call_tool
        self.clients.append(client)
        logger.info("Minted fake SDK client for room=%s", room_id)
        return client


def build_app(voice: bool = False):
    factory = ScriptedFactory()
    stt_engine = ScriptedFakeSTT() if voice else None
    tts_engine = FakeTTS() if voice else None
    app = create_app(
        enable_agent=True,
        agent_client_factory=factory,
        stt_engine=stt_engine,
        tts_engine=tts_engine,
    )

    manager = app.state.agent_session_manager
    factory.manager = manager

    orig_start = manager._start_conversation

    def patched_start(room_id: str):
        factory.current_room = room_id
        return orig_start(room_id)

    manager._start_conversation = patched_start

    orig_connect = manager.connect

    async def patched_connect(room_id: str, socket) -> None:
        # AgentSessionManager sets _main_loop on the first connect(); grab it
        # for ScriptedFactory once it's available.
        await orig_connect(room_id, socket)
        factory.main_loop = manager._main_loop

    manager.connect = patched_connect

    return app


app = build_app(voice=os.environ.get("FAKE_AGENT_VOICE") == "1")


if __name__ == "__main__":
    import uvicorn

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8000, help="Bind port (default: 8000)")
    parser.add_argument(
        "--voice", action="store_true", help="Wire fake STT/TTS engines onto the app"
    )
    args = parser.parse_args()

    app = build_app(voice=args.voice)

    logger.info(
        "Starting fake agent server on http://127.0.0.1:%d (ws: /ws/agent/{room_id}, voice=%s)",
        args.port,
        args.voice,
    )
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="info")
