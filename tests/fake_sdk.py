"""Fake SDK client helper shared by agent_ws tests.

`FakeSDKClient` stands in for `claude_agent_sdk.ClaudeSDKClient` as an async
context manager with `query()`/`receive_response()`. It imports nothing from
`claude_agent_sdk`, so tests using it run without the real SDK or an API key.

Also provides `FakeSDKFactory`, a `client_factory`-shaped callable usable
directly with `AgentSessionManager(client_factory=...)` / `create_app`.
"""

from __future__ import annotations

from types import SimpleNamespace


class ToolTrigger:
    """Scripted step: have the fake actually call out for a client tool.

    Mimics what the real SDK does internally when the model invokes an
    MCP tool -- the call is dispatched (here, via `client.request_client_tool`,
    which a test wires up to `AgentSessionManager.request_client_tool`) and
    awaited before the fake yields the corresponding tool-use message.
    """

    def __init__(self, name: str, args: dict | None = None):
        self.name = name
        self.args = args or {}


class FakeSDKClient:
    """Async context manager fake standing in for ClaudeSDKClient.

    `scripted_messages` is a list of steps to yield from `receive_response()`
    for the *next* query() call. A plain message object (e.g. from
    `text_msg`/`tool_msg`) is yielded as-is. A `ToolTrigger` instead awaits
    `self.request_client_tool(name, args)` (must be set by the caller) and
    then yields a `tool_msg(name)`. `gate` (an asyncio.Event), if set, is
    awaited before receive_response() yields anything, so tests can control
    interleaving without sleeps.
    """

    def __init__(self):
        self.enter_count = 0
        self.exit_count = 0
        self.queries = []
        self.scripted_messages = []
        self.gate = None
        self.request_client_tool = None
        self.tool_results = []

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
            if isinstance(msg, ToolTrigger):
                result = await self.request_client_tool(msg.name, msg.args)
                self.tool_results.append(result)
                yield tool_msg(msg.name)
            else:
                yield msg


def text_msg(text):
    return SimpleNamespace(content=[SimpleNamespace(text=text, name=None)])


def tool_msg(name):
    return SimpleNamespace(content=[SimpleNamespace(text=None, name=name)])


async def fake_request_client_tool(name, args):
    return {"ok": True}


class FakeSDKFactory:
    """A zero-arg `client_factory`-shaped callable that mints `FakeSDKClient`s.

    Each call creates and records a fresh `FakeSDKClient` (one per room, since
    `AgentSessionManager` calls the factory once per room's conversation).
    Access the most recently created one via `.clients[-1]` to script it
    before/while a test drives the conversation.
    """

    def __init__(self):
        self.clients: list[FakeSDKClient] = []

    def __call__(self) -> FakeSDKClient:
        client = FakeSDKClient()
        self.clients.append(client)
        return client
