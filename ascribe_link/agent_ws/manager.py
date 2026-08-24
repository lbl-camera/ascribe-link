"""Room-scoped session manager: socket fan-out and client-tool correlation.

`AgentSessionManager` owns, per room, the set of connected WebSockets and the
single persistent `AgentConversation`. It is the glue between the wire
protocol (`protocol.py`), the worker-thread conversation (`session.py`), and
the MCP tool surface (`tools.py`):

- Incoming client frames are validated and routed to the conversation.
- Frames emitted by the conversation (running on its own worker thread) are
  broadcast to every socket in the room.
- Tool calls that must run on the client (`tools.py`'s client-forwarded
  tools) are correlated request/response style, mirroring
  `FederationHub.proxy_request` (`routes/federation.py:147-161`): a
  `request_id` is minted, a future is parked, a `tool_call` frame is
  broadcast, and the future is awaited with a timeout. `capture_viewport` is
  special-cased: its `tool_result` frame is an ack only, and the future
  actually resolves off the *next* screenshot BINARY frame from the
  executor's socket.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from typing import Any

from ascribe_link.agent_ws import protocol
from ascribe_link.agent_ws.session import AgentConversation

logger = logging.getLogger(__name__)

TOOL_CALL_TIMEOUT = 30.0


class _RoomSink:
    """`tools.ConversationSink` implementation bound to one room's manager."""

    def __init__(self, manager: "AgentSessionManager", room_id: str) -> None:
        self.room_id = room_id
        self._manager = manager

    async def request_client_tool(self, name: str, args: dict) -> Any:
        """Marshal the call onto the manager's main loop and await it there.

        This runs on the conversation's *worker* loop (a different thread and
        a different event loop). `AgentSessionManager.request_client_tool`
        creates a future on, and sends sockets from, the main loop, so it must
        never be awaited directly from here -- doing so raises
        "attached to a different loop" against the real SDK.
        """
        loop = self._manager._main_loop
        if loop is None:
            raise RuntimeError("AgentSessionManager has no main loop yet")
        concurrent_future = asyncio.run_coroutine_threadsafe(
            self._manager.request_client_tool(self.room_id, name, args), loop
        )
        return await asyncio.wrap_future(concurrent_future)

    def stage_result(self, result: Any) -> str:
        return self._manager._stage_result(self.room_id, result)

    def get_staged(self, specimen_id: str) -> Any:
        return self._manager.get_staged_result(self.room_id, specimen_id)


class _RoomState:
    """Per-room bookkeeping: sockets, client ids, the conversation, staged results."""

    __slots__ = ("sockets", "conversation", "staged", "client_ids", "next_client_id")

    def __init__(self) -> None:
        self.sockets: list[Any] = []
        self.conversation: AgentConversation | None = None
        self.staged: dict[str, Any] = {}
        # socket -> persistent monotonic client id (stable for the socket's
        # whole lifetime; never reused, even after a client rejoins).
        self.client_ids: dict[Any, int] = {}
        self.next_client_id: int = 0


class AgentSessionManager:
    """Room -> (sockets, AgentConversation) registry with tool correlation."""

    def __init__(
        self,
        *,
        model: str,
        client_factory=None,
    ) -> None:
        self.model = model
        self._client_factory = client_factory

        self._rooms: dict[str, _RoomState] = {}
        self._main_loop: asyncio.AbstractEventLoop | None = None

        # request_id -> (future, tool_name, room_id, executor_client_id)
        self._pending: dict[str, tuple[asyncio.Future, str, str, int | None]] = {}
        # room_id -> FIFO queue of request_ids waiting on the next
        # screenshot binary frame instead of a tool_result frame. A list
        # (not a single slot) so overlapping capture_viewport calls for the
        # same room each get resolved by their own successive binary frame,
        # in request order, instead of clobbering one another.
        self._capture_pending: dict[str, list[str]] = {}

    # ------------------------------------------------------------------
    # Connection lifecycle
    # ------------------------------------------------------------------

    def _room(self, room_id: str) -> _RoomState:
        room = self._rooms.get(room_id)
        if room is None:
            room = _RoomState()
            self._rooms[room_id] = room
        return room

    async def connect(self, room_id: str, socket: Any) -> None:
        """Register `socket` in `room_id`; start the room's conversation if new."""
        if self._main_loop is None:
            self._main_loop = asyncio.get_running_loop()

        room = self._room(room_id)
        room.sockets.append(socket)
        client_id = room.next_client_id
        room.next_client_id += 1
        room.client_ids[socket] = client_id

        if room.conversation is None:
            room.conversation = self._start_conversation(room_id)

        entries = room.conversation.history()
        frame = protocol.history(entries)
        frame["client_id"] = client_id
        await self._send(socket, frame)

    async def disconnect(self, room_id: str, socket: Any) -> None:
        """Drop `socket` from `room_id` and fail any tool calls it owed us."""
        room = self._rooms.get(room_id)
        if room is None:
            return
        if socket in room.sockets:
            room.sockets.remove(socket)
        client_id = room.client_ids.pop(socket, None)
        if client_id is not None:
            self._fail_pending_for_executor(room_id, client_id)

    def _drop_socket(self, room: _RoomState, socket: Any, room_id: str) -> None:
        """Remove a socket that failed to send (broadcast pruning path)."""
        if socket in room.sockets:
            room.sockets.remove(socket)
        client_id = room.client_ids.pop(socket, None)
        if client_id is not None:
            self._fail_pending_for_executor(room_id, client_id)

    def _fail_pending_for_executor(self, room_id: str, client_id: int) -> None:
        """Fail-fast every pending tool future whose executor just left.

        Without this, a tool call assigned to a socket that disconnects burns
        the full `TOOL_CALL_TIMEOUT` before the agent learns it failed.
        """
        for request_id, (future, _name, pending_room, executor) in list(self._pending.items()):
            if pending_room != room_id or executor != client_id:
                continue
            self._pending.pop(request_id, None)
            queue = self._capture_pending.get(room_id)
            if queue is not None and request_id in queue:
                queue.remove(request_id)
                if not queue:
                    self._capture_pending.pop(room_id, None)
            if not future.done():
                future.set_exception(
                    RuntimeError(f"executing client {client_id} disconnected before replying")
                )

    def _start_conversation(self, room_id: str) -> AgentConversation:
        sink = _RoomSink(self, room_id)

        def emit(frame: dict) -> None:
            # Called from the conversation's worker thread -- marshal onto
            # the main loop instead of touching sockets directly.
            loop = self._main_loop
            if loop is None:
                return
            asyncio.run_coroutine_threadsafe(self.broadcast(room_id, frame), loop)

        client_factory = self._client_factory
        if client_factory is None:
            client_factory = self._build_real_client_factory(sink)

        conversation = AgentConversation(
            room_id=room_id,
            client_factory=client_factory,
            emit=emit,
            # The sink is the single marshalling path onto the main loop.
            request_client_tool=sink.request_client_tool,
            model=self.model,
        )
        conversation.start()
        return conversation

    def _build_real_client_factory(self, sink: _RoomSink):
        """Lazily build a real `ClaudeSDKClient` factory (no SDK import at module load)."""

        def factory():
            from claude_agent_sdk import ClaudeAgentOptions, ClaudeSDKClient

            from ascribe_link.agent_ws.tools import build_conversation_tools

            server, allowed_tools = build_conversation_tools(sink)
            options = ClaudeAgentOptions(
                model=self.model,
                mcp_servers={"scene": server},
                allowed_tools=allowed_tools,
            )
            return ClaudeSDKClient(options=options)

        return factory

    # ------------------------------------------------------------------
    # Frame handling
    # ------------------------------------------------------------------

    async def handle_frame(self, room_id: str, socket: Any, frame: dict) -> None:
        """Validate and route one client TEXT frame."""
        err = protocol.validate_client_frame(frame)
        if err:
            await self._send(socket, protocol.error(err))
            return

        frame_type = frame["type"]
        room = self._room(room_id)

        if frame_type == "text":
            try:
                if room.conversation is None:
                    room.conversation = self._start_conversation(room_id)
                position = room.conversation.submit_text(frame["text"])
            except Exception as err:  # noqa: BLE001 - surface, don't drop the socket
                # A failing client_factory (missing SDK, bad key, ...) used to
                # propagate out of the websocket handler and silently kill the
                # connection; tell the client instead.
                logger.exception("Failed to start/submit conversation in room %s", room_id)
                room.conversation = None
                await self._send(socket, protocol.error(f"agent session failed: {err}"))
                return
            if position > 0:
                await self._send(socket, protocol.turn_queued(position))

        elif frame_type == "interrupt":
            if room.conversation is not None:
                room.conversation.interrupt()

        elif frame_type == "tool_result":
            request_id = frame["request_id"]
            result = frame["result"]
            self._resolve_tool_result(request_id, result)

        elif frame_type == "end_conversation":
            if room.conversation is not None:
                room.conversation.stop()
                room.conversation = None

    def _resolve_tool_result(self, request_id: str, result: Any) -> None:
        entry = self._pending.get(request_id)
        if entry is None:
            return
        future, name, _room_id, _executor = entry
        if name == "capture_viewport":
            # Ack only -- the real resolution comes from the next
            # screenshot binary frame.
            return
        self._pending.pop(request_id, None)
        if not future.done():
            future.set_result(result)

    async def handle_binary(self, room_id: str, socket: Any, data: bytes) -> None:
        """Decode and route one client BINARY frame."""
        try:
            header, payload = protocol.decode_binary(data)
        except ValueError as err:
            await self._send(socket, protocol.error(str(err)))
            return

        if header.get("kind") != "screenshot":
            await self._send(socket, protocol.error(f"unknown binary kind '{header.get('kind')}'"))
            return

        queue = self._capture_pending.get(room_id)
        if queue:
            request_id = queue.pop(0)
            if not queue:
                self._capture_pending.pop(room_id, None)
            entry = self._pending.pop(request_id, None)
            if entry is not None:
                future, _name, _room, _executor = entry
                if not future.done():
                    future.set_result(payload)
            return

        room = self._rooms.get(room_id)
        if room is not None and room.conversation is not None:
            room.conversation.attach_image(payload)

    # ------------------------------------------------------------------
    # Broadcast
    # ------------------------------------------------------------------

    async def broadcast(self, room_id: str, frame: dict) -> None:
        """Send `frame` as JSON to every socket in `room_id`, pruning dead ones."""
        room = self._rooms.get(room_id)
        if room is None:
            return
        dead: list[Any] = []
        for socket in list(room.sockets):
            try:
                await socket.send_json(frame)
            except Exception:
                dead.append(socket)
        for socket in dead:
            self._drop_socket(room, socket, room_id)

    async def _send(self, socket: Any, frame: dict) -> None:
        try:
            await socket.send_json(frame)
        except Exception:
            logger.exception("Failed to send frame type '%s' to socket", frame.get("type"))

    # ------------------------------------------------------------------
    # Tool correlation
    # ------------------------------------------------------------------

    async def request_client_tool(self, room_id: str, name: str, args: dict) -> Any:
        """Broadcast a `tool_call` and await the client's reply (or timeout)."""
        if self._main_loop is None:
            self._main_loop = asyncio.get_running_loop()

        room = self._room(room_id)
        # The executor is the *current* oldest socket's persistent client id,
        # computed at call time -- index 0 is not a stable identity once
        # clients come and go.
        executor = room.client_ids.get(room.sockets[0]) if room.sockets else None

        request_id = uuid.uuid4().hex
        future: asyncio.Future = self._main_loop.create_future()
        self._pending[request_id] = (future, name, room_id, executor)
        if name == "capture_viewport":
            self._capture_pending.setdefault(room_id, []).append(request_id)

        frame = protocol.tool_call(request_id, name, args)
        frame["executor"] = executor
        await self.broadcast(room_id, frame)

        try:
            return await asyncio.wait_for(future, TOOL_CALL_TIMEOUT)
        finally:
            self._pending.pop(request_id, None)
            # Only remove this call's own entry from the queue (e.g. on
            # timeout) -- a concurrent capture_viewport call for the same
            # room may still be waiting and must not be disturbed.
            queue = self._capture_pending.get(room_id)
            if queue is not None and request_id in queue:
                queue.remove(request_id)
                if not queue:
                    self._capture_pending.pop(room_id, None)

    # ------------------------------------------------------------------
    # Staged-result store (backs the ConversationSink for `tools.py`)
    # ------------------------------------------------------------------

    def _stage_result(self, room_id: str, result: Any) -> str:
        room = self._room(room_id)
        specimen_id = uuid.uuid4().hex[:12]
        room.staged[specimen_id] = result
        return specimen_id

    def get_staged_result(self, room_id: str, specimen_id: str) -> Any:
        """Return the MeshResult/VolumeResult staged as `specimen_id` in `room_id`.

        Public because the specimen data route consults it before the
        catalog: agent-staged specimens exist only here (they have no
        catalog entry and no params), so `GET /api/specimens/{id}/data`
        must be able to see them.
        """
        room = self._rooms.get(room_id)
        if room is None:
            return None
        return room.staged.get(specimen_id)

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------

    async def shutdown(self) -> None:
        """Stop every room's conversation (app shutdown hook)."""
        for room in self._rooms.values():
            if room.conversation is not None:
                room.conversation.stop()
                room.conversation = None
