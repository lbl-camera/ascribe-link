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

from ascribe_link.agent_ws import audio, protocol
from ascribe_link.agent_ws.session import AgentConversation
from ascribe_link.agent_ws.stt import STTEngine, UtteranceBuffer
from ascribe_link.agent_ws.tts import SentenceChunker, TTSEngine

logger = logging.getLogger(__name__)

TOOL_CALL_TIMEOUT = 30.0

# Wall-clock ceiling on holding the speaker floor. Endpointing is data-driven
# (`UtteranceBuffer.should_finalize`), so a client that binds and then sends
# nothing -- muted mic, crashed capture thread, dropped audio path -- would
# otherwise hold the floor forever and starve every other client in the room.
BIND_TIMEOUT_S = 90.0

# Sentinel queued onto a room's TTS queue to mark "end of this turn" -- the
# drain task broadcasts agent_audio_end() when it dequeues this instead of a
# sentence string.
_END_TURN = object()


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

    __slots__ = (
        "sockets",
        "conversation",
        "staged",
        "client_ids",
        "next_client_id",
        # Voice floor control / utterance pipeline.
        "speaker",
        "utterance_buffer",
        "bind_timeout_task",
        # TTS fan-out.
        "chunker",
        "tts_queue",
        "tts_task",
        "tts_seq",
        "speaking",
        "barging_in",
    )

    def __init__(self) -> None:
        self.sockets: list[Any] = []
        self.conversation: AgentConversation | None = None
        self.staged: dict[str, Any] = {}
        # socket -> persistent monotonic client id (stable for the socket's
        # whole lifetime; never reused, even after a client rejoins).
        self.client_ids: dict[Any, int] = {}
        self.next_client_id: int = 0

        # The socket currently holding the speaker floor (exclusive per room).
        self.speaker: Any = None
        self.utterance_buffer: UtteranceBuffer | None = None
        # Watchdog releasing the floor BIND_TIMEOUT_S after it was granted if
        # nothing ever finalized the utterance.
        self.bind_timeout_task: asyncio.Task | None = None

        # Sentence-level chunker feeding the per-room TTS drain task.
        self.chunker: SentenceChunker | None = None
        self.tts_queue: asyncio.Queue | None = None
        self.tts_task: asyncio.Task | None = None
        self.tts_seq: int = 0
        # True while the agent's TTS reply for the current turn is still
        # in flight (sentences queued/synthesizing, agent_audio_end not
        # yet sent) -- drives the barge-in decision on `bind`.
        self.speaking: bool = False
        # True for the duration of a barge-in's cancel+interrupt sequence:
        # `_handle_text_delta`/`_finish_tts_turn` must no-op while this is
        # set, otherwise a delta from the turn being interrupted (delivered
        # from the worker thread before `conversation.interrupt()` has
        # actually run) can spawn a fresh TTS task that broadcasts audio
        # AFTER the barge-in's own agent_audio_end.
        self.barging_in: bool = False


class AgentSessionManager:
    """Room -> (sockets, AgentConversation) registry with tool correlation."""

    def __init__(
        self,
        *,
        model: str,
        client_factory=None,
        stt: STTEngine | None = None,
        tts: TTSEngine | None = None,
    ) -> None:
        self.model = model
        self._client_factory = client_factory
        self.stt = stt
        self.tts = tts

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
        # Remove the socket from bookkeeping BEFORE finalizing: finalization
        # broadcasts (transcript/speaker_released), and if this socket is
        # already gone from room.sockets those broadcasts can't fail against
        # it and reenter this same drop path (see _drop_socket below).
        was_speaker = room.speaker is socket
        if socket in room.sockets:
            room.sockets.remove(socket)
        client_id = room.client_ids.pop(socket, None)
        if was_speaker:
            # The speaker's disconnect behaves like unbind: finalize a
            # non-trivial buffered utterance, otherwise just release the floor.
            # `client_id` is passed explicitly because it has already been
            # popped above -- otherwise the transcript broadcast would carry
            # client_id null.
            await self._finalize_or_release(room, room_id, socket, client_id)
        if client_id is not None:
            self._fail_pending_for_executor(room_id, client_id)

    async def _drop_socket(self, room: _RoomState, socket: Any, room_id: str) -> None:
        """Remove a socket that failed to send (broadcast pruning path)."""
        was_speaker = room.speaker is socket
        if socket in room.sockets:
            room.sockets.remove(socket)
        client_id = room.client_ids.pop(socket, None)
        if was_speaker:
            await self._finalize_or_release(room, room_id, socket, client_id)
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

            async def _handle() -> None:
                await self.broadcast(room_id, frame)
                frame_type = frame.get("type")
                if frame_type == "agent_text_done":
                    await self._finish_tts_turn(room_id)
                elif frame_type == "status" and frame.get("text") == "interrupted":
                    # A cancelled turn never reaches agent_text_done, so the
                    # TTS turn would otherwise never be torn down. Idempotent:
                    # the interrupt/barge-in paths usually got here first.
                    await self._cleanup_after_interrupt(room_id)

            asyncio.run_coroutine_threadsafe(_handle(), loop)

        def on_text_delta(text: str) -> None:
            # Also called from the conversation's worker thread.
            loop = self._main_loop
            if loop is None:
                return
            asyncio.run_coroutine_threadsafe(self._handle_text_delta(room_id, text), loop)

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
            on_text_delta=on_text_delta if self.tts is not None else None,
        )
        conversation.start()
        return conversation

    def _build_real_client_factory(self, sink: _RoomSink):
        """Lazily build a real `ClaudeSDKClient` factory (no SDK import at module load)."""

        def factory():
            from claude_agent_sdk import ClaudeAgentOptions, ClaudeSDKClient

            from ascribe_link.agent_ws.tools import build_conversation_tools

            server, allowed_tools, _sdk_tools = build_conversation_tools(sink)
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
            # Same teardown as barge-in: without it a cancelled turn never
            # emits agent_text_done (the session emits status "interrupted"
            # instead), so `_finish_tts_turn` never runs -- already-queued
            # sentences keep synthesizing and broadcasting, and `speaking`
            # stays True forever.
            await self._interrupt_turn(room, room_id)

        elif frame_type == "tool_result":
            request_id = frame["request_id"]
            result = frame["result"]
            self._resolve_tool_result(request_id, result)

        elif frame_type == "end_conversation":
            if room.conversation is not None:
                room.conversation.stop()
                room.conversation = None

        elif frame_type == "bind":
            if self.stt is None or self.tts is None:
                await self._send(socket, protocol.error("voice is not enabled on this server"))
                return
            if room.speaker is not None and room.speaker is not socket:
                await self._send(socket, protocol.error("speaker slot is held"))
                return
            if room.speaking:
                # Barge-in: cancel TTS + clear its queue FIRST, then
                # interrupt the running turn, then tell clients the audio
                # stopped, and only then grant the floor.
                await self._interrupt_turn(room, room_id)
            room.speaker = socket
            room.utterance_buffer = UtteranceBuffer()
            self._arm_bind_timeout(room, room_id, socket)
            client_id = room.client_ids.get(socket)
            await self.broadcast(room_id, protocol.speaker_bound(client_id))

        elif frame_type == "unbind":
            if self.stt is None or self.tts is None:
                await self._send(socket, protocol.error("voice is not enabled on this server"))
                return
            if room.speaker is socket:
                await self._finalize_or_release(room, room_id, socket)

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

        kind = header.get("kind")

        if kind == "audio":
            await self._handle_audio_binary(room_id, socket, header, payload)
            return

        if kind != "screenshot":
            await self._send(socket, protocol.error(f"unknown binary kind '{kind}'"))
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
    # Voice: floor control + utterance pipeline
    # ------------------------------------------------------------------

    async def _handle_audio_binary(
        self, room_id: str, socket: Any, header: dict, payload: bytes
    ) -> None:
        if self.stt is None or self.tts is None:
            await self._send(socket, protocol.error("voice is not enabled on this server"))
            return

        room = self._room(room_id)
        if room.speaker is not socket:
            # Silently drop. This is the normal race after a silence-triggered
            # finalize releases the floor: the client keeps streaming until
            # `speaker_released` reaches it, and every in-flight chunk would
            # otherwise raise an error banner in every panel.
            logger.debug("Dropping audio from non-speaker socket in room %s", room_id)
            return

        if room.utterance_buffer is None:
            room.utterance_buffer = UtteranceBuffer()
        room.utterance_buffer.add(payload, header.get("rate", 48000))

        if room.utterance_buffer.should_finalize():
            await self._finalize_utterance(room, room_id, socket)

    async def _finalize_utterance(
        self, room: _RoomState, room_id: str, socket: Any, client_id: int | None = None
    ) -> None:
        """Transcribe the buffered utterance, release the floor, submit the turn.

        `client_id` may be passed explicitly by callers that have already
        removed the socket from `room.client_ids` (the disconnect paths), so
        the transcript broadcast still carries a real id instead of null.
        """
        buffer = room.utterance_buffer
        room.utterance_buffer = None
        if client_id is None:
            client_id = room.client_ids.get(socket)
        self._cancel_bind_timeout(room)

        # Release the floor BEFORE any broadcast below. A broadcast can fail
        # against this very socket if it's already gone (the speaker's
        # disconnect path) and trigger `_drop_socket`, which must not see
        # this socket as still the current speaker -- that would reenter
        # `_finalize_or_release`/`_finalize_utterance` a second time and
        # double-submit the turn.
        room.speaker = None

        if buffer is None or buffer.duration_s <= 0:
            # Nothing was actually recorded -- treat as silence without
            # bothering the STT engine with an empty array.
            text = ""
        else:
            audio_16k = buffer.take()
            text = await asyncio.to_thread(self.stt.transcribe, audio_16k)
            text = (text or "").strip()

        if not text:
            await self._send(socket, protocol.status("(silence)"))
            await self.broadcast(room_id, protocol.speaker_released())
            return

        await self.broadcast(room_id, protocol.transcript(text, client_id))
        await self.broadcast(room_id, protocol.speaker_released())

        try:
            if room.conversation is None:
                room.conversation = self._start_conversation(room_id)
            position = room.conversation.submit_text(text)
        except Exception as err:  # noqa: BLE001 - surface, don't drop the socket
            logger.exception("Failed to start/submit conversation in room %s", room_id)
            room.conversation = None
            await self._send(socket, protocol.error(f"agent session failed: {err}"))
            return
        if position > 0:
            await self._send(socket, protocol.turn_queued(position))

    async def _release_speaker(self, room: _RoomState, room_id: str, socket: Any) -> None:
        if room.speaker is not socket:
            return
        room.speaker = None
        room.utterance_buffer = None
        self._cancel_bind_timeout(room)
        await self.broadcast(room_id, protocol.speaker_released())

    async def _finalize_or_release(
        self, room: _RoomState, room_id: str, socket: Any, client_id: int | None = None
    ) -> None:
        """`unbind` (or the speaker's disconnect): finalize a non-trivial buffer, else just release."""
        buffer = room.utterance_buffer
        if buffer is not None and buffer.duration_s >= 0.5:
            await self._finalize_utterance(room, room_id, socket, client_id)
        else:
            await self._release_speaker(room, room_id, socket)

    # ------------------------------------------------------------------
    # Voice: bind watchdog
    # ------------------------------------------------------------------

    def _arm_bind_timeout(self, room: _RoomState, room_id: str, socket: Any) -> None:
        """(Re)start the wall-clock watchdog for the floor just granted."""
        self._cancel_bind_timeout(room)
        room.bind_timeout_task = asyncio.ensure_future(
            self._bind_timeout(room, room_id, socket)
        )

    def _cancel_bind_timeout(self, room: _RoomState) -> None:
        task = room.bind_timeout_task
        room.bind_timeout_task = None
        if task is not None and not task.done():
            task.cancel()

    async def _bind_timeout(self, room: _RoomState, room_id: str, socket: Any) -> None:
        try:
            await asyncio.sleep(BIND_TIMEOUT_S)
        except asyncio.CancelledError:
            return
        if room.speaker is not socket:
            return
        logger.info("Speaker floor timed out after %.1fs in room %s", BIND_TIMEOUT_S, room_id)
        room.bind_timeout_task = None
        room.speaker = None
        room.utterance_buffer = None
        await self._send(socket, protocol.status("(silence)"))
        await self.broadcast(room_id, protocol.speaker_released())

    # ------------------------------------------------------------------
    # Voice: TTS fan-out
    # ------------------------------------------------------------------

    async def _interrupt_turn(self, room: _RoomState, room_id: str) -> None:
        """Cancel the in-flight TTS turn and the running conversation turn.

        `barging_in` is set synchronously (no await before it) so any delta
        the worker thread delivers while we're awaiting cancellation below
        (the turn isn't actually interrupted yet) is dropped by
        `_handle_text_delta`/`_finish_tts_turn` instead of spawning a fresh
        TTS task that would outlive this agent_audio_end.
        """
        room.barging_in = True
        try:
            await self._cancel_tts(room, room_id)
            if room.conversation is not None:
                room.conversation.interrupt()
        finally:
            room.barging_in = False
        await self.broadcast(room_id, protocol.agent_audio_end())

    async def _cleanup_after_interrupt(self, room_id: str) -> None:
        """Idempotent TTS teardown for a turn that ended via cancellation."""
        room = self._rooms.get(room_id)
        if room is None:
            return
        if room.barging_in:
            # The interrupting path owns the teardown; don't race it.
            return
        if room.tts_task is None and not room.speaking:
            return
        await self._cancel_tts(room, room_id)
        await self.broadcast(room_id, protocol.agent_audio_end())

    async def _handle_text_delta(self, room_id: str, text: str) -> None:
        if self.tts is None:
            return
        room = self._room(room_id)
        if room.barging_in:
            # A delta from the turn currently being interrupted, delivered
            # from the worker thread before conversation.interrupt() has
            # actually run -- drop it instead of spawning a fresh TTS task.
            return
        # Any delta -- even one that doesn't complete a sentence yet -- means
        # the agent is (about to start) speaking, so `bind` must barge in
        # rather than let a later flush talk over the new speaker.
        room.speaking = True
        if room.chunker is None:
            room.chunker = SentenceChunker()
        sentences = room.chunker.feed(text)
        if not sentences:
            return
        self._ensure_tts_task(room, room_id)
        for sentence in sentences:
            room.tts_queue.put_nowait(sentence)

    async def _finish_tts_turn(self, room_id: str) -> None:
        """Called when the conversation emits agent_text_done.

        Flushes the chunker's trailing fragment, queues it, and queues the
        end-of-turn sentinel so the drain task broadcasts agent_audio_end
        only after every sentence for this turn has been synthesized.
        """
        if self.tts is None:
            return
        room = self._room(room_id)
        if room.barging_in:
            # The turn is being barged into right now -- `_cancel_tts` owns
            # tearing down the queue/task; don't race it by re-queueing.
            return
        remainder = room.chunker.flush() if room.chunker is not None else ""
        self._ensure_tts_task(room, room_id)
        if remainder:
            room.speaking = True
            room.tts_queue.put_nowait(remainder)
        room.tts_queue.put_nowait(_END_TURN)

    def _ensure_tts_task(self, room: _RoomState, room_id: str) -> None:
        if room.tts_task is not None and not room.tts_task.done():
            return
        room.tts_queue = asyncio.Queue()
        room.tts_seq = 0
        room.tts_task = asyncio.ensure_future(self._drain_tts(room, room_id, room.tts_queue))

    async def _cancel_tts(self, room: _RoomState, room_id: str) -> None:
        """Barge-in: cancel the drain task and drop its queue."""
        task = room.tts_task
        room.tts_task = None
        if task is not None and not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            except Exception:  # noqa: BLE001 - the task's own error is irrelevant here
                logger.exception("TTS drain task raised during cancellation in room %s", room_id)
        room.tts_queue = None
        room.tts_seq = 0
        room.speaking = False
        if room.chunker is not None:
            # Reset so stale sentence fragments don't leak into the next turn.
            room.chunker = SentenceChunker()

    async def _drain_tts(
        self, room: _RoomState, room_id: str, queue: asyncio.Queue
    ) -> None:
        while True:
            item = await queue.get()
            if item is _END_TURN:
                room.speaking = False
                room.tts_seq = 0
                await self.broadcast(room_id, protocol.agent_audio_end())
                continue
            try:
                pcm = await asyncio.to_thread(self.tts.synthesize, item)
            except Exception:  # noqa: BLE001 - one bad sentence must not kill the drain loop
                logger.exception("TTS synthesis failed in room %s", room_id)
                continue
            seq = room.tts_seq
            room.tts_seq += 1
            payload = protocol.encode_binary(
                protocol.tts_header(seq), audio.float32_to_pcm16(pcm)
            )
            await self._broadcast_binary(room_id, payload)

    async def _broadcast_binary(self, room_id: str, data: bytes) -> None:
        room = self._rooms.get(room_id)
        if room is None:
            return
        dead: list[Any] = []
        for socket in list(room.sockets):
            try:
                await socket.send_bytes(data)
            except Exception:
                dead.append(socket)
        for socket in dead:
            await self._drop_socket(room, socket, room_id)

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
            await self._drop_socket(room, socket, room_id)

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
            self._cancel_bind_timeout(room)
            if room.conversation is not None:
                room.conversation.stop()
                room.conversation = None
