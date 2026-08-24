"""
Persistent per-room conversation worker with an injectable SDK client.

`AgentConversation` owns a dedicated worker thread running its own asyncio
event loop. The thread enters `client_factory()` as an async context manager
exactly once and keeps it open for the lifetime of the session, so the
underlying SDK client (or a test fake) is reused across turns instead of
being spun up per-message.

Turns are strictly serial: `submit_text` enqueues a turn onto an internal
`asyncio.Queue` and returns the turn's queue position (0 = about to run / running
now). Every emitted event goes through the caller-supplied `emit` callback,
which the caller (the session manager) guarantees is thread-safe.

This module intentionally does not import `claude_agent_sdk` directly --
message translation is duck-typed against the block shapes documented in
`agent_generator.py:43-115` (`_emit_agent_events`), so tests can run entirely
against fakes.
"""

from __future__ import annotations

import asyncio
import base64
import logging
import threading
from typing import Any, Awaitable, Callable

from ascribe_link.agent_ws import protocol

logger = logging.getLogger(__name__)

_HISTORY_CAP = 200


class _Turn:
    """One queued user turn: text plus an optional attached image."""

    __slots__ = ("text", "image")

    def __init__(self, text: str, image: bytes | None):
        self.text = text
        self.image = image


class AgentConversation:
    """A persistent, room-scoped conversational agent session."""

    def __init__(
        self,
        room_id: str,
        *,
        client_factory: Callable[[], Any],
        emit: Callable[[dict], None],
        request_client_tool: Callable[[str, dict], Awaitable[Any]],
        model: str,
        system_prompt: str | None = None,
    ) -> None:
        self.room_id = room_id
        self._client_factory = client_factory
        self._emit = emit
        self._request_client_tool = request_client_tool
        self.model = model
        self.system_prompt = system_prompt

        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._turn_queue: asyncio.Queue[_Turn] | None = None
        self._pending_image: bytes | None = None
        self._in_flight = 0  # turns submitted but not yet finished (running + waiting)
        self._in_flight_lock = threading.Lock()
        self._history: list[dict] = []
        self._history_lock = threading.Lock()
        self._current_turn_task: asyncio.Task | None = None
        self._stopped = threading.Event()
        self._started = threading.Event()

    # ------------------------------------------------------------------
    # Public API (called from the caller's / manager's thread)
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Spawn the worker thread and enter the SDK client context once."""
        if self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._run_worker_loop, name=f"agent-convo-{self.room_id}", daemon=True
        )
        self._thread.start()
        self._started.wait(timeout=5.0)

    def submit_text(self, text: str) -> int:
        """Queue a user turn. Returns the queue position (0 = running now)."""
        image = self._pending_image
        self._pending_image = None

        with self._in_flight_lock:
            position = self._in_flight
            self._in_flight += 1
        self._append_history({"role": "user", "text": text})

        loop = self._loop
        queue_ = self._turn_queue
        if loop is None or queue_ is None:
            raise RuntimeError("AgentConversation.start() must be called before submit_text()")

        def _enqueue() -> None:
            queue_.put_nowait(_Turn(text=text, image=image))

        loop.call_soon_threadsafe(_enqueue)
        return position

    def attach_image(self, jpeg: bytes) -> None:
        """Stash an image to be attached to the next submitted turn."""
        self._pending_image = jpeg

    def interrupt(self) -> None:
        """Cancel the currently running turn, if any."""
        loop = self._loop
        if loop is None:
            return

        def _cancel() -> None:
            task = self._current_turn_task
            if task is not None and not task.done():
                task.cancel()

        loop.call_soon_threadsafe(_cancel)

    def stop(self) -> None:
        """Gracefully stop: cancel the current turn, exit the client context, join."""
        loop = self._loop
        if loop is None:
            return

        def _request_stop() -> None:
            task = self._current_turn_task
            if task is not None and not task.done():
                task.cancel()
            self._stopped.set()

        loop.call_soon_threadsafe(_request_stop)
        if self._thread is not None:
            self._thread.join(timeout=5.0)

    def history(self) -> list[dict]:
        """Return role/text entries (user + agent), capped to the last 200."""
        with self._history_lock:
            return list(self._history)

    def _append_history(self, entry: dict) -> None:
        """Thread-safe append that trims storage to the last _HISTORY_CAP entries."""
        with self._history_lock:
            self._history.append(entry)
            if len(self._history) > _HISTORY_CAP:
                del self._history[: len(self._history) - _HISTORY_CAP]

    def _safe_emit(self, frame: dict) -> None:
        """Call self._emit, swallowing and logging any exception it raises.

        An emit failure (e.g. the caller's websocket send path erroring out)
        must never kill the worker loop/thread -- a subsequent turn should
        still run to completion.
        """
        try:
            self._emit(frame)
        except Exception:
            logger.exception(
                "emit() raised for frame type '%s' in room %s",
                frame.get("type"),
                self.room_id,
            )

    # ------------------------------------------------------------------
    # Worker thread internals
    # ------------------------------------------------------------------

    def _run_worker_loop(self) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self._loop = loop
        self._turn_queue = asyncio.Queue()
        try:
            loop.run_until_complete(self._main())
        finally:
            loop.close()

    async def _main(self) -> None:
        assert self._turn_queue is not None
        stop_event = asyncio.Event()

        async def _watch_stop() -> None:
            # Poll the threading.Event from within the loop without blocking it.
            while not self._stopped.is_set():
                await asyncio.sleep(0.02)
            stop_event.set()

        watcher = asyncio.ensure_future(_watch_stop())

        async with self._client_factory() as client:
            self._started.set()
            try:
                while True:
                    get_task = asyncio.ensure_future(self._turn_queue.get())
                    stop_wait = asyncio.ensure_future(stop_event.wait())
                    done, pending = await asyncio.wait(
                        {get_task, stop_wait}, return_when=asyncio.FIRST_COMPLETED
                    )
                    if stop_wait in done:
                        get_task.cancel()
                        break
                    stop_wait.cancel()
                    turn = get_task.result()

                    task = asyncio.ensure_future(self._run_turn(client, turn))
                    self._current_turn_task = task
                    try:
                        await task
                    except asyncio.CancelledError:
                        self._safe_emit(protocol.status("interrupted"))
                    except Exception as err:  # pragma: no cover - defensive
                        logger.exception("Turn failed in room %s", self.room_id)
                        self._safe_emit(protocol.error(str(err)))
                    finally:
                        self._current_turn_task = None
                        with self._in_flight_lock:
                            self._in_flight = max(0, self._in_flight - 1)
            finally:
                watcher.cancel()

    async def _run_turn(self, client: Any, turn: _Turn) -> None:
        self._safe_emit(protocol.status("thinking"))

        prompt_blocks = self._build_prompt_blocks(turn)
        await client.query(prompt_blocks)

        agent_text_parts: list[str] = []
        async for msg in client.receive_response():
            for frame_text in self._emit_events(msg):
                if frame_text is not None:
                    agent_text_parts.append(frame_text)

        if agent_text_parts:
            self._append_history({"role": "agent", "text": "".join(agent_text_parts)})

        self._safe_emit(protocol.agent_text_done())

    def _build_prompt_blocks(self, turn: _Turn) -> Any:
        """Shape one turn for `client.query`.

        Text-only turns stay a plain string (the common case). A turn with an
        attached screenshot becomes a content-block list, using the same
        base64 image-block shape `tools.py` returns for `capture_viewport`
        (tools.py:87-98).
        """
        if not turn.image:
            return turn.text

        return [
            {"type": "text", "text": turn.text},
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/jpeg",
                    "data": base64.b64encode(turn.image).decode("ascii"),
                },
            },
        ]

    def _emit_events(self, msg: Any) -> list[str | None]:
        """Translate one SDK message into emit() calls.

        Mirrors the block-shape contract of `agent_generator._emit_agent_events`
        (agent_generator.py:43-115): AssistantMessage-shaped messages carry a
        `.content` list of blocks; blocks with non-empty `.text` are agent
        text, blocks with a `.name` are tool-use blocks.

        Returns the list of text fragments emitted (for history accumulation).
        """
        texts: list[str | None] = []
        content = getattr(msg, "content", None)
        if content is None:
            return texts

        for block in content:
            name = getattr(block, "name", None)
            text = getattr(block, "text", None)
            if name:
                self._safe_emit(protocol.status(f"Using the {name} tool..."))
            elif text:
                self._safe_emit(protocol.agent_text(text))
                texts.append(text)
        return texts
