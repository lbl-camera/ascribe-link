"""Federation layer for Ascribe-Link relay mode.

Allows multiple Ascribe-Link instances to federate through a relay:
- Relay mode: Accepts WebSocket connections from workers, aggregates specimens
- Worker mode: Connects outbound to a relay, registers local specimens

Architecture:
    NERSC (worker) ──WS──▶ Neutral Host (relay) ◀──HTTP── Quest (client)
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from litestar.connection import WebSocket

logger = logging.getLogger(__name__)


@dataclass
class WorkerInfo:
    """Information about a connected worker."""

    worker_id: str
    websocket: WebSocket
    specimens: list[dict[str, Any]] = field(default_factory=list)
    functions: list[dict[str, Any]] = field(default_factory=list)


class FederationHub:
    """Relay-side hub that manages connected workers.

    Used when running in relay mode to aggregate specimens from workers.
    """

    def __init__(self) -> None:
        self._workers: dict[str, WorkerInfo] = {}
        self._lock = asyncio.Lock()

    async def register_worker(self, worker_id: str, websocket: WebSocket) -> None:
        """Register a new worker connection."""
        async with self._lock:
            self._workers[worker_id] = WorkerInfo(
                worker_id=worker_id,
                websocket=websocket,
            )
        logger.info("Worker registered: %s", worker_id)

    async def unregister_worker(self, worker_id: str) -> None:
        """Remove a worker connection."""
        async with self._lock:
            if worker_id in self._workers:
                del self._workers[worker_id]
        logger.info("Worker unregistered: %s", worker_id)

    async def update_worker_specimens(
        self, worker_id: str, specimens: list[dict[str, Any]]
    ) -> None:
        """Update the specimen list for a worker."""
        async with self._lock:
            if worker_id in self._workers:
                self._workers[worker_id].specimens = specimens
                logger.debug(
                    "Worker %s updated specimens: %d items",
                    worker_id,
                    len(specimens),
                )

    async def update_worker_functions(
        self, worker_id: str, functions: list[dict[str, Any]]
    ) -> None:
        """Update the function list for a worker."""
        async with self._lock:
            if worker_id in self._workers:
                self._workers[worker_id].functions = functions
                logger.debug(
                    "Worker %s updated functions: %d items",
                    worker_id,
                    len(functions),
                )

    def get_all_specimens(self) -> list[tuple[str, dict[str, Any]]]:
        """Get all specimens from all workers.

        Returns list of (worker_id, specimen_dict) tuples.
        """
        result = []
        for worker_id, info in self._workers.items():
            for specimen in info.specimens:
                result.append((worker_id, specimen))
        return result

    def get_all_functions(self) -> list[tuple[str, dict[str, Any]]]:
        """Get all functions from all workers.

        Returns list of (worker_id, function_dict) tuples.
        """
        result = []
        for worker_id, info in self._workers.items():
            for func in info.functions:
                result.append((worker_id, func))
        return result

    def get_worker(self, worker_id: str) -> WorkerInfo | None:
        """Get a worker by ID."""
        return self._workers.get(worker_id)

    def find_specimen_worker(self, specimen_id: str) -> str | None:
        """Find which worker owns a specimen ID."""
        for worker_id, info in self._workers.items():
            for specimen in info.specimens:
                if specimen.get("id") == specimen_id:
                    return worker_id
        return None

    async def proxy_request(
        self,
        worker_id: str,
        request_type: str,
        payload: dict[str, Any],
        timeout: float = 30.0,
    ) -> dict[str, Any]:
        """Send a request to a worker and wait for response.

        Used to proxy data/thumbnail requests through to workers.
        """
        worker = self._workers.get(worker_id)
        if not worker:
            raise KeyError(f"Worker not found: {worker_id}")

        # Generate request ID for correlation
        import uuid

        request_id = str(uuid.uuid4())

        # Send request to worker
        message = {
            "type": request_type,
            "request_id": request_id,
            "payload": payload,
        }
        await worker.websocket.send_json(message)

        # Wait for response (worker will send back with same request_id)
        # This is handled by the WebSocket handler storing responses
        response = await self._wait_for_response(worker_id, request_id, timeout)
        return response

    async def _wait_for_response(
        self, worker_id: str, request_id: str, timeout: float
    ) -> dict[str, Any]:
        """Wait for a response from a worker.

        The WebSocket handler stores responses in _pending_responses.
        """
        # This will be filled in by the WebSocket message handler
        worker = self._workers.get(worker_id)
        if not worker:
            raise KeyError(f"Worker disconnected: {worker_id}")

        # Poll for response (the WS handler will set it)
        if not hasattr(worker, "_pending_responses"):
            worker._pending_responses = {}  # type: ignore

        start = asyncio.get_event_loop().time()
        while True:
            if request_id in worker._pending_responses:  # type: ignore
                return worker._pending_responses.pop(request_id)  # type: ignore

            if asyncio.get_event_loop().time() - start > timeout:
                raise TimeoutError(f"Timeout waiting for worker response: {request_id}")

            await asyncio.sleep(0.05)

    def store_response(self, worker_id: str, request_id: str, response: dict[str, Any]) -> None:
        """Store a response from a worker (called by WS handler)."""
        worker = self._workers.get(worker_id)
        if worker:
            if not hasattr(worker, "_pending_responses"):
                worker._pending_responses = {}  # type: ignore
            worker._pending_responses[request_id] = response  # type: ignore


class FederationClient:
    """Worker-side client that connects to a relay.

    Used when running in worker mode to register with a relay.
    """

    def __init__(
        self,
        relay_url: str,
        worker_id: str,
        on_request: Any = None,  # Callable for handling proxied requests
    ) -> None:
        self.relay_url = relay_url.rstrip("/")
        self.worker_id = worker_id
        self.on_request = on_request
        self._ws: Any = None  # websockets client connection
        self._running = False
        self._task: asyncio.Task | None = None

    async def start(self) -> None:
        """Start the connection to the relay."""
        self._running = True
        self._task = asyncio.create_task(self._run())

    async def stop(self) -> None:
        """Stop the connection."""
        self._running = False
        if self._ws:
            await self._ws.close()
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    async def _run(self) -> None:
        """Main connection loop with reconnection."""
        import websockets

        ws_url = self.relay_url.replace("http://", "ws://").replace("https://", "wss://")
        ws_url = f"{ws_url}/ws/federation/{self.worker_id}"

        while self._running:
            try:
                async with websockets.connect(ws_url) as ws:
                    self._ws = ws
                    logger.info("Connected to relay: %s", ws_url)

                    # Send initial registration
                    await self._send_registration()

                    # Handle incoming messages
                    async for message in ws:
                        await self._handle_message(json.loads(message))

            except Exception as e:
                logger.warning("Relay connection error: %s, reconnecting...", e)
                await asyncio.sleep(5)

    async def _send_registration(self) -> None:
        """Send initial registration with specimens/functions."""
        # This will be called by the app after connecting
        pass

    async def update_specimens(self, specimens: list[dict[str, Any]]) -> None:
        """Send updated specimen list to relay."""
        if self._ws:
            await self._ws.send(
                json.dumps(
                    {
                        "type": "specimens",
                        "specimens": specimens,
                    }
                )
            )

    async def update_functions(self, functions: list[dict[str, Any]]) -> None:
        """Send updated function list to relay."""
        if self._ws:
            await self._ws.send(
                json.dumps(
                    {
                        "type": "functions",
                        "functions": functions,
                    }
                )
            )

    async def _handle_message(self, message: dict[str, Any]) -> None:
        """Handle a message from the relay."""
        msg_type = message.get("type")
        request_id = message.get("request_id")

        if msg_type and self.on_request:
            # Relay is requesting something (e.g., specimen data)
            try:
                response = await self.on_request(msg_type, message.get("payload", {}))
                if self._ws and request_id:
                    await self._ws.send(
                        json.dumps(
                            {
                                "type": "response",
                                "request_id": request_id,
                                "payload": response,
                            }
                        )
                    )
            except Exception as e:
                logger.error("Error handling relay request: %s", e)
                if self._ws and request_id:
                    await self._ws.send(
                        json.dumps(
                            {
                                "type": "error",
                                "request_id": request_id,
                                "error": str(e),
                            }
                        )
                    )
