"""WebSocket endpoints for federation between Ascribe-Link instances."""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

from litestar import Controller, WebSocket, websocket

from ascribe_link.federation import FederationHub

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class FederationController(Controller):
    """WebSocket endpoint for worker connections."""

    path = "/ws/federation"

    @websocket("/{worker_id:str}")
    async def worker_connection(
        self,
        socket: WebSocket,
        worker_id: str,
        federation_hub: FederationHub,
    ) -> None:
        """Handle a worker WebSocket connection.

        Workers connect here to register their specimens and functions,
        and to receive proxied requests from clients.
        """
        await socket.accept()
        logger.info("Worker connecting: %s", worker_id)

        try:
            await federation_hub.register_worker(worker_id, socket)

            async for message in socket.iter_json():
                await self._handle_worker_message(
                    federation_hub, worker_id, message
                )

        except Exception as e:
            logger.error("Worker %s error: %s", worker_id, e)
        finally:
            await federation_hub.unregister_worker(worker_id)
            logger.info("Worker disconnected: %s", worker_id)

    async def _handle_worker_message(
        self,
        hub: FederationHub,
        worker_id: str,
        message: dict,
    ) -> None:
        """Process a message from a worker."""
        msg_type = message.get("type")

        if msg_type == "specimens":
            # Worker is updating its specimen list
            specimens = message.get("specimens", [])
            await hub.update_worker_specimens(worker_id, specimens)

        elif msg_type == "functions":
            # Worker is updating its function list
            functions = message.get("functions", [])
            await hub.update_worker_functions(worker_id, functions)

        elif msg_type == "response":
            # Worker is responding to a proxied request
            request_id = message.get("request_id")
            payload = message.get("payload", {})
            if request_id:
                hub.store_response(worker_id, request_id, payload)

        elif msg_type == "error":
            # Worker is reporting an error for a proxied request
            request_id = message.get("request_id")
            error = message.get("error", "Unknown error")
            if request_id:
                hub.store_response(worker_id, request_id, {"error": error})

        else:
            logger.warning("Unknown message type from worker %s: %s", worker_id, msg_type)
