"""Litestar websocket controller for the persistent agent conversation.

Mirrors `routes/federation.py`'s controller shape, except the wire protocol
here carries both TEXT (JSON) and BINARY (screenshot) frames, so this uses
`socket.receive()` directly instead of `iter_json()` and branches on the raw
ASGI event.
"""

from __future__ import annotations

import json
import logging

from litestar import Controller, WebSocket, websocket
from litestar.exceptions import WebSocketDisconnect

from ascribe_link.agent_ws import protocol
from ascribe_link.agent_ws.manager import AgentSessionManager

logger = logging.getLogger(__name__)


class AgentWSController(Controller):
    """WebSocket endpoint for the room-scoped conversational agent."""

    path = "/ws/agent"

    @websocket("/{room_id:str}")
    async def agent_socket(
        self,
        socket: WebSocket,
        room_id: str,
        agent_session_manager: AgentSessionManager,
    ) -> None:
        await socket.accept()
        logger.info("Agent conversation client connecting: room=%s", room_id)

        await agent_session_manager.connect(room_id, socket)
        try:
            while True:
                event = await socket.receive()

                if event["type"] == "websocket.disconnect":
                    break

                text = event.get("text")
                data = event.get("bytes")

                if text is not None:
                    try:
                        frame = json.loads(text)
                    except json.JSONDecodeError as err:
                        await socket.send_json(protocol.error(f"invalid JSON: {err}"))
                        continue
                    await agent_session_manager.handle_frame(room_id, socket, frame)
                elif data is not None:
                    await agent_session_manager.handle_binary(room_id, socket, data)
        except WebSocketDisconnect:
            pass
        except Exception:
            logger.exception("Agent conversation error: room=%s", room_id)
        finally:
            await agent_session_manager.disconnect(room_id, socket)
            logger.info("Agent conversation client disconnected: room=%s", room_id)
