"""CLI entrypoint for ascribe-link server."""

from __future__ import annotations

import argparse
import asyncio
import base64
import logging
import mimetypes
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ascribe_link.specimen_store import SpecimenStore


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ascribe-Link specimen & processing server"
    )
    parser.add_argument(
        "--host", default="0.0.0.0", help="Bind host (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--port", type=int, default=8000, help="Bind port (default: 8000)"
    )
    parser.add_argument(
        "--specimens-dir",
        type=Path,
        default=None,
        help="Path to specimens directory (default: ./specimens relative to repo root)",
    )
    parser.add_argument(
        "--reload", action="store_true", help="Enable auto-reload for development"
    )

    # Federation modes
    parser.add_argument(
        "--relay",
        action="store_true",
        help="Enable relay mode: accept worker connections and aggregate specimens",
    )
    parser.add_argument(
        "--worker",
        metavar="URL",
        help="Enable worker mode: connect to relay at URL (e.g., ws://relay.example.com:8000)",
    )
    parser.add_argument(
        "--worker-id",
        default=None,
        help="Worker ID for federation (default: hostname)",
    )

    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose logging"
    )

    # AI agent options
    parser.add_argument(
        "--enable-agent",
        action="store_true",
        help="Enable AI agent-based mesh generation (requires claude-agent-sdk)",
    )
    parser.add_argument(
        "--agent-model",
        default="claude-sonnet-4",
        help="Claude model for agent generation (default: claude-sonnet-4)",
    )
    parser.add_argument(
        "--agent-timeout",
        type=float,
        default=300.0,
        help="Timeout in seconds for agent generation (default: 300)",
    )

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    if args.worker:
        # Worker mode: connect to a relay
        run_worker_mode(args)
    else:
        # Standalone or relay mode
        run_server_mode(args)


def run_server_mode(args: argparse.Namespace) -> None:
    """Run as standalone server or relay."""
    import uvicorn

    from ascribe_link.app import create_app

    mode = "relay" if args.relay else "standalone"
    logging.info(f"Starting Ascribe-Link in {mode} mode")
    
    # If specimens_dir not specified, default to repo root / specimens
    if args.specimens_dir is None:
        import ascribe_link
        module_path = Path(ascribe_link.__file__).parent
        repo_root = module_path.parent
        args.specimens_dir = repo_root / "specimens"
        logging.info(f"Using default specimens directory: {args.specimens_dir}")

    app = create_app(
        specimens_dir=args.specimens_dir,
        relay_mode=args.relay,
        enable_agent=args.enable_agent,
        agent_model=args.agent_model,
        agent_timeout=args.agent_timeout,
    )
    uvicorn.run(app, host=args.host, port=args.port, reload=args.reload)


def run_worker_mode(args: argparse.Namespace) -> None:
    """Run as a worker connected to a relay."""
    import socket

    from ascribe_link.federation import FederationClient
    from ascribe_link.specimen_store import SpecimenStore

    worker_id = args.worker_id or socket.gethostname()
    relay_url = args.worker

    logging.info(f"Starting Ascribe-Link in worker mode")
    logging.info(f"  Worker ID: {worker_id}")
    logging.info(f"  Relay URL: {relay_url}")
    logging.info(f"  Specimens: {args.specimens_dir}")

    # Load local specimens
    store = SpecimenStore(args.specimens_dir)

    async def handle_request(
        request_type: str, payload: dict
    ) -> dict:
        """Handle a proxied request from the relay."""
        specimen_id = payload.get("specimen_id", "")

        if request_type == "get_thumbnail":
            path = store.thumbnail_path(specimen_id)
            if path is None:
                return {"error": f"Thumbnail not found: {specimen_id}"}
            content_type = mimetypes.guess_type(path.name)[0] or "image/png"
            data = path.read_bytes()
            return {
                "data": base64.b64encode(data).decode("utf-8"),
                "content_type": content_type,
            }

        elif request_type == "get_data":
            meta = store.get(specimen_id)
            if meta is None:
                return {"error": f"Specimen not found: {specimen_id}"}
            path = store.data_path(specimen_id)
            if path is None:
                return {"error": f"Data file not found: {specimen_id}"}
            content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
            data = path.read_bytes()
            return {
                "data": base64.b64encode(data).decode("utf-8"),
                "content_type": content_type,
                "filename": meta.data_file,
            }

        else:
            return {"error": f"Unknown request type: {request_type}"}

    async def run() -> None:
        client = FederationClient(
            relay_url=relay_url,
            worker_id=worker_id,
            on_request=handle_request,
        )

        await client.start()

        # Send initial specimen list
        specimens = [
            {
                "id": meta.id,
                "display_name": meta.display_name,
                "description": meta.description,
                "type": meta.type.value,
                "data_file": meta.data_file,
                "thumbnail_file": meta.thumbnail_file,
                "story_text": meta.story_text,
                "tags": meta.tags,
            }
            for meta in store.list()
        ]

        # Wait a moment for connection to establish
        await asyncio.sleep(1)
        await client.update_specimens(specimens)

        logging.info(f"Registered {len(specimens)} specimens with relay")

        # Keep running
        try:
            while True:
                await asyncio.sleep(60)
                # Could periodically re-scan and update specimens here
        except KeyboardInterrupt:
            pass
        finally:
            await client.stop()

    asyncio.run(run())


if __name__ == "__main__":
    main()
