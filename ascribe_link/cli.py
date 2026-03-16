"""CLI entrypoint for ascribe-link server."""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Ascribe-Link specimen & processing server")
    parser.add_argument("--host", default="0.0.0.0", help="Bind host (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="Bind port (default: 8000)")
    parser.add_argument(
        "--specimens-dir",
        type=Path,
        default=Path("./specimens"),
        help="Path to specimens directory (default: ./specimens)",
    )
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")
    args = parser.parse_args()

    import uvicorn

    from ascribe_link.app import create_app

    app = create_app(specimens_dir=args.specimens_dir)
    uvicorn.run(app, host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()
