"""Litestar application factory."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from litestar import Litestar
from litestar.config.cors import CORSConfig
from litestar.di import Provide

from ascribe_link.federation import FederationHub
from ascribe_link.processing import FunctionRegistry
from ascribe_link.routes.federation import FederationController
from ascribe_link.routes.processing import ProcessingController
from ascribe_link.routes.specimens import SpecimenController
from ascribe_link.specimen_store import SpecimenStore

if TYPE_CHECKING:
    from collections.abc import Callable

logger = logging.getLogger(__name__)


def create_app(
    specimens_dir: str | Path = "./specimens",
    mesh_functions: dict[str, Callable] | None = None,
    relay_mode: bool = False,
    enable_agent: bool = False,
    agent_model: str = "claude-sonnet-4-20250514",
    agent_timeout: float = 300.0,
) -> Litestar:
    """Create and configure the Litestar application.

    Parameters
    ----------
    specimens_dir:
        Path to the directory containing curated specimen bundles.
    mesh_functions:
        Optional dict of {name: callable} processing functions to register.
    relay_mode:
        If True, enable federation hub for accepting worker connections.
        Workers can connect via WebSocket to register their specimens.
    enable_agent:
        If True, register the AI agent-based mesh generation function.
        Requires claude-agent-sdk to be installed.
    agent_model:
        Claude model to use for agent-based generation.
    agent_timeout:
        Timeout in seconds for agent-based generation.
    """
    # --- Specimen store ---
    store = SpecimenStore(Path(specimens_dir))

    # --- Function registry ---
    registry = FunctionRegistry()

    # Register built-in example
    from ascribe_link.example import sphere_example

    registry.register_function(sphere_example, "sphere")

    # Register user-provided functions
    if mesh_functions:
        for name, func in mesh_functions.items():
            registry.register_function(func, name)

    # Register AI agent-based generation
    if enable_agent:
        try:
            from ascribe_link.agent_generator import create_agent_function
            from ascribe_link.sandbox import is_firejail_available

            agent_func = create_agent_function(
                model=agent_model,
                timeout=agent_timeout,
                sandbox=True,  # Always try to sandbox
            )
            registry.register_function(agent_func, "ai_generate")
            
            sandbox_status = "enabled" if is_firejail_available() else "disabled (firejail not found)"
            logger.info("AI agent generation enabled (model=%s, sandbox=%s)", agent_model, sandbox_status)
        except ImportError as e:
            logger.warning(
                "AI agent generation disabled: claude-agent-sdk not installed (%s)", e
            )

    # --- Federation hub (relay mode only) ---
    hub: FederationHub | None = None
    if relay_mode:
        hub = FederationHub()

    # --- Dependencies ---
    def provide_specimen_store() -> SpecimenStore:
        return store

    def provide_function_registry() -> FunctionRegistry:
        return registry

    def provide_federation_hub() -> FederationHub | None:
        return hub

    # --- Route handlers ---
    route_handlers = [SpecimenController, ProcessingController]
    if relay_mode:
        route_handlers.append(FederationController)

    app = Litestar(
        route_handlers=route_handlers,
        dependencies={
            "specimen_store": Provide(provide_specimen_store, sync_to_thread=False),
            "function_registry": Provide(provide_function_registry, sync_to_thread=False),
            "federation_hub": Provide(provide_federation_hub, sync_to_thread=False),
        },
        cors_config=CORSConfig(allow_origins=["*"]),
    )

    return app
