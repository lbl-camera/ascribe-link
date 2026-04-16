"""Litestar application factory."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from litestar import Litestar
from litestar.config.cors import CORSConfig
from litestar.di import Provide
from litestar.openapi import OpenAPIConfig
from litestar.openapi.plugins import SwaggerRenderPlugin

from ascribe_link.cache import RoomResultCache
from ascribe_link.federation import FederationHub
from ascribe_link.job_registry import JobRegistry
from ascribe_link.processing import FunctionRegistry
from ascribe_link.routes.federation import FederationController
from ascribe_link.routes.jobs import JobController
from ascribe_link.routes.processing import ProcessingController
from ascribe_link.routes.specimens import SpecimenController
from ascribe_link.specimen_store import SpecimenStore

if TYPE_CHECKING:
    from collections.abc import Callable

logger = logging.getLogger(__name__)


def _default_specimens_dir() -> Path:
    """Return the default specimens directory (repo_root/specimens)."""
    # Go up from ascribe_link/ to repo root, then into specimens/
    module_dir = Path(__file__).parent
    repo_root = module_dir.parent
    return repo_root / "specimens"


def create_app(
    specimens_dir: str | Path | None = None,
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
        If None, defaults to <repo_root>/specimens (relative to this module).
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
    if specimens_dir is None:
        specimens_dir = _default_specimens_dir()
    store = SpecimenStore(Path(specimens_dir))

    # --- Function registry ---
    registry = FunctionRegistry()

    # Register built-in examples
    from ascribe_link.example import sphere_example
    from ascribe_link.parametric import generate_sphere, generate_torus

    registry.register_function(sphere_example, "sphere")

    # Register parametric specimens (fully defined in code)
    registry.register_specimen(
        generate_sphere,
        display_name="Parametric Sphere",
        name="generate_sphere",
        description="Sphere with adjustable radius and resolution",
        return_type="mesh",
        tags=["parametric", "mesh", "dynamic"],
    )
    registry.register_specimen(
        generate_torus,
        display_name="Parametric Torus",
        name="generate_torus",
        description="Torus with adjustable radii and segments",
        return_type="mesh",
        tags=["parametric", "mesh", "dynamic"],
    )

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
            registry.register_specimen(
                agent_func,
                display_name="AI Generate",
                name="ai_generate",
                description="Generate 3D data from natural language prompts using an AI agent",
                return_type="mesh",  # Can also produce volumes, but mesh is default
                tags=["ai", "generative", "dynamic"],
            )
            
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

    # --- Result cache for multiplayer ---
    result_cache = RoomResultCache(ttl_seconds=300.0)  # 5 minute TTL
    logger.info("Room result cache enabled (TTL=300s)")

    # --- Job registry for progress-tracked dynamic loads ---
    job_registry = JobRegistry(ttl_seconds=300.0)
    logger.info("Job registry enabled (TTL=300s)")

    # --- Dependencies ---
    def provide_specimen_store() -> SpecimenStore:
        return store

    def provide_function_registry() -> FunctionRegistry:
        return registry

    def provide_federation_hub() -> FederationHub | None:
        return hub

    def provide_result_cache() -> RoomResultCache:
        return result_cache

    def provide_job_registry() -> JobRegistry:
        return job_registry

    # --- Route handlers ---
    route_handlers = [SpecimenController, ProcessingController, JobController]
    if relay_mode:
        route_handlers.append(FederationController)

    # --- Exception handler for debugging ---
    def log_exception_handler(request, exc: Exception):
        import traceback
        logger.error("Unhandled exception: %s\n%s", exc, traceback.format_exc())
        raise exc  # Re-raise to let Litestar handle the response

    app = Litestar(
        route_handlers=route_handlers,
        dependencies={
            "specimen_store": Provide(provide_specimen_store, sync_to_thread=False),
            "function_registry": Provide(provide_function_registry, sync_to_thread=False),
            "federation_hub": Provide(provide_federation_hub, sync_to_thread=False),
            "result_cache": Provide(provide_result_cache, sync_to_thread=False),
            "job_registry": Provide(provide_job_registry, sync_to_thread=False),
        },
        cors_config=CORSConfig(allow_origins=["*"]),
        exception_handlers={},
        debug=True,
        openapi_config=OpenAPIConfig(
            title="ASCRIBE-Link",
            description="Acribe-link API documentation",
            version="0.0.1",
            render_plugins=[SwaggerRenderPlugin()],
            path="/docs",
        ),
    )

    return app
