"""Litestar application factory."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import TYPE_CHECKING

from litestar import Litestar, Response
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


async def _run_loop_lag_watchdog(
    interval: float = 0.5, threshold: float = 0.25
) -> None:
    """Log a warning whenever the main event loop stalls.

    Sleeps `interval` seconds in a loop and measures how late it wakes up.
    A wake-up more than `threshold` seconds late means something held the
    loop (or the GIL) for that long — anything polling the HTTP API during
    that window got no response. Diagnostic for the ASCRIBE-XR
    "GET /progress ... HTTP 0" errors seen during agent runs.

    Also enables asyncio debug slow-callback reporting so that when the
    stall is caused by a specific callback/coroutine step blocking the
    loop, the asyncio logger names it ("Executing <Task ...> took X.XXXs").
    A stall reported here WITHOUT a matching thread-watchdog stall and
    WITH an asyncio slow-callback line = loop blocked by that callback.
    A stall reported by BOTH watchdogs = process-wide (GIL held, long GC
    pause, or CPU starvation from other processes).
    """
    import time

    from ascribe_link.gen_timing import gt_mark

    loop = asyncio.get_running_loop()
    loop.slow_callback_duration = threshold
    loop.set_debug(True)
    logging.getLogger("asyncio").setLevel(logging.WARNING)

    while True:
        start = time.monotonic()
        await asyncio.sleep(interval)
        lag = time.monotonic() - start - interval
        if lag > threshold:
            logger.warning(
                "Event loop stalled for %.2fs — HTTP requests (including "
                "/progress polls) were blocked during this window",
                lag,
            )
            gt_mark(f"loop watchdog: main event loop stalled {lag:.2f}s")


def _run_thread_lag_watchdog(
    stop_event, interval: float = 0.5, threshold: float = 0.25
) -> None:
    """Plain-OS-thread twin of the loop watchdog (run as a daemon thread).

    Doesn't touch asyncio, so it only stalls when the whole process is
    starved: GIL held by a long C call, a long GC pause, or the OS not
    scheduling this process (e.g. every core saturated by heavy compute).
    Comparing its reports with the loop watchdog's separates "the event
    loop is blocked by a coroutine" from "the process itself is starved".
    """
    import time

    from ascribe_link.gen_timing import gt_mark

    while not stop_event.is_set():
        start = time.monotonic()
        stop_event.wait(interval)
        lag = time.monotonic() - start - interval
        if lag > threshold:
            logger.warning(
                "Process-wide stall of %.2fs (plain thread also starved — "
                "GIL/GC/CPU contention, not an event-loop block)",
                lag,
            )
            gt_mark(f"thread watchdog: process-wide stall {lag:.2f}s")


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
    agent_model: str = "claude-sonnet-4-5",
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
    from ascribe_link.parametric import generate_gaussian_volume, generate_sphere, generate_torus

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
    registry.register_specimen(
        generate_gaussian_volume,
        display_name="Parametric Gaussian Volume",
        name="generate_gaussian_volume",
        description="3D Gaussian blob with adjustable resolution and spread",
        return_type="volume",
        tags=["parametric", "volume", "dynamic"],
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
                return_type=None,  # Agent may return either MeshResult or VolumeResult
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

    # --- Exception handler ---
    # Logs non-HTTP exceptions with a traceback (real runtime errors) and
    # mirrors Litestar's default response for HTTPException (expected control
    # flow: 404, 409, 410, etc.). Returning a Response avoids the re-raise
    # loop that happens when the handler re-enters itself.
    def log_exception_handler(request, exc: Exception) -> Response:
        from litestar.exceptions import HTTPException

        if isinstance(exc, HTTPException):
            return Response(
                content={"status_code": exc.status_code, "detail": exc.detail},
                status_code=exc.status_code,
            )
        import traceback
        logger.error("Unhandled exception: %s\n%s", exc, traceback.format_exc())
        return Response(
            content={"status_code": 500, "detail": "Internal Server Error"},
            status_code=500,
        )

    # --- Lifecycle hooks for background tasks (job TTL sweeper, watchdog) ---
    background_tasks: dict[str, asyncio.Task] = {}

    import threading

    watchdog_stop = threading.Event()

    async def _start_background_tasks(app_: Litestar) -> None:
        background_tasks["sweeper"] = asyncio.create_task(
            job_registry.run_sweeper(interval=30.0)
        )
        background_tasks["loop_watchdog"] = asyncio.create_task(
            _run_loop_lag_watchdog()
        )
        threading.Thread(
            target=_run_thread_lag_watchdog,
            args=(watchdog_stop,),
            name="thread-lag-watchdog",
            daemon=True,
        ).start()

    async def _stop_background_tasks(app_: Litestar) -> None:
        watchdog_stop.set()
        for task in background_tasks.values():
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

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
        exception_handlers={Exception: log_exception_handler},
        debug=True,
        openapi_config=OpenAPIConfig(
            title="ASCRIBE-Link",
            description="Acribe-link API documentation",
            version="0.0.1",
            render_plugins=[SwaggerRenderPlugin()],
            path="/docs",
        ),
        on_startup=[_start_background_tasks],
        on_shutdown=[_stop_background_tasks],
    )

    return app
