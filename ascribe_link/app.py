"""Litestar application factory."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from litestar import Litestar
from litestar.config.cors import CORSConfig
from litestar.di import Provide

from ascribe_link.processing import FunctionRegistry
from ascribe_link.routes.processing import ProcessingController
from ascribe_link.routes.specimens import SpecimenController
from ascribe_link.specimen_store import SpecimenStore

if TYPE_CHECKING:
    from collections.abc import Callable


def create_app(
    specimens_dir: str | Path = "./specimens",
    mesh_functions: dict[str, Callable] | None = None,
) -> Litestar:
    """Create and configure the Litestar application.

    Parameters
    ----------
    specimens_dir:
        Path to the directory containing curated specimen bundles.
    mesh_functions:
        Optional dict of {name: callable} processing functions to register.
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

    # --- Dependencies ---
    def provide_specimen_store() -> SpecimenStore:
        return store

    def provide_function_registry() -> FunctionRegistry:
        return registry

    app = Litestar(
        route_handlers=[SpecimenController, ProcessingController],
        dependencies={
            "specimen_store": Provide(provide_specimen_store, sync_to_thread=False),
            "function_registry": Provide(provide_function_registry, sync_to_thread=False),
        },
        cors_config=CORSConfig(allow_origins=["*"]),
    )

    return app
