"""Processing function endpoints — replaces the old MQTT RPC pattern."""

from __future__ import annotations

from typing import Any

from litestar import Controller, get, post
from litestar.exceptions import NotFoundException

from ascribe_link.models import FunctionInfo, MeshResult, ProcessingRequest
from ascribe_link.processing import FunctionRegistry


class ProcessingController(Controller):
    path = "/api/processing"

    @get("/functions")
    async def list_functions(self, function_registry: FunctionRegistry) -> list[FunctionInfo]:
        """List all registered processing functions."""
        return function_registry.list_functions()

    @get("/functions/{name:str}/schema")
    async def get_function_schema(self, function_registry: FunctionRegistry, name: str) -> dict[str, Any]:
        """Get the JSON Schema for a processing function's parameters."""
        schema = function_registry.get_schema(name)
        if schema is None:
            raise NotFoundException(detail=f"Function not found: {name}")
        return schema

    @post("/invoke")
    async def invoke_function(self, function_registry: FunctionRegistry, data: ProcessingRequest) -> MeshResult:
        """Invoke a processing function and return the mesh result."""
        try:
            return function_registry.invoke(data.function_name, data.args, data.kwargs)
        except KeyError:
            raise NotFoundException(detail=f"Function not found: {data.function_name}")
