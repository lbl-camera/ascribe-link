"""Processing function endpoints — replaces the old MQTT RPC pattern."""

from __future__ import annotations

from typing import Any

from litestar import Controller, get, post
from litestar.exceptions import NotFoundException

from ascribe_link.models import (
    FunctionInfo,
    ProcessingRequest,
    result_to_dict,
)
from ascribe_link.processing import FunctionRegistry


class ProcessingController(Controller):
    path = "/api/processing"

    @get("/functions")
    async def list_functions(self, function_registry: FunctionRegistry) -> list[FunctionInfo]:
        """List all registered processing functions.

        Each function includes:
        - name: Function identifier
        - schema: JSON Schema for parameters (if available)
        - return_type: Expected return type ("mesh", "volume", "point_cloud", "image")
        """
        return function_registry.list_functions()

    @get("/functions/{name:str}/schema")
    async def get_function_schema(self, function_registry: FunctionRegistry, name: str) -> dict[str, Any]:
        """Get the JSON Schema for a processing function's parameters."""
        schema = function_registry.get_schema(name)
        if schema is None:
            raise NotFoundException(detail=f"Function not found: {name}")
        return schema

    @post("/invoke")
    async def invoke_function(
        self,
        function_registry: FunctionRegistry,
        data: ProcessingRequest,
    ) -> dict[str, Any]:
        """Invoke a processing function and return the result.

        The response always includes a 'type' field indicating the data type:
        - "mesh": vertices, indices, optional normals
        - "volume": shape, dtype, base64-encoded data, optional spacing/origin
        - "point_cloud": points, optional colors/scalars
        - "image": width, height, channels, dtype, base64-encoded data

        Example mesh response:
        ```json
        {
            "type": "mesh",
            "vertices": [x1, y1, z1, x2, y2, z2, ...],
            "indices": [i1, i2, i3, ...]
        }
        ```

        Example volume response:
        ```json
        {
            "type": "volume",
            "shape": [64, 64, 64],
            "dtype": "float32",
            "data": "<base64-encoded bytes>",
            "spacing": [1.0, 1.0, 1.0]
        }
        ```
        """
        try:
            result = await function_registry.invoke_async(
                data.function_name,
                data.args,
                data.kwargs,
            )
            return result_to_dict(result)
        except KeyError:
            raise NotFoundException(detail=f"Function not found: {data.function_name}")
        except TypeError as e:
            # Sync function called - fall back to sync invoke
            if "async" in str(e).lower():
                result = function_registry.invoke(
                    data.function_name,
                    data.args,
                    data.kwargs,
                )
                return result_to_dict(result)
            raise
