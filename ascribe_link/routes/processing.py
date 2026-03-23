"""Processing function endpoints — replaces the old MQTT RPC pattern."""

from __future__ import annotations

import logging
from typing import Any

from litestar import Controller, get, post
from litestar.exceptions import NotFoundException

from ascribe_link.cache import RoomResultCache
from ascribe_link.models import (
    FunctionInfo,
    ProcessingRequest,
    result_to_dict,
)
from ascribe_link.processing import FunctionRegistry

logger = logging.getLogger(__name__)


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
        result_cache: RoomResultCache,
        data: ProcessingRequest,
    ) -> dict[str, Any]:
        """Invoke a processing function and return the result.

        Supports multiplayer caching via room_id. If multiple peers in the same room
        request the same function with the same parameters, the result is cached and
        reused. When a new request comes in for a room, the old cache entry is
        invalidated (since the new specimen replaces the old one for all peers).

        Request body:
        ```json
        {
            "function_name": "generate_sphere",
            "args": [],
            "kwargs": {"radius": 2.0, "resolution": 64},
            "room_id": "ascribe"  // optional, defaults to "ascribe"
        }
        ```

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
        """
        room_id = data.room_id or "ascribe"
        
        # Check cache first
        cached_result = result_cache.get(room_id, data.function_name, data.kwargs)
        if cached_result is not None:
            logger.info(
                "Cache hit: room=%s, function=%s, params=%s",
                room_id,
                data.function_name,
                list(data.kwargs.keys()),
            )
            return cached_result
        
        logger.info(
            "Cache miss: room=%s, function=%s, params=%s - computing result",
            room_id,
            data.function_name,
            list(data.kwargs.keys()),
        )
        
        # Compute result
        try:
            result = await function_registry.invoke_async(
                data.function_name,
                data.args,
                data.kwargs,
            )
            result_dict = result_to_dict(result)
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
                result_dict = result_to_dict(result)
            else:
                raise
        except Exception as e:
            import traceback
            logger.error("Error invoking %s: %s\n%s", data.function_name, e, traceback.format_exc())
            raise
        
        # Cache the result
        result_cache.put(room_id, data.function_name, data.kwargs, result_dict)
        logger.info(
            "Cached result: room=%s, function=%s, vertices=%d",
            room_id,
            data.function_name,
            len(result_dict.get("vertices", [])) if result_dict.get("type") == "mesh" else 0,
        )
        
        return result_dict

    @get("/cache/stats")
    async def cache_stats(self, result_cache: RoomResultCache) -> dict[str, Any]:
        """Get cache statistics for debugging.

        Returns information about cached results per room, including:
        - Total entries
        - TTL configuration
        - Per-room details (function, age, access count)
        """
        return result_cache.stats()

    @post("/cache/clear")
    async def clear_cache(self, result_cache: RoomResultCache) -> dict[str, str]:
        """Clear all cached results.

        Useful for testing or forcing recomputation.
        """
        result_cache.clear()
        return {"status": "cleared"}
