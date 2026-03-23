"""Processing function registry.

Replaces the old MQTT-based function_map with a proper registry that
supports schema generation and invocation via HTTP.

Functions can return various data types:
- Mesh: (vertices, indices) tuple or MeshResult
- Volume: VolumeResult or (array, spacing, origin) tuple
- PointCloud: PointCloudResult or (points,) tuple
- Image: ImageResult or 2D/3D numpy array

The return type is detected automatically or can be hinted with type annotations.
"""

from __future__ import annotations

import asyncio
import inspect
from itertools import chain
from typing import Annotated, Any, Callable, Literal, get_args, get_origin, get_type_hints

import numpy as np

from ascribe_link.models import (
    FunctionInfo,
    ImageResult,
    MeshResult,
    PointCloudResult,
    ProcessingResult,
    VolumeResult,
    result_to_dict,
)


class FunctionRegistry:
    """Registry of processing functions."""

    def __init__(self) -> None:
        self._functions: dict[str, Callable] = {}
        self._return_types: dict[str, str | None] = {}

    def register(self, name: str | None = None, return_type: str | None = None) -> Callable:
        """Decorator to register a processing function.

        Parameters
        ----------
        name : str, optional
            Function name (defaults to function.__name__)
        return_type : str, optional
            Hint for return type: "mesh", "volume", "point_cloud", "image"
        """

        def wrapper(func: Callable) -> Callable:
            key = name or func.__name__
            self._functions[key] = func
            self._return_types[key] = return_type
            return func

        return wrapper

    def register_function(
        self,
        func: Callable,
        name: str | None = None,
        return_type: str | None = None,
    ) -> None:
        """Imperatively register a processing function."""
        key = name or func.__name__
        self._functions[key] = func
        self._return_types[key] = return_type

    def list_functions(self) -> list[FunctionInfo]:
        return [
            FunctionInfo(
                name=name,
                schema=self.get_schema(name),
                return_type=self._return_types.get(name),
            )
            for name in sorted(self._functions)
        ]

    def get(self, name: str) -> Callable | None:
        return self._functions.get(name)

    async def invoke_async(
        self,
        name: str,
        args: list | None = None,
        kwargs: dict | None = None,
    ) -> ProcessingResult:
        """Invoke a function (async-aware) and return typed result."""
        func = self._functions.get(name)
        if func is None:
            raise KeyError(f"Unknown function: {name}")

        # Coerce kwargs to match function signature types (e.g., float -> int)
        kwargs = self._coerce_kwargs(func, kwargs or {})

        # Call function (handle both sync and async)
        if asyncio.iscoroutinefunction(func):
            result = await func(*(args or []), **kwargs)
        else:
            result = func(*(args or []), **kwargs)

        # Convert raw result to typed result
        return self._convert_result(result, self._return_types.get(name))

    def invoke(
        self,
        name: str,
        args: list | None = None,
        kwargs: dict | None = None,
    ) -> ProcessingResult:
        """Invoke a function (sync only) and return typed result.

        For async functions, use invoke_async instead.
        """
        func = self._functions.get(name)
        if func is None:
            raise KeyError(f"Unknown function: {name}")

        if asyncio.iscoroutinefunction(func):
            raise TypeError(
                f"Function '{name}' is async. Use invoke_async() instead."
            )

        # Coerce kwargs to match function signature types (e.g., float -> int)
        kwargs = self._coerce_kwargs(func, kwargs or {})

        result = func(*(args or []), **kwargs)
        return self._convert_result(result, self._return_types.get(name))

    def _coerce_kwargs(self, func: Callable, kwargs: dict[str, Any]) -> dict[str, Any]:
        """Coerce kwargs to match function signature types.
        
        Handles common cases like float -> int when the signature expects int.
        """
        try:
            hints = get_type_hints(func)
        except Exception:
            return kwargs
        
        coerced = {}
        for key, value in kwargs.items():
            expected_type = hints.get(key)
            if expected_type is int and isinstance(value, float):
                coerced[key] = int(value)
            elif expected_type is bool and isinstance(value, (int, float)):
                coerced[key] = bool(value)
            else:
                coerced[key] = value
        return coerced

    def _convert_result(
        self,
        result: Any,
        type_hint: str | None,
    ) -> ProcessingResult:
        """Convert a raw function result to a typed ProcessingResult."""

        # Already a ProcessingResult
        if isinstance(result, (MeshResult, VolumeResult, PointCloudResult, ImageResult)):
            return result

        # Dict with 'type' field
        if isinstance(result, dict) and "type" in result:
            from ascribe_link.models import result_from_dict
            return result_from_dict(result)

        # Tuple: detect type based on structure and hints
        if isinstance(result, tuple):
            return self._convert_tuple_result(result, type_hint)

        # NumPy array: could be volume or image
        if isinstance(result, np.ndarray):
            return self._convert_array_result(result, type_hint)

        raise ValueError(
            f"Cannot convert result of type {type(result).__name__}. "
            "Expected tuple, ndarray, or ProcessingResult."
        )

    def _convert_tuple_result(
        self,
        result: tuple,
        type_hint: str | None,
    ) -> ProcessingResult:
        """Convert a tuple result based on structure."""

        # Mesh: (vertices, indices) or (vertices, indices, normals)
        if type_hint == "mesh" or (
            type_hint is None
            and len(result) >= 2
            and self._looks_like_vertices(result[0])
            and self._looks_like_indices(result[1])
        ):
            vertices, indices = result[0], result[1]
            normals = result[2] if len(result) > 2 else None

            # Flatten if nested lists
            if vertices and isinstance(vertices[0], (list, tuple, np.ndarray)):
                vertices = list(chain.from_iterable(vertices))
            if isinstance(vertices, np.ndarray):
                vertices = vertices.flatten().tolist()
            if isinstance(indices, np.ndarray):
                indices = indices.flatten().tolist()
            if normals is not None:
                if isinstance(normals[0], (list, tuple, np.ndarray)):
                    normals = list(chain.from_iterable(normals))
                if isinstance(normals, np.ndarray):
                    normals = normals.flatten().tolist()

            # Validate
            pts = np.array(vertices).reshape(-1, 3)
            if not np.isfinite(pts).all():
                raise ValueError("Mesh contains non-finite vertex values")

            return MeshResult(
                vertices=vertices,
                indices=indices,
                normals=normals,
            )

        # Volume: (array,) or (array, spacing) or (array, spacing, origin)
        if type_hint == "volume" or (
            type_hint is None
            and len(result) >= 1
            and isinstance(result[0], np.ndarray)
            and result[0].ndim == 3
        ):
            arr = result[0]
            spacing = result[1] if len(result) > 1 else None
            origin = result[2] if len(result) > 2 else None
            if spacing is not None:
                spacing = list(spacing)
            if origin is not None:
                origin = list(origin)
            return VolumeResult.from_numpy(arr, spacing=spacing, origin=origin)

        # Point cloud: (points,) or (points, colors) or (points, colors, scalars)
        if type_hint == "point_cloud":
            points = result[0]
            colors = result[1] if len(result) > 1 else None
            scalars = result[2] if len(result) > 2 else None
            return PointCloudResult.from_numpy(points, colors=colors, scalars=scalars)

        raise ValueError(
            f"Cannot interpret tuple of length {len(result)} as a known result type. "
            f"Hint: {type_hint}"
        )

    def _convert_array_result(
        self,
        arr: np.ndarray,
        type_hint: str | None,
    ) -> ProcessingResult:
        """Convert a numpy array result."""

        if type_hint == "volume" or (type_hint is None and arr.ndim == 3):
            return VolumeResult.from_numpy(arr)

        if type_hint == "image" or (type_hint is None and arr.ndim in (2, 3) and arr.ndim != 3):
            # 2D or 3D with channels
            return ImageResult.from_numpy(arr)

        if type_hint == "point_cloud":
            # Assume Nx3 point array
            return PointCloudResult.from_numpy(arr)

        # Default: treat 3D as volume, 2D as image
        if arr.ndim == 3:
            return VolumeResult.from_numpy(arr)
        elif arr.ndim == 2:
            return ImageResult.from_numpy(arr)

        raise ValueError(
            f"Cannot interpret {arr.ndim}D array as a known result type. "
            f"Hint: {type_hint}"
        )

    def _looks_like_vertices(self, data: Any) -> bool:
        """Heuristic: does this look like vertex data?"""
        if isinstance(data, np.ndarray):
            return data.ndim in (1, 2)
        if isinstance(data, (list, tuple)) and len(data) > 0:
            first = data[0]
            # Nested list of [x, y, z]
            if isinstance(first, (list, tuple)) and len(first) == 3:
                return True
            # Flat list of floats
            if isinstance(first, (int, float)):
                return True
        return False

    def _looks_like_indices(self, data: Any) -> bool:
        """Heuristic: does this look like index data?"""
        if isinstance(data, np.ndarray):
            return data.dtype.kind in ("i", "u")  # integer types
        if isinstance(data, (list, tuple)) and len(data) > 0:
            return isinstance(data[0], int)
        return False

    def get_schema(self, name: str) -> dict[str, Any] | None:
        func = self._functions.get(name)
        if func is None:
            return None
        try:
            return create_schema(func)
        except Exception:
            return None


# ---------------------------------------------------------------------------
# Schema generation
# ---------------------------------------------------------------------------


def _type_to_schema(annotation: Any) -> dict[str, Any]:
    if annotation is inspect.Parameter.empty:
        return {}

    origin = get_origin(annotation)
    args = get_args(annotation)

    # Annotated[T, ...metadata...]
    if origin is Annotated:
        base_type = args[0]
        return _type_to_schema(base_type)

    # Literal["a", "b"]
    if origin is Literal:
        values = list(args)
        schema: dict[str, Any] = {"enum": values}
        if values and all(isinstance(v, str) for v in values):
            schema["type"] = "string"
        elif values and all(isinstance(v, bool) for v in values):
            schema["type"] = "boolean"
        elif values and all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in values):
            schema["type"] = "number"
        return schema

    if annotation is str:
        return {"type": "string"}
    if annotation in (int, float):
        return {"type": "number"}
    if annotation is bool:
        return {"type": "boolean"}
    if annotation is type(None):
        return {"type": "null"}

    return {}


def create_schema(func: Callable) -> dict[str, Any]:
    """Generate a JSON Schema for a function's parameters."""
    resolved = get_type_hints(func, include_extras=True)
    sig = inspect.signature(func)

    properties: dict[str, Any] = {}
    required: list[str] = []

    for param_name, param in sig.parameters.items():
        annotation = resolved.get(param_name, param.annotation)
        prop_schema = _type_to_schema(annotation)
        if param.default is not inspect.Parameter.empty:
            prop_schema["default"] = param.default
        else:
            required.append(param_name)
        properties[param_name] = prop_schema

    schema: dict[str, Any] = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": func.__name__,
        "type": "object",
        "properties": properties,
    }
    if required:
        schema["required"] = required
    return schema
