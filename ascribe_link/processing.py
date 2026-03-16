"""Processing function registry.

Replaces the old MQTT-based function_map with a proper registry that
supports schema generation and invocation via HTTP.
"""

from __future__ import annotations

import inspect
from itertools import chain
from typing import Annotated, Any, Callable, Literal, get_args, get_origin, get_type_hints

import numpy as np

from ascribe_link.models import FunctionInfo, MeshResult


class FunctionRegistry:
    """Registry of mesh-processing functions."""

    def __init__(self) -> None:
        self._functions: dict[str, Callable] = {}

    def register(self, name: str | None = None) -> Callable:
        """Decorator to register a processing function."""

        def wrapper(func: Callable) -> Callable:
            key = name or func.__name__
            self._functions[key] = func
            return func

        return wrapper

    def register_function(self, func: Callable, name: str | None = None) -> None:
        """Imperatively register a processing function."""
        key = name or func.__name__
        self._functions[key] = func

    def list_functions(self) -> list[FunctionInfo]:
        return [
            FunctionInfo(name=name, schema=self.get_schema(name))
            for name in sorted(self._functions)
        ]

    def get(self, name: str) -> Callable | None:
        return self._functions.get(name)

    def invoke(self, name: str, args: list | None = None, kwargs: dict | None = None) -> MeshResult:
        func = self._functions.get(name)
        if func is None:
            raise KeyError(f"Unknown function: {name}")

        result = func(*(args or []), **(kwargs or {}))
        vertices, indices = result

        # Validate mesh
        pts = np.array(vertices)
        if not np.isfinite(pts).all():
            raise ValueError("Mesh contains non-finite vertex values")

        return MeshResult(
            vertices=list(chain.from_iterable(vertices)),
            indices=indices if isinstance(indices, list) else indices.tolist(),
        )

    def get_schema(self, name: str) -> dict[str, Any] | None:
        func = self._functions.get(name)
        if func is None:
            return None
        try:
            return create_schema(func)
        except Exception:
            return None


# ---------------------------------------------------------------------------
# Schema generation (cleaned up from original)
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
