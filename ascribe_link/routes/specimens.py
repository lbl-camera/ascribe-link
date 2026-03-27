"""Specimen catalog endpoints."""

from __future__ import annotations

import base64
import json
import logging
import mimetypes
from pathlib import Path
from typing import Any

from litestar import Controller, Response, get, post
from litestar.di import Provide
from litestar.exceptions import NotFoundException
from litestar.response import File

from ascribe_link.cache import RoomResultCache
from ascribe_link.federation import FederationHub
from ascribe_link.models import SpecimenListItem, SpecimenMetadata, SpecimenType, result_to_dict
from ascribe_link.processing import FunctionRegistry
from ascribe_link.specimen_store import SpecimenStore

logger = logging.getLogger(__name__)

# Placeholder SVG for code-registered specimens without thumbnails
_PLACEHOLDER_THUMBNAIL_SVG = b"""<svg xmlns="http://www.w3.org/2000/svg" width="128" height="128" viewBox="0 0 128 128">
  <rect width="128" height="128" fill="#2a2a3e"/>
  <circle cx="64" cy="64" r="40" fill="none" stroke="#6366f1" stroke-width="3"/>
  <circle cx="64" cy="64" r="20" fill="#6366f1" opacity="0.3"/>
</svg>"""


class SpecimenController(Controller):
    path = "/api/specimens"

    @get("/")
    async def list_specimens(
        self,
        specimen_store: SpecimenStore,
        function_registry: FunctionRegistry,
        federation_hub: FederationHub | None = None,
    ) -> list[SpecimenListItem]:
        """List all curated specimens with names and thumbnail URLs.

        Includes:
        - Filesystem specimens (from specimen_store)
        - Code-registered specimens (from function_registry)
        - Federated specimens from connected workers (in relay mode)
        """
        items = []
        seen_ids: set[str] = set()

        # Code-registered specimens (from FunctionRegistry) take priority
        for meta in function_registry.list_specimens():
            seen_ids.add(meta.id)
            items.append(
                SpecimenListItem(
                    id=meta.id,
                    display_name=meta.display_name,
                    description=meta.description,
                    type=meta.type,
                    thumbnail_url=f"/api/specimens/{meta.id}/thumbnail",
                    tags=meta.tags,
                    is_dynamic=True,  # Always dynamic for registry specimens
                )
            )

        # Filesystem specimens (skip if already registered in code)
        for meta in specimen_store.list():
            if meta.id in seen_ids:
                continue
            seen_ids.add(meta.id)
            items.append(
                SpecimenListItem(
                    id=meta.id,
                    display_name=meta.display_name,
                    description=meta.description,
                    type=meta.type,
                    thumbnail_url=f"/api/specimens/{meta.id}/thumbnail",
                    tags=meta.tags,
                    is_dynamic=meta.function_name is not None,
                )
            )

        # Federated specimens from workers
        if federation_hub:
            for worker_id, specimen in federation_hub.get_all_specimens():
                # Prefix ID with worker to ensure uniqueness and enable routing
                federated_id = f"{worker_id}:{specimen.get('id', '')}"
                items.append(
                    SpecimenListItem(
                        id=federated_id,
                        display_name=specimen.get("display_name", federated_id),
                        description=specimen.get("description", ""),
                        type=SpecimenType(specimen.get("type", "mesh")),
                        thumbnail_url=f"/api/specimens/{federated_id}/thumbnail",
                        tags=specimen.get("tags", []),
                    )
                )

        return items

    @get("/{specimen_id:str}")
    async def get_specimen(
        self,
        specimen_store: SpecimenStore,
        specimen_id: str,
        function_registry: FunctionRegistry,
        federation_hub: FederationHub | None = None,
    ) -> SpecimenMetadata:
        """Get full metadata for a specimen."""
        # Check if this is a federated specimen (worker_id:specimen_id)
        if ":" in specimen_id and federation_hub:
            worker_id, actual_id = specimen_id.split(":", 1)
            worker = federation_hub.get_worker(worker_id)
            if worker:
                for specimen in worker.specimens:
                    if specimen.get("id") == actual_id:
                        return SpecimenMetadata(
                            id=specimen_id,  # Keep the prefixed ID
                            display_name=specimen.get("display_name", actual_id),
                            description=specimen.get("description", ""),
                            type=SpecimenType(specimen.get("type", "mesh")),
                            data_file=specimen.get("data_file", ""),
                            thumbnail_file=specimen.get("thumbnail_file", ""),
                            story_text=specimen.get("story_text", []),
                            tags=specimen.get("tags", []),
                        )
            raise NotFoundException(detail=f"Federated specimen not found: {specimen_id}")

        # Check code-registered specimens first
        registry_meta = function_registry.get_specimen(specimen_id)
        if registry_meta is not None:
            return registry_meta

        # Filesystem specimen
        meta = specimen_store.get(specimen_id)
        if meta is None:
            raise NotFoundException(detail=f"Specimen not found: {specimen_id}")

        # For filesystem dynamic specimens, generate schema from function signature
        if meta.function_name:
            dynamic_schema = function_registry.get_schema(meta.function_name)
            if dynamic_schema:
                return SpecimenMetadata(
                    id=meta.id,
                    display_name=meta.display_name,
                    description=meta.description,
                    type=meta.type,
                    data_file=meta.data_file,
                    thumbnail_file=meta.thumbnail_file,
                    story_text=meta.story_text,
                    tags=meta.tags,
                    schema=dynamic_schema,
                    function_name=meta.function_name,
                )

        return meta

    @get("/{specimen_id:str}/thumbnail")
    async def get_thumbnail(
        self,
        specimen_store: SpecimenStore,
        specimen_id: str,
        function_registry: FunctionRegistry,
        federation_hub: FederationHub | None = None,
    ) -> Response | File:
        """Serve the thumbnail image for a specimen."""
        # Check if this is a federated specimen
        if ":" in specimen_id and federation_hub:
            worker_id, actual_id = specimen_id.split(":", 1)
            try:
                response = await federation_hub.proxy_request(
                    worker_id,
                    "get_thumbnail",
                    {"specimen_id": actual_id},
                )
                if "error" in response:
                    raise NotFoundException(detail=response["error"])

                # Response contains base64-encoded data and content_type
                data = base64.b64decode(response.get("data", ""))
                content_type = response.get("content_type", "image/png")
                return Response(
                    content=data,
                    media_type=content_type,
                )
            except TimeoutError:
                raise NotFoundException(detail=f"Timeout fetching thumbnail from worker: {worker_id}")
            except KeyError:
                raise NotFoundException(detail=f"Worker not found: {worker_id}")

        # Check code-registered specimens first (thumbnail as data URI)
        thumbnail_data = function_registry.get_specimen_thumbnail(specimen_id)
        if thumbnail_data is not None:
            # Parse data URI: "data:image/png;base64,..."
            if thumbnail_data.startswith("data:"):
                # Extract content type and base64 data
                header, encoded = thumbnail_data.split(",", 1)
                content_type = header.split(":")[1].split(";")[0]
                data = base64.b64decode(encoded)
                return Response(content=data, media_type=content_type)
            else:
                raise NotFoundException(detail=f"Invalid thumbnail format for: {specimen_id}")

        # Code-registered specimen without thumbnail — return placeholder
        if function_registry.get_specimen(specimen_id) is not None:
            return Response(
                content=_PLACEHOLDER_THUMBNAIL_SVG,
                media_type="image/svg+xml",
            )

        # Filesystem specimen
        path = specimen_store.thumbnail_path(specimen_id)
        if path is None:
            raise NotFoundException(detail=f"Thumbnail not found for: {specimen_id}")
        content_type = mimetypes.guess_type(path.name)[0] or "image/png"
        return File(path=path, content_disposition_type="inline", media_type=content_type)

    async def _get_data_impl(
        self,
        specimen_store: SpecimenStore,
        specimen_id: str,
        function_registry: FunctionRegistry,
        result_cache: RoomResultCache,
        federation_hub: FederationHub | None = None,
        params: dict[str, Any] | None = None,
        room_id: str = "ascribe",
    ) -> Response | File | dict[str, Any]:
        """Internal implementation for fetching specimen data."""
        if params is None:
            params = {}

        # Check if this is a federated specimen
        if ":" in specimen_id and federation_hub:
            worker_id, actual_id = specimen_id.split(":", 1)
            try:
                response = await federation_hub.proxy_request(
                    worker_id,
                    "get_data",
                    {"specimen_id": actual_id},
                )
                if "error" in response:
                    raise NotFoundException(detail=response["error"])

                # Response contains base64-encoded data, content_type, and filename
                data_bytes = base64.b64decode(response.get("data", ""))
                content_type = response.get("content_type", "application/octet-stream")
                filename = response.get("filename", "data")
                return Response(
                    content=data_bytes,
                    media_type=content_type,
                    headers={
                        "Content-Disposition": f'attachment; filename="{filename}"',
                    },
                )
            except TimeoutError:
                raise NotFoundException(detail=f"Timeout fetching data from worker: {worker_id}")
            except KeyError:
                raise NotFoundException(detail=f"Worker not found: {worker_id}")

        # Check code-registered specimens first
        meta = function_registry.get_specimen(specimen_id)
        
        # Fall back to filesystem specimen
        if meta is None:
            meta = specimen_store.get(specimen_id)
        
        if meta is None:
            raise NotFoundException(detail=f"Specimen not found: {specimen_id}")

        # Dynamic specimen: invoke the function
        if meta.function_name:
            # Extract defaults from schema if params not provided
            if not params and meta.schema:
                params = _extract_schema_defaults(meta.schema)
            
            logger.info(
                "Dynamic specimen %s: invoking %s with params=%s",
                specimen_id,
                meta.function_name,
                params,
            )

            # Check cache first
            cached_result = result_cache.get(room_id, meta.function_name, params)
            if cached_result is not None:
                logger.info("Cache hit for %s/%s", room_id, meta.function_name)
                return cached_result

            # Invoke the function
            try:
                result = await function_registry.invoke_async(
                    meta.function_name,
                    [],
                    params,
                )
                result_dict = result_to_dict(result)
            except KeyError:
                raise NotFoundException(detail=f"Function not found: {meta.function_name}")
            except TypeError as e:
                # Sync function - fall back to sync invoke
                if "async" in str(e).lower() or "await" in str(e).lower():
                    result = function_registry.invoke(
                        meta.function_name,
                        [],
                        params,
                    )
                    result_dict = result_to_dict(result)
                else:
                    raise

            # Cache and return
            result_cache.put(room_id, meta.function_name, params, result_dict)
            return result_dict

        # Static specimen: return the file
        path = specimen_store.data_path(specimen_id)
        if path is None:
            raise NotFoundException(detail=f"Data file not found for: {specimen_id}")
        content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        return File(
            path=path,
            filename=meta.data_file,
            content_disposition_type="attachment",
            media_type=content_type,
        )

    @get("/{specimen_id:str}/data")
    async def get_data_get(
        self,
        specimen_store: SpecimenStore,
        specimen_id: str,
        function_registry: FunctionRegistry,
        result_cache: RoomResultCache,
        federation_hub: FederationHub | None = None,
    ) -> Response | File | dict[str, Any]:
        """GET handler for specimen data (uses default parameters for dynamic specimens)."""
        return await self._get_data_impl(
            specimen_store=specimen_store,
            specimen_id=specimen_id,
            function_registry=function_registry,
            result_cache=result_cache,
            federation_hub=federation_hub,
        )

    @post("/{specimen_id:str}/data")
    async def get_data_post(
        self,
        specimen_store: SpecimenStore,
        specimen_id: str,
        function_registry: FunctionRegistry,
        result_cache: RoomResultCache,
        federation_hub: FederationHub | None = None,
        data: dict[str, Any] | None = None,
    ) -> Response | File | dict[str, Any]:
        """POST handler for specimen data (allows custom parameters for dynamic specimens).

        Request body (optional):
        ```json
        {
            "params": {"radius": 2.0, "resolution": 64},
            "room_id": "ascribe"
        }
        ```
        """
        if data is None:
            data = {}
        return await self._get_data_impl(
            specimen_store=specimen_store,
            specimen_id=specimen_id,
            function_registry=function_registry,
            result_cache=result_cache,
            federation_hub=federation_hub,
            params=data.get("params", {}),
            room_id=data.get("room_id", "ascribe"),
        )

    @get("/reload")
    async def reload_specimens(self, specimen_store: SpecimenStore) -> dict[str, int]:
        """Re-scan the specimens directory."""
        specimen_store.reload()
        return {"count": len(specimen_store.list())}


def _extract_schema_defaults(schema: dict[str, Any]) -> dict[str, Any]:
    """Extract default values from a JSON Schema."""
    defaults = {}
    properties = schema.get("properties", {})
    for key, prop in properties.items():
        if "default" in prop:
            defaults[key] = prop["default"]
    return defaults
