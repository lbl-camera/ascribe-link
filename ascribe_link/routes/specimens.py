"""Specimen catalog endpoints."""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import mimetypes
import time
from pathlib import Path
from typing import Any

from litestar import Controller, Response, get, post
from litestar.di import Provide
from litestar.exceptions import HTTPException, NotFoundException
from litestar.response import File

from ascribe_link.cache import RoomResultCache
from ascribe_link.envelope import ENVELOPE_MEDIA_TYPE, encode_envelope
from ascribe_link.federation import FederationHub
from ascribe_link.job_registry import Job, JobRegistry
from ascribe_link.models import SpecimenListItem, SpecimenMetadata, SpecimenType
from ascribe_link.processing import FunctionRegistry
from ascribe_link.progress import JobReporter
from ascribe_link.specimen_store import SpecimenStore

logger = logging.getLogger(__name__)

# Placeholder thumbnail for specimens without thumbnails (Ascribe logo)
# Placeholder thumbnail for specimens without thumbnails (Ascribe-Link logo)
_PLACEHOLDER_THUMBNAIL_PATH = Path(__file__).parent.parent / "assets" / "placeholder.png"


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
            return File(
                path=_PLACEHOLDER_THUMBNAIL_PATH,
                content_disposition_type="inline",
                media_type="image/png",
            )

        # Filesystem specimen
        path = specimen_store.thumbnail_path(specimen_id)
        if path is None:
            # Return placeholder if specimen exists but thumbnail is missing
            if specimen_store.get(specimen_id) is not None:
                return File(
                    path=_PLACEHOLDER_THUMBNAIL_PATH,
                    content_disposition_type="inline",
                    media_type="image/png",
                )
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
                return Response(
                    content=encode_envelope(cached_result),
                    media_type=ENVELOPE_MEDIA_TYPE,
                )

            # Invoke the function
            try:
                result = await function_registry.invoke_async(
                    meta.function_name,
                    [],
                    params,
                )
            except KeyError:
                raise NotFoundException(detail=f"Function not found: {meta.function_name}")
            except TypeError as e:
                if "async" in str(e).lower() or "await" in str(e).lower():
                    result = function_registry.invoke(
                        meta.function_name,
                        [],
                        params,
                    )
                else:
                    raise

            # Cache the raw result object (widened in Task 6); then envelope-serve.
            result_cache.put(room_id, meta.function_name, params, result)
            return Response(
                content=encode_envelope(result),
                media_type=ENVELOPE_MEDIA_TYPE,
            )

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
        params: str | None = None,
        room_id: str = "ascribe",
    ) -> Response | File | dict[str, Any]:
        """GET handler for specimen data.

        Static specimens: returns the data file (no query params used).
        Dynamic specimens: pass ``?params=<JSON>&room_id=<id>`` to invoke the
        function with custom inputs sharing the same room cache as the POST
        endpoint. With no ``params`` query, schema defaults are used.
        """
        parsed_params: dict[str, Any] = {}
        if params:
            try:
                parsed = json.loads(params)
            except json.JSONDecodeError as exc:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid 'params' query string (must be JSON): {exc}",
                ) from exc
            if not isinstance(parsed, dict):
                raise HTTPException(
                    status_code=400,
                    detail="'params' query string must encode a JSON object",
                )
            parsed_params = parsed
        return await self._get_data_impl(
            specimen_store=specimen_store,
            specimen_id=specimen_id,
            function_registry=function_registry,
            result_cache=result_cache,
            federation_hub=federation_hub,
            params=parsed_params,
            room_id=room_id,
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

    @post("/{specimen_id:str}/start", status_code=200)
    async def start_job(
        self,
        specimen_store: SpecimenStore,
        specimen_id: str,
        function_registry: FunctionRegistry,
        result_cache: RoomResultCache,
        job_registry: JobRegistry,
        federation_hub: FederationHub | None = None,
        data: dict[str, Any] | None = None,
    ) -> dict[str, str]:
        """Start a dynamic specimen load as a background job.

        Returns `{job_id, status}`. Status is "done" on cache hit; else "running".
        """
        data = data or {}
        params: dict[str, Any] = data.get("params", {}) or {}
        room_id: str = data.get("room_id", "ascribe")

        # Resolve the specimen and ensure it's dynamic.
        if ":" in specimen_id and federation_hub:
            # Federated — proxy to worker (handled in Task 8).
            worker_id, actual_id = specimen_id.split(":", 1)
            return await _proxy_federated_start(
                federation_hub,
                worker_id,
                actual_id,
                params,
                room_id,
                job_registry,
                specimen_id,
            )

        meta = function_registry.get_specimen(specimen_id)
        if meta is None:
            meta = specimen_store.get(specimen_id)
        if meta is None:
            raise NotFoundException(detail=f"Specimen not found: {specimen_id}")
        if not meta.function_name:
            raise HTTPException(
                status_code=400,
                detail=f"Specimen {specimen_id} is static; use GET /data instead",
            )

        # Extract defaults if params not provided.
        if not params and meta.schema:
            params = _extract_schema_defaults(meta.schema)

        job = await job_registry.create(
            specimen_id=specimen_id, params=params, room_id=room_id
        )

        # Cache hit shortcut — no task needed.
        cached = result_cache.get(room_id, meta.function_name, params)
        if cached is not None:
            job.append_message("cache hit")
            job.result = cached
            job.status = "done"
            job.finished_at = time.monotonic()
            return {"job_id": job.id, "status": "done"}

        # Spawn the runner task; register it so DELETE can cancel.
        job.task = asyncio.create_task(
            _run_job(
                job=job,
                function_registry=function_registry,
                result_cache=result_cache,
                func_name=meta.function_name,
            )
        )
        return {"job_id": job.id, "status": "running"}

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


async def _run_job(
    *,
    job: Job,
    function_registry: FunctionRegistry,
    result_cache: RoomResultCache,
    func_name: str,
) -> None:
    """Execute the specimen function, populating the job's result/status.

    The function is run in a separate thread (with its own event loop) so that
    long-running or event-loop-blocking functions (like the AI agent, which
    uses ClaudeSDKClient) don't prevent the main event loop from serving poll
    requests on /api/jobs/{id}/progress.

    The JobReporter writes to the job's message deque from the worker thread;
    this is safe under CPython's GIL (single-appender deque + atomic append).
    """
    reporter = JobReporter(job)
    job.append_message(f"Starting {job.specimen_id}")
    t0 = time.monotonic()
    try:
        result = await asyncio.to_thread(
            _invoke_in_thread,
            function_registry,
            func_name,
            job.params,
            reporter,
        )
        result_cache.put(job.room_id, func_name, job.params, result)
        job.result = result
        job.status = "done"
        job.append_message(f"Finished in {time.monotonic() - t0:.2f}s")
    except asyncio.CancelledError:
        job.status = "error"
        job.error = "cancelled"
        job.append_message("Cancelled")
        raise
    except Exception as e:
        job.status = "error"
        job.error = str(e)
        job.append_message(f"Error: {e}")
    finally:
        job.finished_at = time.monotonic()


def _invoke_in_thread(
    function_registry: FunctionRegistry,
    func_name: str,
    params: dict,
    reporter: JobReporter,
) -> Any:
    """Run the specimen function in a fresh event loop on a worker thread.

    This keeps the main asyncio event loop free for HTTP request handling
    while the (potentially blocking) function runs.
    """
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(
            function_registry.invoke_async(func_name, [], params, reporter=reporter)
        )
    finally:
        loop.close()


async def _proxy_federated_start(
    federation_hub: FederationHub,
    worker_id: str,
    actual_id: str,
    params: dict,
    room_id: str,
    job_registry: JobRegistry,
    original_specimen_id: str,
) -> dict[str, str]:
    """Start a job on a federated worker and proxy via a local relay-side Job."""
    worker_response = await federation_hub.proxy_request(
        worker_id,
        "start_job",
        {"specimen_id": actual_id, "params": params, "room_id": room_id},
    )
    if "error" in worker_response:
        raise HTTPException(
            status_code=502, detail=f"Worker error: {worker_response['error']}"
        )

    worker_job_id = worker_response["job_id"]
    relay_job = await job_registry.create(
        specimen_id=original_specimen_id, params=params, room_id=room_id
    )
    relay_job.federated_to = (worker_id, worker_job_id)
    # Inherit status from the worker — if the worker said "done" (cache hit),
    # we record that locally so /result is served via a direct proxy fetch.
    if worker_response.get("status") == "done":
        relay_job.status = "done"
        relay_job.finished_at = time.monotonic()
    return {"job_id": relay_job.id, "status": relay_job.status}
