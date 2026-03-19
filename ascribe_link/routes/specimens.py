"""Specimen catalog endpoints."""

from __future__ import annotations

import base64
import mimetypes
from pathlib import Path
from typing import TYPE_CHECKING

from litestar import Controller, Response, get
from litestar.di import Provide
from litestar.exceptions import NotFoundException
from litestar.response import File

from ascribe_link.models import SpecimenListItem, SpecimenMetadata, SpecimenType
from ascribe_link.specimen_store import SpecimenStore

if TYPE_CHECKING:
    from ascribe_link.federation import FederationHub


class SpecimenController(Controller):
    path = "/api/specimens"

    @get("/")
    async def list_specimens(
        self,
        specimen_store: SpecimenStore,
        federation_hub: "FederationHub | None" = None,
    ) -> list[SpecimenListItem]:
        """List all curated specimens with names and thumbnail URLs.

        In relay mode, also includes specimens from connected workers.
        """
        items = []

        # Local specimens
        for meta in specimen_store.list():
            items.append(
                SpecimenListItem(
                    id=meta.id,
                    display_name=meta.display_name,
                    description=meta.description,
                    type=meta.type,
                    thumbnail_url=f"/api/specimens/{meta.id}/thumbnail",
                    tags=meta.tags,
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
        federation_hub: "FederationHub | None" = None,
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

        # Local specimen
        meta = specimen_store.get(specimen_id)
        if meta is None:
            raise NotFoundException(detail=f"Specimen not found: {specimen_id}")
        return meta

    @get("/{specimen_id:str}/thumbnail")
    async def get_thumbnail(
        self,
        specimen_store: SpecimenStore,
        specimen_id: str,
        federation_hub: "FederationHub | None" = None,
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

        # Local specimen
        path = specimen_store.thumbnail_path(specimen_id)
        if path is None:
            raise NotFoundException(detail=f"Thumbnail not found for: {specimen_id}")
        content_type = mimetypes.guess_type(path.name)[0] or "image/png"
        return File(path=path, content_disposition_type="inline", media_type=content_type)

    @get("/{specimen_id:str}/data")
    async def get_data(
        self,
        specimen_store: SpecimenStore,
        specimen_id: str,
        federation_hub: "FederationHub | None" = None,
    ) -> Response | File:
        """Serve the specimen data file (mesh, volume, etc.)."""
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
                data = base64.b64decode(response.get("data", ""))
                content_type = response.get("content_type", "application/octet-stream")
                filename = response.get("filename", "data")
                return Response(
                    content=data,
                    media_type=content_type,
                    headers={
                        "Content-Disposition": f'attachment; filename="{filename}"',
                    },
                )
            except TimeoutError:
                raise NotFoundException(detail=f"Timeout fetching data from worker: {worker_id}")
            except KeyError:
                raise NotFoundException(detail=f"Worker not found: {worker_id}")

        # Local specimen
        meta = specimen_store.get(specimen_id)
        if meta is None:
            raise NotFoundException(detail=f"Specimen not found: {specimen_id}")
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

    @get("/reload")
    async def reload_specimens(self, specimen_store: SpecimenStore) -> dict[str, int]:
        """Re-scan the specimens directory."""
        specimen_store.reload()
        return {"count": len(specimen_store.list())}
