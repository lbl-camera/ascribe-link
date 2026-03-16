"""Specimen catalog endpoints."""

from __future__ import annotations

import mimetypes
from pathlib import Path

from litestar import Controller, Response, get
from litestar.di import Provide
from litestar.exceptions import NotFoundException
from litestar.response import File

from ascribe_link.models import SpecimenListItem, SpecimenMetadata
from ascribe_link.specimen_store import SpecimenStore


class SpecimenController(Controller):
    path = "/api/specimens"

    @get("/")
    async def list_specimens(self, specimen_store: SpecimenStore) -> list[SpecimenListItem]:
        """List all curated specimens with names and thumbnail URLs."""
        items = []
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
        return items

    @get("/{specimen_id:str}")
    async def get_specimen(self, specimen_store: SpecimenStore, specimen_id: str) -> SpecimenMetadata:
        """Get full metadata for a specimen."""
        meta = specimen_store.get(specimen_id)
        if meta is None:
            raise NotFoundException(detail=f"Specimen not found: {specimen_id}")
        return meta

    @get("/{specimen_id:str}/thumbnail")
    async def get_thumbnail(self, specimen_store: SpecimenStore, specimen_id: str) -> File:
        """Serve the thumbnail image for a specimen."""
        path = specimen_store.thumbnail_path(specimen_id)
        if path is None:
            raise NotFoundException(detail=f"Thumbnail not found for: {specimen_id}")
        content_type = mimetypes.guess_type(path.name)[0] or "image/png"
        return File(path=path, content_disposition_type="inline", media_type=content_type)

    @get("/{specimen_id:str}/data")
    async def get_data(self, specimen_store: SpecimenStore, specimen_id: str) -> File:
        """Serve the specimen data file (mesh, volume, etc.)."""
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
