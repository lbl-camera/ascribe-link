"""Specimen store — manages a directory of curated specimens.

Each specimen is a subdirectory containing:
  specimen.json   — metadata (SpecimenMetadata fields)
  thumbnail.*     — thumbnail image (png/jpg)
  <data_file>     — mesh or volume data file
"""

from __future__ import annotations

import json
from pathlib import Path

from ascribe_link.models import SpecimenMetadata, SpecimenType


class SpecimenStore:
    """Read-only store backed by a directory of specimen bundles."""

    def __init__(self, root: Path) -> None:
        self._root = Path(root)
        self._cache: dict[str, SpecimenMetadata] = {}
        self.reload()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def root(self) -> Path:
        return self._root

    def reload(self) -> None:
        """(Re-)scan the specimen directory and rebuild the cache."""
        self._cache.clear()
        if not self._root.is_dir():
            return
        for child in sorted(self._root.iterdir()):
            if not child.is_dir():
                continue
            meta_path = child / "specimen.json"
            if not meta_path.exists():
                continue
            try:
                meta = self._load_metadata(child.name, meta_path)
                self._cache[meta.id] = meta
            except Exception as exc:  # noqa: BLE001
                import logging

                logging.getLogger(__name__).warning(
                    "Skipping specimen %s: %s", child.name, exc
                )

    def list(self) -> list[SpecimenMetadata]:
        return list(self._cache.values())

    def get(self, specimen_id: str) -> SpecimenMetadata | None:
        return self._cache.get(specimen_id)

    def specimen_dir(self, specimen_id: str) -> Path | None:
        if specimen_id in self._cache:
            return self._root / specimen_id
        return None

    def data_path(self, specimen_id: str) -> Path | None:
        meta = self._cache.get(specimen_id)
        if meta and meta.data_file:
            p = self._root / specimen_id / meta.data_file
            if p.is_file():
                return p
        return None

    def thumbnail_path(self, specimen_id: str) -> Path | None:
        meta = self._cache.get(specimen_id)
        if meta and meta.thumbnail_file:
            p = self._root / specimen_id / meta.thumbnail_file
            if p.is_file():
                return p
        # Fallback: look for any thumbnail.* in the directory
        d = self._root / specimen_id
        if d.is_dir():
            for ext in (".png", ".jpg", ".jpeg", ".webp"):
                candidate = d / f"thumbnail{ext}"
                if candidate.is_file():
                    return candidate
        return None

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _load_metadata(dir_name: str, meta_path: Path) -> SpecimenMetadata:
        raw = json.loads(meta_path.read_text())
        return SpecimenMetadata(
            id=raw.get("id", dir_name),
            display_name=raw.get("display_name", dir_name),
            description=raw.get("description", ""),
            type=SpecimenType(raw.get("type", "mesh")),
            data_file=raw.get("data_file", ""),
            thumbnail_file=raw.get("thumbnail_file", ""),
            story_text=raw.get("story_text", []),
            tags=raw.get("tags", []),
            schema=raw.get("schema"),
            function_name=raw.get("function_name"),
        )
