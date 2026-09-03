"""Specimen store — manages a directory of curated specimens.

Each specimen is a subdirectory containing:
  specimen.json   — metadata (SpecimenMetadata fields)
  thumbnail.*     — thumbnail image (png/jpg)
  <data_file>     — mesh or volume data file
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from ascribe_link.models import SpecimenMetadata, SpecimenType

if TYPE_CHECKING:
    from ascribe_link.models import VolumeResult


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
        import logging
        logger = logging.getLogger(__name__)

        self._cache.clear()
        logger.info(f"Scanning specimen directory: {self._root.absolute()}")

        if not self._root.exists():
            logger.warning(f"Specimen directory does not exist: {self._root.absolute()}")
            return

        if not self._root.is_dir():
            logger.warning(f"Specimen path is not a directory: {self._root.absolute()}")
            return

        children = list(self._root.iterdir())
        logger.info(f"Found {len(children)} items in specimen directory")

        for child in sorted(children):
            logger.debug(f"Checking: {child.name}")
            if not child.is_dir():
                logger.debug(f"  Skipping (not a directory): {child.name}")
                continue
            meta_path = child / "specimen.json"
            if not meta_path.exists():
                logger.debug(f"  Skipping (no specimen.json): {child.name}")
                continue
            try:
                meta = self._load_metadata(child.name, meta_path)
                self._cache[meta.id] = meta
                logger.info(f"  Loaded specimen: {meta.id} ({meta.display_name})")
            except Exception as exc:  # noqa: BLE001
                logging.getLogger(__name__).warning(
                    "Skipping specimen %s: %s", child.name, exc
                )

    def list(self) -> list[SpecimenMetadata]:
        import logging
        logger = logging.getLogger(__name__)
        result = list(self._cache.values())
        logger.info(f"SpecimenStore.list() returning {len(result)} specimens")
        return result

    def get(self, specimen_id: str) -> SpecimenMetadata | None:
        meta = self._cache.get(specimen_id)
        if meta is None and self._maybe_rescan_for(specimen_id):
            meta = self._cache.get(specimen_id)
        return meta

    def _maybe_rescan_for(self, specimen_id: str) -> bool:
        """Rescan if a bundle dir for `specimen_id` appeared since the last scan.

        Bundles get written at runtime (the conversational agent does this to
        show real-sized data), and until now they were invisible until someone
        hit ``GET /api/specimens/reload`` by hand -- a lookup miss meant the
        client fell back to the "mesh" renderer for a volume. The check is a
        single ``is_file`` on the expected ``specimen.json`` path, so a miss
        for a genuinely unknown id costs no directory walk.
        """
        if not specimen_id or "/" in specimen_id or "\\" in specimen_id or specimen_id in (".", ".."):
            return False
        if not (self._root / specimen_id / "specimen.json").is_file():
            return False
        self.reload()
        return True

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


def load_static_volume(spec_dir: Path, data_file: str) -> VolumeResult:
    """Load a static volume specimen from disk (currently .npy + optional .json sidecar).

    Parameters
    ----------
    spec_dir : Path
        Directory containing the specimen bundle.
    data_file : str
        Filename (within spec_dir) of the volume data, e.g. "data.npy".

    Returns
    -------
    VolumeResult
    """
    from ascribe_link.models import VolumeResult  # avoid import cycle

    data_path = spec_dir / data_file
    if not data_path.exists():
        raise FileNotFoundError(f"volume data missing: {data_path}")
    if data_path.suffix.lower() != ".npy":
        raise ValueError(f"unsupported static volume format: {data_path.suffix}")

    arr = np.load(data_path, mmap_mode="r")
    if arr.ndim != 3:
        raise ValueError(f"static volume must be 3D, got ndim={arr.ndim}")

    sidecar = data_path.with_suffix(".json")
    spacing: list[float] | None = None
    origin: list[float] | None = None
    if sidecar.exists():
        meta = json.loads(sidecar.read_text())
        spacing = meta.get("spacing")
        origin = meta.get("origin")

    return VolumeResult.from_numpy(np.ascontiguousarray(arr), spacing=spacing, origin=origin)
