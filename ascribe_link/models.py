"""Data models for Ascribe-Link."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


class SpecimenType(str, Enum):
    MESH = "mesh"
    VOLUME = "volume"


@dataclass
class SpecimenMetadata:
    """Metadata for a curated specimen."""

    id: str
    display_name: str
    description: str = ""
    type: SpecimenType = SpecimenType.MESH
    data_file: str = ""  # filename within the specimen directory
    thumbnail_file: str = ""  # filename within the specimen directory
    story_text: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)


@dataclass
class SpecimenListItem:
    """Lightweight specimen info for catalog listings."""

    id: str
    display_name: str
    description: str
    type: SpecimenType
    thumbnail_url: str
    tags: list[str] = field(default_factory=list)


@dataclass
class MeshResult:
    """Result of a mesh processing function."""

    vertices: list[float]
    indices: list[int]


@dataclass
class ProcessingRequest:
    """Request to invoke a registered processing function."""

    function_name: str
    args: list[Any] = field(default_factory=list)
    kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass
class FunctionInfo:
    """Info about a registered processing function."""

    name: str
    schema: dict[str, Any] | None = None
