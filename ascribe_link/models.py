"""Data models for Ascribe-Link."""

from __future__ import annotations

import base64
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal

import numpy as np


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
    schema: dict[str, Any] | None = None  # JSON Schema for dynamic specimens
    function_name: str | None = None  # Function to invoke for dynamic specimens


@dataclass
class SpecimenListItem:
    """Lightweight specimen info for catalog listings."""

    id: str
    display_name: str
    description: str
    type: SpecimenType
    thumbnail_url: str
    tags: list[str] = field(default_factory=list)
    is_dynamic: bool = False  # True if specimen has parameters (schema != None)


# ---------------------------------------------------------------------------
# Processing Results (discriminated union via 'type' field)
# ---------------------------------------------------------------------------


class ResultType(str, Enum):
    """Type discriminator for processing results."""

    MESH = "mesh"
    VOLUME = "volume"
    POINT_CLOUD = "point_cloud"
    IMAGE = "image"


@dataclass
class MeshResult:
    """Mesh data result.

    Attributes
    ----------
    type : Literal["mesh"]
        Discriminator field.
    vertices : list[float]
        Flat list of vertex coordinates [x1,y1,z1, x2,y2,z2, ...].
    indices : list[int]
        Triangle indices (every 3 = one face).
    normals : list[float], optional
        Flat list of vertex normals [nx1,ny1,nz1, ...].
    """

    type: Literal["mesh"] = "mesh"
    vertices: list[float] = field(default_factory=list)
    indices: list[int] = field(default_factory=list)
    normals: list[float] | None = None

    def to_dict(self) -> dict[str, Any]:
        d = {
            "type": self.type,
            "vertices": self.vertices,
            "indices": self.indices,
        }
        if self.normals:
            d["normals"] = self.normals
        return d

    @classmethod
    def from_pyvista(cls, mesh) -> MeshResult:
        """Create from a PyVista mesh."""
        vertices = mesh.points.flatten().tolist()
        # PyVista faces format: [n, i1, i2, ..., in, n, j1, j2, ..., jn, ...]
        # For triangles: [3, i1, i2, i3, 3, j1, j2, j3, ...]
        faces = mesh.faces.reshape(-1, 4)[:, 1:].flatten().tolist()
        normals = None
        if mesh.point_normals is not None:
            normals = mesh.point_normals.flatten().tolist()
        return cls(vertices=vertices, indices=faces, normals=normals)


@dataclass
class VolumeResult:
    """Volumetric data result.

    Attributes
    ----------
    type : Literal["volume"]
        Discriminator field.
    shape : list[int]
        Shape of the volume [depth, height, width] or [z, y, x].
    dtype : str
        NumPy dtype string (e.g., "float32", "uint8").
    data : str
        Base64-encoded raw bytes of the volume array.
    spacing : list[float], optional
        Voxel spacing [sz, sy, sx]. Defaults to [1, 1, 1].
    origin : list[float], optional
        Origin point [oz, oy, ox]. Defaults to [0, 0, 0].
    """

    type: Literal["volume"] = "volume"
    shape: list[int] = field(default_factory=list)
    dtype: str = "float32"
    data: str = ""  # base64-encoded
    spacing: list[float] | None = None
    origin: list[float] | None = None

    def __post_init__(self) -> None:
        # Transient zero-copy handle; populated by from_numpy. Not a
        # dataclass field (no class-level annotation), so it's excluded
        # from to_dict / asdict / repr / eq.
        # NOTE: it IS picked up by copy.deepcopy and pickle — strip it
        # before pickling or caching across process boundaries.
        self._array: np.ndarray | None = None

    def to_dict(self) -> dict[str, Any]:
        d = {
            "type": self.type,
            "shape": self.shape,
            "dtype": self.dtype,
            "data": self.data,
        }
        if self.spacing:
            d["spacing"] = self.spacing
        if self.origin:
            d["origin"] = self.origin
        return d

    def to_numpy(self) -> np.ndarray:
        """Decode to NumPy array."""
        raw = base64.b64decode(self.data)
        arr = np.frombuffer(raw, dtype=np.dtype(self.dtype))
        return arr.reshape(self.shape)

    @classmethod
    def from_numpy(
        cls,
        arr: np.ndarray,
        spacing: list[float] | None = None,
        origin: list[float] | None = None,
    ) -> VolumeResult:
        """Create from a NumPy array."""
        # Ensure C-contiguous for consistent byte order
        arr = np.ascontiguousarray(arr)
        data = base64.b64encode(arr.tobytes()).decode("ascii")
        result = cls(
            shape=list(arr.shape),
            dtype=str(arr.dtype),
            data=data,
            spacing=spacing,
            origin=origin,
        )
        # Cache raw ndarray for zero-copy envelope encoding (avoids base64 round-trip).
        result._array = arr
        return result


@dataclass
class PointCloudResult:
    """Point cloud data result.

    Attributes
    ----------
    type : Literal["point_cloud"]
        Discriminator field.
    points : list[float]
        Flat list of point coordinates [x1,y1,z1, x2,y2,z2, ...].
    colors : list[float], optional
        Flat list of RGB colors [r1,g1,b1, ...] in range [0, 1].
    scalars : list[float], optional
        Per-point scalar values.
    scalar_name : str, optional
        Name of the scalar field.
    """

    type: Literal["point_cloud"] = "point_cloud"
    points: list[float] = field(default_factory=list)
    colors: list[float] | None = None
    scalars: list[float] | None = None
    scalar_name: str | None = None

    def to_dict(self) -> dict[str, Any]:
        d = {
            "type": self.type,
            "points": self.points,
        }
        if self.colors:
            d["colors"] = self.colors
        if self.scalars:
            d["scalars"] = self.scalars
        if self.scalar_name:
            d["scalar_name"] = self.scalar_name
        return d

    @classmethod
    def from_numpy(
        cls,
        points: np.ndarray,
        colors: np.ndarray | None = None,
        scalars: np.ndarray | None = None,
        scalar_name: str | None = None,
    ) -> PointCloudResult:
        """Create from NumPy arrays."""
        return cls(
            points=points.flatten().tolist(),
            colors=colors.flatten().tolist() if colors is not None else None,
            scalars=scalars.flatten().tolist() if scalars is not None else None,
            scalar_name=scalar_name,
        )


@dataclass
class ImageResult:
    """2D image result.

    Attributes
    ----------
    type : Literal["image"]
        Discriminator field.
    width : int
        Image width in pixels.
    height : int
        Image height in pixels.
    channels : int
        Number of channels (1=grayscale, 3=RGB, 4=RGBA).
    dtype : str
        NumPy dtype string (e.g., "uint8", "float32").
    data : str
        Base64-encoded raw bytes (row-major, channels last).
    """

    type: Literal["image"] = "image"
    width: int = 0
    height: int = 0
    channels: int = 1
    dtype: str = "uint8"
    data: str = ""  # base64-encoded

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.type,
            "width": self.width,
            "height": self.height,
            "channels": self.channels,
            "dtype": self.dtype,
            "data": self.data,
        }

    def to_numpy(self) -> np.ndarray:
        """Decode to NumPy array."""
        raw = base64.b64decode(self.data)
        arr = np.frombuffer(raw, dtype=np.dtype(self.dtype))
        if self.channels == 1:
            return arr.reshape((self.height, self.width))
        return arr.reshape((self.height, self.width, self.channels))

    @classmethod
    def from_numpy(cls, arr: np.ndarray) -> ImageResult:
        """Create from a NumPy array (H, W) or (H, W, C)."""
        arr = np.ascontiguousarray(arr)
        if arr.ndim == 2:
            height, width = arr.shape
            channels = 1
        else:
            height, width, channels = arr.shape
        data = base64.b64encode(arr.tobytes()).decode("ascii")
        return cls(
            width=width,
            height=height,
            channels=channels,
            dtype=str(arr.dtype),
            data=data,
        )


# Union type for all results
ProcessingResult = MeshResult | VolumeResult | PointCloudResult | ImageResult


def result_to_dict(result: ProcessingResult) -> dict[str, Any]:
    """Convert any result type to a dictionary."""
    return result.to_dict()


def result_from_dict(data: dict[str, Any]) -> ProcessingResult:
    """Parse a result dictionary based on its 'type' field."""
    result_type = data.get("type", "mesh")

    if result_type == "mesh":
        return MeshResult(
            vertices=data.get("vertices", []),
            indices=data.get("indices", []),
            normals=data.get("normals"),
        )
    elif result_type == "volume":
        return VolumeResult(
            shape=data.get("shape", []),
            dtype=data.get("dtype", "float32"),
            data=data.get("data", ""),
            spacing=data.get("spacing"),
            origin=data.get("origin"),
        )
    elif result_type == "point_cloud":
        return PointCloudResult(
            points=data.get("points", []),
            colors=data.get("colors"),
            scalars=data.get("scalars"),
            scalar_name=data.get("scalar_name"),
        )
    elif result_type == "image":
        return ImageResult(
            width=data.get("width", 0),
            height=data.get("height", 0),
            channels=data.get("channels", 1),
            dtype=data.get("dtype", "uint8"),
            data=data.get("data", ""),
        )
    else:
        raise ValueError(f"Unknown result type: {result_type}")


# ---------------------------------------------------------------------------
# Request Models
# ---------------------------------------------------------------------------


@dataclass
class ProcessingRequest:
    """Request to invoke a registered processing function."""

    function_name: str
    args: list[Any] = field(default_factory=list)
    kwargs: dict[str, Any] = field(default_factory=dict)
    room_id: str = "ascribe"  # Room identifier for multiplayer caching


@dataclass
class FunctionInfo:
    """Info about a registered processing function."""

    name: str
    schema: dict[str, Any] | None = None
    return_type: str | None = None  # "mesh", "volume", "point_cloud", "image", or None for unknown
