"""Binary envelope wire format for MeshResult / VolumeResult.

Layout:
    <4-byte little-endian uint32: preamble_length>
    <preamble_length bytes: UTF-8 JSON preamble>
    <raw bytes: one or more contiguous data blocks>
"""
from __future__ import annotations

import json
import struct
from typing import Any, Union

import numpy as np

from ascribe_link.models import MeshResult, VolumeResult

ENVELOPE_MEDIA_TYPE = "application/x-ascribe-envelope-v1"

Envelopeable = Union[MeshResult, VolumeResult]


def encode_envelope(result: Envelopeable) -> bytes:
    """Serialize a result to the binary envelope format."""
    if isinstance(result, VolumeResult):
        return _encode_volume(result)
    if isinstance(result, MeshResult):
        return _encode_mesh(result)
    raise TypeError(f"Cannot envelope-encode {type(result).__name__}")


def decode_envelope(data: bytes) -> Envelopeable:
    """Parse the binary envelope format back into a typed result."""
    if len(data) < 4:
        raise ValueError("envelope truncated: missing length prefix")
    (preamble_len,) = struct.unpack("<I", data[:4])
    if len(data) < 4 + preamble_len:
        raise ValueError("envelope truncated: preamble incomplete")
    preamble = json.loads(data[4 : 4 + preamble_len].decode("utf-8"))
    offset = 4 + preamble_len
    result_type = preamble.get("type", "")
    if result_type == "volume":
        return _decode_volume(preamble, data, offset)
    if result_type == "mesh":
        return _decode_mesh(preamble, data, offset)
    raise ValueError(f"unknown envelope type: {result_type!r}")


# ---------- volume ----------

def _encode_volume(result: VolumeResult) -> bytes:
    arr = _volume_array(result)
    preamble: dict[str, Any] = {
        "type": "volume",
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
    }
    if result.spacing is not None:
        preamble["spacing"] = list(result.spacing)
    if result.origin is not None:
        preamble["origin"] = list(result.origin)
    # Actual data range, so the client can normalize to [0, 1] on the GPU
    # (the raymarcher expects unit-range scalars) without a per-voxel pass in
    # GDScript. Free here; NaNs are ignored so one bad voxel can't poison it.
    if arr.size:
        vmin, vmax = np.nanmin(arr), np.nanmax(arr)
        if np.isfinite(vmin) and np.isfinite(vmax):
            preamble["value_range"] = [float(vmin), float(vmax)]
    preamble_bytes = json.dumps(preamble, separators=(",", ":")).encode("utf-8")
    header = struct.pack("<I", len(preamble_bytes)) + preamble_bytes
    return header + np.ascontiguousarray(arr).tobytes()


def _decode_volume(preamble: dict, data: bytes, offset: int) -> VolumeResult:
    shape = preamble["shape"]
    dtype = preamble["dtype"]
    count = int(np.prod(shape))
    arr = np.frombuffer(data, dtype=dtype, count=count, offset=offset).reshape(shape).copy()
    return VolumeResult.from_numpy(
        arr,
        spacing=preamble.get("spacing"),
        origin=preamble.get("origin"),
    )


def _volume_array(result: VolumeResult) -> np.ndarray:
    """Get the underlying ndarray, preferring the zero-copy _array if set."""
    arr = getattr(result, "_array", None)
    if arr is not None:
        return arr
    return result.to_numpy()


# ---------- mesh ----------

def _encode_mesh(result: MeshResult) -> bytes:
    vertices = np.asarray(result.vertices, dtype=np.float32)
    indices = np.asarray(result.indices, dtype=np.uint32)
    normals = (
        np.asarray(result.normals, dtype=np.float32)
        if result.normals
        else np.empty(0, dtype=np.float32)
    )
    vertex_count = vertices.size // 3
    index_count = indices.size
    normal_count = normals.size // 3
    preamble = {
        "type": "mesh",
        "vertex_count": vertex_count,
        "vertex_dtype": "float32",
        "index_count": index_count,
        "index_dtype": "uint32",
        "normal_count": normal_count,
        "normal_dtype": "float32",
    }
    preamble_bytes = json.dumps(preamble, separators=(",", ":")).encode("utf-8")
    header = struct.pack("<I", len(preamble_bytes)) + preamble_bytes
    body = vertices.tobytes() + indices.tobytes()
    if normal_count:
        body += normals.tobytes()
    return header + body


def _decode_mesh(preamble: dict, data: bytes, offset: int) -> MeshResult:
    vertex_dtype = preamble.get("vertex_dtype", "float32")
    index_dtype = preamble.get("index_dtype", "uint32")
    normal_dtype = preamble.get("normal_dtype", "float32")
    if vertex_dtype != "float32":
        raise ValueError(f"unsupported mesh vertex_dtype: {vertex_dtype!r}")
    if index_dtype != "uint32":
        raise ValueError(f"unsupported mesh index_dtype: {index_dtype!r}")
    if normal_dtype != "float32":
        raise ValueError(f"unsupported mesh normal_dtype: {normal_dtype!r}")
    vc = preamble["vertex_count"]
    ic = preamble["index_count"]
    nc = preamble["normal_count"]
    vertices = np.frombuffer(data, dtype=np.float32, count=vc * 3, offset=offset).tolist()
    offset += vc * 3 * 4
    indices = np.frombuffer(data, dtype=np.uint32, count=ic, offset=offset).tolist()
    offset += ic * 4
    normals = None
    if nc:
        normals = np.frombuffer(
            data, dtype=np.float32, count=nc * 3, offset=offset
        ).tolist()
    return MeshResult(vertices=vertices, indices=indices, normals=normals)
