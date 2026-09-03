"""Tests for the binary envelope format."""
from __future__ import annotations

import json
import struct

import numpy as np
import pytest

from ascribe_link.envelope import ENVELOPE_MEDIA_TYPE, decode_envelope, encode_envelope
from ascribe_link.models import MeshResult, VolumeResult


def test_envelope_media_type():
    assert ENVELOPE_MEDIA_TYPE == "application/x-ascribe-envelope-v1"


def test_volume_round_trip_float32():
    arr = np.random.RandomState(0).rand(4, 5, 6).astype(np.float32)
    original = VolumeResult.from_numpy(arr, spacing=[1.5, 2.0, 2.5], origin=[0.1, 0.2, 0.3])
    blob = encode_envelope(original)
    decoded = decode_envelope(blob)
    assert isinstance(decoded, VolumeResult)
    assert decoded.dtype == "float32"
    assert decoded.shape == [4, 5, 6]
    np.testing.assert_array_equal(decoded.to_numpy(), arr)
    assert decoded.spacing == [1.5, 2.0, 2.5]
    assert decoded.origin == [0.1, 0.2, 0.3]


def test_volume_round_trip_uint8():
    arr = np.arange(2 * 3 * 4, dtype=np.uint8).reshape(2, 3, 4)
    original = VolumeResult.from_numpy(arr)
    blob = encode_envelope(original)
    decoded = decode_envelope(blob)
    assert decoded.dtype == "uint8"
    np.testing.assert_array_equal(decoded.to_numpy(), arr)


def _preamble(blob: bytes) -> dict:
    (n,) = struct.unpack("<I", blob[:4])
    return json.loads(blob[4 : 4 + n])


def test_volume_preamble_carries_value_range():
    arr = np.array([[[3, 250]], [[7, 90]]], dtype=np.uint8)
    assert _preamble(encode_envelope(VolumeResult.from_numpy(arr)))["value_range"] == [3.0, 250.0]

    farr = np.array([[[-1.5, np.nan]], [[2.25, 0.0]]], dtype=np.float32)
    assert _preamble(encode_envelope(VolumeResult.from_numpy(farr)))["value_range"] == [-1.5, 2.25]

    allnan = np.full((1, 1, 2), np.nan, dtype=np.float32)
    assert "value_range" not in _preamble(encode_envelope(VolumeResult.from_numpy(allnan)))


def test_mesh_round_trip_with_normals():
    original = MeshResult(
        vertices=[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        indices=[0, 1, 2],
        normals=[0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0],
    )
    blob = encode_envelope(original)
    decoded = decode_envelope(blob)
    assert isinstance(decoded, MeshResult)
    assert decoded.vertices == original.vertices
    assert decoded.indices == original.indices
    assert decoded.normals == original.normals


def test_mesh_round_trip_without_normals():
    original = MeshResult(
        vertices=[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        indices=[0, 1, 2],
        normals=None,
    )
    blob = encode_envelope(original)
    decoded = decode_envelope(blob)
    assert isinstance(decoded, MeshResult)
    assert decoded.vertices == original.vertices
    assert decoded.indices == original.indices
    assert decoded.normals is None


def test_truncated_length_prefix():
    with pytest.raises(ValueError, match="missing length prefix"):
        decode_envelope(b"\x00\x00")


def test_truncated_preamble():
    header = struct.pack("<I", 100) + b"{"  # claims 100 bytes, gives 1
    with pytest.raises(ValueError, match="preamble incomplete"):
        decode_envelope(header)


def test_unknown_envelope_type():
    preamble = b'{"type":"nope"}'
    blob = struct.pack("<I", len(preamble)) + preamble
    with pytest.raises(ValueError, match="unknown envelope type"):
        decode_envelope(blob)


def test_encode_rejects_other_types():
    with pytest.raises(TypeError):
        encode_envelope({"type": "volume"})  # dict is not a Result


def test_decode_mesh_rejects_unsupported_vertex_dtype():
    # Build a valid mesh preamble but with a bogus vertex_dtype.
    preamble = json.dumps({
        "type": "mesh",
        "vertex_count": 0, "vertex_dtype": "float64",
        "index_count": 0,  "index_dtype": "uint32",
        "normal_count": 0, "normal_dtype": "float32",
    }, separators=(",", ":")).encode("utf-8")
    blob = struct.pack("<I", len(preamble)) + preamble
    with pytest.raises(ValueError, match="unsupported mesh vertex_dtype"):
        decode_envelope(blob)


def test_volume_round_trip_preserves_none_spacing_origin():
    arr = np.zeros((2, 2, 2), dtype=np.float32)
    original = VolumeResult.from_numpy(arr)  # spacing/origin default to None
    assert original.spacing is None
    assert original.origin is None
    blob = encode_envelope(original)
    decoded = decode_envelope(blob)
    assert decoded.spacing is None
    assert decoded.origin is None


def test_from_numpy_caches_array():
    arr = np.ones((2, 3, 4), dtype=np.float32)
    result = VolumeResult.from_numpy(arr)
    assert getattr(result, "_array", None) is not None
    # Envelope encode should not need to re-decode base64
    blob = encode_envelope(result)
    decoded = decode_envelope(blob)
    np.testing.assert_array_equal(decoded.to_numpy(), arr)


def test_envelope_uses_cached_array_not_base64():
    """If _array is set, encode must not depend on result.data (base64 string).

    Poisoning result.data proves the zero-copy path is the one taken.
    """
    arr = np.ones((2, 3, 4), dtype=np.float32)
    result = VolumeResult.from_numpy(arr)
    result.data = ""  # poison the base64 fallback path
    blob = encode_envelope(result)  # must succeed via _array
    decoded = decode_envelope(blob)
    np.testing.assert_array_equal(decoded.to_numpy(), arr)
