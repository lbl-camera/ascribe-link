"""Tests for the binary envelope format."""
from __future__ import annotations

import struct

import numpy as np
import pytest

from ascribe_link.envelope import decode_envelope, encode_envelope, ENVELOPE_MEDIA_TYPE
from ascribe_link.models import VolumeResult


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


from ascribe_link.models import MeshResult


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
