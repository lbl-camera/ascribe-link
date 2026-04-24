"""Integration tests for envelope-encoded /data responses."""
from __future__ import annotations

import pytest
from litestar.testing import AsyncTestClient

from ascribe_link.app import create_app
from ascribe_link.envelope import ENVELOPE_MEDIA_TYPE, decode_envelope
from ascribe_link.models import MeshResult, VolumeResult


@pytest.fixture
async def client():
    app = create_app()
    async with AsyncTestClient(app=app) as c:
        yield c


async def test_gaussian_volume_data_is_envelope(client: AsyncTestClient):
    r = await client.get("/api/specimens/generate_gaussian_volume/data")
    r.raise_for_status()
    content_type = r.headers["content-type"].split(";")[0].strip()
    assert content_type == ENVELOPE_MEDIA_TYPE
    decoded = decode_envelope(r.content)
    assert isinstance(decoded, VolumeResult)
    assert decoded.shape == [64, 64, 64]
    assert decoded.dtype == "float32"


async def test_sphere_mesh_data_is_envelope(client: AsyncTestClient):
    r = await client.get("/api/specimens/generate_sphere/data")
    r.raise_for_status()
    content_type = r.headers["content-type"].split(";")[0].strip()
    assert content_type == ENVELOPE_MEDIA_TYPE
    decoded = decode_envelope(r.content)
    assert isinstance(decoded, MeshResult)
    assert len(decoded.vertices) > 0
    assert len(decoded.indices) > 0
