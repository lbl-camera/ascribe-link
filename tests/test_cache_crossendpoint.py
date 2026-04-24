"""Regression test: cache consistency across /data and /processing/invoke."""
from __future__ import annotations

import pytest
from litestar.testing import AsyncTestClient

from ascribe_link.app import create_app
from ascribe_link.envelope import ENVELOPE_MEDIA_TYPE, decode_envelope
from ascribe_link.models import VolumeResult


@pytest.fixture
async def client():
    app = create_app()
    async with AsyncTestClient(app=app) as c:
        yield c


async def test_data_then_invoke_cache_consistent(client: AsyncTestClient):
    """Populate cache via /data, then read via /processing/invoke."""
    # Prime cache via /data
    r1 = await client.get("/api/specimens/generate_gaussian_volume/data")
    r1.raise_for_status()
    assert r1.headers["content-type"].startswith(ENVELOPE_MEDIA_TYPE)
    decoded = decode_envelope(r1.content)
    assert isinstance(decoded, VolumeResult)

    # Now read via /processing/invoke with same params -> should get JSON dict
    r2 = await client.post(
        "/api/processing/invoke",
        json={
            "function_name": "generate_gaussian_volume",
            "args": [],
            "kwargs": {},
            "room_id": "ascribe",
        },
    )
    r2.raise_for_status()
    payload = r2.json()
    assert payload["type"] == "volume"
    assert payload["shape"] == [64, 64, 64]
    assert payload["dtype"] == "float32"


async def test_invoke_then_data_cache_consistent(client: AsyncTestClient):
    """Populate cache via /processing/invoke, then read via /data."""
    r1 = await client.post(
        "/api/processing/invoke",
        json={
            "function_name": "generate_gaussian_volume",
            "args": [],
            "kwargs": {},
            "room_id": "ascribe",
        },
    )
    r1.raise_for_status()
    assert r1.headers["content-type"].startswith("application/json")

    r2 = await client.get("/api/specimens/generate_gaussian_volume/data")
    r2.raise_for_status()
    assert r2.headers["content-type"].startswith(ENVELOPE_MEDIA_TYPE)
    decoded = decode_envelope(r2.content)
    assert isinstance(decoded, VolumeResult)
