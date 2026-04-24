"""Tests that specimens are correctly registered at app startup."""
from __future__ import annotations

import pytest
from litestar.testing import AsyncTestClient

from ascribe_link.app import create_app


@pytest.fixture
async def client():
    app = create_app()
    async with AsyncTestClient(app=app) as c:
        yield c


async def test_gaussian_volume_is_registered(client: AsyncTestClient):
    r = await client.get("/api/specimens/")
    r.raise_for_status()
    items = r.json()
    names = {item["id"] for item in items}
    assert "generate_gaussian_volume" in names
    entry = next(x for x in items if x["id"] == "generate_gaussian_volume")
    assert entry["type"] == "volume"
    assert entry["is_dynamic"] is True
    assert "volume" in entry["tags"]
