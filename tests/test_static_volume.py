"""Tests for static .npy volume specimens."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from litestar.testing import AsyncTestClient

from ascribe_link.app import create_app
from ascribe_link.envelope import ENVELOPE_MEDIA_TYPE, decode_envelope
from ascribe_link.models import VolumeResult


@pytest.fixture(scope="module")
def fixture_specimens(tmp_path_factory) -> Path:
    """Write a real static volume specimen fixture once per test module."""
    root = tmp_path_factory.mktemp("specimens")
    spec_dir = root / "gaussian_static"
    spec_dir.mkdir()

    arr = np.random.RandomState(0).rand(16, 16, 16).astype(np.float32)
    np.save(spec_dir / "data.npy", arr)
    (spec_dir / "data.json").write_text(
        json.dumps({"spacing": [0.1, 0.2, 0.3], "origin": [1.0, 2.0, 3.0]})
    )
    (spec_dir / "specimen.json").write_text(
        json.dumps({
            "id": "gaussian_static",
            "display_name": "Static Gaussian",
            "description": "A static .npy volume specimen",
            "type": "volume",
            "data_file": "data.npy",
            "tags": ["static", "volume"],
        })
    )
    return root


@pytest.fixture
async def client(fixture_specimens: Path):
    app = create_app(specimens_dir=fixture_specimens)
    async with AsyncTestClient(app=app) as c:
        yield c


async def test_static_volume_listed(client):
    r = await client.get("/api/specimens/")
    r.raise_for_status()
    items = r.json()
    names = {item["id"] for item in items}
    assert "gaussian_static" in names
    entry = next(x for x in items if x["id"] == "gaussian_static")
    assert entry["type"] == "volume"


async def test_static_volume_data_is_envelope(client, fixture_specimens: Path):
    r = await client.get("/api/specimens/gaussian_static/data")
    r.raise_for_status()
    content_type = r.headers["content-type"].split(";")[0].strip()
    assert content_type == ENVELOPE_MEDIA_TYPE
    decoded = decode_envelope(r.content)
    assert isinstance(decoded, VolumeResult)
    assert decoded.shape == [16, 16, 16]
    assert decoded.spacing == [0.1, 0.2, 0.3]
    assert decoded.origin == [1.0, 2.0, 3.0]

    # Verify the bytes match the source.
    source = np.load(fixture_specimens / "gaussian_static" / "data.npy")
    np.testing.assert_array_equal(decoded.to_numpy(), source)


async def test_static_volume_without_sidecar(tmp_path):
    spec_root = tmp_path / "specimens"
    spec_dir = spec_root / "nosidecar"
    spec_dir.mkdir(parents=True)
    np.save(spec_dir / "data.npy", np.zeros((4, 4, 4), dtype=np.uint8))
    (spec_dir / "specimen.json").write_text(json.dumps({
        "id": "nosidecar",
        "display_name": "No Sidecar",
        "type": "volume",
        "data_file": "data.npy",
    }))
    app = create_app(specimens_dir=spec_root)
    async with AsyncTestClient(app=app) as c:
        r = await c.get("/api/specimens/nosidecar/data")
        r.raise_for_status()
        decoded = decode_envelope(r.content)
        assert decoded.spacing is None
        assert decoded.origin is None
