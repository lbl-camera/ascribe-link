"""Regression test: job-based flow shares cache shape with /data endpoint.

After Task 7, ``_run_job`` writes raw Result objects into ``RoomResultCache``
(matching ``/api/processing/invoke`` and ``/api/specimens/{id}/data``).  This
prevents a cross-endpoint hazard where ``/data`` would crash trying to
envelope-encode a dict left behind by an earlier ``/start`` job.

``/api/jobs/{id}/result`` keeps emitting JSON dicts for back-compat — the XR
client uses it as a job-done signal, not as the primary data channel.
"""
from __future__ import annotations

import asyncio

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


async def _wait_done(client: AsyncTestClient, job_id: str) -> None:
    for _ in range(100):
        r = await client.get(f"/api/jobs/{job_id}/progress")
        r.raise_for_status()
        if r.json().get("status") == "done":
            return
        await asyncio.sleep(0.02)
    pytest.fail("job did not complete in time")


async def test_job_cache_feeds_data_endpoint(client: AsyncTestClient):
    """After /start completes, /data on the same key envelope-serves cleanly.

    Pre-Task-7 this would raise TypeError because the cache held a dict.
    """
    # Kick off via /start with explicit params matching what /data uses.
    r = await client.post(
        "/api/specimens/generate_gaussian_volume/start",
        json={"params": {"resolution": 64, "sigma": 0.25}, "room_id": "ascribe"},
    )
    r.raise_for_status()
    job_id = r.json()["job_id"]

    await _wait_done(client, job_id)

    # Same params, same room_id -> same cache key as /start used.
    r = await client.get(
        "/api/specimens/generate_gaussian_volume/data",
        params={"params": '{"resolution": 64, "sigma": 0.25}', "room_id": "ascribe"},
    )
    r.raise_for_status()
    assert r.headers["content-type"].startswith(ENVELOPE_MEDIA_TYPE)
    decoded = decode_envelope(r.content)
    assert isinstance(decoded, VolumeResult)
    assert decoded.shape == [64, 64, 64]


async def test_job_result_endpoint_returns_json_dict(client: AsyncTestClient):
    """GET /api/jobs/{id}/result still emits JSON dict (back-compat)."""
    r = await client.post(
        "/api/specimens/generate_gaussian_volume/start",
        json={"params": {}, "room_id": "ascribe-result-test"},
    )
    r.raise_for_status()
    job_id = r.json()["job_id"]

    await _wait_done(client, job_id)

    r = await client.get(f"/api/jobs/{job_id}/result")
    r.raise_for_status()
    # JSON, not envelope.
    assert r.headers["content-type"].startswith("application/json")
    payload = r.json()
    # Handler returns the dict directly (no wrapping); accept either shape.
    if isinstance(payload, dict) and "result" in payload and isinstance(payload["result"], dict):
        data = payload["result"]
    else:
        data = payload
    assert data.get("type") == "volume"
    assert data.get("shape") == [64, 64, 64]
