"""Tests for federated job proxying in relay mode."""
from __future__ import annotations

from unittest.mock import AsyncMock

import pytest
from litestar.testing import AsyncTestClient

from ascribe_link.app import create_app
from ascribe_link.federation import FederationHub, WorkerInfo


@pytest.fixture
async def relay_client():
    app = create_app(relay_mode=True)

    # Inject a fake worker into the hub.
    hub: FederationHub = app.dependencies["federation_hub"].dependency()

    # Simulate a registered worker with one specimen.
    worker_id = "worker_a"
    hub._workers[worker_id] = WorkerInfo(
        worker_id=worker_id,
        websocket=None,
        specimens=[{"id": "remote_sphere", "display_name": "Remote", "type": "mesh"}],
    )
    # Mock proxy_request for federated calls.
    hub.proxy_request = AsyncMock(side_effect=_fake_proxy)

    async with AsyncTestClient(app=app) as c:
        yield c, hub


async def _fake_proxy(worker_id, method, payload):
    if method == "start_job":
        return {"job_id": "remote-job-42", "status": "running"}
    if method == "get_progress":
        return {
            "status": "done",
            "messages": [{"seq": 0, "text": "remote msg", "ts": 1.0}],
            "error": None,
        }
    if method == "get_result":
        return {"type": "mesh", "vertices": [0.0] * 9, "indices": [0, 1, 2]}
    raise NotImplementedError(method)


async def test_relay_start_proxies_to_worker(relay_client):
    c, hub = relay_client
    r = await c.post(
        "/api/specimens/worker_a:remote_sphere/start",
        json={"params": {}, "room_id": "ascribe"},
    )
    assert r.status_code == 200
    body = r.json()
    assert "job_id" in body
    assert body["status"] == "running"
    # Proxy was called with start_job
    hub.proxy_request.assert_any_call(
        "worker_a",
        "start_job",
        {"specimen_id": "remote_sphere", "params": {}, "room_id": "ascribe"},
    )


async def test_relay_progress_proxies_to_worker(relay_client):
    c, hub = relay_client
    start = (await c.post(
        "/api/specimens/worker_a:remote_sphere/start",
        json={"params": {}, "room_id": "ascribe"},
    )).json()
    job_id = start["job_id"]

    r = await c.get(f"/api/jobs/{job_id}/progress")
    assert r.status_code == 200
    body = r.json()
    assert body["messages"][0]["text"] == "remote msg"


async def test_relay_result_proxies_to_worker(relay_client):
    c, hub = relay_client
    start = (await c.post(
        "/api/specimens/worker_a:remote_sphere/start",
        json={"params": {}, "room_id": "ascribe"},
    )).json()
    job_id = start["job_id"]

    r = await c.get(f"/api/jobs/{job_id}/result")
    assert r.status_code == 200
    body = r.json()
    assert body["type"] == "mesh"
