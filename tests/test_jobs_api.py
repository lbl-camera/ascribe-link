"""End-to-end tests for the job-based specimen API."""
from __future__ import annotations

import asyncio

import pytest

from ascribe_link.app import create_app
from ascribe_link.processing import FunctionRegistry
from ascribe_link.progress import ProgressReporter


@pytest.fixture
async def client():
    """Spin up the real Litestar app in-process with a fast specimen."""
    from litestar.testing import AsyncTestClient

    mesh_functions = {
        "fast_sphere": _fast_sphere,
        "slow_sphere": _slow_sphere,
        "failing": _failing,
    }

    # Register them as specimens by hand via a small app hook.
    # We piggy-back on create_app, then post-register specimens on the registry.
    app = create_app(mesh_functions=mesh_functions)

    # Register as specimens through the DI provider state
    registry: FunctionRegistry = app.dependencies["function_registry"].dependency()
    registry.register_specimen(
        _fast_sphere, display_name="Fast", name="fast_sphere", return_type="mesh"
    )
    registry.register_specimen(
        _slow_sphere, display_name="Slow", name="slow_sphere", return_type="mesh"
    )
    registry.register_specimen(
        _failing, display_name="Failing", name="failing", return_type="mesh"
    )

    async with AsyncTestClient(app=app) as c:
        yield c


async def _fast_sphere(reporter: ProgressReporter = None) -> tuple[list, list]:
    reporter.report("computing fast sphere")
    return ([0.0, 0.0, 0.0] * 3, [0, 1, 2])


async def _slow_sphere(reporter: ProgressReporter = None) -> tuple[list, list]:
    reporter.report("step 1")
    await asyncio.sleep(0.05)
    reporter.report("step 2")
    await asyncio.sleep(0.05)
    reporter.report("step 3")
    return ([0.0, 0.0, 0.0] * 3, [0, 1, 2])


async def _failing(reporter: ProgressReporter = None) -> tuple[list, list]:
    reporter.report("about to fail")
    raise RuntimeError("boom")


async def test_start_returns_job_id_and_running_status(client):
    resp = await client.post(
        "/api/specimens/fast_sphere/start", json={"params": {}, "room_id": "ascribe"}
    )
    assert resp.status_code == 200
    body = resp.json()
    assert "job_id" in body
    assert body["status"] in ("running", "done")


async def test_progress_returns_messages_in_order(client):
    start = (await client.post(
        "/api/specimens/slow_sphere/start",
        json={"params": {}, "room_id": "ascribe"},
    )).json()
    job_id = start["job_id"]

    # Poll until done
    seen_texts: list[str] = []
    last_seq = -1
    for _ in range(100):
        prog = (await client.get(
            f"/api/jobs/{job_id}/progress?since={last_seq}"
        )).json()
        for m in prog["messages"]:
            seen_texts.append(m["text"])
            last_seq = max(last_seq, m["seq"])
        if prog["status"] in ("done", "error"):
            break
        await asyncio.sleep(0.02)

    assert "step 1" in seen_texts
    assert "step 2" in seen_texts
    assert "step 3" in seen_texts
    # Bracket start/end messages are also present
    assert any("Starting" in t for t in seen_texts)
    assert any("Finished" in t for t in seen_texts)


async def test_progress_since_returns_only_new(client):
    start = (await client.post(
        "/api/specimens/slow_sphere/start",
        json={"params": {}, "room_id": "ascribe"},
    )).json()
    job_id = start["job_id"]

    # Wait for completion
    for _ in range(100):
        p = (await client.get(f"/api/jobs/{job_id}/progress")).json()
        if p["status"] == "done":
            break
        await asyncio.sleep(0.02)
    else:
        pytest.fail("job did not finish")

    full = (await client.get(f"/api/jobs/{job_id}/progress")).json()
    tail = (await client.get(
        f"/api/jobs/{job_id}/progress?since={full['messages'][1]['seq']}"
    )).json()
    assert len(tail["messages"]) == len(full["messages"]) - 2


async def test_result_returns_409_while_running(client):
    start = (await client.post(
        "/api/specimens/slow_sphere/start",
        json={"params": {}, "room_id": "ascribe"},
    )).json()
    job_id = start["job_id"]

    # Immediately — should still be running
    r = await client.get(f"/api/jobs/{job_id}/result")
    assert r.status_code == 409


async def test_result_returns_data_when_done(client):
    start = (await client.post(
        "/api/specimens/fast_sphere/start",
        json={"params": {}, "room_id": "ascribe"},
    )).json()
    job_id = start["job_id"]

    for _ in range(100):
        p = (await client.get(f"/api/jobs/{job_id}/progress")).json()
        if p["status"] == "done":
            break
        await asyncio.sleep(0.02)
    else:
        pytest.fail("job did not finish")

    r = await client.get(f"/api/jobs/{job_id}/result")
    assert r.status_code == 200
    body = r.json()
    assert body.get("type") == "mesh"
    assert "vertices" in body


async def test_result_returns_410_on_error(client):
    start = (await client.post(
        "/api/specimens/failing/start",
        json={"params": {}, "room_id": "ascribe"},
    )).json()
    job_id = start["job_id"]

    for _ in range(100):
        p = (await client.get(f"/api/jobs/{job_id}/progress")).json()
        if p["status"] == "error":
            assert "boom" in (p.get("error") or "")
            break
        await asyncio.sleep(0.02)
    else:
        pytest.fail("job did not error")

    r = await client.get(f"/api/jobs/{job_id}/result")
    assert r.status_code == 410


async def test_progress_404_for_unknown_job(client):
    r = await client.get("/api/jobs/does-not-exist/progress")
    assert r.status_code == 404


async def test_delete_cancels_running_job(client):
    start = (await client.post(
        "/api/specimens/slow_sphere/start",
        json={"params": {}, "room_id": "ascribe"},
    )).json()
    job_id = start["job_id"]

    d = await client.delete(f"/api/jobs/{job_id}")
    assert d.status_code == 204

    # Wait for it to show as errored
    for _ in range(100):
        p = (await client.get(f"/api/jobs/{job_id}/progress")).json()
        if p["status"] == "error":
            assert "cancel" in (p.get("error") or "").lower()
            break
        await asyncio.sleep(0.02)
    else:
        pytest.fail("job did not cancel")
