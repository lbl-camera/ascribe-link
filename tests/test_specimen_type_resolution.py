"""Specimen type resolution for runtime-created and agent-staged specimens.

Two regressions from the same symptom (a volume rendered by the mesh
renderer, i.e. nothing on screen):

- a bundle written after startup was invisible to `SpecimenStore` until a
  manual ``GET /api/specimens/reload``, so the metadata lookup 404'd and the
  client fell back to "mesh";
- an agent-staged specimen has no catalog entry at all, so
  ``GET /api/specimens/{id}`` could never report its type.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
from litestar.testing import AsyncTestClient

from ascribe_link.app import create_app
from ascribe_link.models import VolumeResult
from ascribe_link.specimen_store import SpecimenStore


def _write_bundle(root, specimen_id, type_="volume"):
    d = root / specimen_id
    d.mkdir(parents=True)
    np.save(d / "data.npy", np.zeros((2, 2, 2), np.uint8))
    (d / "specimen.json").write_text(
        json.dumps({"id": specimen_id, "display_name": specimen_id, "type": type_, "data_file": "data.npy"})
    )


def test_store_get_picks_up_bundle_written_after_scan(tmp_path):
    store = SpecimenStore(tmp_path)
    assert store.get("late") is None

    _write_bundle(tmp_path, "late")

    meta = store.get("late")
    assert meta is not None
    assert meta.type.value == "volume"
    assert [m.id for m in store.list()] == ["late"]


def test_store_get_unknown_id_does_not_rescan(tmp_path, monkeypatch):
    _write_bundle(tmp_path, "known")
    store = SpecimenStore(tmp_path)
    calls = []
    monkeypatch.setattr(store, "reload", lambda: calls.append(1))

    assert store.get("nope") is None
    assert store.get("../escape") is None
    assert store.get("known").id == "known"
    assert calls == []


@pytest.mark.anyio
async def test_metadata_route_sees_bundle_written_after_startup(tmp_path):
    app = create_app(specimens_dir=tmp_path)
    async with AsyncTestClient(app=app) as c:
        assert (await c.get("/api/specimens/late")).status_code == 404
        _write_bundle(tmp_path, "late")
        r = await c.get("/api/specimens/late")
        assert r.status_code == 200
        assert r.json()["type"] == "volume"


@pytest.mark.anyio
async def test_metadata_route_resolves_agent_staged_specimen(tmp_path):
    app = create_app(specimens_dir=tmp_path, enable_agent=True, agent_client_factory=lambda: object())
    mgr = app.state.agent_session_manager
    specimen_id = mgr._stage_result("roomA", VolumeResult.from_numpy(np.zeros((2, 2, 2), np.uint8)))

    async with AsyncTestClient(app=app) as c:
        r = await c.get(f"/api/specimens/{specimen_id}", params={"room_id": "roomA"})
        assert r.status_code == 200
        assert r.json()["type"] == "volume"
        assert r.json()["id"] == specimen_id
        # Wrong room: the staged store is room-scoped.
        assert (await c.get(f"/api/specimens/{specimen_id}", params={"room_id": "roomB"})).status_code == 404
