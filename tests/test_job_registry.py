"""Tests for Job and JobRegistry."""
from __future__ import annotations

import asyncio
import time

import pytest

from ascribe_link.job_registry import Job, JobRegistry


@pytest.fixture
def registry():
    return JobRegistry(ttl_seconds=5.0)


async def test_create_returns_job_with_uuid(registry):
    job = await registry.create(specimen_id="sphere", params={"r": 1}, room_id="ascribe")
    assert job.id
    assert len(job.id) >= 32  # UUID-ish
    assert job.specimen_id == "sphere"
    assert job.status == "running"


async def test_get_returns_same_instance(registry):
    job = await registry.create(specimen_id="sphere", params={}, room_id="ascribe")
    assert await registry.get(job.id) is job


async def test_get_unknown_returns_none(registry):
    assert await registry.get("no-such-id") is None


async def test_messages_start_empty_and_seq_zero(registry):
    job = await registry.create(specimen_id="sphere", params={}, room_id="ascribe")
    assert len(job.messages) == 0
    assert job.next_seq == 0


async def test_append_message_increments_seq(registry):
    job = await registry.create(specimen_id="sphere", params={}, room_id="ascribe")
    job.append_message("first")
    job.append_message("second")
    assert [m.seq for m in job.messages] == [0, 1]
    assert [m.text for m in job.messages] == ["first", "second"]
    assert job.next_seq == 2


async def test_message_deque_is_bounded(registry):
    job = await registry.create(specimen_id="sphere", params={}, room_id="ascribe")
    for i in range(60):
        job.append_message(f"msg {i}")
    # Only last 50 retained, but next_seq keeps counting
    assert len(job.messages) == 50
    assert job.next_seq == 60
    assert job.messages[0].text == "msg 10"
    assert job.messages[-1].text == "msg 59"


async def test_messages_since_returns_only_new(registry):
    job = await registry.create(specimen_id="sphere", params={}, room_id="ascribe")
    job.append_message("a")
    job.append_message("b")
    job.append_message("c")
    new = job.messages_since(0)
    assert [m.text for m in new] == ["b", "c"]  # seq > 0
    assert job.messages_since(2) == []
    assert [m.text for m in job.messages_since(-1)] == ["a", "b", "c"]


async def test_delete_removes_job(registry):
    job = await registry.create(specimen_id="sphere", params={}, room_id="ascribe")
    await registry.delete(job.id)
    assert await registry.get(job.id) is None


async def test_expired_jobs_are_swept(registry):
    reg = JobRegistry(ttl_seconds=0.05)
    job = await reg.create(specimen_id="sphere", params={}, room_id="ascribe")
    job.status = "done"
    job.finished_at = time.monotonic()
    await asyncio.sleep(0.1)
    await reg.sweep_expired()
    assert await reg.get(job.id) is None


async def test_running_jobs_not_swept_even_if_old(registry):
    reg = JobRegistry(ttl_seconds=0.05)
    job = await reg.create(specimen_id="sphere", params={}, room_id="ascribe")
    # status stays "running"; no finished_at
    await asyncio.sleep(0.1)
    await reg.sweep_expired()
    assert await reg.get(job.id) is job
