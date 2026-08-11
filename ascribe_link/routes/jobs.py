"""Job endpoints: poll progress, fetch result, cancel."""
from __future__ import annotations

import asyncio
import json
import time
from typing import Any

from litestar import Controller, Response, delete, get
from litestar.exceptions import HTTPException, NotFoundException

from ascribe_link.federation import FederationHub
from ascribe_link.gen_timing import gt_mark
from ascribe_link.job_registry import JobRegistry
from ascribe_link.models import (
    ImageResult,
    MeshResult,
    PointCloudResult,
    VolumeResult,
    result_to_dict,
)


def _coerce_result(value: Any) -> Any:
    """Return a JSON-ready form of a Job.result.

    Result objects are converted via ``result_to_dict``; anything else
    (already a dict, etc.) is passed through unchanged.
    """
    if isinstance(value, (MeshResult, VolumeResult, PointCloudResult, ImageResult)):
        return result_to_dict(value)
    return value


def _encode_result(value: Any) -> bytes:
    """Coerce and JSON-encode a Job.result to response bytes.

    Blocking by design — run via asyncio.to_thread(). Coercion and encoding
    of a large mesh/volume can take seconds; doing it in the handler was
    observed blocking the main event loop for 4.6s (asyncio debug named
    RequestResponseCycle.run_asgi), which starves /progress polls.
    """
    return json.dumps(_coerce_result(value)).encode("utf-8")


class JobController(Controller):
    path = "/api/jobs"

    @get("/{job_id:str}/progress")
    async def get_progress(
        self,
        job_registry: JobRegistry,
        job_id: str,
        federation_hub: FederationHub | None = None,
        since: int = -1,
    ) -> dict[str, Any]:
        """Return new progress messages for this job."""
        job = await job_registry.get(job_id)
        if job is None:
            raise NotFoundException(detail=f"Unknown job: {job_id}")

        # Federated — proxy to worker.
        if job.federated_to is not None and federation_hub is not None:
            worker_id, worker_job_id = job.federated_to
            response = await federation_hub.proxy_request(
                worker_id,
                "get_progress",
                {"job_id": worker_job_id, "since": since},
            )
            # Mirror the terminal status locally so /result can serve,
            # and set finished_at so the TTL sweeper can collect the job.
            if response.get("status") in ("done", "error"):
                job.status = response["status"]
                if response.get("error"):
                    job.error = response["error"]
                if job.finished_at is None:
                    job.finished_at = time.monotonic()
            return response

        messages = [
            {"seq": m.seq, "text": m.text, "ts": m.ts}
            for m in job.messages_since(since)
        ]
        return {
            "status": job.status,
            "messages": messages,
            "error": job.error,
        }

    @get("/{job_id:str}/result")
    async def get_result(
        self,
        job_registry: JobRegistry,
        job_id: str,
        federation_hub: FederationHub | None = None,
    ) -> Response:
        job = await job_registry.get(job_id)
        if job is None:
            raise NotFoundException(detail=f"Unknown job: {job_id}")

        if job.federated_to is not None and federation_hub is not None:
            worker_id, worker_job_id = job.federated_to
            response = await federation_hub.proxy_request(
                worker_id, "get_result", {"job_id": worker_job_id}
            )
            if "error" in response:
                raise HTTPException(status_code=410, detail=response["error"])
            return Response(content=response, media_type="application/json")

        if job.status == "running":
            raise HTTPException(status_code=409, detail="Job still running")
        if job.status == "error":
            raise HTTPException(
                status_code=410, detail=f"Job failed: {job.error}"
            )
        # status == "done" — coerce + encode off-loop so a multi-second
        # serialization of a big mesh/volume can't stall /progress polls.
        gt_mark("/result: request received, encoding off-loop")
        _t0 = time.perf_counter()
        body = await asyncio.to_thread(_encode_result, job.result)
        gt_mark(
            f"/result: encoded {len(body)} bytes "
            f"({time.perf_counter() - _t0:.3f}s), sending"
        )
        return Response(content=body, media_type="application/json")

    @delete("/{job_id:str}", status_code=204)
    async def delete_job(
        self,
        job_registry: JobRegistry,
        job_id: str,
        federation_hub: FederationHub | None = None,
    ) -> None:
        job = await job_registry.get(job_id)
        if job is None:
            raise NotFoundException(detail=f"Unknown job: {job_id}")
        if job.federated_to is not None and federation_hub is not None:
            worker_id, worker_job_id = job.federated_to
            await federation_hub.proxy_request(
                worker_id, "cancel_job", {"job_id": worker_job_id}
            )
            # Federated jobs have no local task; mark finished so the sweeper
            # collects the relay-side job.
            job.status = "error"
            job.error = "cancelled"
            if job.finished_at is None:
                job.finished_at = time.monotonic()
        if job.task is not None and not job.task.done():
            job.task.cancel()
