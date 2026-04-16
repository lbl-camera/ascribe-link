"""Job endpoints: poll progress, fetch result, cancel."""
from __future__ import annotations

import time
from typing import Any

from litestar import Controller, get, delete
from litestar.exceptions import HTTPException, NotFoundException

from ascribe_link.federation import FederationHub
from ascribe_link.job_registry import JobRegistry
from ascribe_link.models import result_to_dict


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
        import logging
        logging.getLogger(__name__).info(
            "GET /progress job=%s since=%d → %d msgs, status=%s, next_seq=%d",
            job_id[:8], since, len(messages), job.status, job.next_seq,
        )
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
    ) -> dict[str, Any]:
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
            return response

        if job.status == "running":
            raise HTTPException(status_code=409, detail="Job still running")
        if job.status == "error":
            raise HTTPException(
                status_code=410, detail=f"Job failed: {job.error}"
            )
        # status == "done"
        if isinstance(job.result, dict):
            return job.result
        return result_to_dict(job.result)

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
