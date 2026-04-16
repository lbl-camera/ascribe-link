"""Job endpoints: poll progress, fetch result, cancel."""
from __future__ import annotations

from typing import Any

from litestar import Controller, get, delete
from litestar.exceptions import HTTPException, NotFoundException

from ascribe_link.job_registry import JobRegistry
from ascribe_link.models import result_to_dict


class JobController(Controller):
    path = "/api/jobs"

    @get("/{job_id:str}/progress")
    async def get_progress(
        self,
        job_registry: JobRegistry,
        job_id: str,
        since: int = -1,
    ) -> dict[str, Any]:
        """Return new progress messages for this job."""
        job = await job_registry.get(job_id)
        if job is None:
            raise NotFoundException(detail=f"Unknown job: {job_id}")

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
    ) -> dict[str, Any]:
        job = await job_registry.get(job_id)
        if job is None:
            raise NotFoundException(detail=f"Unknown job: {job_id}")
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
    ) -> None:
        job = await job_registry.get(job_id)
        if job is None:
            raise NotFoundException(detail=f"Unknown job: {job_id}")
        if job.task is not None and not job.task.done():
            job.task.cancel()
