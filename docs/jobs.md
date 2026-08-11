# Background jobs

Generating a dynamic specimen can take seconds to minutes — a segmentation
pass over a tomography volume, an agent-driven mesh build. Holding an HTTP
request open for that long is a bad deal for a VR client: no progress to show
the user, and a timeout loses the work. The job API splits the work from the
waiting.

## The flow

1. `POST /api/specimens/{specimen_id}/start` with `{"params": …, "room_id": …}`
   → `{"job_id": …, "status": "running"}`. On a cache hit the status is
   already `"done"`.
2. Poll `GET /api/jobs/{job_id}/progress?since={seq}` for new messages.
3. When the status goes terminal, `GET /api/jobs/{job_id}/result`.
4. `DELETE /api/jobs/{job_id}` cancels a job that's still running.

## `GET /api/jobs/{job_id}/progress`

```json
{
  "status": "running",
  "messages": [
    {"seq": 0, "text": "loading volume", "ts": 1234.5},
    {"seq": 1, "text": "thresholding", "ts": 1240.1}
  ],
  "error": null
}
```

`status` is `running`, `done`, or `error`. Pass `since` as the highest `seq`
you've already seen — the default `-1` returns the full history — so a polling
client transfers only what's new.

Progress messages come from the processing function itself: a function that
declares a `ProgressReporter` parameter gets one injected, and each
`reporter.report("…")` call becomes a message here. See
{doc}`dynamic-specimens`.

## `GET /api/jobs/{job_id}/result`

The typed result as JSON, once the job is done.

| Status | Response |
| --- | --- |
| `done` | `200` with the result |
| `running` | `409 Job still running` |
| `error` | `410 Job failed: …` |
| unknown id | `404` |

Result encoding happens on a worker thread rather than the event loop.
Serializing a large mesh was measured blocking the loop for 4.6 seconds, which
starved the `/progress` polls and surfaced in the XR client as `HTTP 0`
errors.

## `DELETE /api/jobs/{job_id}`

Cancels the running task and returns `204`. For a federated job the
cancellation is proxied to the worker that owns it.

## Job lifetime

Finished jobs are swept on a TTL, so a `job_id` is not a durable handle —
fetch the result reasonably soon after the job completes. The result itself
stays in the room cache under its parameters ({doc}`caching`), so a re-request
with the same parameters is a cache hit rather than a recomputation.
