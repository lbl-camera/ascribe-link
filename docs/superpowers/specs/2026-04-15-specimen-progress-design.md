# Specimen Loading Progress — Design

**Date:** 2026-04-15
**Projects:** ascribe-link (server), ascribe-xr (Godot client)

## Problem

When a user in ascribe-xr triggers a specimen load, there is currently no visible feedback while the load is in flight. The server-side handler (`POST /api/specimens/{id}/data` or `POST /api/processing/invoke`) blocks until the full result is computed and serialized, then returns. For most dynamic specimens this is under a second and imperceptible. For the AI Agent specimen (`generate_with_agent` in `ascribe_link/agent_generator.py`) it can take minutes of silent waiting. The agent already produces a rich stream of internal messages (text blocks, tool calls, tool results) via `ClaudeSDKClient.receive_response()`, but those messages are only logged server-side — they never reach the client.

Goal: surface compute-time progress from ascribe-link to ascribe-xr as human-readable text messages, and do so in a way that works across multiplayer rooms (all clients in a room see the same progress).

## Scope

**In scope**
- A job-based API on ascribe-link for starting dynamic specimen computation, polling progress messages, and fetching the final result.
- A `ProgressReporter` injected into dynamic specimen functions via signature detection; the AI agent wires the SDK message stream into it.
- GDScript client changes in ascribe-xr to drive the new API and broadcast progress to peers in a room via Godot multiplayer RPC.
- Bounded per-job message history so late joiners and brief reconnects don't miss context.

**Out of scope**
- Numeric progress values across the wire. The XR client computes bytes-progress locally from `HTTPRequest.get_body_size()` / `get_downloaded_bytes()` during the `/result` download phase. Compute-phase progress is text-only.
- Progress for static specimens. Static specimens keep `GET /api/specimens/{id}/data`; Godot's native HTTP byte-progress covers them.
- Persistence of jobs across server restarts.
- Authorization on job IDs beyond unguessable UUIDs.
- Back-pressure / concurrency limits on dynamic jobs (the agent is the only realistically expensive case; add a semaphore if needed).

## API Surface

Four endpoints under a new `/api/jobs` resource.

```
POST   /api/specimens/{specimen_id}/start
         body: {params: {...}, room_id: "ascribe"}
         → 200 {job_id: "uuid", status: "running" | "done"}
               status "done" is returned immediately on cache hit

GET    /api/jobs/{job_id}/progress?since={seq}
         → 200 {status: "running" | "done" | "error",
                messages: [{seq: int, text: str, ts: float}, ...],
                error: str | null}
               messages only includes entries with seq > since

GET    /api/jobs/{job_id}/result
         → 200 <mesh/volume/image/point_cloud JSON>  when status=done
         → 409 Conflict                               when status=running
         → 410 Gone                                   when status=error
         → 404 Not Found                              when job_id unknown or expired

DELETE /api/jobs/{job_id}
         → 204 No Content  (best-effort cancellation via asyncio.Task.cancel())
```

Design notes:
- `job_id` is server-generated (UUID4).
- `since=seq` lets poll responses return only new messages, bounded wire size.
- `seq` is a monotonic integer starting at 0 per job.
- `status: "done"` returned by `/start` on cache hit allows the client to skip `/progress` polling and call `/result` directly.
- Static specimens stay on `GET /api/specimens/{id}/data`. Only specimens with `function_name` set route through `/start`.
- The old `POST /api/specimens/{id}/data` and `POST /api/processing/invoke` endpoints are removed once ascribe-xr has migrated.

## Server Architecture

### Job registry

In-memory, ephemeral, no persistence.

```python
@dataclass
class Job:
    id: str                            # UUID4
    specimen_id: str
    params: dict
    room_id: str
    status: Literal["running", "done", "error"]
    messages: deque[ProgressMessage]   # maxlen=50
    next_seq: int
    result: ProcessingResult | None
    error: str | None
    task: asyncio.Task
    created_at: float
    finished_at: float | None


@dataclass(frozen=True)
class ProgressMessage:
    seq: int
    text: str
    ts: float  # epoch seconds
```

`JobRegistry` is a plain dict guarded by an `asyncio.Lock` for mutations, touched only from the event loop. The per-job `messages` deque has a single appender (the job's own task) and multiple readers (poll handlers); Python's `collections.deque` is safe for this pattern without additional locking, and poll handlers snapshot `(next_seq, list(messages))` atomically before serializing.

### Lifecycle

1. **Create.** The `POST /start` handler:
   1. Resolves the specimen (filesystem / registry / federated) and validates that it is dynamic.
   2. Checks `RoomResultCache` (existing cache keyed by `(room_id, function_name, params)`); on hit, creates a `Job` with `status="done"`, preloaded `result`, and a single `"cache hit"` message; returns `{job_id, status: "done"}`.
   3. Otherwise creates a `Job` with `status="running"`, registers it (so `/progress` polls can never race), spawns `asyncio.create_task(_run_job(...))`, and returns `{job_id, status: "running"}`.
2. **Run.** `_run_job` invokes `function_registry.invoke_async(func_name, [], params, reporter=reporter)`. Before invocation, it appends a `"Starting {specimen_id}"` bracket message; on return it appends `"Finished in {t:.2f}s"`.
3. **Finish.** On return, write `result`, set `status="done"`, set `finished_at`. On exception, set `status="error"` with `error=str(exc)` and append `"Error: {exc}"`. On `asyncio.CancelledError`, set `status="error"` with `error="cancelled"` and append `"Cancelled"`.
4. **TTL / GC.** Completed jobs stay in the registry for 5 minutes. A background sweeper (asyncio task started at app startup) deletes expired jobs every 30 seconds. After deletion, `/progress` and `/result` return 404.

### ProgressReporter plumbing

```python
class ProgressReporter:
    """Passed into dynamic specimen functions. Appends to the job's message deque."""
    def report(self, text: str) -> None: ...
```

A concrete `JobReporter(job)` appends to `job.messages` and increments `job.next_seq`. The default `ProgressReporter()` with no job is a no-op, which lets callers invoke these functions directly from tests or a REPL.

Functions opt in by declaring a parameter annotated as `ProgressReporter`:

```python
@registry.register(display_name="CT Isosurface")
async def ct_isosurface(
    threshold: Annotated[float, Range(0, 255)] = 100.0,
    reporter: ProgressReporter = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    reporter.report("Loading volume...")
    ...
    reporter.report(f"Running marching cubes at threshold={threshold}")
    ...
```

Injection rules (in `FunctionRegistry.invoke_async`, which gains a new `reporter: ProgressReporter | None = None` kwarg):
- Inspect `inspect.signature(func).parameters`.
- For any parameter whose annotation is `ProgressReporter`, inject the caller-supplied reporter (or a no-op if none) under that parameter name.
- Functions without a `ProgressReporter` parameter are unchanged; the reporter kwarg is simply ignored.

Schema filter (in `create_schema`): skip any parameter whose annotation is `ProgressReporter`, so these do not appear in the XR parameter UI and cannot be set by clients.

Rationale for parameter injection over a `ContextVar`: explicit dependencies, trivial to mock in unit tests, and mirrors Litestar's own DI pattern.

### AI agent integration

`generate_with_agent` in `ascribe_link/agent_generator.py` is modified to accept a `reporter: ProgressReporter | None = None` parameter (default no-op). The wrapper returned by `create_agent_function(...)` now includes `reporter` in its signature so registry injection wires it automatically.

Inside the existing `process_responses()` loop:

| SDK event | Reporter message |
|---|---|
| `AssistantMessage` + `TextBlock` | First non-empty line, trimmed to 200 chars |
| `AssistantMessage` + `ToolUseBlock` | `Tool: {name}` (e.g., `Tool: Bash`, `Tool: submit_mesh_file`) |
| `ToolResultBlock` with error | `Tool error: {brief reason}` |
| `submit_mesh` / `submit_mesh_file` / `submit_volume` success | `Mesh submitted` / `Volume submitted` |
| Timeout | `_run_job` catches `TimeoutError` and writes `Error: Agent timed out after {timeout}s` |

Deliberately not emitted:
- Full text blocks (can be paragraphs of reasoning; too noisy for a loading panel).
- Bash command contents (noisy; `Tool: Bash` alone is enough signal).
- Tool input/output payloads (large payloads would chew through the 50-message buffer and evict earlier context).

### Federation

When a specimen ID contains `:` (federated), the relay's `_run_job`:
1. Calls the worker's `POST /start` via `federation_hub.proxy_request`.
2. Records the mapping `relay_job_id → (worker_id, worker_job_id)` on the relay's `Job`.
3. Relay's `/progress` and `/result` handlers proxy straight to the worker.

The bounded message history lives on the worker; the relay is a stateless pass-through for federated jobs.

## Godot Client Architecture

### Files affected

| File | Change |
|---|---|
| `scripts/DataSources/ascribe_link_client.gd` | Replace `invoke_processing_function()` with `run_job()`; add `job_progress`, `job_complete`, `job_error` signals. |
| `scripts/DataSources/http_source.gd` | Route through `run_job()` instead of `/invoke`. |
| `scripts/Specimen/dynamic_mesh_specimen.gd` | Wire progress signals to RPC broadcast and UI. |
| `scenes/UI/LoadingLayer` | Add a `RichTextLabel` ("MessageLog") and a `ProgressBar` bound to `/result` download bytes. |

### Client flow

```gdscript
# In AscribeLinkClient
signal job_progress(text: String)
signal job_complete(result: Dictionary)
signal job_error(error: String)

func run_job(specimen_id: String, params: Dictionary, room_id: String = "ascribe") -> void
```

`run_job()`:
1. POST `/start` with `{params, room_id}` → `{job_id, status}`. If `status == "done"` (cache hit), skip to step 4.
2. Start a `Timer` (0.5s interval) calling `GET /api/jobs/{job_id}/progress?since={last_seq}`. For each new message, emit `job_progress(text)` and update `last_seq`.
3. When poll sees `status == "done"`, stop timer and go to step 4. If `"error"`, emit `job_error` and stop.
4. `GET /api/jobs/{job_id}/result` → payload. Emit `job_complete(result)`. During this phase, `LoadingLayer` can show a real progress bar from `HTTPRequest.get_downloaded_bytes()` / `get_body_size()`.

### Authority vs. peers

Only the multiplayer authority (`multiplayer.get_unique_id() == 1`) calls `run_job()`. It broadcasts progress to peers via RPC and they render it in their own `LoadingLayer`:

```gdscript
# Authority
func _load_dynamic(specimen_id: String, params: Dictionary) -> void:
    _link_client.job_progress.connect(_on_progress)
    _link_client.job_complete.connect(_on_complete)
    _link_client.run_job(specimen_id, params, _room_id)

func _on_progress(text: String) -> void:
    _message_log.append(text)   # cap at 50
    _rpc_progress.rpc(text)

func _on_complete(result: Dictionary) -> void:
    _handle_result(result)       # existing mesh dispatch + RPC mesh sync
    _rpc_job_done.rpc()

# Peers
@rpc("authority", "call_remote", "reliable")
func _rpc_progress(text: String) -> void:
    _append_message_to_ui(text)

@rpc("authority", "call_remote", "reliable")
func _rpc_job_done() -> void:
    ui_instance.get_node("LoadingLayer").hide()
```

Peers never poll ascribe-link for progress; all progress arrives through Godot RPC from the authority.

### Late-joiner handling

Authority tracks `var _active_job_id: String` and `var _message_log: Array[String]` (capped at 50, matching the server buffer). On `multiplayer.peer_connected`:

```gdscript
func _on_peer_connected(peer_id: int) -> void:
    if not is_multiplayer_authority() or _active_job_id.is_empty():
        return
    _rpc_sync_state.rpc_id(peer_id, _active_job_id, _message_log)
```

The newly-joined peer renders the backlog immediately and keeps receiving live messages.

### Cancellation

If the user backs out of a specimen before it finishes, the authority calls `DELETE /api/jobs/{job_id}` and RPCs `_rpc_job_done` to peers. Server-side `task.cancel()` aborts the underlying `asyncio.Task` (including the agent's `ClaudeSDKClient` context). v1 feature; can be gated behind a UI control.

## Error Handling

| Failure | Server | XR |
|---|---|---|
| Function raises | `status="error"`, `error=str(e)`, terminal `"Error: …"` message | `job_error` → `load_error` signal → LoadingLayer shows error and a close affordance |
| Agent timeout | `TimeoutError` wrapped as error | Same as above |
| DELETE during run | `task.cancel()`; `status="error"`, `error="cancelled"` | Authority already tearing down; peers see `_rpc_job_done` |
| Poll 404 (job expired past TTL) | — | Stop polling; if authority and still mid-load, try one fresh `/start`, else surface error |
| Poll network fail | — | Retry up to 3 times with backoff; then `job_error` |
| `/result` fails on done job | Shouldn't happen unless TTL expired mid-handoff | Retry once; else `job_error` |
| Server restart mid-job | All jobs dropped from registry; next poll 404s | Same as "poll 404" |

Message eviction: the 50-message deque will never drop unseen messages for a single active polling client (poll is 500ms; agent messages arrive at human-reading pace). Peers catching up via `_rpc_sync_state` get whatever the authority has in its local 50-cap log.

## Testing

Server-side unit tests (in `ascribe-link/tests/`):
- `test_job_registry.py` — lifecycle transitions, TTL sweep, cancellation, message deque bound, cache-hit shortcut.
- `test_reporter_injection.py` — signature-driven detection, schema filter, no-op fallback when no job.
- `test_specimens_api.py` — `/start` creates a job; `/progress?since=N` returns only new messages; `/result` state machine (404/409/410/200).
- `test_agent_progress.py` — mock `ClaudeSDKClient.receive_response()` to yield a known sequence of SDK messages; assert the reporter receives the expected summaries. No real agent calls.
- `test_federation_jobs.py` — relay proxies `/start`, `/progress`, `/result` to a mocked worker.

End-to-end smoke test (scripted, separate from unit tests): start a job against the real AI agent with a trivial prompt (`"make a unit sphere"`), poll to completion, fetch the result, assert mesh validity. This is the test that catches integration drift across the pipeline.

Godot-side testing is manual against a local ascribe-link run for v1. A `gdUnit4` test for `AscribeLinkClient.run_job` against a mocked HTTP server is a nice-to-have follow-up.

## Rollout

- **PR 1 (ascribe-link):** job registry, new endpoints, `ProgressReporter` injection, agent wiring, server tests. Old `POST /data` and `POST /invoke` remain as thin shims that auto-run the job pattern server-side and return a normal blocking response — so the server PR is backwards-compatible.
- **PR 2 (ascribe-xr):** GDScript client migration to `run_job()`, RPC plumbing, UI updates.
- **PR 3 (ascribe-link, follow-up):** remove the deprecated `POST /data` and `POST /invoke` shims.

## Open questions

None blocking v1. Deferred decisions:
- Whether to add per-specimen concurrency limits if the agent becomes a DoS vector in shared deployments.
- Whether to surface tool input/output excerpts (truncated) in progress messages — useful for debugging, noisy for end users. Defer until we have real usage feedback.
