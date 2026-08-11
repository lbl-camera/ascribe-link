# AI agent generation

With the agent enabled, the catalog gains a specimen whose parameter is a
sentence. The client sends a prompt; a Claude agent writes and runs code to
produce the mesh or volume, reporting progress as it goes.

```bash
pip install "ascribe-link[agent]"
ascribe-link --enable-agent
```

Or in code:

```python
app = create_app(
    enable_agent=True,
    agent_model="claude-opus-4-8",
    agent_timeout=300.0,
)
```

This registers a function named `ai_generate`, tagged `ai`, `generative`,
`dynamic`:

```json
{
  "function_name": "ai_generate",
  "kwargs": {"prompt": "Create a DNA double helix mesh with 10 base pairs"}
}
```

Because generation takes minutes rather than milliseconds, drive it through
`POST /api/specimens/{id}/start` and poll the job ({doc}`jobs`) — the agent's
tool calls and intermediate steps arrive as progress messages.

The agent submits its output through tools rather than returning a value:
mesh or volume, inline or from a file it wrote (`.npy` preferred for volumes,
with `.npz` and a JSON envelope also accepted). Whatever it submits is wrapped
into a `MeshResult` or `VolumeResult` and cached like any other result.

## Sandboxing

Code written by an agent is untrusted code. When
[Firejail](https://firejail.wordpress.com/) is on the PATH, execution is
confined: private tmpfs filesystem, **no network**, dropped capabilities,
seccomp filtering, and memory/CPU/wall-clock limits (4 GB, 5 min CPU, 10 min
wall by default, via `SandboxConfig`).

Firejail is Linux-only. Elsewhere — including Windows — the server logs
`sandbox=disabled (firejail not found)` at startup and runs the code
unconfined. Read that line before enabling the agent on a machine you care
about, and prefer a Linux host with Firejail installed for anything exposed
beyond your own workstation.

`allow_paths` on `SandboxConfig` opens specific read-only paths into the
sandbox — the way to let an agent reach a dataset without opening the
filesystem generally.

## Timeouts

`--agent-timeout` bounds a generation run; exceeding it fails the job with a
timeout error. The agent also runs in a subprocess, so a crash in agent code
surfaces as a failed job rather than taking the server down.
