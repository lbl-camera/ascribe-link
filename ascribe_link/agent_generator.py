"""AI-powered mesh generation using Claude Agent SDK.

Provides an agent that can generate 3D meshes from natural language prompts,
optionally processing input data files.

Code execution is sandboxed via Firejail when available, providing:
- Filesystem isolation
- Network isolation
- Resource limits (memory, CPU time)
- Capability dropping and seccomp filtering
"""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Annotated

from ascribe_link.gen_timing import gt_mark
from ascribe_link.progress import ProgressReporter
from ascribe_link.sandbox import (
    SandboxConfig,
    is_firejail_available,
    build_firejail_command,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Agent event emission
# ---------------------------------------------------------------------------


def _emit_agent_events(msg: Any, reporter: ProgressReporter) -> None:
    """Translate an SDK message to one or more reporter.report() calls.

    Called from the receive_response() loop. Kept as a module-level function
    so it can be unit-tested with mocked SDK types without spinning up a
    real ClaudeSDKClient.

    Handles the full set of SDK message types:
    - AssistantMessage: text blocks, tool-use blocks (ThinkingBlock skipped)
    - TaskProgressMessage: agent progress updates with description
    - TaskNotificationMessage: task completed/failed summaries
    - ToolResultBlock: only surfaces errors
    - SessionMessage: extracts nested AssistantMessage content
    - SystemMessage, UserMessage, etc.: silently ignored
    """
    # Import lazily so the module still imports when claude_agent_sdk
    # isn't installed (agent is an optional extra).
    from claude_agent_sdk import (
        AssistantMessage,
        TextBlock,
        ToolResultBlock,
        TaskProgressMessage,
        TaskNotificationMessage,
        SessionMessage,
    )

    if isinstance(msg, AssistantMessage):
        for block in msg.content:
            if isinstance(block, TextBlock):
                first_line = (block.text or "").strip().splitlines()
                if first_line:
                    reporter.report(first_line[0][:200])
            elif hasattr(block, "name"):
                # ToolUseBlock — report just the tool name.
                reporter.report(f"Tool: {block.name}")
            # ThinkingBlock — skip (internal reasoning, not user-facing)

    elif isinstance(msg, TaskProgressMessage):
        desc = getattr(msg, "description", "")
        tool = getattr(msg, "last_tool_name", "")
        if desc:
            reporter.report(desc[:200])
        elif tool:
            reporter.report(f"Tool: {tool}")

    elif isinstance(msg, TaskNotificationMessage):
        status = getattr(msg, "status", "")
        summary = getattr(msg, "summary", "")
        if summary:
            reporter.report(summary[:200])
        elif status:
            reporter.report(f"Task {status}")

    elif isinstance(msg, SessionMessage):
        # SessionMessage wraps a full message; extract content if it's
        # an assistant message with blocks we can surface.
        inner = getattr(msg, "message", None)
        if inner and hasattr(inner, "content"):
            for block in inner.content:
                if isinstance(block, TextBlock):
                    first_line = (block.text or "").strip().splitlines()
                    if first_line:
                        reporter.report(first_line[0][:200])
                elif hasattr(block, "name"):
                    reporter.report(f"Tool: {block.name}")

    elif isinstance(msg, ToolResultBlock):
        # Only surface errors; successful tool results are noisy.
        is_error = getattr(msg, "is_error", False)
        if is_error:
            content = getattr(msg, "content", None)
            summary = str(content)[:120] if content else "unknown"
            reporter.report(f"Tool error: {summary}")


# ---------------------------------------------------------------------------
# Mesh Generation Skill (system prompt for the agent)
# ---------------------------------------------------------------------------

MESH_GENERATION_SKILL = """# 3D Data Generation Assistant

Generate 3D meshes for Ascribe-XR.

IMPORTANT: Do NOT use ToolSearch or any tool-listing commands.

## Known Environment Issues (handle upfront, do not re-diagnose)

This environment inherits PyCharm helper paths that inject a broken
`sitecustomize` (it reloads numpy, corrupting it). Start EVERY Python
script you write with this preamble — do not wait for an ImportError to
discover it:

```python
import sys
sys.path[:] = [p for p in sys.path if "pycharm" not in p.lower()]
sys.modules.pop("sitecustomize", None)
```

Also known and harmless: skimage import may print a matplotlib traceback.
It is an optional-dependency check that skimage swallows — if your script
printed its expected output, processing succeeded. Ignore the traceback.

## How to Submit

Save mesh to JSON file with FLATTENED vertices, indices, and normals:

```python
import json

# vertices, indices, normals must all be FLAT lists:
# vertices: [x, y, z, x, y, z, ...] not [[x,y,z], [x,y,z], ...]
# normals:  [nx, ny, nz, nx, ny, nz, ...] not [[nx,ny,nz], ...]
# indices:  [i, j, k, ...] (already flat)

with open("mesh.json", "w") as f:
    json.dump({"vertices": vertices, "indices": indices, "normals": normals}, f)
print(f"Saved {len(vertices)//3} vertices")
```

Then call: `submit_mesh_file(file_path="mesh.json")`

## PyVista Example

```python
import pyvista as pv
from ascribe_link.mesh_utils import extract_mesh_data

mesh = pv.Sphere(radius=1.0)
vertices, indices, normals = extract_mesh_data(mesh)  # Includes normals
```

## Marching Cubes Example (isosurface from volume)

```python
from skimage.measure import marching_cubes

# volume is a 3D numpy array
verts, faces, norms, values = marching_cubes(volume, level=threshold)
vertices = verts.flatten().tolist()  # MUST flatten to [x, y, z, x, y, z, ...]
indices = faces.flatten().tolist()
normals = norms.flatten().tolist()   # MUST flatten to [nx, ny, nz, nx, ny, nz, ...]
```

## PyVista Primitives

```python
pv.Sphere(radius=1.0, center=(0, 0, 0))
pv.Box(bounds=(xmin, xmax, ymin, ymax, zmin, zmax))
pv.Cylinder(radius=0.5, height=2.0)
```

## extract_mesh_data

Triangulates and converts to Python lists. Always use it before submitting.

## How to Submit a Volume (volumetric / voxel data)

For ANY real-sized volume, save it to a NumPy `.npy` file and submit the file.
Do NOT base64-encode the volume yourself, and do NOT try to pass the volume
data inline — a real volume is far too large to fit in a tool-call argument.

```python
import numpy as np

# `volume` is your 3D array, shape [depth, height, width]
np.save("volume.npy", np.ascontiguousarray(volume))
print(f"Saved volume {volume.shape} ({volume.dtype})")
```

Then call: `submit_volume_file(file_path="volume.npy")`
(optionally pass `spacing=[sz, sy, sx]`).

ONLY for tiny volumes (e.g. a small synthetic test) may you submit inline with
`submit_volume` using this schema:

```json
{
  "shape": [depth, height, width],  // 3 integers
  "dtype": "float32",               // numpy dtype string
  "data": "base64...",              // base64-encoded raw bytes
  "spacing": [sz, sy, sx]           // optional, 3 numbers
}
```
"""


# ---------------------------------------------------------------------------
# Result holder for capturing mesh from tool call
# ---------------------------------------------------------------------------


@dataclass
class AgentResult:
    """Holds the result from the agent (mesh or volume)."""

    result_type: str | None = None  # "mesh" or "volume"
    # Mesh data
    vertices: list[list[float]] | None = None
    indices: list[int] | None = None
    normals: list[list[float]] | None = None
    # Volume data
    volume_shape: list[int] | None = None
    volume_dtype: str | None = None
    volume_data: str | None = None  # base64
    volume_spacing: list[float] | None = None
    # Status
    error: str | None = None
    submitted: bool = False


def _load_volume_array(full_path: Path):
    """Load a 3D volume array (and optional spacing) from a file on disk.

    This is the file-based counterpart to inline base64 submission: the agent
    writes the volume to disk and we read it back here, so it never has to
    inline tens-to-hundreds of MB of base64 into a tool-call argument.

    Kept module-level (like ``_emit_agent_events``) so it can be unit-tested
    without spinning up a real SDK client.

    Supported formats (by extension):
    - ``.npy``  : NumPy-native; carries shape and dtype directly (preferred).
    - ``.npz``  : NumPy archive; the first array is used.
    - ``.json`` : envelope ``{"shape", "dtype", "data"(base64), "spacing"?}``
      mirroring the inline ``submit_volume`` schema.

    Returns
    -------
    (array, spacing)
        ``array`` is a numpy ndarray; ``spacing`` is a list of 3 floats or None.

    Raises
    ------
    ValueError
        With a user-facing message if the file can't be read or is invalid.
        ``allow_pickle=False`` is enforced so a malicious .npy/.npz can't
        execute code inside the agent's working directory.
    """
    import numpy as np

    suffix = full_path.suffix.lower()

    if suffix == ".npy":
        try:
            arr = np.load(full_path, allow_pickle=False)
        except Exception as e:
            raise ValueError(f"could not load .npy file: {e}") from e
        return np.asarray(arr), None

    if suffix == ".npz":
        try:
            with np.load(full_path, allow_pickle=False) as npz:
                keys = list(npz.files)
                if not keys:
                    raise ValueError(".npz file contains no arrays")
                arr = np.asarray(npz[keys[0]])
        except ValueError:
            raise
        except Exception as e:
            raise ValueError(f"could not load .npz file: {e}") from e
        return arr, None

    if suffix == ".json":
        import base64
        import json

        try:
            with open(full_path, "r") as f:
                env = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"invalid JSON: {e}") from e
        shape = env.get("shape")
        dtype = env.get("dtype", "float32")
        data = env.get("data", "")
        spacing = env.get("spacing")
        if not shape or len(shape) != 3:
            raise ValueError("envelope 'shape' must be [depth, height, width]")
        if not data:
            raise ValueError("envelope 'data' is empty")
        try:
            raw = base64.b64decode(data)
            arr = np.frombuffer(raw, dtype=np.dtype(dtype)).reshape(shape)
        except Exception as e:
            raise ValueError(f"could not decode envelope data: {e}") from e
        return arr, spacing

    raise ValueError(
        f"unsupported volume file type '{suffix}'. Use .npy, .npz, or .json"
    )


# ---------------------------------------------------------------------------
# Agent-based mesh generation
# ---------------------------------------------------------------------------


async def generate_with_agent(
    prompt: str,
    file_path: str | None = None,
    model: str = "claude-opus-4-8",
    timeout: float = 3000.0,
    working_dir: str | None = None,
    sandbox: bool = True,
    sandbox_config: SandboxConfig | None = None,
    reporter: ProgressReporter | None = None,
) -> dict[str, Any]:
    """Generate data (mesh, volume, etc.) using an AI agent.

    Parameters
    ----------
    prompt : str
        Natural language description of what to generate.
    file_path : str, optional
        Path to an input data file for the agent to process.
    model : str
        Claude model to use.
    timeout : float
        Maximum time in seconds to wait for the agent.
    working_dir : str, optional
        Working directory for the agent. Defaults to a temp directory.
    sandbox : bool
        If True, wrap Bash commands in Firejail sandbox.
    sandbox_config : SandboxConfig, optional
        Configuration for sandbox. Uses defaults if None.

    Returns
    -------
    dict
        Result dictionary with 'type' field indicating the data type:
        - "mesh": contains vertices, indices
        - "volume": contains shape, dtype, data (base64), spacing

    Raises
    ------
    ValueError
        If the agent fails to produce valid data.
    TimeoutError
        If the agent exceeds the timeout.
    """
    from claude_agent_sdk import (
        ClaudeAgentOptions,
        ClaudeSDKClient,
        AssistantMessage,
        ResultMessage,
        TextBlock,
        ToolResultBlock,
        HookMatcher,
        tool,
        create_sdk_mcp_server,
    )

    result = AgentResult()
    reporter = reporter or ProgressReporter()
    sandbox_config = sandbox_config or SandboxConfig()

    # Check sandbox availability
    use_sandbox = sandbox and is_firejail_available()
    if sandbox and not use_sandbox:
        logger.warning("Firejail not available, running without sandbox")

    # Define the submit_mesh tool with explicit JSON Schema
    submit_mesh_schema = {
        "type": "object",
        "properties": {
            "vertices": {
                "type": "array",
                "description": "List of [x, y, z] coordinates for each vertex",
                "items": {
                    "type": "array",
                    "items": {"type": "number"},
                    "minItems": 3,
                    "maxItems": 3,
                },
            },
            "indices": {
                "type": "array",
                "description": "Flat list of vertex indices (every 3 = one triangle)",
                "items": {"type": "integer"},
            },
        },
        "required": ["vertices", "indices"],
    }

    @tool(
        "submit_mesh",
        "Submit a triangular mesh. vertices is an array of [x,y,z] points. indices is a flat array of integers (every 3 indices form one triangle face).",
        submit_mesh_schema,
    )
    async def submit_mesh(args: dict) -> dict:
        """Capture the mesh submitted by the agent."""
        gt_mark("submit_mesh: tool entry")
        vertices = args.get("vertices", [])
        indices = args.get("indices", [])

        # Validate
        if not vertices:
            return {
                "content": [{"type": "text", "text": "Error: vertices list is empty"}]
            }
        if not indices:
            return {
                "content": [{"type": "text", "text": "Error: indices list is empty"}]
            }

        # Check vertex format
        if not all(isinstance(v, (list, tuple)) and len(v) == 3 for v in vertices):
            return {
                "content": [
                    {"type": "text", "text": "Error: each vertex must be [x, y, z]"}
                ]
            }

        # Check for non-finite values
        import math

        for v in vertices:
            if not all(isinstance(c, (int, float)) and math.isfinite(c) for c in v):
                return {
                    "content": [
                        {
                            "type": "text",
                            "text": "Error: vertices contain non-finite values",
                        }
                    ]
                }

        # Check indices
        if not all(isinstance(i, int) and 0 <= i < len(vertices) for i in indices):
            return {
                "content": [
                    {
                        "type": "text",
                        "text": f"Error: indices must be integers in range [0, {len(vertices)})",
                    }
                ]
            }

        if len(indices) % 3 != 0:
            return {
                "content": [
                    {
                        "type": "text",
                        "text": "Error: indices length must be divisible by 3 (triangles)",
                    }
                ]
            }

        result.result_type = "mesh"
        result.vertices = vertices
        result.indices = indices
        result.submitted = True
        gt_mark("submit_mesh: tool done")

        return {
            "content": [
                {
                    "type": "text",
                    "text": f"Mesh submitted successfully: {len(vertices)} vertices, {len(indices) // 3} triangles",
                }
            ]
        }

    # Define submit_mesh_file tool for large meshes
    submit_mesh_file_schema = {
        "type": "object",
        "properties": {
            "file_path": {
                "type": "string",
                "description": "Path to JSON file with 'vertices' and 'indices' keys",
            },
        },
        "required": ["file_path"],
    }

    @tool(
        "submit_mesh_file",
        "Submit a mesh from a JSON file. The file must have 'vertices' and 'indices', and optionally 'normals'.",
        submit_mesh_file_schema,
    )
    async def submit_mesh_file(args: dict) -> dict:
        """Load mesh from JSON file and submit it."""
        gt_mark("submit_mesh_file: tool entry")
        import json
        import math

        file_path = args.get("file_path", "")
        if not file_path:
            return {
                "content": [{"type": "text", "text": "Error: file_path is required"}]
            }

        # Resolve relative to working directory
        full_path = working_dir_path / file_path
        if not full_path.exists():
            return {
                "content": [
                    {"type": "text", "text": f"Error: file not found: {file_path}"}
                ]
            }

        try:
            with open(full_path, "r") as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            return {"content": [{"type": "text", "text": f"Error: invalid JSON: {e}"}]}

        vertices = data.get("vertices", [])
        indices = data.get("indices", [])
        normals = data.get("normals")  # Optional

        if not vertices:
            return {
                "content": [{"type": "text", "text": "Error: vertices list is empty"}]
            }
        if not indices:
            return {
                "content": [{"type": "text", "text": "Error: indices list is empty"}]
            }

        # Basic validation (skip full validation for large meshes for performance)
        if len(indices) % 3 != 0:
            return {
                "content": [
                    {
                        "type": "text",
                        "text": "Error: indices length must be divisible by 3",
                    }
                ]
            }

        result.result_type = "mesh"
        result.vertices = vertices
        result.indices = indices
        result.normals = normals
        result.submitted = True
        gt_mark("submit_mesh_file: tool done")

        normals_info = f", {len(normals)} normals" if normals else ""
        return {
            "content": [
                {
                    "type": "text",
                    "text": f"Mesh submitted from file: {len(vertices)} vertices, {len(indices) // 3} triangles{normals_info}",
                }
            ]
        }

    # Define the submit_volume tool with explicit JSON Schema
    submit_volume_schema = {
        "type": "object",
        "properties": {
            "shape": {
                "type": "array",
                "description": "Volume dimensions [depth, height, width]",
                "items": {"type": "integer"},
                "minItems": 3,
                "maxItems": 3,
            },
            "dtype": {
                "type": "string",
                "description": "NumPy dtype string, e.g. 'float32', 'uint8'",
            },
            "data": {
                "type": "string",
                "description": "Base64-encoded raw bytes of the volume data",
            },
            "spacing": {
                "type": "array",
                "description": "Optional voxel spacing [sz, sy, sx]",
                "items": {"type": "number"},
                "minItems": 3,
                "maxItems": 3,
            },
        },
        "required": ["shape", "dtype", "data"],
    }

    @tool(
        "submit_volume",
        "Submit volumetric (voxel) data. shape is [depth, height, width], dtype is a numpy dtype string, data is base64-encoded bytes.",
        submit_volume_schema,
    )
    async def submit_volume(args: dict) -> dict:
        """Capture the volume submitted by the agent."""
        gt_mark("submit_volume: tool entry")
        shape = args.get("shape", [])
        dtype = args.get("dtype", "float32")
        data = args.get("data", "")
        spacing = args.get("spacing")

        # Validate
        if not shape or len(shape) != 3:
            return {
                "content": [
                    {
                        "type": "text",
                        "text": "Error: shape must be [depth, height, width]",
                    }
                ]
            }
        if not data:
            return {"content": [{"type": "text", "text": "Error: data is empty"}]}

        # Validate base64
        import base64

        try:
            decoded = base64.b64decode(data)
            import numpy as np

            expected_size = np.prod(shape) * np.dtype(dtype).itemsize
            if len(decoded) != expected_size:
                return {
                    "content": [
                        {
                            "type": "text",
                            "text": f"Error: data size mismatch. Expected {expected_size} bytes, got {len(decoded)}",
                        }
                    ]
                }
        except Exception as e:
            return {"content": [{"type": "text", "text": f"Error decoding data: {e}"}]}

        result.result_type = "volume"
        result.volume_shape = shape
        result.volume_dtype = dtype
        result.volume_data = data
        result.volume_spacing = spacing
        result.submitted = True
        gt_mark("submit_volume: tool done")

        total_voxels = shape[0] * shape[1] * shape[2]
        return {
            "content": [
                {
                    "type": "text",
                    "text": f"Volume submitted successfully: {shape} ({total_voxels:,} voxels, {dtype})",
                }
            ]
        }

    # Define submit_volume_file tool for real-sized volumes (avoids inlining
    # tens-to-hundreds of MB of base64 into a tool-call argument).
    submit_volume_file_schema = {
        "type": "object",
        "properties": {
            "file_path": {
                "type": "string",
                "description": "Path to a volume file: .npy (preferred), .npz, or a .json envelope with shape/dtype/data(base64)",
            },
            "spacing": {
                "type": "array",
                "description": "Optional voxel spacing [sz, sy, sx]; overrides any spacing in the file",
                "items": {"type": "number"},
                "minItems": 3,
                "maxItems": 3,
            },
        },
        "required": ["file_path"],
    }

    @tool(
        "submit_volume_file",
        "Submit volumetric data from a file on disk. Use this instead of submit_volume for any real-sized volume. Preferred format is a NumPy .npy file written with np.save('volume.npy', arr); .npz and .json envelopes are also accepted.",
        submit_volume_file_schema,
    )
    async def submit_volume_file(args: dict) -> dict:
        """Load a volume from a file and submit it (no inline base64 needed)."""
        gt_mark("submit_volume_file: tool entry")
        from ascribe_link.models import VolumeResult

        file_path = args.get("file_path", "")
        if not file_path:
            return {
                "content": [{"type": "text", "text": "Error: file_path is required"}]
            }

        # Resolve relative to working dir; fall back to an absolute path the
        # agent may have written elsewhere.
        full_path = working_dir_path / file_path
        if not full_path.exists():
            full_path = Path(file_path)
        if not full_path.exists():
            return {
                "content": [
                    {"type": "text", "text": f"Error: file not found: {file_path}"}
                ]
            }

        try:
            arr, file_spacing = _load_volume_array(full_path)
        except ValueError as e:
            return {"content": [{"type": "text", "text": f"Error: {e}"}]}

        if arr.ndim != 3:
            return {
                "content": [
                    {
                        "type": "text",
                        "text": f"Error: volume must be 3D, got shape {list(arr.shape)}",
                    }
                ]
            }

        spacing = args.get("spacing") or file_spacing
        # from_numpy handles C-contiguity and base64 encoding server-side.
        vr = VolumeResult.from_numpy(arr, spacing=spacing)

        result.result_type = "volume"
        result.volume_shape = vr.shape
        result.volume_dtype = vr.dtype
        result.volume_data = vr.data
        result.volume_spacing = spacing
        result.submitted = True
        gt_mark("submit_volume_file: tool done")

        total_voxels = int(arr.shape[0] * arr.shape[1] * arr.shape[2])
        return {
            "content": [
                {
                    "type": "text",
                    "text": f"Volume submitted from file: {vr.shape} ({total_voxels:,} voxels, {vr.dtype})",
                }
            ]
        }

    # Create MCP server with our tools
    mesh_server = create_sdk_mcp_server(
        name="mesh-tools",
        version="1.0.0",
        tools=[submit_mesh, submit_mesh_file, submit_volume, submit_volume_file],
    )

    # Set up working directory
    if working_dir is None:
        working_dir = tempfile.mkdtemp(prefix="ascribe_agent_")

    working_dir_path = Path(working_dir)

    # Build the user prompt
    user_prompt = prompt
    if file_path:
        # Copy input file to working directory
        file_path_obj = Path(file_path)
        if file_path_obj.exists():
            dest = working_dir_path / file_path_obj.name
            shutil.copy2(file_path_obj, dest)
            user_prompt = f"Input file: {dest.name}\n\n{prompt}"
        else:
            user_prompt = f"Input file (not found): {file_path}\n\n{prompt}"

    # Create a hook that wraps Bash commands in Firejail
    async def sandbox_bash_hook(
        input_data: dict, tool_use_id: str, context: dict
    ) -> dict:
        """Intercept Bash commands and wrap them in Firejail."""
        tool_name = input_data.get("tool_name", "")
        tool_input = input_data.get("tool_input", {})

        if tool_name != "Bash" or not use_sandbox:
            return {}  # Allow through unchanged

        command = tool_input.get("command", "")
        if not command:
            return {}

        # Build Firejail-wrapped command
        # The command runs in the working directory which Firejail will isolate
        firejail_cmd = build_firejail_command(
            command=["bash", "-c", command],
            working_dir=working_dir_path,
            config=sandbox_config,
        )

        # Replace the command with the sandboxed version
        sandboxed_command = " ".join(
            f'"{arg}"' if " " in arg else arg for arg in firejail_cmd
        )

        logger.debug("Sandboxing Bash command: %s -> firejail ...", command[:50])

        return {
            "hookSpecificOutput": {
                "hookEventName": "PreToolUse",
                "toolInput": {
                    "command": sandboxed_command,
                },
            }
        }

    # Configure hooks
    hooks = {}
    if use_sandbox:
        hooks["PreToolUse"] = [
            HookMatcher(matcher="Bash", hooks=[sandbox_bash_hook]),
        ]
        logger.info("Sandbox enabled for Bash commands (Firejail)")

    # Strip PyCharm helper paths from PYTHONPATH so the agent's Python
    # never imports PyCharm's sitecustomize (it reloads numpy, corrupting it).
    agent_env = {}
    pythonpath = os.environ.get("PYTHONPATH", "")
    if pythonpath:
        cleaned = os.pathsep.join(
            p for p in pythonpath.split(os.pathsep)
            if p and "pycharm" not in p.lower()
        )
        if cleaned != pythonpath:
            agent_env["PYTHONPATH"] = cleaned
            logger.info("Removed PyCharm helper paths from agent PYTHONPATH")

    # Configure agent options
    options = ClaudeAgentOptions(
        cli_path=os.environ.get("ASCRIBE_LINK_CLAUDE_CLI") or None,
        env=agent_env,
        model=model,
        system_prompt=MESH_GENERATION_SKILL,
        cwd=working_dir,
        mcp_servers={"mesh": mesh_server},
        allowed_tools=[
            "Read",
            "Write",
            "Edit",
            "Bash",
            "mcp__mesh__submit_mesh",
            "mcp__mesh__submit_mesh_file",
            "mcp__mesh__submit_volume",
            "mcp__mesh__submit_volume_file",
        ],
        disallowed_tools=[
            "ToolSearch",  # Schema is already in the prompt
        ],
        permission_mode="acceptEdits",
        # acceptEdits only auto-approves edits under cwd; include the repo so
        # the agent can modify its own source without a permission prompt.
        add_dirs=[str(Path(__file__).resolve().parent.parent)],
        max_turns=25,  # Encourage efficiency
        hooks=hooks if hooks else None,
    )

    logger.info("Starting generation agent: %s", prompt[:100])
    gt_mark("agent: starting (spawning Claude CLI)")

    try:
        async with ClaudeSDKClient(options=options) as client:
            logger.info("ClaudeSDKClient connected, sending query...")
            gt_mark("agent: SDK client connected")
            await client.query(user_prompt)
            logger.info("Query sent, waiting for responses...")
            gt_mark("agent: query sent")

            # Process responses with timeout
            async def process_responses():
                msg_count = 0
                async for msg in client.receive_response():
                    msg_count += 1
                    msg_type = type(msg).__name__
                    if msg_count == 1:
                        gt_mark("agent: first SDK message received")
                    logger.info("Received message #%d: %s", msg_count, msg_type)
                    try:
                        _emit_agent_events(msg, reporter)
                    except Exception as _emit_err:
                        # Never let progress emission break the agent loop.
                        logger.warning("_emit_agent_events failed: %s", _emit_err, exc_info=True)

                    if isinstance(msg, AssistantMessage):
                        for block in msg.content:
                            if isinstance(block, TextBlock):
                                logger.info("Agent text: %s", block.text[:500])
                            elif hasattr(block, "name"):
                                # ToolUseBlock - log the tool being called
                                tool_name = getattr(block, "name", "unknown")
                                tool_input = getattr(block, "input", {})
                                logger.info(
                                    "Agent tool call: %s, input_keys=%s",
                                    tool_name,
                                    list(tool_input.keys())
                                    if isinstance(tool_input, dict)
                                    else "N/A",
                                )
                    elif isinstance(msg, ResultMessage):
                        logger.info(
                            "Result message received: %s",
                            getattr(msg, "result", "no result attr"),
                        )

                    # Check if we got a result
                    if result.submitted:
                        reporter.report(
                            f"{result.result_type.capitalize()} submitted"
                            if result.result_type
                            else "Result submitted"
                        )
                        logger.info("Result submitted, exiting response loop")
                        gt_mark("agent: submission observed, exiting response loop")
                        return

                logger.warning(
                    "Response loop ended without submission (processed %d messages)",
                    msg_count,
                )

            try:
                await asyncio.wait_for(process_responses(), timeout=timeout)
            except asyncio.TimeoutError:
                raise TimeoutError(f"Agent timed out after {timeout}s")

    except Exception as e:
        logger.error("Agent error: %s", e, exc_info=True)
        raise

    if not result.submitted:
        raise ValueError(
            "Agent did not submit any data. It may have encountered an error."
        )

    # Build result dictionary based on type
    if result.result_type == "mesh":
        logger.info(
            "Mesh generated: %d vertices, %d triangles, normals=%s",
            len(result.vertices),
            len(result.indices) // 3,
            len(result.normals) if result.normals else 0,
        )
        output = {
            "type": "mesh",
            "vertices": result.vertices,
            "indices": result.indices,
        }
        if result.normals:
            output["normals"] = result.normals
        return output
    elif result.result_type == "volume":
        logger.info(
            "Volume generated: %s (%s)",
            result.volume_shape,
            result.volume_dtype,
        )
        output = {
            "type": "volume",
            "shape": result.volume_shape,
            "dtype": result.volume_dtype,
            "data": result.volume_data,
        }
        if result.volume_spacing:
            output["spacing"] = result.volume_spacing
        return output
    else:
        raise ValueError(f"Unknown result type: {result.result_type}")


# ---------------------------------------------------------------------------
# Wrapper for FunctionRegistry integration
# ---------------------------------------------------------------------------


def create_agent_function(
    model: str = "claude-opus-4-8",
    timeout: float = 3000.0,
    sandbox: bool = True,
    sandbox_config: SandboxConfig | None = None,
):
    """Create an agent-based generation function for the registry.

    Parameters
    ----------
    model : str
        Default Claude model to use.
    timeout : float
        Default timeout in seconds.
    sandbox : bool
        If True, wrap Bash commands in Firejail sandbox.
    sandbox_config : SandboxConfig, optional
        Configuration for sandbox limits.

    Returns
    -------
    callable
        Async function compatible with FunctionRegistry.
    """

    async def agent_generate(
        prompt: Annotated[str, "textarea"] = r"Read the tif stacks at ~/Downloads/5dry.tif and ~/Downloads/60dry.tif and slice out a cube of equal length/width/height from the center. Subtract the 10th percentile from each; that will be the 'raw' data. Perform threshold using Yen method from skimage, assuming the background is dark. Then use skimage.regionprops to filter out objects smaller than 500 voxels. Use that result to mask the original 'cube' data. Return a 2x2 stack of the raw and masked data with a small gap between each. Each dataset should be processed individually."
                                             r"Note, this is a dry run. Afterwards, any issues you encounter should be investigated, and the system prompt in agent_generator.py (C:\Users\rp\PycharmProjects\ascribe-link\ascribe_link\agent_generator.py) should be modified to reduce friction in future runs. Do not actually submit the result until you've updated the system prompt.",
        file_path: str = "",
        reporter: ProgressReporter | None = None,
    ) -> dict[str, Any]:
        """Generate data (mesh, volume, etc.) using an AI agent.

        Parameters
        ----------
        prompt : str
            Natural language description of what to generate.
            Examples:
            - "Create a DNA double helix mesh"
            - "Generate a torus with major radius 2 and minor radius 0.5"
            - "Create a 64x64x64 volume with a sphere in the center"
            - "Process this volume data with marching cubes at threshold 0.5"
        file_path : str, optional
            Path to an input data file for the agent to process.

        Returns
        -------
        dict
            Result with 'type' field ("mesh" or "volume") and corresponding data.
        """
        return await generate_with_agent(
            prompt=prompt,
            file_path=file_path,
            model=model,
            timeout=timeout,
            sandbox=sandbox,
            sandbox_config=sandbox_config,
            reporter=reporter,
        )

    return agent_generate


# Backwards compatibility alias
generate_mesh_with_agent = generate_with_agent


# ---------------------------------------------------------------------------
# Output dispatcher: coerce arbitrary agent return values into a typed Result
# ---------------------------------------------------------------------------


def wrap_agent_output(value: Any):
    """Coerce an arbitrary agent Python return value into a typed Result.

    Dispatch:
    - VolumeResult / MeshResult -> passthrough
    - numpy.ndarray (ndim == 3) -> VolumeResult.from_numpy (cast to float32)
    - pyvista.PolyData / similar -> MeshResult.from_pyvista
    - anything else -> TypeError
    """
    import numpy as np
    from ascribe_link.models import MeshResult, VolumeResult

    if isinstance(value, (MeshResult, VolumeResult)):
        return value
    if isinstance(value, np.ndarray) and value.ndim == 3:
        return VolumeResult.from_numpy(
            np.ascontiguousarray(value.astype(np.float32))
        )
    try:
        import pyvista as pv

        if isinstance(value, pv.PolyData) or (
            hasattr(value, "points") and hasattr(value, "faces")
        ):
            return MeshResult.from_pyvista(value)
    except ImportError:
        pass
    raise TypeError(
        f"cannot wrap agent output of type {type(value).__name__}"
    )
