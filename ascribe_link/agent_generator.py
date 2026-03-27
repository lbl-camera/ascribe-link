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
from typing import Any

from ascribe_link.sandbox import (
    SandboxConfig,
    is_firejail_available,
    build_firejail_command,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Mesh Generation Skill (system prompt for the agent)
# ---------------------------------------------------------------------------

MESH_GENERATION_SKILL = """# 3D Data Generation Assistant

Generate 3D meshes for Ascribe-XR.

IMPORTANT: Do NOT use mcp__mesh__list_tools or any tool-listing commands. The schemas are provided below. Just generate the mesh and call submit_mesh directly.

## submit_mesh Schema

```json
{
  "vertices": [[x, y, z], ...],  // array of 3-number arrays
  "indices": [i, j, k, ...]      // flat array of integers, every 3 = one triangle
}
```

## Quick Pattern

Run with `python script.py` (not python3), then call submit_mesh with the printed JSON:

```python
import pyvista as pv
import json
from ascribe_link.mesh_utils import extract_mesh_data

mesh = pv.Box(bounds=(-0.5, 0.5, -0.5, 0.5, -0.5, 0.5))  # <-- modify for request
vertices, indices = extract_mesh_data(mesh)
print(json.dumps({"vertices": vertices, "indices": indices}))
```

Use the output directly in your submit_mesh call. Do not fetch tool schemas — they're documented above.

## PyVista Primitives

```python
pv.Sphere(radius=1.0, center=(0, 0, 0))
pv.Box(bounds=(xmin, xmax, ymin, ymax, zmin, zmax))
pv.Cylinder(radius=0.5, height=2.0)
pv.ParametricTorus(ringradius=1.0, crosssectionradius=0.3)

# Combine: mesh1 + mesh2.translate((x, y, z))
# Boolean: mesh1.boolean_difference(mesh2)
```

## extract_mesh_data

Triangulates and converts to Python lists. Always use it before submitting.

## submit_volume Schema (for volumetric data)

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
    # Volume data
    volume_shape: list[int] | None = None
    volume_dtype: str | None = None
    volume_data: str | None = None  # base64
    volume_spacing: list[float] | None = None
    # Status
    error: str | None = None
    submitted: bool = False


# ---------------------------------------------------------------------------
# Agent-based mesh generation
# ---------------------------------------------------------------------------


async def generate_with_agent(
    prompt: str,
    file_path: str | None = None,
    model: str = "claude-sonnet-4-20250514",
    timeout: float = 300.0,
    working_dir: str | None = None,
    sandbox: bool = True,
    sandbox_config: SandboxConfig | None = None,
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

        return {
            "content": [
                {
                    "type": "text",
                    "text": f"Mesh submitted successfully: {len(vertices)} vertices, {len(indices) // 3} triangles",
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

        total_voxels = shape[0] * shape[1] * shape[2]
        return {
            "content": [
                {
                    "type": "text",
                    "text": f"Volume submitted successfully: {shape} ({total_voxels:,} voxels, {dtype})",
                }
            ]
        }

    # Create MCP server with our tools
    mesh_server = create_sdk_mcp_server(
        name="mesh-tools",
        version="1.0.0",
        tools=[submit_mesh, submit_volume],
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

    # Configure agent options
    options = ClaudeAgentOptions(
        model=model,
        system_prompt=MESH_GENERATION_SKILL,
        cwd=working_dir,
        mcp_servers={"mesh": mesh_server},
        allowed_tools=[
            "Read",
            "Write",
            "Bash",
            "mcp__mesh__submit_mesh",
            "mcp__mesh__submit_volume",
        ],
        disallowed_tools=[
            "mcp__*__list_tools",  # Don't introspect MCP tools
            "mcp__*__get_schema",  # Schema is in the prompt
        ],
        permission_mode="acceptEdits",
        max_turns=25,  # Encourage efficiency
        hooks=hooks if hooks else None,
    )

    logger.info("Starting generation agent: %s", prompt[:100])

    try:
        async with ClaudeSDKClient(options=options) as client:
            logger.info("ClaudeSDKClient connected, sending query...")
            await client.query(user_prompt)
            logger.info("Query sent, waiting for responses...")

            # Process responses with timeout
            async def process_responses():
                msg_count = 0
                async for msg in client.receive_response():
                    msg_count += 1
                    msg_type = type(msg).__name__
                    logger.info("Received message #%d: %s", msg_count, msg_type)

                    if isinstance(msg, AssistantMessage):
                        for block in msg.content:
                            if isinstance(block, TextBlock):
                                logger.info("Agent text: %s", block.text[:500])
                    elif isinstance(msg, ResultMessage):
                        logger.info(
                            "Result message received: %s",
                            getattr(msg, "result", "no result attr"),
                        )

                    # Check if we got a result
                    if result.submitted:
                        logger.info("Result submitted, exiting response loop")
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
            "Mesh generated: %d vertices, %d triangles",
            len(result.vertices),
            len(result.indices) // 3,
        )
        return {
            "type": "mesh",
            "vertices": result.vertices,
            "indices": result.indices,
        }
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
    model: str = "claude-sonnet-4-20250514",
    timeout: float = 300.0,
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
        prompt: str = "Create a sphere",
        file_path: str | None = None,
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
        )

    return agent_generate


# Backwards compatibility alias
generate_mesh_with_agent = generate_with_agent
