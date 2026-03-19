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

You are a data generation assistant for Ascribe-XR, a scientific visualization platform.
Your job is to create 3D data (meshes or volumes) based on user prompts.

## Your Task

Generate 3D data and submit it using either `submit_mesh` or `submit_volume`.

### For Meshes (surfaces, objects):
Use `submit_mesh` with:
- vertices: list of [x, y, z] coordinate lists
- indices: flat list of triangle vertex indices (every 3 = one face)

### For Volumes (3D arrays, voxel data):
Use `submit_volume` with:
- shape: [depth, height, width] 
- dtype: numpy dtype string (e.g., "float32", "uint8")
- data: base64-encoded raw bytes
- spacing: optional voxel spacing [sz, sy, sx]

## Tools Available

You have access to standard tools (Read, Write, Bash) plus:

- **submit_mesh**: Submit a triangular mesh surface
- **submit_volume**: Submit volumetric (voxel) data

## Environment

- Bash commands run in a sandboxed environment (isolated filesystem, no network)
- All work happens in the current working directory
- Input files are copied to the working directory
- PyVista, NumPy, SciPy, and scikit-image are available

## Recommended Libraries

PyVista is installed and highly recommended for mesh generation:

```python
import pyvista as pv
import numpy as np

# Create primitives
sphere = pv.Sphere(radius=1.0, center=(0, 0, 0))
box = pv.Box(bounds=(-1, 1, -1, 1, -1, 1))
cylinder = pv.Cylinder(radius=0.5, height=2.0)
torus = pv.ParametricTorus(ringradius=1.0, crosssectionradius=0.3)

# Combine meshes
combined = sphere + box.translate((2, 0, 0))

# Boolean operations
result = sphere.boolean_difference(box)

# Smoothing, decimation
smoothed = mesh.smooth(n_iter=100)
decimated = mesh.decimate(0.5)

# Extract for submission
vertices = mesh.points.tolist()
faces = mesh.faces.reshape(-1, 4)[:, 1:].flatten().tolist()  # Remove face counts
```

Other useful libraries:
- NumPy for numerical operations
- SciPy for algorithms (scipy.ndimage.label, etc.)
- skimage.measure.marching_cubes for volumetric data

## Working with Input Files

If a file path is provided, read it first to understand the data:
- `.npy` files: NumPy arrays (could be point clouds, volumes, etc.)
- `.stl/.obj/.vtk`: Mesh files (load with pyvista)
- `.csv`: Tabular data (load with numpy or pandas)

## Guidelines

1. Write Python code to generate the mesh
2. Execute the code using Bash
3. Extract vertices and indices
4. Call submit_mesh with the result

Keep meshes reasonable in size (< 1M triangles) unless specifically requested.
Ensure all coordinates are finite (no NaN or inf values).

## Example: Simple Sphere (Mesh)

```python
import pyvista as pv

sphere = pv.Sphere(radius=1.0, theta_resolution=30, phi_resolution=30)
vertices = sphere.points.tolist()
faces = sphere.faces.reshape(-1, 4)[:, 1:].flatten().tolist()

# Save for submission
import json
with open('/tmp/mesh_result.json', 'w') as f:
    json.dump({'vertices': vertices, 'indices': faces}, f)
print("Mesh saved to /tmp/mesh_result.json")
```

After running the code, read the JSON and call submit_mesh.

## Example: Volume Data

```python
import numpy as np

# Create a 3D sphere volume
size = 64
x, y, z = np.ogrid[-1:1:size*1j, -1:1:size*1j, -1:1:size*1j]
volume = (x**2 + y**2 + z**2 < 0.5).astype(np.float32)

# Save for submission
import json
import base64
data = {
    'type': 'volume',
    'shape': list(volume.shape),
    'dtype': str(volume.dtype),
    'data': base64.b64encode(volume.tobytes()).decode('ascii'),
    'spacing': [1.0, 1.0, 1.0]
}
with open('result.json', 'w') as f:
    json.dump(data, f)
```

Then call submit_volume with the result, or read the JSON.
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

    # Define the submit_mesh tool
    @tool(
        "submit_mesh",
        "Submit a generated mesh. Call this when your mesh is ready.",
        {
            "vertices": list,  # list of [x, y, z] coordinates
            "indices": list,   # flat list of triangle indices
        }
    )
    async def submit_mesh(args: dict) -> dict:
        """Capture the mesh submitted by the agent."""
        vertices = args.get("vertices", [])
        indices = args.get("indices", [])

        # Validate
        if not vertices:
            return {"content": [{"type": "text", "text": "Error: vertices list is empty"}]}
        if not indices:
            return {"content": [{"type": "text", "text": "Error: indices list is empty"}]}

        # Check vertex format
        if not all(isinstance(v, (list, tuple)) and len(v) == 3 for v in vertices):
            return {"content": [{"type": "text", "text": "Error: each vertex must be [x, y, z]"}]}

        # Check for non-finite values
        import math
        for v in vertices:
            if not all(isinstance(c, (int, float)) and math.isfinite(c) for c in v):
                return {"content": [{"type": "text", "text": "Error: vertices contain non-finite values"}]}

        # Check indices
        if not all(isinstance(i, int) and 0 <= i < len(vertices) for i in indices):
            return {"content": [{"type": "text", "text": f"Error: indices must be integers in range [0, {len(vertices)})"}]}

        if len(indices) % 3 != 0:
            return {"content": [{"type": "text", "text": "Error: indices length must be divisible by 3 (triangles)"}]}

        result.result_type = "mesh"
        result.vertices = vertices
        result.indices = indices
        result.submitted = True

        return {
            "content": [{
                "type": "text",
                "text": f"Mesh submitted successfully: {len(vertices)} vertices, {len(indices) // 3} triangles"
            }]
        }

    # Define the submit_volume tool
    @tool(
        "submit_volume",
        "Submit generated volumetric data. Call this when your 3D volume is ready.",
        {
            "shape": list,    # [depth, height, width] or [z, y, x]
            "dtype": str,     # numpy dtype string, e.g. "float32"
            "data": str,      # base64-encoded raw bytes
            "spacing": list,  # optional voxel spacing [sz, sy, sx]
        }
    )
    async def submit_volume(args: dict) -> dict:
        """Capture the volume submitted by the agent."""
        shape = args.get("shape", [])
        dtype = args.get("dtype", "float32")
        data = args.get("data", "")
        spacing = args.get("spacing")

        # Validate
        if not shape or len(shape) != 3:
            return {"content": [{"type": "text", "text": "Error: shape must be [depth, height, width]"}]}
        if not data:
            return {"content": [{"type": "text", "text": "Error: data is empty"}]}

        # Validate base64
        import base64
        try:
            decoded = base64.b64decode(data)
            import numpy as np
            expected_size = np.prod(shape) * np.dtype(dtype).itemsize
            if len(decoded) != expected_size:
                return {"content": [{"type": "text", "text": f"Error: data size mismatch. Expected {expected_size} bytes, got {len(decoded)}"}]}
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
            "content": [{
                "type": "text",
                "text": f"Volume submitted successfully: {shape} ({total_voxels:,} voxels, {dtype})"
            }]
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
    async def sandbox_bash_hook(input_data: dict, tool_use_id: str, context: dict) -> dict:
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
        sandboxed_command = " ".join(f'"{arg}"' if " " in arg else arg for arg in firejail_cmd)
        
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
        permission_mode="acceptEdits",
        max_turns=50,  # Reasonable limit for generation
        hooks=hooks if hooks else None,
    )

    logger.info("Starting generation agent: %s", prompt[:100])

    try:
        async with ClaudeSDKClient(options=options) as client:
            await client.query(user_prompt)

            # Process responses with timeout
            async def process_responses():
                async for msg in client.receive_response():
                    if isinstance(msg, AssistantMessage):
                        for block in msg.content:
                            if isinstance(block, TextBlock):
                                logger.debug("Agent: %s", block.text[:200])
                    elif isinstance(msg, ResultMessage):
                        # Check if this indicates completion
                        pass

                    # Check if we got a result
                    if result.submitted:
                        return

            try:
                await asyncio.wait_for(process_responses(), timeout=timeout)
            except asyncio.TimeoutError:
                raise TimeoutError(f"Agent timed out after {timeout}s")

    except Exception as e:
        logger.error("Agent error: %s", e)
        raise

    if not result.submitted:
        raise ValueError("Agent did not submit any data. It may have encountered an error.")

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
        prompt: str,
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
