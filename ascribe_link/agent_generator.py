"""AI-powered mesh generation using Claude Agent SDK.

Provides an agent that can generate 3D meshes from natural language prompts,
optionally processing input data files.
"""

from __future__ import annotations

import asyncio
import logging
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Mesh Generation Skill (system prompt for the agent)
# ---------------------------------------------------------------------------

MESH_GENERATION_SKILL = """# Mesh Generation Assistant

You are a mesh generation assistant for Ascribe-XR, a scientific visualization platform.
Your job is to create 3D meshes based on user prompts.

## Your Task

Generate a 3D mesh and submit it using the `submit_mesh` tool. The mesh must be:
- A list of vertices (each vertex is [x, y, z])
- A list of triangle indices (flat list of vertex indices, every 3 indices form a triangle)

## Tools Available

You have access to standard tools (Read, Write, Bash) plus:

- **submit_mesh**: Submit your final mesh. Call this when done.
  - vertices: list of [x, y, z] coordinate lists
  - indices: flat list of integers (every 3 = one triangle face)

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

## Example: Simple Sphere

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
"""


# ---------------------------------------------------------------------------
# Result holder for capturing mesh from tool call
# ---------------------------------------------------------------------------

@dataclass
class MeshResult:
    """Holds the mesh result from the agent."""
    vertices: list[list[float]] | None = None
    indices: list[int] | None = None
    error: str | None = None
    submitted: bool = False


# ---------------------------------------------------------------------------
# Agent-based mesh generation
# ---------------------------------------------------------------------------

async def generate_mesh_with_agent(
    prompt: str,
    file_path: str | None = None,
    model: str = "claude-sonnet-4-20250514",
    timeout: float = 300.0,
    working_dir: str | None = None,
) -> tuple[list[list[float]], list[int]]:
    """Generate a mesh using an AI agent.

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

    Returns
    -------
    tuple[list[list[float]], list[int]]
        (vertices, indices) where vertices is a list of [x,y,z] coords
        and indices is a flat list of triangle vertex indices.

    Raises
    ------
    ValueError
        If the agent fails to produce a valid mesh.
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
        tool,
        create_sdk_mcp_server,
    )

    result = MeshResult()

    # Define the submit_mesh tool
    @tool(
        "submit_mesh",
        "Submit the generated mesh. Call this when your mesh is ready.",
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

        result.vertices = vertices
        result.indices = indices
        result.submitted = True

        return {
            "content": [{
                "type": "text",
                "text": f"Mesh submitted successfully: {len(vertices)} vertices, {len(indices) // 3} triangles"
            }]
        }

    # Create MCP server with our tool
    mesh_server = create_sdk_mcp_server(
        name="mesh-tools",
        version="1.0.0",
        tools=[submit_mesh],
    )

    # Set up working directory
    if working_dir is None:
        working_dir = tempfile.mkdtemp(prefix="ascribe_agent_")

    # Build the user prompt
    user_prompt = prompt
    if file_path:
        user_prompt = f"Input file: {file_path}\n\n{prompt}"

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
        ],
        permission_mode="acceptEdits",
        max_turns=50,  # Reasonable limit for mesh generation
    )

    logger.info("Starting mesh generation agent: %s", prompt[:100])

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
        raise ValueError("Agent did not submit a mesh. It may have encountered an error.")

    logger.info(
        "Mesh generated: %d vertices, %d triangles",
        len(result.vertices),
        len(result.indices) // 3,
    )

    return result.vertices, result.indices


# ---------------------------------------------------------------------------
# Wrapper for FunctionRegistry integration
# ---------------------------------------------------------------------------

def create_agent_function(
    model: str = "claude-sonnet-4-20250514",
    timeout: float = 300.0,
):
    """Create an agent-based mesh generation function for the registry.

    Parameters
    ----------
    model : str
        Default Claude model to use.
    timeout : float
        Default timeout in seconds.

    Returns
    -------
    callable
        Async function compatible with FunctionRegistry.
    """
    async def agent_generate(
        prompt: str,
        file_path: str | None = None,
    ) -> tuple[list, list]:
        """Generate a mesh using an AI agent.

        Parameters
        ----------
        prompt : str
            Natural language description of what to generate.
            Examples:
            - "Create a DNA double helix"
            - "Generate a torus with major radius 2 and minor radius 0.5"
            - "Process this volume data with marching cubes at threshold 0.5"
        file_path : str, optional
            Path to an input data file for the agent to process.

        Returns
        -------
        tuple[list, list]
            (vertices, indices) mesh data.
        """
        return await generate_mesh_with_agent(
            prompt=prompt,
            file_path=file_path,
            model=model,
            timeout=timeout,
        )

    return agent_generate
