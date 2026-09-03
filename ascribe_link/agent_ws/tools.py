"""Conversational MCP tool surface for the persistent agent-conversation session.

Registers eleven tools on an in-process MCP server named ``"scene"``:

- Server-compute: ``submit_mesh``, ``submit_volume``, ``submit_mesh_file``,
  ``submit_volume_file`` -- de-closured ports of ``agent_generator.py``'s
  tools of the same names (same JSON schemas and validation), except instead
  of setting a one-shot ``AgentResult`` they call ``sink.stage_result(...)``
  and return, so the conversation continues. The ``*_file`` variants read the
  data from disk so a real-sized volume never has to be inlined as base64 in
  a tool-call argument. ``analyze_specimen`` looks up a previously staged
  result and reports basic statistics.
- Client-forwarded: ``load_specimen``, ``set_active_specimen``,
  ``remove_specimen``, ``set_room_scene``, ``set_display_param``,
  ``capture_viewport`` -- each is a thin, timeout-wrapped call to
  ``sink.request_client_tool(name, args)``. The manager pairs the reply with
  the real client RPC / binary frame. ``load_specimen`` additionally resolves
  the specimen's renderer type server-side (via ``sink.resolve_specimen_type``)
  and injects it into the forwarded args: the client has no catalog entry for
  agent-staged specimens, so without this it can only guess -- and its guess
  is "mesh", which silently renders nothing for a volume.

``claude_agent_sdk`` is imported lazily inside ``build_conversation_tools`` so
importing this module never requires the SDK to be installed.
"""

from __future__ import annotations

import asyncio
import base64
import logging
import math
from pathlib import Path
from typing import Any, Awaitable, Callable, Protocol

logger = logging.getLogger(__name__)

# Wrapping timeout for every client-forwarded tool call (request_client_tool).
CLIENT_TOOL_TIMEOUT = 30.0

_DISPLAY_PARAM_NAMES = ("gamma", "opacity", "color_scalar", "max_steps", "step_size", "zoom")
_SCENE_NAMES = ("lab", "black", "passthrough", "world_scale")


class ConversationSink(Protocol):
    """What `tools.py` needs from the room/session manager.

    The manager (Task 5) provides the real implementation; tests use a fake.
    """

    room_id: str

    async def request_client_tool(self, name: str, args: dict) -> Any:
        """Forward a tool call to the connected client and await its reply."""
        ...

    def stage_result(self, result: Any) -> str:
        """Insert a MeshResult/VolumeResult into the room cache; return its id."""
        ...

    def get_staged(self, specimen_id: str) -> Any:
        """Look up a previously staged result by specimen id, or None."""
        ...

    def resolve_specimen_type(self, specimen_id: str) -> str | None:
        """Return "mesh"/"volume" for a staged or catalog specimen, else None."""
        ...


def _error(text: str) -> dict:
    return {"content": [{"type": "text", "text": text}], "is_error": True}


def _text(text: str) -> dict:
    return {"content": [{"type": "text", "text": text}]}


def _client_forward(
    sink: ConversationSink, name: str
) -> Callable[[dict], Awaitable[dict]]:
    """Build a handler that forwards `args` to `sink.request_client_tool(name, args)`.

    Wraps the await in a timeout; any timeout or exception becomes an
    `is_error` text-content result instead of propagating.
    """

    async def handler(args: dict) -> dict:
        if name == "load_specimen":
            # Resolve the renderer type here, where both the staged store and
            # the catalog are visible. A specimen id that resolves to nothing
            # is an error *now* -- the client used to reply ok:true for any
            # string (file paths included) and the agent reasoned from that
            # phantom success for turns.
            specimen_id = str(args.get("specimen_id", ""))
            specimen_type = sink.resolve_specimen_type(specimen_id)
            if specimen_type is None:
                return _error(
                    f"Error: unknown specimen id '{specimen_id}'. load_specimen takes an "
                    "id returned by submit_mesh/submit_volume/submit_*_file or a catalog "
                    "specimen id -- not a file path. To show a file, use "
                    "submit_volume_file / submit_mesh_file first."
                )
            args = {**args, "type": specimen_type}

        try:
            result = await asyncio.wait_for(
                sink.request_client_tool(name, args), CLIENT_TOOL_TIMEOUT
            )
        except Exception as err:  # noqa: BLE001 - deliberately broad: never raise into the SDK
            return _error(f"Tool {name} failed: {err}")

        if name == "capture_viewport":
            if not isinstance(result, (bytes, bytearray)):
                return _error(f"Tool {name} failed: expected JPEG bytes, got {type(result).__name__}")
            # MCP ImageContent shape ({data, mimeType}), NOT the Anthropic
            # Messages API {source: {...}} shape: claude_agent_sdk's
            # _convert_tool_content indexes item["data"] / item["mimeType"]
            # directly, and the API shape surfaced to the agent as the bare
            # error text "'data'" (a KeyError) on every capture.
            return {
                "content": [
                    {
                        "type": "image",
                        "data": base64.b64encode(bytes(result)).decode("ascii"),
                        "mimeType": "image/jpeg",
                    }
                ]
            }

        return _text(str(result))

    return handler


async def _show_staged(sink: ConversationSink, specimen_id: str, specimen_type: str) -> str:
    """Push a just-staged specimen to the client (the same path as load_specimen).

    Staging alone only parks the result in the room store; nothing reaches
    the headset until the client is told to fetch it. The submit tools used
    to stop there while claiming "now visible", so the agent -- reasonably --
    never followed up with load_specimen. Returns a short suffix describing
    the outcome for the tool's text result.
    """
    try:
        await asyncio.wait_for(
            sink.request_client_tool(
                "load_specimen", {"specimen_id": specimen_id, "type": specimen_type}
            ),
            CLIENT_TOOL_TIMEOUT,
        )
    except Exception as err:  # noqa: BLE001 - staging succeeded; report, don't fail the tool
        return (
            f" Staged, but the viewer could not be told to load it ({err}); "
            f"call load_specimen('{specimen_id}') to retry."
        )
    return " It is now visible to the user."


def _resolve_submit_path(file_path: str) -> Path | None:
    """Resolve a ``submit_*_file`` path: relative to the server cwd, else as given."""
    candidate = Path.cwd() / file_path
    if candidate.is_file():
        return candidate
    candidate = Path(file_path)
    if candidate.is_file():
        return candidate
    return None


def _analyze_volume(vr: Any) -> str:
    import numpy as np

    arr = vr.to_numpy()
    hist, edges = np.histogram(arr, bins=16)
    return (
        f"Volume: shape={list(arr.shape)}, dtype={arr.dtype}, "
        f"min={float(arr.min()):.6g}, max={float(arr.max()):.6g}, "
        f"mean={float(arr.mean()):.6g}, std={float(arr.std()):.6g}, "
        f"histogram(16 bins)={hist.tolist()}, bin_edges={[float(e) for e in edges]}"
    )


def _analyze_mesh(mr: Any) -> str:
    verts = mr.vertices
    n_verts = len(verts) // 3
    n_faces = len(mr.indices) // 3
    if n_verts == 0:
        return f"Mesh: {n_verts} vertices, {n_faces} faces (empty, no bounding box)"
    xs = verts[0::3]
    ys = verts[1::3]
    zs = verts[2::3]
    bbox_min = [min(xs), min(ys), min(zs)]
    bbox_max = [max(xs), max(ys), max(zs)]
    return (
        f"Mesh: {n_verts} vertices, {n_faces} faces, "
        f"bounding_box_min={bbox_min}, bounding_box_max={bbox_max}"
    )


def build_conversation_tools(sink: ConversationSink) -> tuple[dict, list[str], list]:
    """Build the "scene" MCP server + allowed_tools list for one conversation.

    Returns
    -------
    tuple[dict, list[str], list]
        (server, allowed_tools, sdk_tools) -- `server` is the
        `McpSdkServerConfig` dict from `create_sdk_mcp_server`, ready for
        `ClaudeAgentOptions.mcp_servers`. `allowed_tools` is
        `["mcp__scene__" + name, ...]` for all eleven tools. `sdk_tools` is the
        raw `SdkMcpTool` list, returned separately so tests can invoke
        handlers directly -- it MUST NOT be stored inside `server`: the SDK
        json.dumps-es that dict verbatim when building the CLI command, and
        any non-JSON value in it crashes `ClaudeSDKClient.connect()`.
    """
    from claude_agent_sdk import create_sdk_mcp_server, tool

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
        "Submit a triangular mesh. vertices is an array of [x,y,z] points. indices is a "
        "flat array of integers (every 3 indices form one triangle face). The mesh becomes "
        "visible to the user; the conversation continues afterward.",
        submit_mesh_schema,
    )
    async def submit_mesh(args: dict) -> dict:
        from ascribe_link.models import MeshResult

        vertices = args.get("vertices", [])
        indices = args.get("indices", [])

        if not vertices:
            return _error("Error: vertices list is empty")
        if not indices:
            return _error("Error: indices list is empty")
        if not all(isinstance(v, (list, tuple)) and len(v) == 3 for v in vertices):
            return _error("Error: each vertex must be [x, y, z]")
        for v in vertices:
            if not all(isinstance(c, (int, float)) and math.isfinite(c) for c in v):
                return _error("Error: vertices contain non-finite values")
        if not all(isinstance(i, int) and 0 <= i < len(vertices) for i in indices):
            return _error(f"Error: indices must be integers in range [0, {len(vertices)})")
        if len(indices) % 3 != 0:
            return _error("Error: indices length must be divisible by 3 (triangles)")

        flat_vertices = [c for v in vertices for c in v]
        mesh_result = MeshResult(vertices=flat_vertices, indices=list(indices))
        specimen_id = sink.stage_result(mesh_result)
        shown = await _show_staged(sink, specimen_id, "mesh")

        return _text(
            f"Mesh submitted as specimen '{specimen_id}': {len(vertices)} vertices, "
            f"{len(indices) // 3} triangles.{shown}"
        )

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
        "Submit volumetric (voxel) data. shape is [depth, height, width], dtype is a numpy "
        "dtype string, data is base64-encoded bytes. The volume becomes visible to the user; "
        "the conversation continues afterward.",
        submit_volume_schema,
    )
    async def submit_volume(args: dict) -> dict:
        from ascribe_link.models import VolumeResult

        shape = args.get("shape", [])
        dtype = args.get("dtype", "float32")
        data = args.get("data", "")
        spacing = args.get("spacing")

        if not shape or len(shape) != 3:
            return _error("Error: shape must be [depth, height, width]")
        if not data:
            return _error("Error: data is empty")

        try:
            decoded = base64.b64decode(data)
            import numpy as np

            expected_size = int(np.prod(shape)) * np.dtype(dtype).itemsize
            if len(decoded) != expected_size:
                return _error(
                    f"Error: data size mismatch. Expected {expected_size} bytes, "
                    f"got {len(decoded)}"
                )
        except Exception as e:  # noqa: BLE001
            return _error(f"Error decoding data: {e}")

        volume_result = VolumeResult(
            shape=list(shape), dtype=dtype, data=data, spacing=spacing
        )
        specimen_id = sink.stage_result(volume_result)
        shown = await _show_staged(sink, specimen_id, "volume")

        total_voxels = int(shape[0]) * int(shape[1]) * int(shape[2])
        return _text(
            f"Volume submitted as specimen '{specimen_id}': {shape} "
            f"({total_voxels:,} voxels, {dtype}).{shown}"
        )

    submit_mesh_file_schema = {
        "type": "object",
        "properties": {
            "file_path": {
                "type": "string",
                "description": "Path to a JSON file with 'vertices' ([[x,y,z], ...]) and "
                "'indices' (flat int list) keys, optionally 'normals'",
            },
        },
        "required": ["file_path"],
    }

    @tool(
        "submit_mesh_file",
        "Submit a triangular mesh from a JSON file on disk. Use this instead of "
        "submit_mesh for any mesh too large to inline. The file must have 'vertices' "
        "and 'indices', and optionally 'normals'. The mesh becomes visible to the "
        "user; the conversation continues afterward.",
        submit_mesh_file_schema,
    )
    async def submit_mesh_file(args: dict) -> dict:
        import json

        from ascribe_link.models import MeshResult

        file_path = str(args.get("file_path", ""))
        if not file_path:
            return _error("Error: file_path is required")
        full_path = _resolve_submit_path(file_path)
        if full_path is None:
            return _error(f"Error: file not found: {file_path}")

        try:
            data = await asyncio.to_thread(lambda: json.loads(full_path.read_text()))
        except (json.JSONDecodeError, OSError) as err:
            return _error(f"Error: could not read mesh file: {err}")
        if not isinstance(data, dict):
            return _error("Error: mesh file must be a JSON object")

        vertices = data.get("vertices", [])
        indices = data.get("indices", [])
        normals = data.get("normals")
        if not vertices:
            return _error("Error: vertices list is empty")
        if not indices:
            return _error("Error: indices list is empty")
        if len(indices) % 3 != 0:
            return _error("Error: indices length must be divisible by 3 (triangles)")

        # Accept either nested [[x,y,z], ...] or an already-flat list.
        if vertices and isinstance(vertices[0], (list, tuple)):
            flat_vertices = [float(c) for v in vertices for c in v]
        else:
            flat_vertices = [float(c) for c in vertices]
        if len(flat_vertices) % 3 != 0:
            return _error("Error: vertices must be [x, y, z] triples")
        flat_normals = None
        if normals:
            if isinstance(normals[0], (list, tuple)):
                flat_normals = [float(c) for n in normals for c in n]
            else:
                flat_normals = [float(c) for c in normals]

        mesh_result = MeshResult(
            vertices=flat_vertices, indices=[int(i) for i in indices], normals=flat_normals
        )
        specimen_id = sink.stage_result(mesh_result)
        shown = await _show_staged(sink, specimen_id, "mesh")
        return _text(
            f"Mesh submitted from file as specimen '{specimen_id}': "
            f"{len(flat_vertices) // 3} vertices, {len(indices) // 3} triangles.{shown}"
        )

    submit_volume_file_schema = {
        "type": "object",
        "properties": {
            "file_path": {
                "type": "string",
                "description": "Path to a volume file: .npy (preferred, written with "
                "np.save), .npz, or a .json envelope with shape/dtype/data(base64)",
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
        "Submit volumetric (voxel) data from a file on disk. Use this instead of "
        "submit_volume for any real-sized volume -- write the 3D array with "
        "np.save('volume.npy', arr) and pass the path; .npz and .json envelopes are also "
        "accepted. The volume becomes visible to the user; the conversation continues "
        "afterward.",
        submit_volume_file_schema,
    )
    async def submit_volume_file(args: dict) -> dict:
        from ascribe_link.agent_generator import _load_volume_array
        from ascribe_link.models import VolumeResult

        file_path = str(args.get("file_path", ""))
        if not file_path:
            return _error("Error: file_path is required")
        full_path = _resolve_submit_path(file_path)
        if full_path is None:
            return _error(f"Error: file not found: {file_path}")

        try:
            arr, file_spacing = await asyncio.to_thread(_load_volume_array, full_path)
        except ValueError as err:
            return _error(f"Error: {err}")
        if arr.ndim != 3:
            return _error(f"Error: volume must be 3D, got shape {list(arr.shape)}")

        spacing = args.get("spacing") or file_spacing
        volume_result = await asyncio.to_thread(VolumeResult.from_numpy, arr, spacing=spacing)
        specimen_id = sink.stage_result(volume_result)
        shown = await _show_staged(sink, specimen_id, "volume")

        total_voxels = int(arr.shape[0] * arr.shape[1] * arr.shape[2])
        return _text(
            f"Volume submitted from file as specimen '{specimen_id}': {volume_result.shape} "
            f"({total_voxels:,} voxels, {volume_result.dtype}).{shown}"
        )

    @tool(
        "analyze_specimen",
        "Compute basic statistics on a previously submitted/loaded specimen: for volumes, "
        "shape/dtype/min/max/mean/std and a 16-bin histogram; for meshes, vertex/face counts "
        "and bounding box.",
        {
            "type": "object",
            "properties": {
                "specimen_id": {"type": "string", "description": "Specimen id to analyze"},
            },
            "required": ["specimen_id"],
        },
    )
    async def analyze_specimen(args: dict) -> dict:
        from ascribe_link.models import MeshResult, VolumeResult

        specimen_id = args.get("specimen_id", "")
        staged = sink.get_staged(specimen_id)
        if staged is None:
            return _error(f"Error: no staged specimen with id '{specimen_id}'")

        if isinstance(staged, VolumeResult):
            summary = await asyncio.to_thread(_analyze_volume, staged)
        elif isinstance(staged, MeshResult):
            summary = _analyze_mesh(staged)
        else:
            return _error(f"Error: unsupported specimen type {type(staged).__name__}")

        return _text(summary)

    client_forwarded_specs = [
        (
            "load_specimen",
            "Load a specimen into the scene by its id: either an id returned by "
            "submit_mesh/submit_volume/submit_mesh_file/submit_volume_file, or a catalog "
            "specimen id. This is NOT a file path -- to display a file, submit it with "
            "submit_volume_file / submit_mesh_file first. The renderer (mesh vs volume) is "
            "chosen from the specimen's type automatically.",
            {
                "type": "object",
                "properties": {"specimen_id": {"type": "string"}},
                "required": ["specimen_id"],
            },
        ),
        (
            "set_active_specimen",
            "Set which loaded specimen (by scene index) is currently active.",
            {
                "type": "object",
                "properties": {"index": {"type": "integer"}},
                "required": ["index"],
            },
        ),
        (
            "remove_specimen",
            "Remove a loaded specimen from the scene by index.",
            {
                "type": "object",
                "properties": {"index": {"type": "integer"}},
                "required": ["index"],
            },
        ),
        (
            "set_room_scene",
            "Change the room's environment/backdrop.",
            {
                "type": "object",
                "properties": {"name": {"type": "string", "enum": list(_SCENE_NAMES)}},
                "required": ["name"],
            },
        ),
        (
            "set_display_param",
            "Set a display parameter on a loaded specimen by index.",
            {
                "type": "object",
                "properties": {
                    "index": {"type": "integer"},
                    "name": {"type": "string", "enum": list(_DISPLAY_PARAM_NAMES)},
                    "value": {"type": "number"},
                },
                "required": ["index", "name", "value"],
            },
        ),
        (
            "capture_viewport",
            "Capture the user's current viewport as a JPEG image, returned to you inline.",
            {"type": "object", "properties": {}},
        ),
    ]

    sdk_tools = [submit_mesh, submit_volume, submit_mesh_file, submit_volume_file, analyze_specimen]
    for name, description, schema in client_forwarded_specs:
        sdk_tools.append(tool(name, description, schema)(_client_forward(sink, name)))

    server = create_sdk_mcp_server(name="scene", version="1.0.0", tools=sdk_tools)

    allowed_tools = [f"mcp__scene__{t.name}" for t in sdk_tools]
    return server, allowed_tools, sdk_tools
