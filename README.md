# Ascribe-Link

**HTTP-based specimen server for Ascribe-XR with dynamic data generation**

Ascribe-Link provides a REST API for serving scientific datasets (meshes, volumes, point clouds) to VR clients. Features include parametric specimen generation, JSON Schema-driven UI generation, and multiplayer result caching.

## Features

- **Specimen Catalog:** Curated collection of 3D datasets with metadata
- **Dynamic Specimens:** Generate data on-demand from Python functions
- **JSON Schema Generation:** Auto-generate parameter schemas from function signatures
- **Processing API:** Invoke functions with parameters, return typed results
- **Multiplayer Caching:** Room-based result caching for collaborative sessions
- **Type System:** Support for mesh, volume, point cloud, and image data
- **Federation:** Relay mode for aggregating specimens from worker nodes

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/ronpandolfi/Ascribe-Link.git
cd Ascribe-Link

# Install dependencies
pip install -e .
```

### Run the Server

```bash
python -m ascribe_link
```

Server starts at `http://localhost:8000`

### Test the API

```bash
# Run comprehensive test suite
./test_dynamic_specimen.py

# Or manually test endpoints
curl http://localhost:8000/api/specimens/
curl http://localhost:8000/api/processing/functions
```

## API Documentation

### Specimen Endpoints

**List all specimens:**
```http
GET /api/specimens/
```
Returns array of specimen metadata with `is_dynamic` flag.

**Get specimen details:**
```http
GET /api/specimens/{specimen_id}
```
Returns full metadata including JSON Schema for dynamic specimens.

**Get specimen data:**
```http
GET /api/specimens/{specimen_id}/data
```
Downloads the specimen data file (mesh, volume, etc.).

**Get thumbnail:**
```http
GET /api/specimens/{specimen_id}/thumbnail
```
Returns specimen preview image.

### Processing Endpoints

**List processing functions:**
```http
GET /api/processing/functions
```
Returns all registered functions with schemas and return types.

**Get function schema:**
```http
GET /api/processing/functions/{name}/schema
```
Returns JSON Schema for function parameters.

**Invoke function:**
```http
POST /api/processing/invoke
Content-Type: application/json

{
  "function_name": "generate_sphere",
  "args": [],
  "kwargs": {"radius": 2.0, "resolution": 64},
  "room_id": "ascribe"
}
```

Returns typed result (mesh, volume, etc.) with automatic caching.

**Cache stats:**
```http
GET /api/processing/cache/stats
```
Returns cache usage statistics per room.

**Clear cache:**
```http
POST /api/processing/cache/clear
```
Invalidates all cached results.

## Dynamic Specimens

### Creating a Dynamic Specimen

**1. Write a processing function:**

```python
# ascribe_link/parametric.py
import pyvista as pv
from ascribe_link.models import MeshResult

def generate_torus(
    major_radius: float = 1.0,
    minor_radius: float = 0.3,
    segments: int = 32,
) -> MeshResult:
    """Generate a parametric torus mesh."""
    torus = pv.ParametricTorus(
        ringradius=major_radius,
        crosssectionradius=minor_radius,
    )
    return MeshResult.from_pyvista(torus)
```

**2. Register the function:**

```python
# ascribe_link/app.py
from ascribe_link.parametric import generate_torus

registry.register_function(generate_torus, "generate_torus", return_type="mesh")
```

**3. Create specimen metadata:**

```json
// specimens/parametric_torus/specimen.json
{
  "id": "parametric_torus",
  "display_name": "Parametric Torus",
  "description": "Torus with adjustable radii and resolution",
  "type": "mesh",
  "thumbnail_file": "thumbnail.png",
  "function_name": "generate_torus",
  "tags": ["parametric", "mesh", "dynamic"],
  "schema": {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "generate_torus",
    "type": "object",
    "properties": {
      "major_radius": {
        "type": "number",
        "default": 1.0,
        "minimum": 0.5,
        "maximum": 3.0
      },
      "minor_radius": {
        "type": "number",
        "default": 0.3,
        "minimum": 0.1,
        "maximum": 1.0
      },
      "segments": {
        "type": "number",
        "default": 32,
        "minimum": 8,
        "maximum": 128
      }
    }
  }
}
```

The schema is **automatically generated** from function signatures if omitted, but can be customized for better UX (e.g., min/max ranges for sliders).

### Supported Types

**Type Hints → JSON Schema:**
- `float`, `int` → `"number"` or `"integer"`
- `str` → `"string"`
- `bool` → `"boolean"`
- `Literal["a", "b"]` → `{"enum": ["a", "b"]}`
- Defaults extracted from function signature

**Return Types:**
- `MeshResult` → vertices, indices, normals
- `VolumeResult` → shape, dtype, base64 data, spacing/origin
- `PointCloudResult` → points, colors, scalars
- `ImageResult` → width, height, channels, data

## Multiplayer Caching

Ascribe-Link implements **room-based result caching** to optimize collaborative VR sessions:

### How It Works

1. **Room-Scoped Cache:** Each room (e.g., "ascribe") has one cached result
2. **First Peer:** Computes result, stores in cache
3. **Subsequent Peers:** Get instant cached result (no recomputation)
4. **Auto-Invalidation:** New request with different parameters wipes old cache

### Example

```
Room: "ascribe"

Peer A: generate_sphere(radius=2.0, resolution=64)
→ Cache miss → Compute (500ms) → Store

Peer B: generate_sphere(radius=2.0, resolution=64)  [same params]
→ Cache hit! → Return cached result (<10ms)

Peer C: generate_sphere(radius=3.0, resolution=64)  [different params]
→ Cache miss → Invalidates old cache → Compute new result → Store
```

### Configuration

```python
# ascribe_link/app.py
result_cache = RoomResultCache(ttl_seconds=300.0)  # 5 minute TTL
```

## Project Structure

```
ascribe_link/
├── app.py              # Litestar application factory
├── models.py           # Data models (specimens, results, requests)
├── processing.py       # FunctionRegistry and schema generation
├── cache.py            # RoomResultCache for multiplayer
├── specimen_store.py   # Specimen directory management
├── parametric.py       # Built-in parametric functions
├── example.py          # Example functions
├── routes/
│   ├── specimens.py    # Specimen catalog endpoints
│   ├── processing.py   # Function invocation endpoints
│   └── federation.py   # Worker federation (relay mode)
└── utils.py            # Helper functions

specimens/              # Specimen data directory
├── brain/
│   ├── specimen.json   # Metadata
│   ├── brain.stl       # Data file
│   └── thumbnail.png   # Preview image
└── parametric_sphere/
    ├── specimen.json   # Metadata with schema + function_name
    └── thumbnail.png
```

## Advanced Usage

### AI Agent Generation

Enable AI-powered mesh generation (requires `claude-agent-sdk`):

```python
app = create_app(
    enable_agent=True,
    agent_model="claude-sonnet-4",
    agent_timeout=300.0
)
```

Clients can invoke:
```json
{
  "function_name": "ai_generate",
  "kwargs": {
    "prompt": "Create a DNA double helix mesh with 10 base pairs"
  }
}
```

### Federation (Relay Mode)

Run as a relay to aggregate specimens from worker nodes:

```python
app = create_app(relay_mode=True)
```

Workers connect via WebSocket and register their specimens. The relay aggregates all specimens into a unified catalog.

### Custom Functions

Register your own processing functions:

```python
from ascribe_link.models import MeshResult

def my_function(param1: float, param2: int) -> MeshResult:
    # ... generate mesh
    return MeshResult(vertices=..., indices=...)

app = create_app(
    mesh_functions={"my_function": my_function}
)
```

## Development

### Running Tests

```bash
# API validation + cache tests
./test_dynamic_specimen.py

# Unit tests (if available)
pytest
```

### Environment Variables

```bash
# Override default port
export PORT=8080
python -m ascribe_link

# Specify specimens directory
export SPECIMENS_DIR=/path/to/specimens
python -m ascribe_link
```

## Dependencies

Core:
- `litestar` - Modern ASGI web framework
- `pyvista` - 3D mesh processing
- `numpy` - Numerical arrays

Optional:
- `claude-agent-sdk` - AI agent generation
- `firejail` - Sandbox for agent code execution

See `pyproject.toml` for full dependency list.

## Architecture

```
┌─────────────┐
│ VR Client   │  (Ascribe-XR)
│ (Godot)     │
└──────┬──────┘
       │ HTTP/REST
       ▼
┌─────────────────────────────┐
│  Ascribe-Link (Litestar)    │
│                             │
│  ┌──────────────────────┐   │
│  │ Specimen Catalog     │   │
│  │ (Static Files)       │   │
│  └──────────────────────┘   │
│                             │
│  ┌──────────────────────┐   │
│  │ Processing Functions │   │
│  │ (Dynamic Generation) │   │
│  └──────────────────────┘   │
│                             │
│  ┌──────────────────────┐   │
│  │ Room Result Cache    │   │
│  │ (Multiplayer Sync)   │   │
│  └──────────────────────┘   │
└─────────────────────────────┘
```

## Performance

**Benchmarks (MacBook Pro M1):**
- Sphere generation (32 resolution): ~50ms
- Sphere generation (128 resolution): ~500ms
- Cache hit latency: <10ms
- Typical speedup: 10-50x on cache hits

## Contributing

1. Add new parametric functions to `ascribe_link/parametric.py`
2. Register in `app.py`
3. Create specimen.json with schema
4. Test with `./test_dynamic_specimen.py`
5. Submit PR

## Related Projects

- [Ascribe-XR](https://github.com/lbl-camera/Ascribe-XR) - VR client (Godot)
- [Paper (ACM)](https://dl.acm.org/doi/10.1145/3731599.3767368) - VRST 2024 publication
