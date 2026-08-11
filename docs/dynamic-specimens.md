# Dynamic specimens

A dynamic specimen is a Python function the client can call with parameters.
The server derives a JSON Schema from the function's signature, the client
renders that schema as a parameter panel, and each parameter change is a new
request against the room cache.

## 1. Write the function

Type hints and defaults are the interface — they become the schema.

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

## 2. Register it

```python
registry.register_function(generate_torus, "generate_torus", return_type="mesh")
```

`register_specimen()` goes further and registers the function *as* a specimen
in one call, with display name, description, tags, thumbnail, and story text —
no `specimen.json` on disk required.

Functions can also be passed straight to `create_app(mesh_functions={...})`.

## 3. Describe it (optional)

The schema is generated automatically from the signature, but writing
`specimen.json` yourself buys better UX — slider bounds, in particular, which
a type hint can't express.

```json
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
      "major_radius": {"type": "number", "default": 1.0, "minimum": 0.5, "maximum": 3.0},
      "minor_radius": {"type": "number", "default": 0.3, "minimum": 0.1, "maximum": 1.0},
      "segments":     {"type": "number", "default": 32,  "minimum": 8,   "maximum": 128}
    }
  }
}
```

## Type mapping

| Python annotation | JSON Schema |
| --- | --- |
| `float` | `"number"` |
| `int` | `"integer"` |
| `str` | `"string"` |
| `bool` | `"boolean"` |
| `Literal["a", "b"]` | `{"enum": ["a", "b"]}` |

Defaults are taken from the signature.

## Return types

| Return annotation | Payload |
| --- | --- |
| `MeshResult` | vertices, indices, normals |
| `VolumeResult` | shape, dtype, data, spacing, origin |
| `PointCloudResult` | points, colors, scalars |
| `ImageResult` | width, height, channels, data |

`MeshResult.from_pyvista()` and `VolumeResult.from_numpy()` cover the common
construction paths.

## Reporting progress

Declare a `ProgressReporter` parameter and the server injects one — the
parameter is not part of the generated schema, so it stays invisible to the
client:

```python
from ascribe_link.models import VolumeResult
from ascribe_link.progress import ProgressReporter


def segment_volume(
    threshold: float = 0.5,
    reporter: ProgressReporter | None = None,
) -> VolumeResult:
    if reporter:
        reporter.report("loading volume")
    ...
    if reporter:
        reporter.report("thresholding")
    ...
```

Each `report()` call becomes a message on `GET /api/jobs/{job_id}/progress`
({doc}`jobs`). Without a job — a plain `POST /invoke` — the reporter is a
no-op, so the same function works either way.

## Async functions

`async def` functions are supported directly; the server awaits them. Use
async for anything I/O-bound, and keep in mind that a long CPU-bound `async`
function blocks the event loop — push that work to a thread.
