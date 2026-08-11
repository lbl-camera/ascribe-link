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
- **Background Jobs:** Poll progress and fetch results for long-running generation
- **Federation:** Relay mode for aggregating specimens from worker nodes
- **AI Agent Generation:** Optional Claude-driven mesh/volume generation, Firejail-sandboxed

## Quick Start

```bash
pip install ascribe-link
ascribe-link
```

Server starts at `http://localhost:8000`. Check it:

```bash
curl http://localhost:8000/api/specimens/
curl http://localhost:8000/api/processing/functions
```

From a source checkout:

```bash
git clone https://github.com/lbl-camera/ascribe-link.git
cd ascribe-link
pip install -e ".[test]"
pytest
```

## Documentation

Full documentation lives in `docs/` and covers the REST API, writing dynamic
specimens, room caching, the background job flow, the binary envelope wire
format, federation, and AI agent generation.

```bash
pip install -e ".[docs]"
sphinx-build -W -b html docs docs/_build/html
```

| Topic | Page |
| --- | --- |
| Install, run, embed, CLI flags | `docs/getting-started.md` |
| Every HTTP endpoint | `docs/rest-api.md` |
| Writing parametric functions | `docs/dynamic-specimens.md` |
| Room-based result caching | `docs/caching.md` |
| Long-running jobs and progress | `docs/jobs.md` |
| Binary envelope wire format | `docs/envelope.md` |
| Relay and worker modes | `docs/federation.md` |
| AI agent generation and sandboxing | `docs/agent.md` |
| Python API | `docs/api.md` |

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

## Project Structure

```
ascribe_link/
├── app.py              # Litestar application factory
├── models.py           # Data models (specimens, results, requests)
├── processing.py       # FunctionRegistry and schema generation
├── cache.py            # RoomResultCache for multiplayer
├── job_registry.py     # Background job store with TTL sweeping
├── envelope.py         # Binary wire format for mesh/volume payloads
├── specimen_store.py   # Specimen directory management
├── parametric.py       # Built-in parametric functions
├── agent_generator.py  # AI agent generation
├── sandbox.py          # Firejail sandbox for agent code
├── routes/
│   ├── specimens.py    # Specimen catalog endpoints
│   ├── processing.py   # Function invocation endpoints
│   ├── jobs.py         # Job progress / result / cancel
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

## Dependencies

Core:
- `litestar` - Modern ASGI web framework
- `pyvista` - 3D mesh processing
- `numpy` - Numerical arrays

Optional:
- `claude-agent-sdk` - AI agent generation
- `firejail` - Sandbox for agent code execution

See `pyproject.toml` for full dependency list.

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
4. Test with `pytest`
5. Submit PR

Releases are cut by pushing a `vX.Y.Z` tag; the version comes from the tag via
hatch-vcs and GitHub Actions publishes to PyPI.

## Related Projects

- [Ascribe-XR](https://github.com/lbl-camera/Ascribe-XR) - VR client (Godot)
- [Paper (ACM)](https://dl.acm.org/doi/10.1145/3731599.3767368) - VRST 2024 publication
