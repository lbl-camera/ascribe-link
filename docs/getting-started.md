# Getting started

## Install

```bash
pip install ascribe-link
```

For AI agent generation (see {doc}`agent`), install the optional extra:

```bash
pip install "ascribe-link[agent]"
```

From a source checkout:

```bash
git clone https://github.com/lbl-camera/ascribe-link.git
cd ascribe-link
pip install -e ".[test]"
```

## Run the server

Installing puts an `ascribe-link` console script on your PATH:

```bash
ascribe-link
```

The server binds `0.0.0.0:8000` by default. Check that it came up:

```bash
curl http://localhost:8000/api/specimens/
curl http://localhost:8000/api/processing/functions
```

### Command-line options

| Flag | Default | Meaning |
| --- | --- | --- |
| `--host` | `0.0.0.0` | Bind host |
| `--port` | `8000` | Bind port |
| `--specimens-dir` | `./specimens` | Specimen directory to scan |
| `--reload` | off | Auto-reload on source changes (development) |
| `--relay` | off | Relay mode — accept worker connections, aggregate their specimens |
| `--worker URL` | — | Worker mode — connect to the relay at `URL` |
| `--worker-id` | hostname | Identity this worker registers under |
| `--enable-agent` | off | Enable AI agent generation (needs `claude-agent-sdk`) |
| `--agent-model` | `claude-opus-4-8` | Model used for agent generation |
| `--agent-timeout` | `300` | Agent timeout, seconds |
| `--verbose` / `-v` | off | Verbose logging |

See {doc}`federation` for relay and worker modes.

## Embed it in your own application

`create_app()` returns a configured Litestar app, so you can register your own
functions and serve them with any ASGI server:

```python
import uvicorn

from ascribe_link import create_app
from ascribe_link.models import MeshResult


def my_function(param1: float = 1.0, param2: int = 32) -> MeshResult:
    ...  # generate a mesh
    return MeshResult(vertices=..., indices=...)


app = create_app(
    specimens_dir="/data/specimens",
    mesh_functions={"my_function": my_function},
)

uvicorn.run(app, host="0.0.0.0", port=8000)
```

`create_app()` accepts:

`specimens_dir`
: Directory of specimen bundles to scan. Defaults to `./specimens`.

`mesh_functions`
: Mapping of name → callable, registered as processing functions.

`relay_mode`
: Accept worker connections and aggregate their catalogs.

`enable_agent`, `agent_model`, `agent_timeout`
: AI agent generation — see {doc}`agent`.

## Specimen directory layout

Each specimen is a subdirectory holding a `specimen.json` metadata file, a
thumbnail, and (for static specimens) a data file:

```text
specimens/
├── brain/
│   ├── specimen.json   # metadata
│   ├── brain.stl       # data file
│   └── thumbnail.png   # preview image
└── parametric_sphere/
    ├── specimen.json   # metadata with schema + function_name
    └── thumbnail.png
```

The directory is scanned at startup. `GET /api/specimens/reload` re-scans it
without a restart.

## Run the tests

```bash
pip install -e ".[test]"
pytest
```
