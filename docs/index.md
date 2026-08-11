# ascribe-link

HTTP server bridging Python processing functions and curated specimens to
[Ascribe-XR](https://github.com/lbl-camera/Ascribe-XR), the Godot/OpenXR
scientific visualization client.

A running server exposes two kinds of things to a VR client:

**Specimens** — a catalog of 3D datasets. A *static* specimen is a file on
disk (an `.stl` mesh, an `.npy` volume). A *dynamic* specimen is a Python
function plus a JSON Schema describing its parameters, so the client can
render a parameter panel and re-generate the data on demand.

**Processing functions** — callables registered with the server, invocable
over HTTP with typed results (mesh, volume, point cloud, image). Results are
cached per *room* so every peer in a multiplayer session gets the same data
without recomputing it.

```{toctree}
:maxdepth: 2
:caption: Using the server

getting-started
rest-api
dynamic-specimens
caching
```

```{toctree}
:maxdepth: 2
:caption: Advanced

jobs
envelope
federation
agent
volume-transmission-smoke
```

```{toctree}
:maxdepth: 2
:caption: Reference

api
```

## Indices

- {ref}`genindex`
- {ref}`modindex`
- {ref}`search`
