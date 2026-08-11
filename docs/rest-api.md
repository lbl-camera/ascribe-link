# REST API

All endpoints are JSON unless noted. Mesh and volume payloads from the
specimen `data` endpoints use the binary {doc}`envelope` format rather than
JSON, because base64-in-JSON roughly doubles the bytes on the wire for
multi-hundred-megabyte volumes.

## Specimens

### `GET /api/specimens/`

List every specimen in the catalog — those found in the specimens directory,
those registered in code, and (in relay mode) those aggregated from workers.
Each entry carries an `is_dynamic` flag.

### `GET /api/specimens/{specimen_id}`

Full metadata for one specimen, including the JSON Schema for dynamic
specimens.

### `GET /api/specimens/{specimen_id}/thumbnail`

The preview image. Code-registered specimens may return a data URI instead of
a file.

### `GET /api/specimens/{specimen_id}/data`

Fetch the specimen data.

- **Static specimen** — returns the data file.
- **Dynamic specimen** — invokes the function and returns an envelope. With no
  query string, the schema defaults are used.

| Query param | Default | Meaning |
| --- | --- | --- |
| `params` | schema defaults | JSON object of function parameters |
| `room_id` | `ascribe` | Cache room (see {doc}`caching`) |

```bash
curl -G http://localhost:8000/api/specimens/parametric_sphere/data \
  --data-urlencode 'params={"radius": 2.0, "resolution": 64}' \
  --data-urlencode 'room_id=ascribe'
```

A `params` value that isn't valid JSON, or that isn't a JSON object, is a
`400`.

### `POST /api/specimens/{specimen_id}/data`

The same operation with the parameters in the body — the practical choice once
the parameter set is large enough to be awkward in a query string:

```json
{
  "params": {"radius": 2.0, "resolution": 64},
  "room_id": "ascribe"
}
```

### `POST /api/specimens/{specimen_id}/start`

Start a dynamic specimen load as a **background job** and return immediately:

```json
{"job_id": "…", "status": "running"}
```

`status` is `"done"` right away on a cache hit. Static specimens are a `400` —
use `GET /data` for those. Poll the job as described in {doc}`jobs`. This is
the endpoint to use for anything slow enough that a client would otherwise sit
on a blocked request.

### `GET /api/specimens/reload`

Re-scan the specimens directory. Returns the resulting counts.

## Processing

### `GET /api/processing/functions`

Every registered function, with its parameter schema and return type
(`mesh`, `volume`, `point_cloud`, `image`).

### `GET /api/processing/functions/{name}/schema`

Just the JSON Schema for one function's parameters. `404` if the function
isn't registered.

### `POST /api/processing/invoke`

Invoke a function and get the typed result as JSON.

```json
{
  "function_name": "generate_sphere",
  "args": [],
  "kwargs": {"radius": 2.0, "resolution": 64},
  "room_id": "ascribe"
}
```

The result is cached per room, so a second identical request is served from
cache. Both sync and async functions are supported. An unknown
`function_name` is a `404`.

### `GET /api/processing/cache/stats`

Cache usage per room — entry count, function names, access counts.

### `POST /api/processing/cache/clear`

Drop every cached result.

## Jobs

`GET /api/jobs/{job_id}/progress`, `GET /api/jobs/{job_id}/result`, and
`DELETE /api/jobs/{job_id}` are documented in {doc}`jobs`.

## Federation

`WS /ws/federation/{worker_id}` is the worker↔relay channel — see
{doc}`federation`.
