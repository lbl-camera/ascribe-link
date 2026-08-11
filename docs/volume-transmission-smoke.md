# Volume Transmission — Server Smoke Test

Run date: 2026-04-24
Commit: `5053d37` (branch `volume-transmission`)

Smoke test exercising the new binary envelope wire format on a running server (port 8001).

## Results

### 1. Gaussian volume listed in `/api/specimens/`

```
{"id":"generate_gaussian_volume","display_name":"Parametric Gaussian Volume",
 "type":"volume","is_dynamic":true,"tags":["parametric","volume","dynamic"]}
```

Pass.

### 2. `/api/specimens/generate_gaussian_volume/data` — envelope

- `content-type: application/x-ascribe-envelope-v1`
- `content-length: 1048696` (= 1048576 raw float32 bytes + 120-byte preamble, ~0 overhead)
- Preamble: `{"type": "volume", "shape": [64, 64, 64], "dtype": "float32", "spacing": [0.015625, 0.015625, 0.015625], "origin": [0.0, 0.0, 0.0]}`
- Decoded volume: peak 1.0 at center (32,32,32), min 0.0155, all voxels in [0,1]

Pass.

### 3. `/api/specimens/generate_sphere/data` — mesh via envelope

- `content-type: application/x-ascribe-envelope-v1`
- Decoded: `MeshResult` with 962 vertices, 1920 triangles, 2886 normals
- Payload 46,277 bytes

Pass. Confirms the previously-JSON-bottlenecked mesh path now ships as raw bytes.

### 4. `/api/specimens/brain/data` — static STL still streams raw

- `content-type: application/octet-stream`
- `content-disposition: attachment; filename="brain-1-no-cerebellum.stl"`
- `content-length: 35096184`
- Body starts with `STL File created by netfabb` (ASCII STL file)

Pass. Static mesh file path untouched by the envelope change.

## Test suite

```
pytest -q
```

All 73 tests pass, including:
- 13 envelope tests (round-trip, error handling, zero-copy cache)
- 4 parametric volume tests
- 2 envelope-endpoint integration tests
- 1 cache widening regression test
- 2 cross-endpoint cache consistency tests
- 2 job-cache normalization regression tests
- 3 static `.npy` specimen tests
- 5 agent output dispatch tests

Baseline before this branch: 40 tests. Net added: +33.
