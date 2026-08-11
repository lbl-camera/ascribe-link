# Binary envelope format

Media type: `application/x-ascribe-envelope-v1`

Mesh and volume payloads are returned from the specimen `data` endpoints in a
binary envelope rather than JSON. A 512³ `uint16` volume is 256 MB of raw
bytes; base64 inside JSON inflates that by a third before the client has
parsed anything, and the JSON parse itself is slow. The envelope is a small
JSON header followed by the array bytes verbatim, so the client can point a
buffer at it.

## Layout

```text
<4-byte little-endian uint32: preamble_length>
<preamble_length bytes: UTF-8 JSON preamble>
<raw bytes: one or more contiguous data blocks>
```

The preamble's `type` field selects how the trailing bytes are read.

### Volume

```json
{"type": "volume", "shape": [128, 128, 128], "dtype": "uint16",
 "spacing": [1.0, 1.0, 1.0], "origin": [0.0, 0.0, 0.0]}
```

`spacing` and `origin` are present only when set. One data block follows: the
array in C order, `dtype` as given.

### Mesh

```json
{"type": "mesh",
 "vertex_count": 1234, "vertex_dtype": "float32",
 "index_count": 6000,  "index_dtype": "uint32",
 "normal_count": 1234, "normal_dtype": "float32"}
```

Up to three blocks follow, in order: vertices (`vertex_count × 3` float32),
indices (`index_count` uint32), normals (`normal_count × 3` float32). The
normals block is omitted when `normal_count` is 0.

The dtypes are fixed in v1 — the decoder rejects anything but float32
vertices/normals and uint32 indices. They appear in the preamble so a v2 can
widen them without a flag day.

## Reading one in Python

```python
from ascribe_link.envelope import decode_envelope

result = decode_envelope(response.content)   # MeshResult | VolumeResult
```

`encode_envelope()` is the other direction. Both raise `ValueError` on a
truncated buffer or an unknown `type`.

## Which endpoints use it

`GET`/`POST /api/specimens/{id}/data` for dynamic mesh and volume specimens.
`POST /api/processing/invoke` and `GET /api/jobs/{id}/result` return JSON —
convenient for scripting and small results. Encoding runs on a worker thread,
never the event loop: a multi-second encode on the loop was observed starving
job progress polls.
