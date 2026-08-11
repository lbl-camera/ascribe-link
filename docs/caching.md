# Room caching

Several people in a shared VR session look at the same specimen. Computing it
once per peer wastes time and, worse, risks handing peers subtly different
data. `RoomResultCache` makes the result a property of the *room* instead of
the request.

## How it works

Each room holds **one** cached result, keyed by function name and a hash of
the parameters:

- **First peer** — cache miss, computes, stores.
- **Later peers, same parameters** — cache hit, served immediately.
- **Any peer, different parameters** — miss; the room's old entry is replaced,
  because the new specimen is about to replace the old one for everyone.

```text
Room: "ascribe"

Peer A: generate_sphere(radius=2.0, resolution=64)
→ miss → compute (500 ms) → store

Peer B: generate_sphere(radius=2.0, resolution=64)   [same params]
→ hit → cached result (<10 ms)

Peer C: generate_sphere(radius=3.0, resolution=64)   [different params]
→ miss → replaces the room's entry → compute → store
```

One entry per room is a deliberate bound: it keeps memory flat when the
payloads are hundreds of megabytes, and it matches the usage pattern, where a
room is looking at one thing at a time.

Entries also expire on a TTL, five minutes by default:

```python
result_cache = RoomResultCache(ttl_seconds=300.0)
```

The cache is thread-safe, and stores the raw result object rather than a
serialized form so that every endpoint can normalize on read into whatever
shape it needs — JSON for `/invoke`, an {doc}`envelope` for `/data`.

## Choosing a room id

`room_id` defaults to `ascribe` on every endpoint that takes one. Pass a
distinct id per multiplayer session if you run more than one at a time;
sessions that share an id share a cache slot and will evict each other.

## Inspecting it

```bash
curl http://localhost:8000/api/processing/cache/stats   # per-room entries, access counts
curl -X POST http://localhost:8000/api/processing/cache/clear
```

Cache hits and misses are also logged at INFO, with the room, function name,
and parameter keys.
