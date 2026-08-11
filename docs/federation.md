# Federation

Data often can't come to the client. It sits on a machine inside a facility
network — a NERSC node, a beamline workstation — that a headset on a
conference-center wifi can't route to. Federation inverts the direction: the
data-side server dials **out** to a relay the client can reach.

```text
NERSC (worker) ──WS──▶ Neutral host (relay) ◀──HTTP── Quest (client)
```

The worker holds the specimens and does the computation. The relay holds no
data; it aggregates catalogs and proxies requests. Only the outbound WebSocket
has to get through the facility firewall.

## Running it

Relay, on a host the clients can reach:

```bash
ascribe-link --relay --port 8000
```

Worker, on the machine with the data:

```bash
ascribe-link --worker ws://relay.example.com:8000 \
             --worker-id nersc-01 \
             --specimens-dir /data/specimens
```

Equivalently, in code: `create_app(relay_mode=True)`.

The worker connects, registers its specimens and functions, and re-registers
on reconnect if the link drops.

## What clients see

A client talks only to the relay, over the same REST API as a standalone
server ({doc}`rest-api`). The relay's catalog is the union of its own
specimens and every worker's.

Federated specimen ids are namespaced `{worker_id}:{specimen_id}`. The relay
splits on the colon to route. `GET /data`, `POST /start`, thumbnails, job
progress, job results, and job cancellation are all proxied to the owning
worker; the relay mirrors terminal job status locally so a completed federated
job can be served and swept like a local one.

## The worker protocol

`WS /ws/federation/{worker_id}`, JSON messages both ways. The worker sends
registration and catalog updates; the relay sends proxied requests tagged with
a request id, and the worker replies with the matching id. A worker that
vanishes is unregistered, and its specimens leave the aggregated catalog.

## Timeouts

Proxied requests are bounded. A worker that stops answering surfaces to the
client as a `404` naming the worker rather than a hung request — for example
`Timeout fetching data from worker: nersc-01`.
