#!/usr/bin/env python3
"""End-to-end smoke test for job-based AI agent specimen loading.

Run against a live ascribe-link server with the agent enabled:

    ascribe-link --enable-agent

Then:

    python test_jobs_e2e.py
"""
import asyncio

import httpx


async def main() -> None:
    base = "http://localhost:8000"
    async with httpx.AsyncClient(timeout=None) as c:
        print("1. POST /api/specimens/ai_generate/start ...")
        r = await c.post(
            f"{base}/api/specimens/ai_generate/start",
            json={
                "params": {"prompt": "make a unit sphere"},
                "room_id": "ascribe",
            },
        )
        assert r.status_code == 200, r.text
        start = r.json()
        job_id = start["job_id"]
        print(f"   job_id={job_id} status={start['status']}")

        print("\n2. Poll /progress ...")
        last_seq = -1
        while True:
            p = (await c.get(
                f"{base}/api/jobs/{job_id}/progress?since={last_seq}"
            )).json()
            for m in p["messages"]:
                print(f"   [{m['seq']}] {m['text']}")
                last_seq = max(last_seq, m["seq"])
            if p["status"] in ("done", "error"):
                print(f"   final status: {p['status']}")
                if p["status"] == "error":
                    print(f"   error: {p['error']}")
                    return
                break
            await asyncio.sleep(0.5)

        print("\n3. GET /result ...")
        r = await c.get(f"{base}/api/jobs/{job_id}/result")
        assert r.status_code == 200, r.text
        result = r.json()
        print(f"   type: {result.get('type')}")
        assert result.get("type") == "mesh"
        verts = result.get("vertices", [])
        idx = result.get("indices", [])
        assert len(verts) > 0 and len(verts) % 3 == 0
        assert len(idx) > 0 and len(idx) % 3 == 0
        print(f"   vertices: {len(verts) // 3}, triangles: {len(idx) // 3}")

        print("\nE2E OK.")


if __name__ == "__main__":
    asyncio.run(main())
