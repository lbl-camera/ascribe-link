#!/usr/bin/env python3
"""Quick test script to verify dynamic specimen API endpoints."""

import asyncio

import httpx


async def test_dynamic_specimen():
    """Test the dynamic specimen endpoints."""
    base_url = "http://localhost:8000"

    async with httpx.AsyncClient() as client:
        print("1. Testing GET /api/specimens/ ...")
        response = await client.get(f"{base_url}/api/specimens/")
        specimens = response.json()
        print(f"   Found {len(specimens)} specimens")

        # Find dynamic specimens
        dynamic = [s for s in specimens if s.get("is_dynamic")]
        print(f"   {len(dynamic)} are dynamic:")
        for s in dynamic:
            print(f"     - {s['id']}: {s['display_name']}")

        print("\n2. Testing GET /api/specimens/parametric_sphere ...")
        response = await client.get(f"{base_url}/api/specimens/parametric_sphere")
        if response.status_code == 200:
            meta = response.json()
            print(f"   Display name: {meta['display_name']}")
            print(f"   Function: {meta.get('function_name')}")
            print(f"   Schema present: {meta.get('schema') is not None}")
            if meta.get("schema"):
                print(f"   Parameters: {list(meta['schema'].get('properties', {}).keys())}")
        else:
            print(f"   ERROR: {response.status_code}")

        print("\n3. Testing GET /api/processing/functions ...")
        response = await client.get(f"{base_url}/api/processing/functions")
        functions = response.json()
        print(f"   Found {len(functions)} registered functions")
        for f in functions:
            print(f"     - {f['name']}: {f.get('return_type', 'unknown')}")

        print("\n4. Testing GET /api/processing/functions/generate_sphere/schema ...")
        response = await client.get(f"{base_url}/api/processing/functions/generate_sphere/schema")
        if response.status_code == 200:
            schema = response.json()
            print(f"   Schema title: {schema.get('title')}")
            print(f"   Properties: {list(schema.get('properties', {}).keys())}")
        else:
            print(f"   ERROR: {response.status_code}")

        print("\n5. Testing POST /api/processing/invoke (generate_sphere) ...")
        response = await client.post(
            f"{base_url}/api/processing/invoke",
            json={
                "function_name": "generate_sphere",
                "args": [],
                "kwargs": {"radius": 2.0, "resolution": 16}
            }
        )
        if response.status_code == 200:
            result = response.json()
            print(f"   Result type: {result.get('type')}")
            print(f"   Vertices: {len(result.get('vertices', []))}")
            print(f"   Indices: {len(result.get('indices', []))}")
            print(f"   Normals present: {result.get('normals') is not None}")
        else:
            print(f"   ERROR: {response.status_code}")
            print(f"   {response.text}")

        print("\n6. Testing cache behavior (invoke same request twice) ...")
        # First request - should miss cache
        response1 = await client.post(
            f"{base_url}/api/processing/invoke",
            json={
                "function_name": "generate_sphere",
                "args": [],
                "kwargs": {"radius": 1.5, "resolution": 32},
                "room_id": "test_room"
            }
        )
        print(f"   First request: {response1.status_code}")

        # Second request - should hit cache
        response2 = await client.post(
            f"{base_url}/api/processing/invoke",
            json={
                "function_name": "generate_sphere",
                "args": [],
                "kwargs": {"radius": 1.5, "resolution": 32},
                "room_id": "test_room"
            }
        )
        print(f"   Second request (same params): {response2.status_code}")

        # Check results are identical
        if response1.json() == response2.json():
            print("   ✅ Cache working: identical results")

        # Different params - should invalidate cache
        response3 = await client.post(
            f"{base_url}/api/processing/invoke",
            json={
                "function_name": "generate_sphere",
                "args": [],
                "kwargs": {"radius": 2.0, "resolution": 32},
                "room_id": "test_room"
            }
        )
        print(f"   Third request (different params): {response3.status_code}")

        print("\n7. Testing GET /api/processing/cache/stats ...")
        response = await client.get(f"{base_url}/api/processing/cache/stats")
        if response.status_code == 200:
            stats = response.json()
            print(f"   Cache entries: {stats.get('total_entries')}")
            for entry in stats.get("entries", []):
                print(f"     - Room: {entry['room_id']}, Function: {entry['function_name']}, Accesses: {entry['access_count']}")
        else:
            print(f"   ERROR: {response.status_code}")

        print("\n✅ All tests completed!")


if __name__ == "__main__":
    asyncio.run(test_dynamic_specimen())
