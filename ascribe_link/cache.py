"""Result cache for multiplayer synchronization.

Caches processing results per room to ensure all peers receive identical data
without redundant computation.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from dataclasses import dataclass
from typing import Any


@dataclass
class CachedResult:
    """A cached processing result."""

    room_id: str
    function_name: str
    params_hash: str
    result: Any
    timestamp: float
    access_count: int = 0


class RoomResultCache:
    """Cache for processing results, keyed by room ID.

    Each room can have one cached result at a time. When a new request comes in
    for a room, the old cached result for that room is invalidated (since the new
    specimen will replace the old one for all peers).

    Thread-safe for concurrent access.
    """

    def __init__(self, ttl_seconds: float = 300.0) -> None:
        """Initialize the cache.

        Parameters
        ----------
        ttl_seconds : float
            Time-to-live for cache entries (default 5 minutes).
            Entries older than this are automatically evicted.
        """
        self._cache: dict[str, CachedResult] = {}  # room_id -> CachedResult
        self._lock = threading.RLock()
        self._ttl = ttl_seconds

    def get(
        self,
        room_id: str,
        function_name: str,
        params: dict[str, Any],
    ) -> Any | None:
        """Get a cached result if available.

        Parameters
        ----------
        room_id : str
            Room identifier (e.g., "ascribe")
        function_name : str
            Name of the processing function
        params : dict
            Function parameters (must match exactly)

        Returns
        -------
        Any | None
            Cached result if found and valid, otherwise None.
            Stored objects may be raw Result instances or dicts;
            callers must handle either shape.
        """
        params_hash = self._hash_params(params)

        with self._lock:
            entry = self._cache.get(room_id)

            if entry is None:
                return None

            # Check if function and params match
            if entry.function_name != function_name or entry.params_hash != params_hash:
                # Different request for this room — invalidate
                del self._cache[room_id]
                return None

            # Check TTL
            age = time.time() - entry.timestamp
            if age > self._ttl:
                del self._cache[room_id]
                return None

            # Valid cache hit
            entry.access_count += 1
            return entry.result

    def put(
        self,
        room_id: str,
        function_name: str,
        params: dict[str, Any],
        result: Any,
    ) -> None:
        """Store a processing result in the cache.

        Any previous cached result for this room is replaced.

        Parameters
        ----------
        room_id : str
            Room identifier
        function_name : str
            Name of the processing function
        params : dict
            Function parameters
        result : Any
            Processing result to cache. May be a raw Result
            (MeshResult/VolumeResult/...) or a JSON-ready dict;
            consumers are responsible for normalizing on read.
        """
        params_hash = self._hash_params(params)

        with self._lock:
            self._cache[room_id] = CachedResult(
                room_id=room_id,
                function_name=function_name,
                params_hash=params_hash,
                result=result,
                timestamp=time.time(),
                access_count=0,
            )

    def invalidate_room(self, room_id: str) -> bool:
        """Explicitly invalidate the cached result for a room.

        Parameters
        ----------
        room_id : str
            Room identifier

        Returns
        -------
        bool
            True if an entry was removed, False if no entry existed
        """
        with self._lock:
            if room_id in self._cache:
                del self._cache[room_id]
                return True
            return False

    def clear(self) -> None:
        """Clear all cached results."""
        with self._lock:
            self._cache.clear()

    def cleanup_expired(self) -> int:
        """Remove expired entries from the cache.

        Returns
        -------
        int
            Number of entries removed
        """
        now = time.time()
        with self._lock:
            expired = [
                room_id
                for room_id, entry in self._cache.items()
                if now - entry.timestamp > self._ttl
            ]
            for room_id in expired:
                del self._cache[room_id]
            return len(expired)

    def stats(self) -> dict[str, Any]:
        """Get cache statistics.

        Returns
        -------
        dict
            Statistics including entry count, room IDs, access counts
        """
        with self._lock:
            entries = []
            for room_id, entry in self._cache.items():
                age = time.time() - entry.timestamp
                entries.append({
                    "room_id": room_id,
                    "function_name": entry.function_name,
                    "age_seconds": round(age, 2),
                    "access_count": entry.access_count,
                })

            return {
                "total_entries": len(self._cache),
                "ttl_seconds": self._ttl,
                "entries": entries,
            }

    @staticmethod
    def _hash_params(params: dict[str, Any]) -> str:
        """Create a deterministic hash of parameters.

        Parameters
        ----------
        params : dict
            Function parameters

        Returns
        -------
        str
            Hex digest of parameter hash
        """
        # Sort keys for deterministic serialization
        json_str = json.dumps(params, sort_keys=True, separators=(',', ':'))
        return hashlib.sha256(json_str.encode('utf-8')).hexdigest()[:16]
