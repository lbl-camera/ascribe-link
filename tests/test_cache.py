"""Tests that RoomResultCache stores arbitrary result objects."""
from __future__ import annotations

import numpy as np

from ascribe_link.cache import RoomResultCache
from ascribe_link.models import VolumeResult


def test_cache_stores_volume_result():
    cache = RoomResultCache()
    arr = np.ones((2, 2, 2), dtype=np.float32)
    result = VolumeResult.from_numpy(arr)
    cache.put("room1", "fn", {"a": 1}, result)
    got = cache.get("room1", "fn", {"a": 1})
    assert got is result  # same object, no serialization round-trip
