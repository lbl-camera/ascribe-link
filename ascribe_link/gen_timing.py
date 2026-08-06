"""Coarse end-to-end timing marks for the AI Generate flow.

Single-run, order-of-magnitude instrumentation: gt_reset() is called when a
new job is submitted, then gt_mark() prints elapsed time since reset and
since the previous mark. Output goes to the logger. Grep for [GENTIMING].
"""
from __future__ import annotations

import logging
import time

logger = logging.getLogger(__name__)

_t0: float | None = None
_last: float | None = None


def gt_reset() -> None:
    global _t0, _last
    _t0 = None
    _last = None


def gt_mark(label: str) -> None:
    global _t0, _last
    now = time.perf_counter()
    if _t0 is None:
        _t0 = now
        _last = now
    line = "[GENTIMING] %-48s t=%9.3fs (+%.3fs)" % (label, now - _t0, now - _last)
    # Logger only — a direct print() blocks the calling thread (and with it
    # the event loop) whenever the Windows console is paused, e.g. during
    # quick-edit text selection. The CLI routes logging through a queue.
    logger.info(line)
    _last = now
