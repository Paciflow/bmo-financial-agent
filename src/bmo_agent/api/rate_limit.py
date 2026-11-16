from __future__ import annotations

import time
from collections import defaultdict, deque

from fastapi import HTTPException, Request

# In-memory only: correct for a single-task deployment (no horizontal scaling),
# and resets on restart -- a deliberate tradeoff for a demo, not a production limiter.
_PER_IP_LIMIT = 20
_PER_IP_WINDOW_SECONDS = 300
_GLOBAL_LIMIT = 150
_GLOBAL_WINDOW_SECONDS = 300

_per_ip_hits: dict[str, deque[float]] = defaultdict(deque)
_global_hits: deque[float] = deque()


def _prune(hits: deque[float], window_seconds: int, now: float) -> None:
    while hits and now - hits[0] > window_seconds:
        hits.popleft()


def enforce_rate_limit(request: Request) -> None:
    """Dependency for endpoints that call the LLM or embedding models -- the
    costly, billable paths. Bounds how much a single open demo link can cost
    if it gets scraped or abused, independent of AWS-side autoscaling (there is
    none; this is the only backstop)."""
    now = time.monotonic()
    client_ip = request.client.host if request.client else "unknown"

    ip_hits = _per_ip_hits[client_ip]
    _prune(ip_hits, _PER_IP_WINDOW_SECONDS, now)
    _prune(_global_hits, _GLOBAL_WINDOW_SECONDS, now)

    if len(_global_hits) >= _GLOBAL_LIMIT:
        raise HTTPException(status_code=429, detail="Demo is at capacity, try again shortly.")
    if len(ip_hits) >= _PER_IP_LIMIT:
        raise HTTPException(status_code=429, detail="Too many requests, slow down a little.")

    ip_hits.append(now)
    _global_hits.append(now)
