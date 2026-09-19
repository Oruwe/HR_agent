"""Health, readiness, and Prometheus metrics.

Kept as a separate, unauthenticated, unprefixed router: these are the
endpoints an orchestrator (Kubernetes, a load balancer, Docker Compose's
``healthcheck``) polls constantly, and they must never fail from a transient
blip in a way that could create a restart loop -- ``/health`` is "the process is alive", ``/ready`` is
"the process can serve a request right now".
"""

from __future__ import annotations

import json
import time

from fastapi import APIRouter, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from sqlalchemy import text

from app.db.engine import SessionLocal

router = APIRouter(tags=["health"])

_START_TIME = time.time()

# -- Prometheus metrics -------------------------------------------------------
# Wired from real request handling (see the middleware in app/api/app.py),
# not placeholder values.

HTTP_REQUESTS_TOTAL = Counter(
    "hrte_http_requests_total", "Total HTTP requests", ["method", "path", "status"]
)
HTTP_REQUEST_DURATION_SECONDS = Histogram(
    "hrte_http_request_duration_seconds", "HTTP request duration", ["method", "path"]
)
CANDIDATES_IMPORTED_TOTAL = Counter("hrte_candidates_imported_total", "Candidate records ingested")
ANALYSIS_RUNS_TOTAL = Counter("hrte_analysis_runs_total", "Pool analysis runs")
CHAT_MESSAGES_TOTAL = Counter("hrte_chat_messages_total", "Manager questions answered")


@router.get("/health")
def health() -> dict[str, object]:
    return {"status": "ok", "uptime_s": round(time.time() - _START_TIME, 1)}


@router.get("/ready")
def ready() -> Response:
    """True readiness: can this process actually reach its dependencies?

    Checks the one hard dependency this app cannot run without -- the
    database. The model provider is intentionally excluded: it has a
    documented offline fallback, so its absence degrades quality, not
    availability, and should never fail a readiness probe.
    """
    checks: dict[str, bool] = {}
    try:
        with SessionLocal() as db:
            db.execute(text("SELECT 1"))
        checks["database"] = True
    except Exception:
        checks["database"] = False

    all_ok = all(checks.values())
    return Response(
        content=json.dumps({"ready": all_ok, "checks": checks}),
        media_type="application/json",
        status_code=200 if all_ok else 503,
    )


@router.get("/metrics", include_in_schema=False)
def metrics() -> Response:
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


__all__ = [
    "ANALYSIS_RUNS_TOTAL",
    "CANDIDATES_IMPORTED_TOTAL",
    "CHAT_MESSAGES_TOTAL",
    "HTTP_REQUESTS_TOTAL",
    "HTTP_REQUEST_DURATION_SECONDS",
    "router",
]
