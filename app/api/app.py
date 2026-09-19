"""FastAPI application factory.

One surface, one audience: a hiring manager reviewing scraped candidate
records. There is no candidate-facing side to this service.
"""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.api.routes_candidates import router as candidates_router
from app.api.routes_health import HTTP_REQUEST_DURATION_SECONDS, HTTP_REQUESTS_TOTAL
from app.api.routes_health import router as health_router
from app.config import get_settings
from app.db.engine import init_db

logger = logging.getLogger(__name__)


@asynccontextmanager
async def _lifespan(app: FastAPI):
    # Convenience for local/dev/demo so the app runs against a fresh SQLite
    # file with zero setup. Real deployments run `alembic upgrade head` as
    # part of the release, as the Dockerfile's CMD does.
    init_db()
    settings = get_settings()
    logger.info(
        "Hiring analyst API starting. offline=%s model=%s",
        settings.offline,
        settings.model,
    )
    yield


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title="Hiring Analyst API",
        description="Scraped candidate records in, hiring recommendations out.",
        version="2.0.0",
        docs_url="/api/docs",
        openapi_url="/api/openapi.json",
        lifespan=_lifespan,
    )

    origins = (
        ["*"]
        if settings.cors_origins.strip() == "*"
        else [o.strip() for o in settings.cors_origins.split(",") if o.strip()]
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=origins != ["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def _metrics_middleware(request: Request, call_next):
        start = time.perf_counter()
        response = await call_next(request)
        duration = time.perf_counter() - start
        # Route template, not the raw path, so /api/candidates/<uuid> does not
        # explode Prometheus's label cardinality.
        path_label = (
            request.scope.get("route").path if request.scope.get("route") else request.url.path
        )
        HTTP_REQUESTS_TOTAL.labels(
            method=request.method, path=path_label, status=response.status_code
        ).inc()
        HTTP_REQUEST_DURATION_SECONDS.labels(method=request.method, path=path_label).observe(
            duration
        )
        return response

    @app.exception_handler(StarletteHTTPException)
    async def _http_exception_handler(request: Request, exc: StarletteHTTPException):
        return JSONResponse(
            status_code=exc.status_code, content={"error": exc.detail, "detail": ""}
        )

    @app.exception_handler(RequestValidationError)
    async def _validation_exception_handler(request: Request, exc: RequestValidationError):
        return JSONResponse(
            status_code=422, content={"error": "validation_error", "detail": str(exc.errors())}
        )

    app.include_router(health_router)
    app.include_router(candidates_router)

    return app


app = create_app()

__all__ = ["app", "create_app"]
