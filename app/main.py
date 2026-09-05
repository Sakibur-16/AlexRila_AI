"""Application entry point.

Builds the FastAPI application, wires middleware and error handlers, and
constructs the pipeline once during startup so that no request pays engine
initialisation cost.
"""

from __future__ import annotations

import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware

from app.api.dependencies import (
    REQUEST_ID_HEADER,
    build_ocr_provider,
    build_pipeline,
    sanitize_request_id,
)
from app.api.errors import register_error_handlers
from app.api.routes import health, receipts, reviews
from app.api.security import RequireAPIKey
from app.core.config import Settings, get_settings
from app.core.logging import (
    bind_request_context,
    clear_request_context,
    configure_logging,
    get_logger,
)
from app.core.versions import API_VERSION, PIPELINE_VERSION, SCHEMA_VERSION

logger = get_logger(__name__)

API_PREFIX = f"/api/{API_VERSION}"

_DESCRIPTION = """\
Receipt OCR and structured extraction.

Submit a receipt image, receive validated JSON. The response contract is stable
and independent of which OCR engine produced it.

**Monetary values are JSON strings** (`"25.99"`), not numbers, so that decimal
precision survives `JSON.parse`. Parse them into your own decimal type.

**`success: true` means processed, not certain.** Check `warnings`,
`data.confidence` and `data.review.review_required` before acting on a value.

**Absent fields are `null`.** Never `"N/A"`, never `0`. A `null` means the
receipt did not carry that information.
"""


class RequestContextMiddleware(BaseHTTPMiddleware):
    """Assign a correlation id and bind it to the logging context.

    Runs before routing so that even a 404 or a validation failure carries the
    same id the client sees in the response header.
    """

    async def dispatch(self, request: Request, call_next):  # type: ignore[no-untyped-def]
        request_id = sanitize_request_id(request.headers.get(REQUEST_ID_HEADER))
        request.state.request_id = request_id
        request.state.started_at = time.perf_counter()

        bind_request_context(
            request_id=request_id,
            method=request.method,
            path=request.url.path,
        )
        try:
            response = await call_next(request)
        finally:
            clear_request_context()

        response.headers[REQUEST_ID_HEADER] = request_id
        return response


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Build shared collaborators at startup and report engine readiness.

    Settings were stamped onto application state by :func:`create_app`, so an
    explicitly configured app builds its collaborators from that configuration
    rather than re-reading the environment here.
    """
    settings: Settings = getattr(app.state, "settings", None) or get_settings()
    configure_logging(settings, force=True)

    app.state.settings = settings
    app.state.ocr_provider = build_ocr_provider(settings)
    app.state.pipeline = build_pipeline(settings)

    ready, detail = app.state.ocr_provider.health_check()
    logger.info(
        "application_started",
        api_version=API_VERSION,
        pipeline_version=PIPELINE_VERSION,
        schema_version=SCHEMA_VERSION,
        ocr_provider=settings.ocr_provider,
        ocr_ready=ready,
        ocr_detail=detail,
        llm_enabled=settings.llm_enabled,
        environment=settings.app_env,
    )
    if not ready:
        # Startup deliberately succeeds: /ready reports the problem and the
        # orchestrator withholds traffic, which is more diagnosable than a
        # container that will not start.
        logger.error("ocr_provider_not_ready", detail=detail)

    yield

    logger.info("application_stopping")


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build the FastAPI application."""
    settings = settings or get_settings()
    configure_logging(settings)

    app = FastAPI(
        title="Receipt Intelligence API",
        description=_DESCRIPTION,
        version=PIPELINE_VERSION,
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )

    # Stamped before the lifespan runs so that dependencies, error handlers
    # and collaborator construction all see the same configuration.
    app.state.settings = settings

    app.add_middleware(RequestContextMiddleware)

    if settings.cors_allow_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=list(settings.cors_allow_origins),
            allow_credentials=False,
            allow_methods=["GET", "POST"],
            allow_headers=["Content-Type", REQUEST_ID_HEADER],
            expose_headers=[REQUEST_ID_HEADER],
        )

    register_error_handlers(app)

    # Operational endpoints are unversioned and unauthenticated: probes must
    # not move with the API, and an orchestrator has no credential to present.
    app.include_router(health.router)
    # Versioned endpoints require the API key when one is configured.
    app.include_router(receipts.router, prefix=API_PREFIX, dependencies=[RequireAPIKey])
    app.include_router(reviews.router, prefix=API_PREFIX, dependencies=[RequireAPIKey])

    return app


app = create_app()


def _get_lan_ip() -> str:
    """Retrieve primary local network IPv4 address."""
    import socket

    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip: str = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


def main() -> None:
    """Run a development server.

    Provided because ``python app/main.py`` is the natural thing to try, and
    silently constructing an ASGI app and exiting is a poor answer. Production
    runs uvicorn (or gunicorn) directly so the process manager owns worker
    count, restarts and signal handling -- see docs/deployment.md.
    """
    import uvicorn

    settings = get_settings()
    configure_logging(settings)
    lan_ip = _get_lan_ip() if settings.host in ("0.0.0.0", "::") else settings.host

    logger.info(
        "starting_development_server",
        host=settings.host,
        port=settings.port,
        reload=settings.reload,
        lan_ip=lan_ip,
        docs_url=f"http://{lan_ip}:{settings.port}/docs",
    )
    print("\n=======================================================")
    print(f" [API] Server running on LAN IP: http://{lan_ip}:{settings.port}")
    print(f" [DOCS] Swagger Docs (LAN):       http://{lan_ip}:{settings.port}/docs")
    print("=======================================================\n", flush=True)

    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.reload,
        log_config=None,  # structlog owns log formatting
    )


if __name__ == "__main__":
    main()
