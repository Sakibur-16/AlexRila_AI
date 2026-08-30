"""API error handling.

Every exception leaves the application through one of these handlers, so a
client always receives the same envelope shape and never receives a stack
trace, an internal path or a provider's raw error text.

The mapping is deliberate:

* :class:`~app.core.exceptions.PipelineError` carries its own status and code.
* A request-validation failure becomes ``INVALID_IMAGE`` / 422 with field
  paths but no echoed values.
* Anything unexpected becomes ``INTERNAL_ERROR`` / 500 with the request id --
  the full exception goes to the logs, where it belongs, and the client gets a
  correlation handle instead.
"""

from __future__ import annotations

import time
from typing import Any

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.api.dependencies import settings_for
from app.core.exceptions import ErrorCode, PipelineError
from app.core.logging import get_logger
from app.schemas.response import APIResponse, ErrorDetail, ResponseProcessing

logger = get_logger(__name__)

#: Starlette renamed this constant; resolve it at import so the module works on
#: either version without emitting a deprecation warning per request.
_UNPROCESSABLE_CONTENT: int = getattr(status, "HTTP_422_UNPROCESSABLE_CONTENT", 422)


def _request_id(request: Request) -> str:
    """Correlation id stamped on the request by the middleware."""
    return getattr(request.state, "request_id", "unknown")


def _elapsed_ms(request: Request) -> float:
    started: float | None = getattr(request.state, "started_at", None)
    if started is None:
        return 0.0
    return (time.perf_counter() - started) * 1000.0


def _envelope(
    request: Request,
    errors: list[ErrorDetail],
    http_status: int,
) -> JSONResponse:
    """Render the failure envelope."""
    payload: APIResponse[Any] = APIResponse(
        success=False,
        data=None,
        warnings=(),
        errors=tuple(errors),
        processing=ResponseProcessing(
            request_id=_request_id(request),
            processing_time_ms=round(_elapsed_ms(request), 2),
        ),
    )
    return JSONResponse(
        status_code=http_status,
        content=payload.model_dump(mode="json"),
        headers={"X-Request-ID": _request_id(request)},
    )


async def pipeline_error_handler(request: Request, exc: PipelineError) -> JSONResponse:
    """Render a structured pipeline failure."""
    settings = settings_for(request)
    logger.warning(
        "request_failed",
        error_code=exc.code.value,
        http_status=exc.http_status,
        details=exc.details or None,
        path=request.url.path,
    )
    return _envelope(
        request,
        [
            ErrorDetail(
                code=exc.code,
                message=exc.message,
                details=exc.details if settings.debug_errors and exc.details else None,
            )
        ],
        exc.http_status,
    )


async def validation_error_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    """Render a request-validation failure.

    Reports which field was rejected and why, but never echoes the submitted
    value -- it could be document content or a credential.
    """
    errors = [
        ErrorDetail(
            code=ErrorCode.INVALID_IMAGE,
            message=str(error.get("msg", "Invalid request.")),
            field=".".join(str(part) for part in error.get("loc", ()) if part != "body"),
        )
        for error in exc.errors()
    ] or [ErrorDetail(code=ErrorCode.INVALID_IMAGE, message="Invalid request.")]

    return _envelope(request, errors, _UNPROCESSABLE_CONTENT)


async def http_exception_handler(request: Request, exc: StarletteHTTPException) -> JSONResponse:
    """Render framework HTTP errors in the standard envelope."""
    code = {
        status.HTTP_404_NOT_FOUND: ErrorCode.INVALID_IMAGE,
        413: ErrorCode.REQUEST_TOO_LARGE,
        status.HTTP_415_UNSUPPORTED_MEDIA_TYPE: ErrorCode.UNSUPPORTED_FILE_TYPE,
        status.HTTP_429_TOO_MANY_REQUESTS: ErrorCode.RATE_LIMITED,
    }.get(exc.status_code, ErrorCode.INTERNAL_ERROR)

    return _envelope(
        request,
        [ErrorDetail(code=code, message=str(exc.detail))],
        exc.status_code,
    )


async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Render an unexpected failure without leaking internals.

    The exception and its traceback are logged; the client receives only a
    generic message and the request id needed to find that log entry.
    """
    logger.error(
        "unhandled_exception",
        error_type=type(exc).__name__,
        path=request.url.path,
        exc_info=exc,
    )
    return _envelope(
        request,
        [
            ErrorDetail(
                code=ErrorCode.INTERNAL_ERROR,
                message="An internal error occurred. Quote the request id when reporting it.",
            )
        ],
        status.HTTP_500_INTERNAL_SERVER_ERROR,
    )


def register_error_handlers(app: FastAPI) -> None:
    """Attach every handler to ``app``."""
    app.add_exception_handler(PipelineError, pipeline_error_handler)  # type: ignore[arg-type]
    app.add_exception_handler(RequestValidationError, validation_error_handler)  # type: ignore[arg-type]
    app.add_exception_handler(StarletteHTTPException, http_exception_handler)  # type: ignore[arg-type]
    app.add_exception_handler(Exception, unhandled_exception_handler)
