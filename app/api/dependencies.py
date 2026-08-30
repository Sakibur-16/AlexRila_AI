"""FastAPI dependency providers.

Expensive collaborators -- the OCR provider, the pipeline -- are built once at
application startup and shared, because constructing them per request would
add engine initialisation to every call. They are exposed as dependencies
rather than module globals so a test can override them through FastAPI's
``dependency_overrides`` without monkey-patching.
"""

from __future__ import annotations

import uuid
from typing import Annotated

from fastapi import Depends, Header, Request

from app.core.config import Settings, get_settings
from app.core.logging import get_logger
from app.llm.extractor import LLMFallbackExtractor
from app.llm.factory import create_llm_provider
from app.ocr.base import OCRProvider
from app.ocr.factory import create_ocr_provider
from app.pipeline.receipt_pipeline import ReceiptPipeline

logger = get_logger(__name__)

#: Header clients may set to propagate their own correlation id.
REQUEST_ID_HEADER = "X-Request-ID"

#: Bound on an accepted client-supplied request id, to keep log fields sane.
_MAX_REQUEST_ID_LENGTH = 128


def get_app_settings(request: Request) -> Settings:
    """Return the settings this application instance was built with.

    Read from application state rather than the process-wide singleton so that
    ``create_app(settings)`` genuinely governs request handling. Without this,
    an explicitly configured app would silently serve requests using whatever
    the environment happened to say.
    """
    configured = getattr(request.app.state, "settings", None)
    return configured if configured is not None else get_settings()


def settings_for(request: Request) -> Settings:
    """Settings lookup for contexts without dependency injection.

    Used by exception handlers, which receive a request but not the dependency
    graph.
    """
    configured = getattr(request.app.state, "settings", None)
    return configured if configured is not None else get_settings()


SettingsDep = Annotated[Settings, Depends(get_app_settings)]


def build_ocr_provider(settings: Settings) -> OCRProvider:
    """Construct the configured OCR provider."""
    return create_ocr_provider(settings=settings)


def build_llm_extractor(settings: Settings) -> LLMFallbackExtractor | None:
    """Construct the LLM fallback, or ``None`` when disabled.

    A misconfigured LLM must not prevent the service from starting: the
    deterministic path is fully functional without it, so a construction
    failure is logged and the fallback is simply disabled.
    """
    if not settings.llm_enabled:
        return None
    try:
        provider = create_llm_provider(settings=settings)
    except Exception as exc:
        logger.error(
            "llm_provider_unavailable",
            error_type=type(exc).__name__,
            hint="LLM fallback disabled; deterministic extraction continues.",
        )
        return None
    return LLMFallbackExtractor(settings, provider)


def build_pipeline(settings: Settings) -> ReceiptPipeline:
    """Construct the receipt pipeline with its collaborators."""
    return ReceiptPipeline(
        settings=settings,
        ocr_provider=build_ocr_provider(settings),
        llm_extractor=build_llm_extractor(settings),
    )


def get_pipeline(request: Request) -> ReceiptPipeline:
    """Return the process-wide pipeline built during startup."""
    return request.app.state.pipeline  # type: ignore[no-any-return]


PipelineDep = Annotated[ReceiptPipeline, Depends(get_pipeline)]


def get_ocr_provider(request: Request) -> OCRProvider:
    """Return the process-wide OCR provider built during startup."""
    return request.app.state.ocr_provider  # type: ignore[no-any-return]


OCRProviderDep = Annotated[OCRProvider, Depends(get_ocr_provider)]


def sanitize_request_id(raw: str | None) -> str:
    """Return a safe correlation id, generating one when none was supplied.

    A client-supplied id is honoured so traces span services, but it is
    length-capped and stripped of non-printable characters first -- it ends up
    in log fields and a response header, and an unbounded client string in
    either is an injection vector.
    """
    if raw:
        cleaned = "".join(char for char in raw.strip() if char.isprintable())[
            :_MAX_REQUEST_ID_LENGTH
        ]
        if cleaned:
            return cleaned
    return str(uuid.uuid4())


def get_request_id(
    request: Request,
    x_request_id: Annotated[str | None, Header(alias=REQUEST_ID_HEADER)] = None,
) -> str:
    """Return this request's correlation id.

    The middleware already assigned one before routing, so it is reused here
    rather than generated again -- otherwise the id in the response body would
    not match the one in the header or the logs.
    """
    existing = getattr(request.state, "request_id", None)
    if existing:
        return str(existing)
    return sanitize_request_id(x_request_id)


RequestIdDep = Annotated[str, Depends(get_request_id)]
