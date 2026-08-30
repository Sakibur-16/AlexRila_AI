"""Operational endpoints.

The distinction between the two probes matters for orchestration:

* ``/health`` is a **liveness** probe -- is the process alive? It never touches
  a dependency, so a slow OCR engine can never cause a restart loop.
* ``/ready`` is a **readiness** probe -- can this instance serve traffic? It
  checks the OCR provider (and the LLM provider when enabled) and returns 503
  when it cannot, so a container missing its engine binary is removed from the
  load balancer instead of failing every request it receives.
"""

from __future__ import annotations

from fastapi import APIRouter, Response, status

from app.api.dependencies import OCRProviderDep, SettingsDep
from app.core.metrics import get_metrics_sink
from app.llm.factory import create_llm_provider
from app.schemas.response import (
    ComponentStatus,
    HealthResponse,
    ReadinessResponse,
    VersionResponse,
)

router = APIRouter(tags=["operations"])


@router.get("/health", response_model=HealthResponse, summary="Liveness probe")
async def health(settings: SettingsDep) -> HealthResponse:
    """Report that the process is alive. Never touches a dependency."""
    return HealthResponse(status="ok", service=settings.app_name, version=settings.app_version)


@router.get("/ready", response_model=ReadinessResponse, summary="Readiness probe")
async def ready(
    settings: SettingsDep, ocr_provider: OCRProviderDep, response: Response
) -> ReadinessResponse:
    """Report whether this instance can serve extraction requests."""
    components: list[ComponentStatus] = []

    ocr_ready, ocr_detail = ocr_provider.health_check()
    components.append(
        ComponentStatus(name=f"ocr:{ocr_provider.name}", ready=ocr_ready, detail=ocr_detail)
    )

    if settings.llm_enabled:
        try:
            llm_provider = create_llm_provider(settings=settings)
            llm_ready, llm_detail = llm_provider.health_check()
        except Exception as exc:
            llm_ready, llm_detail = False, f"unavailable: {type(exc).__name__}"
        components.append(
            ComponentStatus(name=f"llm:{settings.llm_provider}", ready=llm_ready, detail=llm_detail)
        )

    # The LLM is an optional enhancement: the deterministic path works without
    # it, so its absence does not make the instance unready.
    is_ready = ocr_ready
    if not is_ready:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE

    return ReadinessResponse(ready=is_ready, components=tuple(components))


@router.get("/version", response_model=VersionResponse, summary="Version information")
async def version(settings: SettingsDep) -> VersionResponse:
    """Report the API, pipeline and schema versions in use."""
    return VersionResponse(
        service=settings.app_name,
        ocr_provider=settings.ocr_provider,
        llm_enabled=settings.llm_enabled,
    )


@router.get("/metrics", summary="Pipeline metrics snapshot", include_in_schema=False)
async def metrics() -> dict[str, object]:
    """Return a metrics snapshot from the active sink.

    A dependency-free view intended for development and smoke tests. A
    production deployment installs a real sink (Prometheus, OTLP) via
    :func:`~app.core.metrics.set_metrics_sink` and scrapes that instead.
    """
    return get_metrics_sink().snapshot()
