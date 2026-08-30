"""HTTP response envelopes.

One shape for success, one for failure, both carrying the same ``processing``
block so a client can correlate any response with server logs. Stack traces and
internal exception text never appear; clients get a stable ``code`` and a safe
``message``.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from app.core.exceptions import ErrorCode
from app.core.versions import API_VERSION, PIPELINE_VERSION, SCHEMA_VERSION
from app.schemas.receipt import Receipt
from app.schemas.validation import IssueSeverity, ValidationResult


class ErrorDetail(BaseModel):
    """A single error returned to the client."""

    model_config = ConfigDict(frozen=True)

    code: ErrorCode
    message: str
    field: str | None = None
    #: Populated only when ``DEBUG_ERRORS`` is enabled. Never contains secrets.
    details: dict[str, Any] | None = None


class WarningDetail(BaseModel):
    """A non-fatal finding. Mirrors ``ValidationIssue`` on the wire."""

    model_config = ConfigDict(frozen=True)

    code: str
    message: str
    field: str | None = None


class ResponseProcessing(BaseModel):
    """Correlation and version metadata, present on every response."""

    model_config = ConfigDict(frozen=True)

    request_id: str
    api_version: str = API_VERSION
    pipeline_version: str = PIPELINE_VERSION
    schema_version: str = SCHEMA_VERSION
    processing_time_ms: float = Field(default=0.0, ge=0)


class APIResponse[T](BaseModel):
    """Standard envelope.

    ``success`` answers "did processing complete?", not "is the result
    certain?". A result with warnings is still ``success: true`` -- callers
    decide what to do with the warnings.
    """

    model_config = ConfigDict(frozen=True)

    success: bool
    data: T | None = None
    warnings: tuple[WarningDetail, ...] = ()
    errors: tuple[ErrorDetail, ...] = ()
    processing: ResponseProcessing


class ReceiptExtractionResponse(APIResponse[Receipt]):
    """Response for ``POST /api/v1/receipts/extract``."""


def warnings_from_validation(validation: ValidationResult) -> tuple[WarningDetail, ...]:
    """Project validation warnings and errors onto the envelope.

    Validation *errors* are surfaced in ``warnings`` rather than ``errors``:
    they describe an inconsistent document, not a failed request. The
    ``data.validation`` block retains the full severity breakdown.
    """
    return tuple(
        WarningDetail(
            code=issue.code.value,
            message=issue.message,
            field=issue.field,
        )
        for issue in validation.all_issues
        if issue.severity in (IssueSeverity.WARNING, IssueSeverity.ERROR)
    )


class HealthResponse(BaseModel):
    """``GET /health`` payload."""

    model_config = ConfigDict(frozen=True)

    status: str = "ok"
    service: str
    version: str


class ComponentStatus(BaseModel):
    """Readiness of one dependency."""

    model_config = ConfigDict(frozen=True)

    name: str
    ready: bool
    detail: str | None = None


class ReadinessResponse(BaseModel):
    """``GET /ready`` payload. Non-ready responses use HTTP 503."""

    model_config = ConfigDict(frozen=True)

    ready: bool
    components: tuple[ComponentStatus, ...] = ()


class VersionResponse(BaseModel):
    """``GET /version`` payload."""

    model_config = ConfigDict(frozen=True)

    service: str
    api_version: str = API_VERSION
    pipeline_version: str = PIPELINE_VERSION
    schema_version: str = SCHEMA_VERSION
    ocr_provider: str
    llm_enabled: bool
