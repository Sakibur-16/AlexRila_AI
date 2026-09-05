"""Structured error taxonomy.

Every failure surfaces as a machine-readable :class:`ErrorCode`. Application
code raises :class:`PipelineError` subclasses; the API layer converts them into
the error envelope. Messages are safe for external consumption -- secrets and
stack traces never reach ``message``; debugging detail goes into ``details``
which is logged and only echoed when ``settings.debug_errors`` is enabled.
"""

from __future__ import annotations

import enum
from typing import Any


class ErrorCode(enum.StrEnum):
    """Stable, machine-readable failure identifiers.

    Values are part of the public API contract: never rename one, only add.
    """

    # --- input / image ---------------------------------------------------
    INVALID_IMAGE = "INVALID_IMAGE"
    UNSUPPORTED_FILE_TYPE = "UNSUPPORTED_FILE_TYPE"
    IMAGE_TOO_LARGE = "IMAGE_TOO_LARGE"
    IMAGE_TOO_SMALL = "IMAGE_TOO_SMALL"
    EMPTY_FILE = "EMPTY_FILE"
    UNSAFE_FILENAME = "UNSAFE_FILENAME"

    # --- OCR -------------------------------------------------------------
    OCR_FAILED = "OCR_FAILED"
    OCR_TIMEOUT = "OCR_TIMEOUT"
    OCR_EMPTY_RESULT = "OCR_EMPTY_RESULT"
    PROVIDER_UNAVAILABLE = "PROVIDER_UNAVAILABLE"
    PROVIDER_NOT_REGISTERED = "PROVIDER_NOT_REGISTERED"

    # --- extraction ------------------------------------------------------
    EXTRACTION_FAILED = "EXTRACTION_FAILED"
    INVALID_RECEIPT = "INVALID_RECEIPT"

    # --- LLM -------------------------------------------------------------
    LLM_FAILED = "LLM_FAILED"
    LLM_TIMEOUT = "LLM_TIMEOUT"
    LLM_INVALID_OUTPUT = "LLM_INVALID_OUTPUT"
    #: No model provider is configured, or the feature is switched off.
    LLM_UNAVAILABLE = "LLM_UNAVAILABLE"

    # --- transport / infra ----------------------------------------------
    UNAUTHORIZED = "UNAUTHORIZED"
    REQUEST_TOO_LARGE = "REQUEST_TOO_LARGE"
    RATE_LIMITED = "RATE_LIMITED"
    INTERNAL_ERROR = "INTERNAL_ERROR"


#: Codes for which a retry can plausibly succeed. Consulted by the retry
#: helper so that deterministic failures (bad image, bad credentials) are
#: never retried -- see ``app.core.retry``.
RETRYABLE_ERROR_CODES: frozenset[ErrorCode] = frozenset(
    {
        ErrorCode.OCR_TIMEOUT,
        ErrorCode.PROVIDER_UNAVAILABLE,
        ErrorCode.LLM_TIMEOUT,
    }
)


class PipelineError(Exception):
    """Base class for all pipeline failures.

    Args:
        code: Stable machine-readable identifier.
        message: Human-readable, externally safe summary.
        details: Structured debugging context. Must not contain secrets or
            raw document content; it may be logged.
        http_status: Status the API layer should use when this escapes.
    """

    code: ErrorCode = ErrorCode.INTERNAL_ERROR
    http_status: int = 500

    def __init__(
        self,
        message: str,
        *,
        code: ErrorCode | None = None,
        details: dict[str, Any] | None = None,
        http_status: int | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        if code is not None:
            self.code = code
        self.details: dict[str, Any] = details or {}
        if http_status is not None:
            self.http_status = http_status

    @property
    def retryable(self) -> bool:
        """Whether retrying the same operation could plausibly succeed."""
        return self.code in RETRYABLE_ERROR_CODES

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"{type(self).__name__}(code={self.code.value!r}, message={self.message!r})"


class InputValidationError(PipelineError):
    """The submitted document was rejected before any processing began."""

    code = ErrorCode.INVALID_IMAGE
    http_status = 400


class UnsupportedFileTypeError(InputValidationError):
    code = ErrorCode.UNSUPPORTED_FILE_TYPE
    http_status = 415


class FileTooLargeError(InputValidationError):
    code = ErrorCode.IMAGE_TOO_LARGE
    http_status = 413


class ImageTooSmallError(InputValidationError):
    code = ErrorCode.IMAGE_TOO_SMALL
    http_status = 422


class OCRError(PipelineError):
    """OCR execution failed."""

    code = ErrorCode.OCR_FAILED
    http_status = 502


class OCRTimeoutError(OCRError):
    code = ErrorCode.OCR_TIMEOUT
    http_status = 504


class ProviderUnavailableError(PipelineError):
    """A configured provider cannot service requests (missing binary, creds...)."""

    code = ErrorCode.PROVIDER_UNAVAILABLE
    http_status = 503


class ProviderNotRegisteredError(PipelineError):
    """Configuration names a provider that no factory knows about."""

    code = ErrorCode.PROVIDER_NOT_REGISTERED
    http_status = 500


class ExtractionError(PipelineError):
    code = ErrorCode.EXTRACTION_FAILED
    http_status = 500


class LLMError(PipelineError):
    code = ErrorCode.LLM_FAILED
    http_status = 502


class LLMInvalidOutputError(LLMError):
    code = ErrorCode.LLM_INVALID_OUTPUT
