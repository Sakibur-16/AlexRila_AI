"""Fallback chain for OCR providers.

Runs a primary provider and, when it fails for a reason a different engine
could survive, retries with a secondary one. The intended configuration is a
vision model in front of Tesseract: the model reads messy photographs far
better, and the local engine keeps the service answering when the model is
rate-limited, out of credit, timing out, or simply unreachable.

**Only provider-side failures fall through.** A corrupt upload or an image
below the size limit fails identically on every engine, so retrying it just
doubles the latency of a request that was always going to fail. The same
reasoning governs :mod:`app.core.retry`, and the two use the same idea of
which errors are worth another attempt.

The result records which engine actually served the request, so a spike in
fallbacks is visible in metrics and in every response's processing metadata
rather than being silently absorbed.
"""

from __future__ import annotations

from typing import Any

from app.core.exceptions import ErrorCode, PipelineError
from app.core.logging import get_logger
from app.core.metrics import increment
from app.ocr.base import OCRProvider, OCRRequest
from app.schemas.ocr import OCRResult

logger = get_logger(__name__)

#: Failures that say something about the *provider*, not the document. A
#: different engine has a real chance with these.
_FALLBACK_CODES: frozenset[ErrorCode] = frozenset(
    {
        ErrorCode.PROVIDER_UNAVAILABLE,
        ErrorCode.PROVIDER_QUOTA_EXHAUSTED,
        ErrorCode.PROVIDER_NOT_REGISTERED,
        ErrorCode.OCR_TIMEOUT,
        ErrorCode.OCR_FAILED,
        # An engine returning nothing is worth a second opinion: the image may
        # be legible to a different one. A genuinely blank page then fails
        # twice, which is cheap and correct.
        ErrorCode.OCR_EMPTY_RESULT,
    }
)


class FallbackOCRProvider(OCRProvider):
    """Tries ``primary``, then ``secondary`` on a provider-side failure."""

    def __init__(self, primary: OCRProvider, secondary: OCRProvider) -> None:
        self._primary = primary
        self._secondary = secondary
        self.name = f"{primary.name}+{secondary.name}"

    def health_check(self) -> tuple[bool, str | None]:
        """Ready when *either* engine can serve.

        A chain whose primary is misconfigured is degraded, not broken, so it
        stays ready and says so. Reporting it unready would take an instance
        out of rotation that can still answer every request.
        """
        primary_ready, primary_detail = self._primary.health_check()
        if primary_ready:
            return True, None

        secondary_ready, secondary_detail = self._secondary.health_check()
        if secondary_ready:
            return True, (
                f"primary ({self._primary.name}) unavailable, serving from "
                f"{self._secondary.name}: {primary_detail}"
            )
        return False, (
            f"{self._primary.name}: {primary_detail} / {self._secondary.name}: {secondary_detail}"
        )

    def describe(self) -> dict[str, Any]:
        return {
            "provider": self.name,
            "primary": self._primary.describe(),
            "fallback": self._secondary.describe(),
        }

    def extract(self, request: OCRRequest) -> OCRResult:
        try:
            return self._primary.extract(request)
        except PipelineError as exc:
            if exc.code not in _FALLBACK_CODES:
                # A document-side failure. The other engine would fail too.
                raise

            logger.warning(
                "ocr_falling_back",
                primary=self._primary.name,
                fallback=self._secondary.name,
                error_code=exc.code.value,
            )
            increment(
                "ocr.fallback.total",
                labels={"from": self._primary.name, "code": exc.code.value},
            )

            result = self._secondary.extract(request)
            # Record the degradation on the result itself: a response that
            # silently came from the weaker engine is a response nobody can
            # interpret correctly later.
            return result.model_copy(
                update={
                    "provider_metadata": {
                        **result.provider_metadata,
                        "served_by_fallback": True,
                        "primary_provider": self._primary.name,
                        "primary_error": exc.code.value,
                    }
                }
            )
