"""Validation orchestration.

Runs every validator and merges their findings into a single
:class:`~app.schemas.validation.ValidationResult`.

The layering is deliberate: quality validators explain *why* extraction may
have gone wrong, semantic validators check individual values, and financial
validators check the relationships between them. Running all three -- rather
than stopping at the first failure -- means one response tells a consumer
everything that is questionable about a document, which is what makes the
warnings actionable.
"""

from __future__ import annotations

from datetime import date

from app.core.config import Settings
from app.core.metrics import MetricNames, increment
from app.extraction.receipt import ExtractionResult
from app.schemas.ocr import OCRResult
from app.schemas.quality import ImageQuality
from app.schemas.validation import IssueSeverity, ValidationIssue, ValidationResult
from app.validation.financial import validate_financials
from app.validation.quality import validate_image_quality, validate_ocr_quality
from app.validation.semantic import validate_semantics


class ValidationEngine:
    """Runs the full validation suite over one extracted document."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    def validate(
        self,
        extraction: ExtractionResult,
        *,
        ocr: OCRResult | None = None,
        quality: ImageQuality | None = None,
        today: date | None = None,
    ) -> ValidationResult:
        """Validate an extraction result.

        Args:
            extraction: Fields read from the document.
            ocr: Recognition output, for OCR-quality checks.
            quality: Pre-OCR image measurements.
            today: Reference date, injectable for deterministic tests.

        Returns:
            The merged result. ``is_valid`` is ``False`` only when at least one
            ``ERROR`` was raised; warnings leave it ``True``.
        """
        issues: list[ValidationIssue] = []
        issues.extend(validate_image_quality(quality, self._settings))
        issues.extend(validate_ocr_quality(ocr, self._settings))
        issues.extend(validate_semantics(extraction, self._settings, today=today))
        issues.extend(validate_financials(extraction, self._settings))

        deduplicated = _deduplicate(issues)
        result = ValidationResult.from_issues(deduplicated)

        for issue in deduplicated:
            if issue.severity is IssueSeverity.ERROR:
                increment(MetricNames.VALIDATION_FAILURE, labels={"code": issue.code.value})
            elif issue.severity is IssueSeverity.WARNING:
                increment(MetricNames.VALIDATION_WARNING, labels={"code": issue.code.value})

        return result


def _deduplicate(issues: list[ValidationIssue]) -> list[ValidationIssue]:
    """Collapse issues sharing a code and field, keeping the highest severity.

    Two validators can legitimately reach the same conclusion by different
    routes -- the tender check and the identity check both report
    ``TOTAL_MISMATCH``. A consumer should see that once, at its worst
    severity, not twice.
    """
    ranked: dict[tuple[str, str | None], ValidationIssue] = {}
    order = {IssueSeverity.INFO: 0, IssueSeverity.WARNING: 1, IssueSeverity.ERROR: 2}

    for issue in issues:
        key = (issue.code.value, issue.field)
        existing = ranked.get(key)
        if existing is None or order[issue.severity] > order[existing.severity]:
            ranked[key] = issue

    return list(ranked.values())
