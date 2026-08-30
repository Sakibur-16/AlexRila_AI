"""Validation result contract.

The distinction that drives this module: a *warning* means "processed, but be
careful"; an *error* means "the extracted document is internally inconsistent
or semantically impossible". Neither is a crash -- both come back inside a
``success: true`` response with structured codes. Only a genuine processing
failure produces ``success: false`` (see ``app.schemas.response``).

Issue codes are part of the public contract: add freely, never rename.
"""

from __future__ import annotations

import enum

from pydantic import BaseModel, ConfigDict, Field


class IssueSeverity(enum.StrEnum):
    """How much an issue should worry a consumer."""

    #: Informational: extraction succeeded but a heuristic was involved.
    INFO = "info"
    #: The value may be wrong; a human should look if the amount matters.
    WARNING = "warning"
    #: The document is internally inconsistent or semantically invalid.
    ERROR = "error"


class IssueCode(enum.StrEnum):
    """Machine-readable validation findings."""

    # --- image quality ---------------------------------------------------
    IMAGE_LOW_RESOLUTION = "IMAGE_LOW_RESOLUTION"
    IMAGE_BLURRY = "IMAGE_BLURRY"
    IMAGE_TOO_DARK = "IMAGE_TOO_DARK"
    IMAGE_TOO_BRIGHT = "IMAGE_TOO_BRIGHT"
    IMAGE_LOW_CONTRAST = "IMAGE_LOW_CONTRAST"
    IMAGE_SKEWED = "IMAGE_SKEWED"

    # --- OCR quality -----------------------------------------------------
    OCR_LOW_CONFIDENCE = "OCR_LOW_CONFIDENCE"
    OCR_SPARSE_TEXT = "OCR_SPARSE_TEXT"
    OCR_SUSPICIOUS_CHARACTERS = "OCR_SUSPICIOUS_CHARACTERS"
    OCR_NO_GEOMETRY = "OCR_NO_GEOMETRY"

    # --- missing fields --------------------------------------------------
    MISSING_MERCHANT_NAME = "MISSING_MERCHANT_NAME"
    MISSING_TOTAL = "MISSING_TOTAL"
    MISSING_DATE = "MISSING_DATE"
    MISSING_CURRENCY = "MISSING_CURRENCY"
    MISSING_ITEMS = "MISSING_ITEMS"
    MISSING_SUBTOTAL = "MISSING_SUBTOTAL"

    # --- financial consistency -------------------------------------------
    TOTAL_MISMATCH = "TOTAL_MISMATCH"
    SUBTOTAL_MISMATCH = "SUBTOTAL_MISMATCH"
    ITEM_ARITHMETIC_MISMATCH = "ITEM_ARITHMETIC_MISMATCH"
    TAX_MISMATCH = "TAX_MISMATCH"
    SUBTOTAL_EXCEEDS_TOTAL = "SUBTOTAL_EXCEEDS_TOTAL"
    NEGATIVE_TOTAL = "NEGATIVE_TOTAL"
    NEGATIVE_SUBTOTAL = "NEGATIVE_SUBTOTAL"
    DISCOUNT_EXCEEDS_SUBTOTAL = "DISCOUNT_EXCEEDS_SUBTOTAL"

    # --- semantic --------------------------------------------------------
    INVALID_QUANTITY = "INVALID_QUANTITY"
    NEGATIVE_ITEM_PRICE = "NEGATIVE_ITEM_PRICE"
    INVALID_CURRENCY = "INVALID_CURRENCY"
    AMBIGUOUS_CURRENCY = "AMBIGUOUS_CURRENCY"
    MULTIPLE_CURRENCIES_DETECTED = "MULTIPLE_CURRENCIES_DETECTED"
    AMBIGUOUS_DATE_FORMAT = "AMBIGUOUS_DATE_FORMAT"
    INVALID_DATE = "INVALID_DATE"
    FUTURE_DATE = "FUTURE_DATE"
    IMPLAUSIBLE_DATE = "IMPLAUSIBLE_DATE"
    INVALID_TIME = "INVALID_TIME"

    # --- confidence / review ---------------------------------------------
    LOW_FIELD_CONFIDENCE = "LOW_FIELD_CONFIDENCE"
    LOW_OVERALL_CONFIDENCE = "LOW_OVERALL_CONFIDENCE"
    REVIEW_RECOMMENDED = "REVIEW_RECOMMENDED"

    # --- privacy ---------------------------------------------------------
    SENSITIVE_DATA_REDACTED = "SENSITIVE_DATA_REDACTED"

    # --- LLM -------------------------------------------------------------
    LLM_FALLBACK_USED = "LLM_FALLBACK_USED"
    LLM_OUTPUT_REJECTED = "LLM_OUTPUT_REJECTED"
    LLM_UNAVAILABLE = "LLM_UNAVAILABLE"


class ValidationIssue(BaseModel):
    """One validation finding.

    ``field`` uses dotted paths matching the receipt schema (``tax.total``,
    ``items[2].quantity``) so a consumer can map an issue straight onto the
    value it concerns.
    """

    model_config = ConfigDict(frozen=True)

    code: IssueCode
    severity: IssueSeverity
    message: str
    field: str | None = None
    #: Structured, non-sensitive context: expected vs actual amounts, etc.
    context: dict[str, str] = Field(default_factory=dict)


class ValidationResult(BaseModel):
    """Aggregated outcome of the validation stage."""

    model_config = ConfigDict(frozen=True)

    #: False when at least one ``ERROR`` severity issue was raised.
    is_valid: bool = True
    warnings: tuple[ValidationIssue, ...] = ()
    errors: tuple[ValidationIssue, ...] = ()
    infos: tuple[ValidationIssue, ...] = ()

    @property
    def all_issues(self) -> tuple[ValidationIssue, ...]:
        return self.errors + self.warnings + self.infos

    def codes(self) -> tuple[str, ...]:
        """All issue codes, in severity order. Useful for assertions."""
        return tuple(issue.code.value for issue in self.all_issues)

    def fields_with_issues(self, *severities: IssueSeverity) -> frozenset[str]:
        """Dotted field paths touched by issues of the given severities."""
        wanted = set(severities) or {IssueSeverity.ERROR, IssueSeverity.WARNING}
        return frozenset(
            issue.field for issue in self.all_issues if issue.field and issue.severity in wanted
        )

    @classmethod
    def from_issues(cls, issues: list[ValidationIssue]) -> ValidationResult:
        """Partition a flat issue list into the severity buckets."""
        errors = tuple(i for i in issues if i.severity is IssueSeverity.ERROR)
        warnings = tuple(i for i in issues if i.severity is IssueSeverity.WARNING)
        infos = tuple(i for i in issues if i.severity is IssueSeverity.INFO)
        return cls(is_valid=not errors, warnings=warnings, errors=errors, infos=infos)
