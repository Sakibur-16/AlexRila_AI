"""Semantic and completeness validation.

Checks that extracted values are *meaningful*, not merely well-typed: a
quantity of zero, a date twenty years in the future, a currency code outside
ISO-4217. Pydantic guarantees shape; this module guarantees sense.

It also reports what is missing. A receipt with no total is not an error --
the pipeline did its job and the field genuinely could not be read -- but a
consumer must be told, so absence is surfaced as a structured warning rather
than left for them to discover.
"""

from __future__ import annotations

from datetime import date, timedelta
from decimal import Decimal

from app.core.config import Settings
from app.extraction.receipt import ExtractionResult
from app.normalization.currency import KNOWN_CODES
from app.schemas.validation import IssueCode, IssueSeverity, ValidationIssue

#: A receipt dated further ahead than this is a misread, not a future receipt.
_MAX_FUTURE_DAYS = 2

#: Receipts older than this are implausible for an expense pipeline and
#: usually indicate a two-digit-year misparse.
_MAX_AGE_YEARS = 25

#: A single line item above this magnitude is suspicious on a retail receipt.
#: Reported, never rejected -- legitimate high-value receipts exist.
_IMPLAUSIBLE_ITEM_TOTAL = Decimal("1000000")


def _issue(
    code: IssueCode,
    severity: IssueSeverity,
    message: str,
    field: str | None = None,
    **context: object,
) -> ValidationIssue:
    return ValidationIssue(
        code=code,
        severity=severity,
        message=message,
        field=field,
        context={k: str(v) for k, v in context.items() if v is not None},
    )


def validate_semantics(
    extraction: ExtractionResult, settings: Settings, *, today: date | None = None
) -> list[ValidationIssue]:
    """Check that extracted values are semantically plausible.

    Args:
        extraction: Fields read from the document.
        settings: Active configuration.
        today: Reference date for the future-date check. Injectable so tests
            are not time-dependent.
    """
    issues: list[ValidationIssue] = []
    issues.extend(_check_required(extraction))
    issues.extend(_check_items(extraction))
    issues.extend(_check_currency(extraction))
    issues.extend(_check_date(extraction, today or date.today()))
    issues.extend(_check_field_warnings(extraction))
    del settings
    return issues


def _check_required(extraction: ExtractionResult) -> list[ValidationIssue]:
    """Report absent fields that a usable receipt would normally carry."""
    issues: list[ValidationIssue] = []

    if extraction.merchant_name.value is None:
        issues.append(
            _issue(
                IssueCode.MISSING_MERCHANT_NAME,
                IssueSeverity.WARNING,
                "No merchant name could be identified.",
                "merchant.name",
            )
        )
    if extraction.total.value is None:
        issues.append(
            _issue(
                IssueCode.MISSING_TOTAL,
                IssueSeverity.WARNING,
                "No total could be identified.",
                "total",
            )
        )
    if extraction.transaction_date.value is None:
        issues.append(
            _issue(
                IssueCode.MISSING_DATE,
                IssueSeverity.WARNING,
                "No transaction date could be determined."
                if extraction.transaction_date.raw_value is None
                else "A date was found but could not be resolved unambiguously.",
                "transaction.date",
                raw=extraction.transaction_date.raw_value,
            )
        )
    if extraction.currency.value is None:
        issues.append(
            _issue(
                IssueCode.MISSING_CURRENCY,
                IssueSeverity.WARNING,
                "Currency could not be determined.",
                "currency",
                symbol=extraction.currency_symbol.value,
            )
        )
    if not extraction.items:
        issues.append(
            _issue(
                IssueCode.MISSING_ITEMS,
                IssueSeverity.WARNING,
                "No line items were extracted.",
                "items",
            )
        )
    if extraction.subtotal.value is None and extraction.items:
        issues.append(
            _issue(
                IssueCode.MISSING_SUBTOTAL,
                IssueSeverity.INFO,
                "No subtotal was printed on the receipt.",
                "subtotal",
            )
        )
    return issues


def _check_items(extraction: ExtractionResult) -> list[ValidationIssue]:
    """Check each item for implausible quantities and prices."""
    issues: list[ValidationIssue] = []

    for index, item in enumerate(extraction.items):
        path = f"items[{index}]"

        if item.quantity is not None and item.quantity <= 0:
            issues.append(
                _issue(
                    IssueCode.INVALID_QUANTITY,
                    IssueSeverity.ERROR,
                    "Item quantity must be positive.",
                    f"{path}.quantity",
                    quantity=item.quantity,
                    description=item.description,
                )
            )

        # A negative line is normal for a discount or return row, so it is only
        # reported when nothing marks it as one.
        if item.total_price is not None and item.total_price < 0 and not item.discount:
            issues.append(
                _issue(
                    IssueCode.NEGATIVE_ITEM_PRICE,
                    IssueSeverity.WARNING,
                    "Item price is negative and is not marked as a discount.",
                    f"{path}.total_price",
                    total_price=item.total_price,
                    description=item.description,
                )
            )

        if item.total_price is not None and abs(item.total_price) > _IMPLAUSIBLE_ITEM_TOTAL:
            issues.append(
                _issue(
                    IssueCode.NEGATIVE_ITEM_PRICE,
                    IssueSeverity.WARNING,
                    "Item total is implausibly large and was probably misread.",
                    f"{path}.total_price",
                    total_price=item.total_price,
                )
            )

    return issues


def _check_currency(extraction: ExtractionResult) -> list[ValidationIssue]:
    """Check that a detected currency code is a real ISO-4217 code."""
    code = extraction.currency.value
    if code is None:
        return []
    if code in KNOWN_CODES:
        return []
    return [
        _issue(
            IssueCode.INVALID_CURRENCY,
            IssueSeverity.WARNING,
            "Detected currency is not a recognised ISO-4217 code.",
            "currency",
            currency=code,
        )
    ]


def _check_date(extraction: ExtractionResult, today: date) -> list[ValidationIssue]:
    """Check that a resolved date is plausible for a receipt."""
    value = extraction.transaction_date.value
    if value is None:
        return []

    issues: list[ValidationIssue] = []
    if value > today + timedelta(days=_MAX_FUTURE_DAYS):
        issues.append(
            _issue(
                IssueCode.FUTURE_DATE,
                IssueSeverity.WARNING,
                "Transaction date is in the future.",
                "transaction.date",
                date=value.isoformat(),
            )
        )
    elif value.year < today.year - _MAX_AGE_YEARS:
        issues.append(
            _issue(
                IssueCode.IMPLAUSIBLE_DATE,
                IssueSeverity.WARNING,
                "Transaction date is implausibly old; the year may be misread.",
                "transaction.date",
                date=value.isoformat(),
            )
        )
    return issues


def _check_field_warnings(extraction: ExtractionResult) -> list[ValidationIssue]:
    """Promote per-field warnings raised during normalisation into issues.

    Normalisers record uncertainty on the field itself (an ambiguous date, a
    contested currency symbol). This lifts those notes into the shared
    validation vocabulary so consumers see one uniform list.
    """
    issues: list[ValidationIssue] = []
    seen: set[tuple[str, str]] = set()

    for path, extracted in extraction.field_map().items():
        for warning in extracted.warnings:
            key = (path, warning)
            if key in seen:
                continue
            seen.add(key)
            try:
                code = IssueCode(warning)
            except ValueError:
                continue
            if code in _ALREADY_REPORTED:
                continue
            issues.append(
                _issue(
                    code,
                    IssueSeverity.WARNING,
                    _WARNING_MESSAGES.get(code, "Value could not be resolved confidently."),
                    path,
                    raw=extracted.raw_value,
                )
            )
    return issues


#: Codes emitted by the completeness checks above; not duplicated from field
#: warnings.
_ALREADY_REPORTED = frozenset(
    {
        IssueCode.MISSING_CURRENCY,
        IssueCode.MISSING_MERCHANT_NAME,
        IssueCode.MISSING_DATE,
        IssueCode.MISSING_TOTAL,
    }
)

_WARNING_MESSAGES: dict[IssueCode, str] = {
    IssueCode.AMBIGUOUS_DATE_FORMAT: (
        "Date is ambiguous between day-first and month-first order and was not "
        "converted. The printed text is preserved in raw_date."
    ),
    IssueCode.INVALID_DATE: "The printed date is not a valid calendar date.",
    IssueCode.INVALID_TIME: "The printed time is not a valid time of day.",
    IssueCode.IMPLAUSIBLE_DATE: "The parsed date falls outside a plausible range.",
    IssueCode.AMBIGUOUS_CURRENCY: (
        "The currency symbol is used by several currencies and no context "
        "resolved it. The symbol is reported in currency_symbol."
    ),
    IssueCode.MULTIPLE_CURRENCIES_DETECTED: (
        "More than one currency indicator appears on the document."
    ),
}
