"""Financial consistency validation.

Two identities are checked, both in :class:`~decimal.Decimal` arithmetic:

.. code-block:: text

    sum(item totals)                        ~= subtotal
    subtotal - discount + tax + service
        + shipping + tip + rounding         ~= total

"``~=``" is a tolerance comparison: receipts round each line independently, so
an exact match is the exception rather than the rule. Tolerance is absolute
(``TOTAL_TOLERANCE``) plus a proportional component (``TOTAL_TOLERANCE_RATIO``)
so that a 5-cent allowance on a 20-unit receipt does not become an absurdly
tight constraint on a 5,000-unit one.

**Discrepancies are reported, never repaired.** A mismatch means one of the
inputs was misread, and silently rewriting the printed total to make the
arithmetic work would destroy the only signal that anything went wrong.
"""

from __future__ import annotations

from decimal import Decimal

from app.core.config import Settings
from app.extraction.items import sum_item_totals
from app.extraction.receipt import ExtractionResult
from app.schemas.validation import IssueCode, IssueSeverity, ValidationIssue

#: Tax-inclusive receipts print a subtotal that already contains tax, so the
#: additive identity does not hold. Detected by testing both readings.
_INCLUSIVE_TAX_HINT = "tax_inclusive_pricing"


def tolerance_for(amount: Decimal | None, settings: Settings) -> Decimal:
    """Return the permitted absolute discrepancy for ``amount``."""
    base = settings.total_tolerance
    if amount is None:
        return base
    proportional = abs(amount) * Decimal(str(settings.total_tolerance_ratio))
    return max(base, proportional)


def _issue(
    code: IssueCode,
    severity: IssueSeverity,
    message: str,
    field: str | None = None,
    **context: Decimal | str | None,
) -> ValidationIssue:
    """Build an issue with numeric context rendered as strings."""
    return ValidationIssue(
        code=code,
        severity=severity,
        message=message,
        field=field,
        context={k: str(v) for k, v in context.items() if v is not None},
    )


def validate_financials(extraction: ExtractionResult, settings: Settings) -> list[ValidationIssue]:
    """Check the arithmetic of an extracted receipt.

    Args:
        extraction: Fields read from the document.
        settings: Supplies the tolerance configuration.

    Returns:
        Issues found. An empty list means every checkable identity held --
        which is also the strongest available evidence that the individual
        amounts were read correctly.
    """
    issues: list[ValidationIssue] = []

    subtotal = extraction.subtotal.value
    total = extraction.total.value
    tax_total = extraction.tax.value.total if extraction.tax.value else None
    discount = extraction.discount.value.amount if extraction.discount.value else None
    service = extraction.service_charge.value
    shipping = extraction.shipping.value
    tip = extraction.tip.value
    rounding = extraction.rounding.value
    items_sum = sum_item_totals(extraction.items)

    issues.extend(_check_signs(subtotal, total, discount))
    issues.extend(_check_items_against_subtotal(items_sum, subtotal, total, tax_total, settings))
    issues.extend(
        _check_total_identity(
            subtotal=subtotal,
            total=total,
            tax_total=tax_total,
            discount=discount,
            service=service,
            shipping=shipping,
            tip=tip,
            rounding=rounding,
            settings=settings,
        )
    )
    issues.extend(_check_ordering(subtotal, total, discount, settings))
    issues.extend(
        _check_tender(
            total=total,
            tendered=extraction.amount_tendered.value,
            change=extraction.change.value,
            settings=settings,
        )
    )
    issues.extend(_check_tax_details(extraction, settings))
    return issues


def _check_signs(
    subtotal: Decimal | None, total: Decimal | None, discount: Decimal | None
) -> list[ValidationIssue]:
    """Reject negative aggregates.

    A refund receipt legitimately carries a negative total, but it is rare
    enough -- and consequential enough -- that it is surfaced as an error for
    the consumer to acknowledge rather than passed through silently.
    """
    issues: list[ValidationIssue] = []
    if total is not None and total < 0:
        issues.append(
            _issue(
                IssueCode.NEGATIVE_TOTAL,
                IssueSeverity.ERROR,
                "Total is negative; the document may be a refund or was misread.",
                "total",
                total=total,
            )
        )
    if subtotal is not None and subtotal < 0:
        issues.append(
            _issue(
                IssueCode.NEGATIVE_SUBTOTAL,
                IssueSeverity.ERROR,
                "Subtotal is negative.",
                "subtotal",
                subtotal=subtotal,
            )
        )
    if discount is not None and subtotal is not None and discount > abs(subtotal):
        issues.append(
            _issue(
                IssueCode.DISCOUNT_EXCEEDS_SUBTOTAL,
                IssueSeverity.ERROR,
                "Discount exceeds the subtotal.",
                "discount",
                discount=discount,
                subtotal=subtotal,
            )
        )
    return issues


def _check_items_against_subtotal(
    items_sum: Decimal | None,
    subtotal: Decimal | None,
    total: Decimal | None,
    tax_total: Decimal | None,
    settings: Settings,
) -> list[ValidationIssue]:
    """Compare the item sum with the subtotal.

    Falls back to comparing against the total when no subtotal was printed --
    common on small receipts -- but only when there is no tax to account for,
    since otherwise the comparison is meaningless.
    """
    if items_sum is None:
        return []

    reference = subtotal
    field = "subtotal"
    if reference is None:
        if tax_total is not None and tax_total != 0:
            return []
        reference = total
        field = "total"
    if reference is None:
        return []

    difference = abs(items_sum - reference)
    if difference <= tolerance_for(reference, settings):
        return []

    return [
        _issue(
            IssueCode.ITEM_ARITHMETIC_MISMATCH
            if field == "subtotal"
            else IssueCode.SUBTOTAL_MISMATCH,
            IssueSeverity.WARNING,
            f"Sum of line items does not match the {field}.",
            field,
            items_sum=items_sum,
            reported=reference,
            difference=difference,
        )
    ]


def _check_total_identity(
    *,
    subtotal: Decimal | None,
    total: Decimal | None,
    tax_total: Decimal | None,
    discount: Decimal | None,
    service: Decimal | None,
    shipping: Decimal | None,
    tip: Decimal | None,
    rounding: Decimal | None,
    settings: Settings,
) -> list[ValidationIssue]:
    """Check ``subtotal - discount + charges ~= total``.

    Several readings are legitimate, and the identity is satisfied if *any* of
    them holds. Reporting a mismatch requires all of them to fail.

    * **Tax-exclusive vs tax-inclusive.** A European VAT receipt prints a
      subtotal that already contains the tax, so adding it again would flag
      every such receipt.
    * **Discount deducted vs already applied.** Many receipts print a savings
      summary ("TOTAL SAVINGS THIS TRIP $0.88") that is *informational* -- the
      reduction is already reflected in the item prices. Subtracting it would
      manufacture a mismatch on a perfectly consistent receipt. Distinguishing
      the two by wording is unreliable across chains, so both readings are
      tried and the discount is still reported either way.
    """
    if subtotal is None or total is None:
        return []

    zero = Decimal("0")
    additions = (service or zero) + (shipping or zero) + (tip or zero) + (rounding or zero)

    candidates = []
    for deduction in {zero, discount or zero}:
        base = subtotal - deduction + additions
        candidates.append(base + (tax_total or zero))  # tax added on top
        candidates.append(base)  # tax already included

    tolerance = tolerance_for(total, settings)
    differences = [abs(candidate - total) for candidate in candidates]

    if min(differences) <= tolerance:
        return []

    # Report the reading that came closest, so the context names the figure a
    # human should actually compare against the printed total.
    best_index = differences.index(min(differences))
    return [
        _issue(
            IssueCode.TOTAL_MISMATCH,
            IssueSeverity.ERROR,
            "Calculated total does not match the total printed on the receipt.",
            "total",
            calculated=candidates[best_index],
            reported=total,
            difference=min(differences),
            tolerance=tolerance,
        )
    ]


def _check_ordering(
    subtotal: Decimal | None,
    total: Decimal | None,
    discount: Decimal | None,
    settings: Settings,
) -> list[ValidationIssue]:
    """Check that the subtotal does not exceed the total.

    Only applies when no discount was found: with a discount, a subtotal above
    the total is exactly what is expected.
    """
    if subtotal is None or total is None or discount:
        return []
    if subtotal <= total + tolerance_for(total, settings):
        return []
    return [
        _issue(
            IssueCode.SUBTOTAL_EXCEEDS_TOTAL,
            IssueSeverity.WARNING,
            "Subtotal is greater than the total but no discount was found.",
            "subtotal",
            subtotal=subtotal,
            total=total,
        )
    ]


def _check_tender(
    *,
    total: Decimal | None,
    tendered: Decimal | None,
    change: Decimal | None,
    settings: Settings,
) -> list[ValidationIssue]:
    """Corroborate the total against the cash tendered and change given.

    ``tendered - change == total`` is an independent confirmation of the
    total, derived from figures the total extractor never touched. When it
    holds it is strong evidence; when it fails it is a warning rather than an
    error, because partial and split tenders break the identity legitimately.
    """
    if total is None or tendered is None or change is None:
        return []
    implied = tendered - change
    if abs(implied - total) <= tolerance_for(total, settings):
        return []
    return [
        _issue(
            IssueCode.TOTAL_MISMATCH,
            IssueSeverity.WARNING,
            "Amount tendered minus change does not equal the total.",
            "payment.amount_paid",
            implied_total=implied,
            reported_total=total,
        )
    ]


def _check_tax_details(extraction: ExtractionResult, settings: Settings) -> list[ValidationIssue]:
    """Check that itemised tax lines sum to the reported tax total."""
    tax = extraction.tax.value
    if tax is None or tax.total is None or len(tax.details) < 2:
        return []

    detail_sum = sum(
        (detail.amount for detail in tax.details if detail.amount is not None),
        Decimal("0"),
    )
    if abs(detail_sum - tax.total) <= tolerance_for(tax.total, settings):
        return []

    return [
        _issue(
            IssueCode.TAX_MISMATCH,
            IssueSeverity.WARNING,
            "Itemised tax lines do not sum to the reported tax total.",
            "tax.total",
            detail_sum=detail_sum,
            reported=tax.total,
        )
    ]
