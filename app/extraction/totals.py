"""Monetary total extraction.

The rule that matters most: **never take the largest number on the receipt.**

A receipt's largest number is routinely the cash tendered, a loyalty point
balance, a phone number fragment or a transaction id. Totals are identified by
their *label*, and the label vocabulary is ordered so that ``SUBTOTAL`` can
never satisfy a request for ``TOTAL`` (see :mod:`app.extraction.lexicon`).

When several candidates share a label, the last one wins: receipts print
running or per-tender totals above the final figure, and the bottom-most is
authoritative.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from decimal import Decimal

from app.domain.evidence import ExtractedField, ExtractionMethod
from app.extraction.context import LineView, ReceiptContext
from app.extraction.lexicon import LabelCategory, LabelMatch
from app.normalization.money import parse_percentage
from app.schemas.receipt import Discount, ReceiptSection, Tax, TaxDetail

#: Above this a figure is a loyalty balance, a store number or a misread,
#: not a receipt total. Only used by the unlabelled fallback.
_MAX_PLAUSIBLE_TOTAL = Decimal("100000")


@dataclass(frozen=True, slots=True)
class TotalsResult:
    """Every monetary aggregate read from the totals block."""

    subtotal: ExtractedField[Decimal]
    total: ExtractedField[Decimal]
    tax: ExtractedField[Tax]
    discount: ExtractedField[Discount]
    service_charge: ExtractedField[Decimal]
    shipping: ExtractedField[Decimal]
    tip: ExtractedField[Decimal]
    rounding: ExtractedField[Decimal]
    #: Cash or card amount presented, used to sanity-check the total.
    tendered: ExtractedField[Decimal]
    change: ExtractedField[Decimal]


def extract_totals(context: ReceiptContext) -> TotalsResult:
    """Read all monetary aggregates from ``context``."""
    return TotalsResult(
        subtotal=_labelled_amount(context, LabelCategory.SUBTOTAL),
        total=_extract_total(context),
        tax=_extract_tax(context),
        discount=_extract_discount(context),
        service_charge=_labelled_amount(context, LabelCategory.SERVICE_CHARGE),
        shipping=_labelled_amount(context, LabelCategory.SHIPPING),
        tip=_labelled_amount(context, LabelCategory.TIP),
        rounding=_labelled_amount(context, LabelCategory.ROUNDING, allow_negative=True),
        tendered=_labelled_amount(context, LabelCategory.TENDERED),
        change=_labelled_amount(context, LabelCategory.CHANGE),
    )


def _labelled_amount(
    context: ReceiptContext,
    category: LabelCategory,
    *,
    allow_negative: bool = False,
    prefer_last: bool = True,
) -> ExtractedField[Decimal]:
    """Extract the amount carried by the last line labelled ``category``.

    Args:
        context: Document context.
        category: Label category to look for.
        allow_negative: Keep a negative sign. Most aggregates are magnitudes
            even when printed as ``-2.00``; rounding adjustments are the
            exception and genuinely can be negative.
        prefer_last: Take the bottom-most labelled line.

    Returns:
        The field, absent when no labelled line carries an associable amount.
    """
    candidates = context.find_all(category)
    if not candidates:
        return ExtractedField.absent()

    for line in reversed(candidates) if prefer_last else candidates:
        label = line.label_of(category)
        if label is None:
            continue
        found = context.value_for_label(line, label)
        if found is None:
            continue
        amount, evidence = found
        value = amount.value if allow_negative else abs(amount.value)
        return ExtractedField(
            value=value,
            confidence=0.0,  # assigned by the confidence scorer
            evidence=(evidence,),
            raw_value=amount.raw,
        )

    return ExtractedField.absent()


def _extract_total(context: ReceiptContext) -> ExtractedField[Decimal]:
    """Extract the grand total.

    Prefers an explicitly emphatic label (``GRAND TOTAL``, ``AMOUNT DUE``) over
    a bare ``TOTAL``, since a receipt printing both means the emphatic one.
    """
    candidates = [
        line
        for line in context.find_all(LabelCategory.TOTAL)
        if not line.has(LabelCategory.CHANGE, LabelCategory.DISCOUNT, LabelCategory.TENDERED)
        and not _NON_MONETARY_BALANCE.search(line.keyword_text)
    ]

    if candidates:
        emphatic = [
            line
            for line in candidates
            if (label := line.label_of(LabelCategory.TOTAL)) is not None
            and label.keyword in _EMPHATIC_TOTALS
        ]
        ordered = emphatic or candidates

        for line in reversed(ordered):
            label = line.label_of(LabelCategory.TOTAL)
            if label is None:
                continue
            found = context.value_for_label(line, label)
            if found is None:
                continue
            amount, evidence = found
            return ExtractedField(
                value=amount.value,
                evidence=(evidence,),
                raw_value=amount.raw,
            )

    return _fallback_total_from_bottom(context)


def _fallback_total_from_bottom(context: ReceiptContext) -> ExtractedField[Decimal]:
    """Last resort when no ``TOTAL`` label was found anywhere on the receipt.

    Deliberately narrow. The module rule is *never take the largest number*,
    and the largest number in a totals block is routinely the cash tendered:

    .. code-block:: text

        SUBTOTAL 5.50
        CASH    20.00      <- largest, and not the total
        CHANGE  14.50

    So a tender line is excluded along with change, discounts and metadata. The
    figure is taken from the **bottom-most** qualifying line rather than the
    largest one, because the payable amount is printed last, and it is scored
    ``HEURISTIC`` so the confidence layer and review flag reflect the guess.
    """
    if not context.lines:
        return ExtractedField.absent()

    region = context.in_section(
        ReceiptSection.TOTALS, ReceiptSection.PAYMENT, ReceiptSection.FOOTER
    )
    if not region:
        return ExtractedField.absent()

    excluded = (
        LabelCategory.CHANGE,
        LabelCategory.TENDERED,  # cash presented is not the amount owed
        LabelCategory.DISCOUNT,
        LabelCategory.ITEM_COUNT,
        LabelCategory.DATE,
        LabelCategory.TIME,
        LabelCategory.RECEIPT_ID,
        LabelCategory.SUBTOTAL,
        LabelCategory.MERCHANT_CONTACT,
    )

    for line in reversed(region):
        if line.has(*excluded):
            continue
        # Require a real monetary figure: a bare integer here is a loyalty
        # balance, a store number or a phone fragment far more often than a total.
        monetary = [
            amount
            for amount in line.amounts
            if ("." in amount.raw or "," in amount.raw)
            and Decimal("0") < amount.value < _MAX_PLAUSIBLE_TOTAL
        ]
        if not monetary:
            continue
        chosen = monetary[-1]
        return ExtractedField(
            value=chosen.value,
            evidence=(line.evidence(ExtractionMethod.HEURISTIC, notes="unlabelled_bottom_total"),),
            raw_value=chosen.raw,
        )

    return ExtractedField.absent()


#: Footer savings summaries. They report what was saved across the whole trip
#: and are informational: the reductions are already reflected in the prices
#: above. Summing them into the discount produced a figure many times the
#: subtotal.
_SAVINGS_SUMMARY = re.compile(
    r"\b(?:TODAY\s+YOU\s+SAVED|YOU\s+SAVED\s+TODAY|TOTAL\s+SAVINGS"
    r"|SAVINGS\s+VALUE|TRIP\s+SUMMARY|YOUR\s+SAVINGS)\b",
    re.IGNORECASE,
)

#: Qualifiers that turn a TOTAL-vocabulary word into something else entirely.
#: "BALANCE" is a legitimate total label, but "POINTS BALANCE 9500" is a
#: loyalty statement -- and being the largest figure on the receipt, it would
#: win outright unless excluded here.
_NON_MONETARY_BALANCE = re.compile(
    r"\b(?:POINTS?|REWARDS?|LOYALTY|STARS?|MILES?|CREDITS?|GIFT\s*CARD"
    r"|VOUCHER|ACCOUNT|MEMBERSHIP)\b",
    re.IGNORECASE,
)

#: Labels that unambiguously mark the final payable figure.
_EMPHATIC_TOTALS = frozenset(
    {
        "GRAND TOTAL",
        "TOTAL DUE",
        "AMOUNT DUE",
        "BALANCE DUE",
        "TOTAL PAYABLE",
        "AMOUNT PAYABLE",
        "NET PAYABLE",
        "TOTAL AMOUNT",
        "BAL DUE",
        "G.TOTAL",
        "G TOTAL",
        "NET AMT",
        "TOTAL RS",
        "TOTAL USD",
        "TOTALL",
        "TO TAL",
        "FINAL TOTAL",
        "TOTAL COST",
    }
)


def _extract_tax(context: ReceiptContext) -> ExtractedField[Tax]:
    """Extract tax lines and their total.

    Receipts print tax in three shapes, all handled here:

    * one ``TAX 2.10`` line;
    * several rate-specific lines (``VAT 20% 4.00`` / ``VAT 5% 0.50``);
    * a breakdown plus an explicit ``TOTAL TAX`` line.

    When an explicit total is printed it is used verbatim. Otherwise the
    detail lines are summed -- and that sum is marked ``DERIVED``, so the
    confidence layer scores it below a directly-read value.
    """
    tax_lines = context.find_all(LabelCategory.TAX)
    if not tax_lines:
        return ExtractedField.absent()

    details: list[TaxDetail] = []
    explicit_total: Decimal | None = None
    explicit_line: LineView | None = None

    for line in tax_lines:
        label = line.label_of(LabelCategory.TAX)
        if label is None:
            continue
        found = context.value_for_label(line, label)
        if found is None:
            continue
        amount, _ = found
        rate = parse_percentage(line.text_after(label)) or parse_percentage(line.normalized)

        if label.keyword in {"TOTAL TAX", "TAX TOTAL"}:
            explicit_total = abs(amount.value)
            explicit_line = line
            continue

        details.append(
            TaxDetail(
                name=label.keyword,
                rate=rate,
                amount=abs(amount.value),
                taxable_amount=None,
            )
        )

    if explicit_total is not None and explicit_line is not None:
        label = explicit_line.label_of(LabelCategory.TAX)
        evidence = explicit_line.evidence(
            ExtractionMethod.KEYWORD_ANCHORED,
            notes=f"label={label.keyword if label else 'TAX'}",
        )
        return ExtractedField(
            value=Tax(total=explicit_total, details=tuple(details)),
            evidence=(evidence,),
        )

    if not details:
        return ExtractedField.absent()

    if len(details) == 1:
        line = tax_lines[0]
        label = line.label_of(LabelCategory.TAX)
        evidence = line.evidence(
            ExtractionMethod.KEYWORD_ANCHORED,
            notes=f"label={label.keyword if label else 'TAX'}",
        )
        return ExtractedField(
            value=Tax(total=details[0].amount, details=tuple(details)),
            evidence=(evidence,),
        )

    summed = sum((d.amount for d in details if d.amount is not None), Decimal("0"))
    evidence = tax_lines[0].evidence(
        ExtractionMethod.DERIVED, notes=f"sum_of_{len(details)}_tax_lines"
    )
    return ExtractedField(
        value=Tax(total=summed, details=tuple(details)),
        evidence=(evidence,),
    )


def _extract_discount(context: ReceiptContext) -> ExtractedField[Discount]:
    """Extract the total discount.

    Several discount lines are summed into one aggregate; the individual
    descriptions are not preserved on the aggregate because the schema models
    a single receipt-level discount. Per-item discounts live on their items.

    Amounts are stored as positive magnitudes even though receipts print them
    negatively, so that the financial validator can apply a consistent
    ``subtotal - discount`` rule without sign guesswork.
    """
    lines = context.find_all(LabelCategory.DISCOUNT)
    if not lines:
        return ExtractedField.absent()

    collected: list[tuple[Decimal, LineView, str]] = []
    for line in lines:
        label = line.label_of(LabelCategory.DISCOUNT)
        if label is None:
            continue
        if _SAVINGS_SUMMARY.search(line.normalized) or _is_item_level_saving(line, label):
            # Belongs to the item above it, and the item extractor already
            # records it. Counting it here as well is double-counting.
            continue
        found = context.value_for_label(line, label)
        if found is None:
            continue
        amount, _ = found
        collected.append((abs(amount.value), line, label.keyword))

    if not collected:
        return ExtractedField.absent()

    if len(collected) == 1:
        value, line, keyword = collected[0]
        rate = parse_percentage(line.normalized)
        return ExtractedField(
            value=Discount(description=keyword, amount=value, rate=rate),
            evidence=(line.evidence(ExtractionMethod.KEYWORD_ANCHORED, notes=f"label={keyword}"),),
        )

    total = sum((value for value, _, _ in collected), Decimal("0"))
    return ExtractedField(
        value=Discount(description="TOTAL DISCOUNT", amount=total, rate=None),
        evidence=(
            collected[0][1].evidence(
                ExtractionMethod.DERIVED, notes=f"sum_of_{len(collected)}_discount_lines"
            ),
        ),
    )


def _is_item_level_saving(line: LineView, label: LabelMatch) -> bool:
    """Whether a discount keyword is an annotation on an item line.

    A receipt-level discount leads its line, allowing for a quantity:
    "DISCOUNT 5.00", "1 MFR COUPON 5.00 -". An item-level one trails the
    item's own price: "PNTNE PRO-V DMR CD 12Z 4.49T SAVED .50".

    The discriminator is a *monetary* figure before the keyword. A bare
    integer there is a quantity, and treating it as a price wrongly excluded
    every coupon row.
    """
    if label.start == 0:
        return False
    return any(
        ("." in amount.raw or "," in amount.raw)
        and 0 <= line.normalized.find(amount.raw) < label.start
        for amount in line.amounts
    )
