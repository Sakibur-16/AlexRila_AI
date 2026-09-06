"""Payment extraction.

Two constraints shape this module.

**Only a masked tail is ever captured.** Card numbers are extracted as four
digits and nothing else. Even when a receipt prints a full PAN, redaction runs
before extraction, so there is no code path by which a full number reaches the
schema.

**Payment lines are not totals.** ``CASH 30.00`` and ``CHANGE 4.01`` are
tender amounts, and confusing either with the receipt total is a classic
extraction bug. They are captured here so that the validator can use them as
corroboration (``tendered - change ≈ total``) without them ever competing for
the ``total`` field.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from decimal import Decimal

from app.domain.evidence import ExtractedField, ExtractionMethod
from app.extraction.context import LineView, ReceiptContext
from app.extraction.lexicon import LabelCategory
from app.normalization.numbers import extract_auth_code, extract_card_type
from app.schemas.receipt import PaymentMethod, PaymentSplit

#: Printed forms mapped to the normalised method enum. Ordered most specific
#: first so "CREDIT CARD" is not shortened to "CARD".
_METHOD_PATTERNS: tuple[tuple[re.Pattern[str], PaymentMethod], ...] = (
    (re.compile(r"\b(CREDIT\s*CARD|CREDIT)\b", re.IGNORECASE), PaymentMethod.CREDIT_CARD),
    (re.compile(r"\b(DEBIT\s*CARD|DEBIT|EFTPOS)\b", re.IGNORECASE), PaymentMethod.DEBIT_CARD),
    (
        re.compile(
            r"\b(APPLE\s*PAY|GOOGLE\s*PAY|SAMSUNG\s*PAY|PAYPAL|BKASH|NAGAD|UPI|PAYTM"
            r"|ALIPAY|WECHAT\s*PAY|SWISH|MOBILE\s*PAY|CONTACTLESS)\b",
            re.IGNORECASE,
        ),
        PaymentMethod.MOBILE,
    ),
    (
        re.compile(r"\b(GIFT\s*CARD|VOUCHER|COUPON\s*PAYMENT)\b", re.IGNORECASE),
        PaymentMethod.VOUCHER,
    ),
    (
        re.compile(r"\b(BANK\s*TRANSFER|WIRE|IBAN|NEFT|IMPS)\b", re.IGNORECASE),
        PaymentMethod.BANK_TRANSFER,
    ),
    (re.compile(r"\b(CHEQUE|CHECK\s*NO|CHECK\s*PAYMENT)\b", re.IGNORECASE), PaymentMethod.CHECK),
    (
        re.compile(
            r"\b(VISA|MASTERCARD|MASTER\s*CARD|AMEX|AMERICAN\s*EXPRESS|DISCOVER"
            r"|MAESTRO|JCB|UNIONPAY|RUPAY|CARD)\b",
            re.IGNORECASE,
        ),
        PaymentMethod.CARD,
    ),
    (re.compile(r"\bCASH\b", re.IGNORECASE), PaymentMethod.CASH),
)

#: A masked card tail as printed by terminals: ****1234, XXXX1234, ...1234,
#: and the single-asterisk form ("*5025") that some registers use.
_MASKED_TAIL = re.compile(
    r"(?:[*xX#•]{2,}[\s-]?){1,4}(\d{4})(?!\d)"
    r"|\.{3,}\s?(\d{4})(?!\d)"
    r"|(?<![\w*])\*(\d{4})(?!\d)"
)

#: Loyalty, membership and rewards lines. They carry a masked number and the
#: word "card", so without an explicit exclusion they are indistinguishable
#: from a payment card -- and a store loyalty number reported as card_last4
#: is worse than no value at all.
_LOYALTY_LINE = re.compile(
    r"\b(?:EXTRACARE|LOYALTY|REWARDS?|MEMBER(?:SHIP)?|CLUB|POINTS?|ADVANTAGE"
    r"|BONUS\s*CARD|STORE\s*CARD|GIFT\s*CARD\s*BALANCE)\b",
    re.IGNORECASE,
)

#: Permitted discrepancy when checking that split amounts sum to the total.
_SPLIT_TOLERANCE = Decimal("0.05")

#: A card tail introduced by a label rather than a mask.
_LABELLED_TAIL = re.compile(
    r"\b(?:CARD|ACCT|ACCOUNT|PAN)\s*(?:NO|NUMBER|#)?\s*[:#]?\s*(?:ENDING\s*(?:IN)?\s*)?(\d{4})(?!\d)",
    re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class PaymentResult:
    """Extracted payment details."""

    method: ExtractedField[PaymentMethod]
    card_type: ExtractedField[str]
    card_last4: ExtractedField[str]
    authorization_code: ExtractedField[str]
    amount_paid: ExtractedField[Decimal]
    change: ExtractedField[Decimal]
    splits: tuple[PaymentSplit, ...] = ()


def extract_payment(
    context: ReceiptContext,
    *,
    tendered: ExtractedField[Decimal],
    change: ExtractedField[Decimal],
    total: ExtractedField[Decimal] | None = None,
) -> PaymentResult:
    """Extract how the receipt was settled.

    Args:
        context: Document context.
        tendered: Amount-tendered field already read by the totals extractor,
            reused rather than re-parsed so both agree by construction.
        change: Change-due field, likewise.
        total: Receipt total, used to corroborate a suspected split payment.
    """
    payment_lines = _payment_lines(context)

    method = _extract_method(payment_lines)
    card_type = _scan(payment_lines, extract_card_type, ExtractionMethod.REGEX)
    card_last4 = _extract_card_last4(payment_lines)
    auth = _scan(payment_lines, extract_auth_code, ExtractionMethod.REGEX)
    splits = _extract_splits(payment_lines, total.value if total else None)

    return PaymentResult(
        method=method,
        card_type=card_type,
        card_last4=card_last4,
        authorization_code=auth,
        amount_paid=tendered,
        change=change,
        splits=splits,
    )


def _payment_lines(context: ReceiptContext) -> list[LineView]:
    """Lines that plausibly describe payment, best first."""
    labelled = context.find_all(
        LabelCategory.PAYMENT_METHOD, LabelCategory.TENDERED, LabelCategory.CHANGE
    )
    tail = context.lines[max(0, context.items_end) :]
    ordered: list[LineView] = list(labelled)
    for line in tail:
        if line not in ordered:
            ordered.append(line)
    return ordered or list(context.lines)


def _extract_method(lines: list[LineView]) -> ExtractedField[PaymentMethod]:
    """Identify the payment instrument."""
    for line in lines:
        if _LOYALTY_LINE.search(line.normalized):
            continue
        for pattern, method in _METHOD_PATTERNS:
            if pattern.search(line.normalized):
                return ExtractedField(
                    value=method,
                    evidence=(
                        line.evidence(
                            ExtractionMethod.KEYWORD_ANCHORED, notes=f"method={method.value}"
                        ),
                    ),
                    raw_value=line.raw,
                )
    return ExtractedField.absent()


def _extract_card_last4(lines: list[LineView]) -> ExtractedField[str]:
    """Extract the last four digits of the card used, if printed."""
    for line in lines:
        if _LOYALTY_LINE.search(line.normalized):
            continue
        match = _MASKED_TAIL.search(line.normalized)
        if match:
            digits = match.group(1) or match.group(2) or match.group(3)
            if digits:
                return ExtractedField(
                    value=digits,
                    evidence=(line.evidence(ExtractionMethod.REGEX, notes="masked_tail"),),
                )
        labelled = _LABELLED_TAIL.search(line.normalized)
        if labelled:
            return ExtractedField(
                value=labelled.group(1),
                evidence=(line.evidence(ExtractionMethod.REGEX, notes="labelled_tail"),),
            )
    return ExtractedField.absent()


def _extract_splits(lines: list[LineView], total: Decimal | None) -> tuple[PaymentSplit, ...]:
    """Detect a receipt settled with more than one instrument.

    Split detection is held to a deliberately high bar, because the cost of a
    false positive is a fabricated payment record:

    * each candidate must carry a *monetary* amount -- a masked card tail such
      as ``****4321`` parses as the integer 4321 and must never be read as an
      amount;
    * at least two distinct methods must be present;
    * their amounts must sum to the receipt total. Without that corroboration
      a "CASH 20.00" tender line beside a "VISA" footer looks identical to a
      genuine split, and only the arithmetic tells them apart.
    """
    if total is None:
        return ()

    found: list[PaymentSplit] = []
    seen: set[PaymentMethod] = set()

    for line in lines:
        if line.has(LabelCategory.CHANGE) or _LOYALTY_LINE.search(line.normalized):
            continue
        monetary = [amount for amount in line.amounts if "." in amount.raw or "," in amount.raw]
        if not monetary:
            continue
        for pattern, method in _METHOD_PATTERNS:
            if not pattern.search(line.normalized):
                continue
            if method in seen:
                break
            seen.add(method)
            tail = _MASKED_TAIL.search(line.normalized)
            found.append(
                PaymentSplit(
                    method=method,
                    amount=abs(monetary[-1].value),
                    card_last4=(
                        (tail.group(1) or tail.group(2) or tail.group(3)) if tail else None
                    ),
                )
            )
            break

    if len(found) < 2:
        return ()

    combined = sum((split.amount or Decimal("0") for split in found), Decimal("0"))
    if abs(combined - total) > _SPLIT_TOLERANCE:
        return ()
    return tuple(found)


def _scan(
    lines: list[LineView],
    extractor: Callable[[str], str | None],
    method: ExtractionMethod,
) -> ExtractedField[str]:
    """Apply a line-level extractor across ``lines``, first match wins."""
    for line in lines:
        value = extractor(line.normalized)
        if value:
            return ExtractedField(
                value=value,
                evidence=(line.evidence(method),),
                raw_value=line.raw,
            )
    return ExtractedField.absent()
