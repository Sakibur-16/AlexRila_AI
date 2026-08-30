"""Locale-aware monetary parsing.

Receipts print the same amount four different ways::

    1,234.50    (en-US, en-GB, en-IN)
    1.234,50    (de, es, it, pt, id)
    1 234,50    (fr, ru, sv -- space or NBSP as group separator)
    1'234.50    (de-CH)

Guessing wrongly turns 1.234,50 into 1.23 -- a three-order-of-magnitude error
that arithmetic validation would then "confirm" against an equally misparsed
total. So separator interpretation is decided **per document**, not per number:
:func:`detect_separator_style` samples every amount on the receipt and picks
the reading that is consistent across all of them, which is far more reliable
than examining any single value.
"""

from __future__ import annotations

import enum
import re
from decimal import Decimal, InvalidOperation
from typing import Final, NamedTuple

from app.normalization.text import normalize_numeric_tokens, normalize_unicode

#: An amount: optional sign/paren, digits with optional separators, optional
#: trailing minus (used by some registers for refunds).
AMOUNT_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"""
    (?P<open>\()?                       # accounting-style negative
    (?P<sign>[-+])?
    (?P<digits>
        \d{1,3}(?:[,.\s'  ]\d{3})+(?:[.,]\d{1,3})?   # grouped
        | \d+[.,]\d{1,3}                                        # simple decimal
        | \d+                                                   # bare integer
    )
    (?P<close>\))?
    (?P<trailing_sign>-)?
    """,
    re.VERBOSE,
)

#: Amount preceded or followed by a currency indicator. Used to prefer real
#: money over incidental numbers such as quantities and phone fragments.
CURRENCY_ADJACENT: Final[re.Pattern[str]] = re.compile(
    r"(?:(?P<symbol_before>[$€£¥₹৳₺₩₪฿]|[A-Z]{3})\s*)?"
    r"(?P<amount>[\d][\d,.\s'  ]*)"
    r"(?:\s*(?P<symbol_after>[$€£¥₹৳₺₩₪฿]|[A-Z]{3}))?"
)


class SeparatorStyle(enum.StrEnum):
    """How to read ``,`` and ``.`` in a document."""

    #: ``,`` groups thousands, ``.`` marks decimals (1,234.50).
    DOT_DECIMAL = "dot_decimal"
    #: ``.`` groups thousands, ``,`` marks decimals (1.234,50).
    COMMA_DECIMAL = "comma_decimal"
    #: Not enough evidence; treat a lone separator as decimal.
    UNKNOWN = "unknown"


class ParsedAmount(NamedTuple):
    """A parsed monetary value with its source text."""

    value: Decimal
    raw: str
    #: True when the value was written in accounting-negative form.
    negative: bool = False


def detect_separator_style(texts: list[str], configured: str = "auto") -> SeparatorStyle:
    """Infer the document's decimal convention from all its amounts.

    Args:
        texts: Every line of the document.
        configured: ``dot``, ``comma`` or ``auto``. An explicit setting wins,
            because an operator processing a known locale should not be
            second-guessed.

    Returns:
        The inferred style. ``UNKNOWN`` when the document contains no
        disambiguating evidence, which is common for small receipts where
        every amount is under 1000.
    """
    if configured == "dot":
        return SeparatorStyle.DOT_DECIMAL
    if configured == "comma":
        return SeparatorStyle.COMMA_DECIMAL

    dot_votes = 0
    comma_votes = 0

    for text in texts:
        for match in AMOUNT_PATTERN.finditer(normalize_unicode(text)):
            digits = match.group("digits")
            last_dot = digits.rfind(".")
            last_comma = digits.rfind(",")

            if last_dot >= 0 and last_comma >= 0:
                # Both present: whichever comes last is the decimal separator.
                if last_dot > last_comma:
                    dot_votes += 3
                else:
                    comma_votes += 3
                continue

            # A single separator followed by exactly three digits is ambiguous
            # ("1,234" could be 1234 or 1.234), so it casts no vote. Followed
            # by one or two digits it can only be a decimal separator.
            if last_comma >= 0:
                tail = len(digits) - last_comma - 1
                if tail in (1, 2):
                    comma_votes += 1
                elif tail == 3:
                    dot_votes += 1  # comma as a group separator implies dot decimals
            elif last_dot >= 0:
                tail = len(digits) - last_dot - 1
                if tail in (1, 2):
                    dot_votes += 1
                elif tail == 3:
                    comma_votes += 1

    if dot_votes == comma_votes:
        return SeparatorStyle.UNKNOWN
    return SeparatorStyle.DOT_DECIMAL if dot_votes > comma_votes else SeparatorStyle.COMMA_DECIMAL


def parse_amount(
    text: str,
    *,
    style: SeparatorStyle = SeparatorStyle.UNKNOWN,
    repair_ocr: bool = True,
) -> ParsedAmount | None:
    """Parse the first monetary amount in ``text``.

    Args:
        text: Source text, typically one OCR line or a fragment of one.
        style: Document separator convention from :func:`detect_separator_style`.
        repair_ocr: Apply context-aware glyph repair first. Disable when the
            text is known clean, e.g. when re-parsing LLM output.

    Returns:
        The parsed amount, or ``None`` when ``text`` holds no amount. Returning
        ``None`` rather than zero is deliberate: a missing amount and a zero
        amount mean entirely different things on a receipt.
    """
    if not text:
        return None

    source = normalize_unicode(text)
    if repair_ocr:
        source, _ = normalize_numeric_tokens(source)

    match = AMOUNT_PATTERN.search(source)
    if match is None:
        return None

    raw = match.group(0)
    digits = match.group("digits")
    negative = bool(
        match.group("sign") == "-"
        or (match.group("open") and match.group("close"))
        or match.group("trailing_sign")
    )

    normalized = _to_canonical(digits, style)
    if normalized is None:
        return None

    try:
        value = Decimal(normalized)
    except InvalidOperation:
        return None

    if negative:
        value = -value
    return ParsedAmount(value=value, raw=raw, negative=negative)


def parse_all_amounts(
    text: str, *, style: SeparatorStyle = SeparatorStyle.UNKNOWN, repair_ocr: bool = True
) -> list[ParsedAmount]:
    """Parse every amount in ``text``, left to right.

    Item lines carry several numbers (``2 x 4.00  8.00``), and which one is the
    unit price versus the line total is a *positional* decision made by the
    item extractor -- so this returns all of them in order.
    """
    if not text:
        return []

    source = normalize_unicode(text)
    if repair_ocr:
        source, _ = normalize_numeric_tokens(source)

    results: list[ParsedAmount] = []
    for match in AMOUNT_PATTERN.finditer(source):
        parsed = parse_amount(match.group(0), style=style, repair_ocr=False)
        if parsed is not None:
            results.append(parsed)
    return results


def _to_canonical(digits: str, style: SeparatorStyle) -> str | None:
    """Rewrite a grouped number into plain ``-?\\d+(\\.\\d+)?`` form."""
    cleaned = digits.strip()
    if not cleaned:
        return None

    # Spaces and apostrophes are only ever group separators.
    cleaned = re.sub(r"[\s'  ]", "", cleaned)

    has_dot = "." in cleaned
    has_comma = "," in cleaned

    if has_dot and has_comma:
        # Unambiguous: the rightmost separator is the decimal point.
        if cleaned.rfind(".") > cleaned.rfind(","):
            return cleaned.replace(",", "")
        return cleaned.replace(".", "").replace(",", ".")

    if not has_dot and not has_comma:
        return cleaned

    separator = "." if has_dot else ","
    parts = cleaned.split(separator)

    if len(parts) > 2:
        # Repeated separator can only be grouping: 1.234.567
        return "".join(parts)

    tail = parts[1]
    if len(tail) == 3:
        # The genuinely ambiguous case. Resolve by document style; with no
        # style evidence, treat it as grouping, which is the safer error: a
        # value read as 1234 instead of 1.234 fails arithmetic validation
        # loudly, whereas the reverse silently under-reports by 1000x.
        if separator == "." and style is SeparatorStyle.COMMA_DECIMAL:
            return "".join(parts)
        if separator == "," and style is SeparatorStyle.DOT_DECIMAL:
            return "".join(parts)
        if style is SeparatorStyle.UNKNOWN:
            return "".join(parts)
        return f"{parts[0]}.{tail}"

    if len(tail) in (1, 2):
        return f"{parts[0]}.{tail}"

    return "".join(parts)


def parse_percentage(text: str) -> Decimal | None:
    """Parse a percentage rate such as ``VAT 20%`` or ``Tax @ 8.25 %``."""
    if not text:
        return None
    match = re.search(r"(\d{1,2}(?:[.,]\d{1,3})?)\s*%", normalize_unicode(text))
    if match is None:
        return None
    try:
        return Decimal(match.group(1).replace(",", "."))
    except InvalidOperation:
        return None


def parse_quantity(text: str) -> Decimal | None:
    """Parse a quantity, which may be fractional (``1.5 kg``, ``2``)."""
    if not text:
        return None
    match = re.search(r"(\d+(?:[.,]\d{1,3})?)", normalize_unicode(text))
    if match is None:
        return None
    try:
        return Decimal(match.group(1).replace(",", "."))
    except InvalidOperation:
        return None
