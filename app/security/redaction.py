"""Sensitive-data redaction for receipt text.

Card receipts sometimes print more than they should: a full PAN on an older
terminal, a truncated PAN in an unmasked form, occasionally a PIN prompt echo.
Once such a string is in OCR text it can reach logs, storage and LLM providers,
so it is masked as early as possible.

The design bias is deliberate: **over-redact rather than under-redact**. A
masked value that turns out to have been a harmless order number costs nothing;
a leaked PAN is a compliance incident. The last four digits are preserved
because they are the one part the schema legitimately captures.
"""

from __future__ import annotations

import re
from typing import Final, NamedTuple

#: Digit groups of 13-19 digits, optionally separated by spaces or hyphens.
#: Anchored on non-digit boundaries so long numeric IDs are not split into a
#: false positive.
_PAN_PATTERN: Final[re.Pattern[str]] = re.compile(r"(?<![\d])(?:\d[ -]?){12,18}\d(?![\d])")

#: Explicit CVV/PIN labels followed by a short numeric value.
_CVV_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"\b(CVV2?|CVC2?|CID|PIN)\b\s*[:#]?\s*(\d{3,6})\b", re.IGNORECASE
)

#: Already-masked forms such as ``****1234`` or ``XXXX-XXXX-XXXX-1234``.
_MASKED_TAIL_PATTERN: Final[re.Pattern[str]] = re.compile(r"(?:[*xX#]{2,}[ -]?){2,}(\d{4})(?![\d])")

_MASK_CHAR: Final[str] = "*"


class RedactionResult(NamedTuple):
    """Redacted text plus what was removed."""

    text: str
    redacted: bool
    #: Counts by category, e.g. ``{"pan": 1}``. Safe to log.
    counts: dict[str, int]


def _luhn_valid(digits: str) -> bool:
    """Luhn checksum test.

    Used to decide whether a long digit run is *probably* a card number. It is
    a filter for precision only -- failing Luhn does not exempt a 16-digit run
    from masking when it also looks like a card (see :func:`redact_text`).
    """
    total = 0
    parity = len(digits) % 2
    for index, char in enumerate(digits):
        digit = ord(char) - 48
        if index % 2 == parity:
            digit *= 2
            if digit > 9:
                digit -= 9
        total += digit
    return total % 10 == 0


def mask_pan(digits: str) -> str:
    """Mask all but the final four digits of a card-like number."""
    if len(digits) <= 4:
        return _MASK_CHAR * len(digits)
    return _MASK_CHAR * (len(digits) - 4) + digits[-4:]


def extract_card_last4(text: str) -> str | None:
    """Return the last four digits of a card reference, if one is present.

    Reads masked forms (``****1234``) first because that is the common case on
    a modern receipt, then falls back to an unmasked PAN.
    """
    masked = _MASKED_TAIL_PATTERN.search(text)
    if masked:
        return masked.group(1)
    for match in _PAN_PATTERN.finditer(text):
        digits = re.sub(r"[ -]", "", match.group(0))
        if 13 <= len(digits) <= 19 and _luhn_valid(digits):
            return digits[-4:]
    return None


def redact_text(text: str | None) -> RedactionResult:
    """Mask card numbers and labelled CVV/PIN values in ``text``.

    Returns the original object unchanged (and ``redacted=False``) when there
    was nothing to mask, so callers can cheaply detect the common case.
    """
    if not text:
        return RedactionResult(text=text or "", redacted=False, counts={})

    counts: dict[str, int] = {}

    def _replace_pan(match: re.Match[str]) -> str:
        raw = match.group(0)
        digits = re.sub(r"[ -]", "", raw)
        if not (13 <= len(digits) <= 19):
            return raw
        # A 16-digit run on a payment receipt is masked whether or not it
        # passes Luhn: OCR digit errors would otherwise defeat the check.
        if not _luhn_valid(digits) and len(digits) not in (15, 16):
            return raw
        counts["pan"] = counts.get("pan", 0) + 1
        return mask_pan(digits)

    def _replace_cvv(match: re.Match[str]) -> str:
        counts["cvv"] = counts.get("cvv", 0) + 1
        return f"{match.group(1)} {_MASK_CHAR * len(match.group(2))}"

    redacted = _PAN_PATTERN.sub(_replace_pan, text)
    redacted = _CVV_PATTERN.sub(_replace_cvv, redacted)

    return RedactionResult(
        text=redacted,
        redacted=bool(counts),
        counts=counts,
    )
