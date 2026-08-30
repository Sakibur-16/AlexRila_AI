"""Identifier and contact-detail normalisation.

These fields are the clearest case for deterministic extraction: a phone
number, an email address and a VAT number all have shape, and a regular
expression that matches that shape is more reliable, cheaper and more
reproducible than asking a model.

Every function here returns ``None`` rather than a best guess when the input
does not match, so that an unmatched field stays absent instead of becoming
plausible-looking noise.
"""

from __future__ import annotations

import re
from typing import Final

from app.normalization.text import normalize_unicode

#: Phone numbers: optional country prefix, 7-15 significant digits, with the
#: separators receipts actually use. Bounded so that a long transaction id is
#: not mistaken for a phone number.
_PHONE: Final[re.Pattern[str]] = re.compile(
    r"(?<![\d])(\+?\d{1,3}[\s.\-]?)?(\(?\d{2,4}\)?[\s.\-]?)?\d{3}[\s.\-]?\d{2,4}"
    r"([\s.\-]?\d{2,4})?(?![\d])"
)

#: Labels that mark a line as carrying a phone number.
_PHONE_LABELS: Final[re.Pattern[str]] = re.compile(
    r"\b(TEL|TELEPHONE|PHONE|PH|MOB|MOBILE|CONTACT|CALL|FAX)\b[.:\s#]*", re.IGNORECASE
)

_EMAIL: Final[re.Pattern[str]] = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")

_URL: Final[re.Pattern[str]] = re.compile(
    r"\b((?:https?://)?(?:www\.)[A-Za-z0-9.\-]+\.[A-Za-z]{2,}(?:/\S*)?)\b"
    r"|\b((?:https?://)[A-Za-z0-9.\-]+\.[A-Za-z]{2,}(?:/\S*)?)\b"
)

#: Tax registration identifiers, keyed by the label that precedes them.
_TAX_ID_LABELS: Final[re.Pattern[str]] = re.compile(
    r"\b(VAT(?:\s*(?:NO|NUMBER|REG|ID))?|GST(?:\s*(?:NO|NUMBER|IN))?|GSTIN|TIN|TAX\s*ID"
    r"|TAX\s*NO|ABN|ACN|NIF|CIF|CUIT|BIN|EIN|UID|USt-IdNr|SIRET)\b[.:\s#]*",
    re.IGNORECASE,
)

#: Business registration identifiers.
_REG_ID_LABELS: Final[re.Pattern[str]] = re.compile(
    r"\b(REG(?:\.|ISTRATION)?\s*(?:NO|NUMBER|ID)?|COMPANY\s*(?:NO|NUMBER)|CRN|BRN"
    r"|TRADE\s*LICENSE|LICENSE\s*NO)\b[.:\s#]*",
    re.IGNORECASE,
)

#: Receipt / invoice / transaction identifiers.
#:
#: The label is followed by an optional separator, an optional "number" word,
#: then the separator run before the value. That three-part shape is what lets
#: a single pattern read "Receipt No: X", "Beleg-Nr: X" and a bare
#: "Rechnung X" without a variant per language.
_RECEIPT_ID_LABELS: Final[re.Pattern[str]] = re.compile(
    r"\b("
    r"RECEIPT|RCPT|INVOICE|INV|BILL|ORDER|TRANS(?:ACTION)?|TXN|TRX|REF(?:ERENCE)?"
    r"|TICKET|CHECK|DOC(?:UMENT)?|SALE|REG(?:ISTER)?"
    r"|BELEG|KASSENBON|RECHNUNG|QUITTUNG"
    r"|FACTURE|RECU|FACTURA|RECIBO|NOTA|FATTURA|SCONTRINO"
    r")"
    r"\s*[-.]?\s*(?:NO|NR|NUM(?:BER)?|ID|N|#)?\b[.:\s#-]*",
    re.IGNORECASE,
)

#: A plausible identifier value: alphanumeric with optional dashes/slashes.
_ID_VALUE: Final[re.Pattern[str]] = re.compile(r"([A-Z0-9][A-Z0-9\-/_:]{1,35})", re.IGNORECASE)


#: Card-scheme names as printed by terminals.
_CARD_TYPES: Final[dict[str, str]] = {
    "VISA": "VISA",
    "MASTERCARD": "MASTERCARD",
    "MASTER CARD": "MASTERCARD",
    "MC": "MASTERCARD",
    "AMEX": "AMEX",
    "AMERICAN EXPRESS": "AMEX",
    "DISCOVER": "DISCOVER",
    "DINERS": "DINERS",
    "JCB": "JCB",
    "UNIONPAY": "UNIONPAY",
    "MAESTRO": "MAESTRO",
    "RUPAY": "RUPAY",
}

#: Words that, on a phone-shaped match, mean it is not a phone number.
_PHONE_NEGATIVE_CONTEXT: Final[re.Pattern[str]] = re.compile(
    r"\b(TOTAL|SUBTOTAL|TAX|VAT|CASH|CHANGE|CARD|QTY|ITEM|DATE|TIME|AUTH|APPR)\b",
    re.IGNORECASE,
)


def extract_phone(text: str) -> str | None:
    """Extract a phone number from ``text``.

    Requires either an explicit label (``TEL:``) or an international prefix.
    Without one of those, a bare digit run on a receipt is far more likely to
    be a transaction id, so nothing is returned -- precision over recall.
    """
    if not text:
        return None
    source = normalize_unicode(text)

    labelled = _PHONE_LABELS.search(source)
    if labelled:
        remainder = source[labelled.end() :]
        match = _PHONE.search(remainder)
        if match:
            return _clean_phone(match.group(0))
        return None

    if _PHONE_NEGATIVE_CONTEXT.search(source):
        return None

    stripped = source.strip()
    if stripped.startswith("+"):
        match = _PHONE.search(stripped)
        if match:
            return _clean_phone(match.group(0))
    return None


def _clean_phone(raw: str) -> str | None:
    """Normalise spacing and validate digit count."""
    cleaned = re.sub(r"[^\d+]", "", raw)
    digits = cleaned.lstrip("+")
    if not 7 <= len(digits) <= 15:
        return None
    return cleaned


def extract_email(text: str) -> str | None:
    """Extract the first email address in ``text``."""
    if not text:
        return None
    match = _EMAIL.search(normalize_unicode(text))
    return match.group(0).lower() if match else None


def extract_website(text: str) -> str | None:
    """Extract the first website URL in ``text``.

    Requires a scheme or a ``www.`` prefix: matching any bare ``x.y`` token
    would turn abbreviations and prices into URLs.
    """
    if not text:
        return None
    match = _URL.search(normalize_unicode(text))
    if not match:
        return None
    return (match.group(1) or match.group(2)).rstrip(".,;").lower()


def _extract_labelled_id(text: str, labels: re.Pattern[str]) -> tuple[str, str] | None:
    """Extract ``(label, value)`` for a labelled identifier."""
    if not text:
        return None
    source = normalize_unicode(text)
    match = labels.search(source)
    if match is None:
        return None
    remainder = source[match.end() :].strip()
    value_match = _ID_VALUE.search(remainder)
    if value_match is None:
        return None
    value = value_match.group(1).strip(" -/_")
    if not value or not any(ch.isdigit() for ch in value):
        # An identifier with no digits is almost always a mis-split word.
        return None
    return match.group(1).strip(), value


def extract_tax_id(text: str) -> str | None:
    """Extract a VAT/GST/TIN registration number."""
    found = _extract_labelled_id(text, _TAX_ID_LABELS)
    return found[1] if found else None


def extract_registration_id(text: str) -> str | None:
    """Extract a company registration number."""
    found = _extract_labelled_id(text, _REG_ID_LABELS)
    return found[1] if found else None


def extract_receipt_number(text: str) -> str | None:
    """Extract a receipt / invoice / transaction identifier."""
    found = _extract_labelled_id(text, _RECEIPT_ID_LABELS)
    return found[1] if found else None


def extract_card_type(text: str) -> str | None:
    """Identify the card scheme named in ``text``."""
    if not text:
        return None
    upper = normalize_unicode(text).upper()
    for needle, scheme in _CARD_TYPES.items():
        if re.search(rf"\b{re.escape(needle)}\b", upper):
            return scheme
    return None


def extract_auth_code(text: str) -> str | None:
    """Extract a card authorisation / approval code."""
    if not text:
        return None
    source = normalize_unicode(text)
    match = re.search(
        r"\b(?:AUTH(?:ORI[SZ]ATION)?|APPROVAL|APPR)\s*(?:CODE|NO|#)?\b[.:\s#]*([A-Z0-9]{4,12})\b",
        source,
        re.IGNORECASE,
    )
    return match.group(1).upper() if match else None
