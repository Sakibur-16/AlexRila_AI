"""Merchant identification.

The merchant name is the one important receipt field with no label to anchor
on -- receipts do not print ``MERCHANT:``. It is identified structurally
instead: it is the first substantial line of the header block, typically the
largest text on the page.

Where geometry is available, the tallest header line is preferred, because
point-of-sale software prints the store name in a larger font than the address
beneath it. Without geometry the extractor falls back to the first header line
that is not an address, phone number or greeting -- a weaker signal, and it is
scored accordingly.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass

from app.domain.evidence import ExtractedField, ExtractionMethod
from app.extraction.context import LineView, ReceiptContext
from app.extraction.lexicon import LabelCategory
from app.normalization.numbers import (
    extract_email,
    extract_phone,
    extract_registration_id,
    extract_tax_id,
    extract_website,
)
from app.schemas.receipt import ReceiptSection

#: A line that looks like a postal address.
#:
#: The street-number alternative carries no trailing word boundary on purpose:
#: it ends mid-word (the "P" of "5959 Poplar"), where a boundary can never
#: hold. An earlier revision had one, matched nothing, and silently dropped
#: every street line from the merchant address.
_ADDRESS_HINT = re.compile(
    r"\b\d{1,5}\s+[A-Za-z]"
    r"|\b(?:STREET|ST\.|ROAD|RD\.|AVENUE|AVE|LANE|LN\.|BLVD|BOULEVARD"
    r"|DRIVE|DR\.|SUITE|STE\.|FLOOR|UNIT|BLOCK|SECTOR|PLOT|HOUSE"
    r"|HIGHWAY|HWY|P\.?O\.?\s*BOX|ZIP|POSTCODE)\b",
    re.IGNORECASE,
)

#: A postal-code-like token, which strengthens an address judgement.
_POSTCODE = re.compile(r"\b(\d{4,6}|[A-Z]{1,2}\d[A-Z\d]?\s*\d[A-Z]{2})\b")

#: Greetings and boilerplate that are never a merchant name.
_BOILERPLATE = re.compile(
    r"\b(WELCOME|THANK|THANKS|HELLO|RECEIPT|INVOICE|TAX INVOICE|CUSTOMER COPY"
    r"|MERCHANT COPY|ORIGINAL|DUPLICATE|CASH MEMO|BILL)\b",
    re.IGNORECASE,
)

#: A name must contain at least this many letters to be plausible.
_MIN_NAME_LETTERS = 3

#: Only the first few lines are considered; a merchant name never appears
#: halfway down a receipt.
_HEADER_SEARCH_DEPTH = 10

#: How much taller than its neighbours a line must be to be read as the store
#: name on font size alone.
_NAME_HEIGHT_RATIO = 1.15


_GREETING_PREFIX = re.compile(
    r"^(?:WELCOME\s+(?:TO|AT)?|THANKS?\s+(?:FOR\s+(?:SHOPPING|VISITING))?\s*(?:AT|WITH)?|VISIT\s+US\s+AT)\s+",
    re.IGNORECASE,
)

_STORE_ID_REGEX = re.compile(
    r"\b(?:STORE|STR|BRANCH|LOC|LOCATION|SITE)\s*[:#.\s]*([A-Za-z0-9\-_]{1,15})\b",
    re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class MerchantResult:
    """Extracted merchant details."""

    name: ExtractedField[str]
    address: ExtractedField[str]
    phone: ExtractedField[str]
    email: ExtractedField[str]
    website: ExtractedField[str]
    tax_id: ExtractedField[str]
    registration_id: ExtractedField[str]
    store_id: ExtractedField[str]


def extract_merchant(context: ReceiptContext) -> MerchantResult:
    """Extract merchant identity and contact details."""
    return MerchantResult(
        name=_extract_name(context),
        address=_extract_address(context),
        phone=_scan(context, extract_phone, ExtractionMethod.REGEX),
        email=_scan(context, extract_email, ExtractionMethod.REGEX),
        website=_scan(context, extract_website, ExtractionMethod.REGEX),
        tax_id=_scan(context, extract_tax_id, ExtractionMethod.REGEX),
        registration_id=_scan(context, extract_registration_id, ExtractionMethod.REGEX),
        store_id=_extract_store_id(context),
    )


def _extract_store_id(context: ReceiptContext) -> ExtractedField[str]:
    """Extract store or branch identifier from header or metadata."""
    for line in _header_lines(context) + context.in_section(ReceiptSection.METADATA):
        match = _STORE_ID_REGEX.search(line.normalized)
        if match:
            val = match.group(1).strip()
            if val and val.upper() not in {"NO", "NUMBER", "ID", "TAX", "TEL", "FAX"}:
                return ExtractedField(
                    value=val,
                    evidence=(line.evidence(ExtractionMethod.REGEX, notes="store_id"),),
                    raw_value=match.group(0),
                )
    return ExtractedField.absent()


def _header_lines(context: ReceiptContext) -> list[LineView]:
    """Candidate header lines, in document order."""
    header = context.in_section(ReceiptSection.MERCHANT)
    if header:
        return header
    return list(context.lines[:_HEADER_SEARCH_DEPTH])


def _is_name_candidate(line: LineView) -> bool:
    """Whether a header line could be the merchant name."""
    text = line.normalized.strip()
    if not text or line.is_divider:
        return False
    if len(re.findall(r"[A-Za-z€-]", text)) < _MIN_NAME_LETTERS:
        return False
    if line.has_monetary_amount:
        return False
    cleaned_text = _GREETING_PREFIX.sub("", text).strip()
    if _BOILERPLATE.search(cleaned_text):
        return False

    if line.has(LabelCategory.MERCHANT_CONTACT, LabelCategory.DATE, LabelCategory.RECEIPT_ID):
        return False
    if _looks_like_address(text):
        return False
    # A line that is mostly digits is an identifier, not a name.
    digits = sum(1 for c in text if c.isdigit())
    return digits <= len(text) / 3


def _looks_like_address(text: str) -> bool:
    """Whether a line reads as a postal address."""
    if _ADDRESS_HINT.search(text):
        return True
    # A comma-separated line ending in a postcode is an address even without
    # an explicit street keyword.
    return "," in text and bool(_POSTCODE.search(text))


def _extract_name(context: ReceiptContext) -> ExtractedField[str]:
    """Identify the merchant name from the header block."""
    candidates = [line for line in _header_lines(context) if _is_name_candidate(line)]
    if not candidates:
        return ExtractedField.absent("MISSING_MERCHANT_NAME")

    # With geometry, the tallest line is the store name: POS software prints
    # it in a larger font than everything around it.
    with_geometry = [line for line in candidates if line.bbox is not None]
    if len(with_geometry) >= 2:
        tallest = max(with_geometry, key=lambda line: line.bbox.height)  # type: ignore[union-attr]
        # Compare against the other candidates, excluding the tallest itself.
        # Including it would make the comparison unwinnable whenever there are
        # only two candidates, which is the common case for a short header.
        others = sorted(
            line.bbox.height  # type: ignore[union-attr]
            for line in with_geometry
            if line is not tallest
        )
        reference = others[len(others) // 2]
        if tallest.bbox is not None and tallest.bbox.height > reference * _NAME_HEIGHT_RATIO:
            return ExtractedField(
                value=_clean_name(tallest.normalized),
                evidence=(
                    tallest.evidence(ExtractionMethod.SPATIAL, notes="largest_text_in_header"),
                ),
                raw_value=tallest.raw,
            )

    first = candidates[0]
    return ExtractedField(
        value=_clean_name(first.normalized),
        evidence=(first.evidence(ExtractionMethod.HEURISTIC, notes="first_header_line"),),
        raw_value=first.raw,
    )


def _clean_name(text: str) -> str:
    """Trim decoration and greeting prefixes from a merchant name."""
    text = _GREETING_PREFIX.sub("", text).strip()
    cleaned = re.sub(r"^[\s*#=\-]+|[\s*#=\-]+$", "", text).strip()
    return re.sub(r"\s{2,}", " ", cleaned)


def _extract_address(context: ReceiptContext) -> ExtractedField[str]:
    """Join the consecutive header lines that form the postal address."""
    header = _header_lines(context)
    address_lines: list[LineView] = []

    for line in header:
        text = line.normalized.strip()
        if not text or line.is_divider:
            continue
        if _looks_like_address(text):
            address_lines.append(line)
        elif address_lines:
            # Address lines are contiguous; the first non-address line after
            # them ends the block.
            break

    if not address_lines:
        return ExtractedField.absent()

    joined = ", ".join(line.normalized.strip().strip(",") for line in address_lines)
    return ExtractedField(
        value=re.sub(r"\s{2,}", " ", joined),
        evidence=tuple(
            line.evidence(ExtractionMethod.HEURISTIC, notes="address_block")
            for line in address_lines
        ),
        raw_value=" / ".join(line.raw for line in address_lines),
    )


def _scan(
    context: ReceiptContext,
    extractor: Callable[[str], str | None],
    method: ExtractionMethod,
) -> ExtractedField[str]:
    """Apply a line-level extractor across the document, first match wins.

    The header is searched first because merchant contact details belong
    there; the footer often repeats them, and a customer-service number in the
    footer is a worse answer than the store's own number in the header.
    """
    ordered = _header_lines(context) + [
        line for line in context.lines if line.section is not ReceiptSection.MERCHANT
    ]
    for line in ordered:
        value = extractor(line.normalized)
        if value:
            return ExtractedField(
                value=value,
                evidence=(line.evidence(method),),
                raw_value=line.raw,
            )
    return ExtractedField.absent()
