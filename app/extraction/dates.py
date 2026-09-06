"""Transaction date and time extraction.

A receipt often carries several dates: the transaction date, a card expiry, a
"return by" date in the footer, sometimes a printed-on timestamp. Picking the
wrong one silently misfiles the expense, so selection is ordered by evidence
strength: an explicitly labelled date beats an unlabelled one, and a date in
the metadata block beats one in the footer.

Ambiguity is never resolved by guessing. See :mod:`app.normalization.dates`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date as date_type
from datetime import datetime as datetime_type
from datetime import time as time_type

from app.domain.evidence import ExtractedField, ExtractionMethod
from app.extraction.context import LineView, ReceiptContext
from app.extraction.lexicon import LabelCategory
from app.normalization import dates as date_norm
from app.schemas.receipt import ReceiptSection

#: Any month name or three-letter abbreviation. A date containing one is
#: unambiguous, which is what lets it outrank a numeric candidate.
_MONTH_NAME = re.compile(
    r"\b(?:JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)[A-Z]*\b",
    re.IGNORECASE,
)

#: Contexts in which a date is *not* the transaction date.
_DISQUALIFYING_CONTEXT = re.compile(
    r"\b(EXP|EXPIRY|EXPIRES|VALID\s*(?:UNTIL|THRU|TILL)|RETURN\s*BY|BEST\s*BEFORE"
    r"|USE\s*BY|WARRANTY|DUE\s*DATE|BIRTH)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class DateTimeResult:
    """Extracted transaction timing."""

    date: ExtractedField[date_type]
    time: ExtractedField[time_type]
    datetime: ExtractedField[datetime_type]


def extract_datetime(context: ReceiptContext) -> DateTimeResult:
    """Extract the transaction date and time.

    Returns fields whose ``value`` may be ``None`` while ``raw_value`` holds
    the printed text -- that combination means "found but not confidently
    interpretable", which is materially different from "not present".
    """
    date_field = _extract_date(context)
    time_field = _extract_time(context)

    combined: ExtractedField[datetime_type]
    if date_field.value is not None and time_field.value is not None:
        combined = ExtractedField(
            value=date_norm.combine(date_field.value, time_field.value),
            evidence=date_field.evidence + time_field.evidence,
        )
    else:
        combined = ExtractedField.absent()

    return DateTimeResult(date=date_field, time=time_field, datetime=combined)


def _candidate_lines(context: ReceiptContext) -> list[LineView]:
    """Lines worth searching for a transaction date, best first."""
    labelled = context.find_all(LabelCategory.DATE, LabelCategory.TIME)
    metadata = context.in_section(ReceiptSection.METADATA)
    header = context.in_section(ReceiptSection.MERCHANT)

    ordered: list[LineView] = []
    for group in (labelled, metadata, header, context.lines):
        for line in group:
            if line not in ordered and not _DISQUALIFYING_CONTEXT.search(line.normalized):
                ordered.append(line)
    return ordered


def _extract_date(context: ReceiptContext) -> ExtractedField[date_type]:
    """Find the transaction date."""
    date_order = context.settings.date_order

    # A month-name date cannot be mistaken for a price, so it outranks every
    # numeric candidate regardless of where it appears. Receipts often print
    # it in the footer, well below price lines that are date-shaped.
    textual = _extract_textual_date(context)
    if textual is not None:
        return textual

    ambiguous_fallback: ExtractedField[date_type] | None = None

    for line in _candidate_lines(context):
        label = line.label_of(LabelCategory.DATE)
        search_text = line.text_after(label) if label else line.normalized
        parsed = date_norm.parse_date(search_text, date_order=date_order)
        if parsed is None and label is not None:
            # The label may sit on its own with the date beside it elsewhere on
            # the line; retry against the whole line before giving up.
            parsed = date_norm.parse_date(line.normalized, date_order=date_order)
        if parsed is None:
            continue

        method = ExtractionMethod.KEYWORD_ANCHORED if label is not None else ExtractionMethod.REGEX
        evidence = line.evidence(method, notes=f"raw={parsed.raw}")

        if parsed.value is not None:
            return ExtractedField(
                value=parsed.value,
                evidence=(evidence,),
                raw_value=parsed.raw,
                warnings=parsed.warnings,
            )

        # Found a date we refuse to guess at. Keep looking for an unambiguous
        # one elsewhere on the receipt before settling for this.
        if ambiguous_fallback is None:
            ambiguous_fallback = ExtractedField(
                value=None,
                evidence=(evidence,),
                raw_value=parsed.raw,
                warnings=parsed.warnings or ("AMBIGUOUS_DATE_FORMAT",),
            )

    if ambiguous_fallback is not None:
        return ambiguous_fallback
    return ExtractedField.absent("MISSING_DATE")


def _extract_time(context: ReceiptContext) -> ExtractedField[time_type]:
    """Find the transaction time."""
    for line in _candidate_lines(context):
        label = line.label_of(LabelCategory.TIME)
        search_text = line.text_after(label) if label else line.normalized
        parsed = date_norm.parse_time(search_text)
        if parsed is None and label is not None:
            parsed = date_norm.parse_time(line.normalized)
        if parsed is None:
            continue

        method = ExtractionMethod.KEYWORD_ANCHORED if label is not None else ExtractionMethod.REGEX
        return ExtractedField(
            value=parsed.value,
            evidence=(line.evidence(method, notes=f"raw={parsed.raw}"),),
            raw_value=parsed.raw,
            warnings=parsed.warnings,
        )

    return ExtractedField.absent()


def _extract_textual_date(context: ReceiptContext) -> ExtractedField[date_type] | None:
    """Find a date written with a month name, anywhere in the document.

    Returns ``None`` when there is none, leaving the numeric scan to run.
    """
    for line in context.lines:
        if _DISQUALIFYING_CONTEXT.search(line.normalized):
            continue
        if not _MONTH_NAME.search(line.normalized):
            continue
        parsed = date_norm.parse_date(line.normalized, date_order="none")
        if parsed is None or parsed.value is None:
            continue
        label = line.label_of(LabelCategory.DATE)
        method = ExtractionMethod.KEYWORD_ANCHORED if label is not None else ExtractionMethod.REGEX
        return ExtractedField(
            value=parsed.value,
            evidence=(line.evidence(method, notes=f"month_name raw={parsed.raw}"),),
            raw_value=parsed.raw,
            warnings=parsed.warnings,
        )
    return None
