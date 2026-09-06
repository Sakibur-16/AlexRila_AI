"""Date and time normalisation.

The hard rule: **never resolve an ambiguous date by guessing.**

``08/09/26`` is 8 September in most of the world and 9 August in the United
States. Silently picking one produces data that looks authoritative and is
wrong half the time -- far worse than returning ``null``. So a date is only
converted when it is *self-disambiguating* (a component above 12, a four-digit
year in an unambiguous position, a named month, ISO order) or when the operator
has declared a locale via ``DATE_ORDER``. Otherwise the parsed value is
``None``, the printed text is preserved in ``raw_date``, and
``AMBIGUOUS_DATE_FORMAT`` is raised.
"""

from __future__ import annotations

import re
from contextlib import suppress
from datetime import date, datetime, time
from typing import Final, Literal, NamedTuple

from app.normalization.text import normalize_unicode

DateOrder = Literal["DMY", "MDY", "none"]

#: Month names and common abbreviations. English first; extend per language.
_MONTH_NAMES: Final[dict[str, int]] = {
    "jan": 1,
    "january": 1,
    "feb": 2,
    "february": 2,
    "mar": 3,
    "march": 3,
    "apr": 4,
    "april": 4,
    "may": 5,
    "jun": 6,
    "june": 6,
    "jul": 7,
    "july": 7,
    "aug": 8,
    "august": 8,
    "sep": 9,
    "sept": 9,
    "september": 9,
    "oct": 10,
    "october": 10,
    "nov": 11,
    "november": 11,
    "dec": 12,
    "december": 12,
    # Common non-English abbreviations that do not collide with the above.
    "ene": 1,
    "abr": 4,
    "ago": 8,
    "dic": 12,  # es
    "fev": 2,
    "avr": 4,
    "mai": 5,
    "juin": 6,
    "juil": 7,
    "aout": 8,
    "déc": 12,
    "dez": 12,
    "mrz": 3,
    "okt": 10,
    "mai_de": 5,  # fr/de/pt
}

#: Numeric date: three groups separated by / - . or space.
_NUMERIC_DATE: Final[re.Pattern[str]] = re.compile(
    r"(?<!\d)(\d{1,4})\s*[/\-.]\s*(\d{1,2})\s*[/\-.]\s*(\d{2,4})(?!\d)"
)

#: Textual date in either order: "25 Aug 2026" / "Aug 25, 2026".
_TEXT_DATE_DMY: Final[re.Pattern[str]] = re.compile(
    r"(?<!\d)(\d{1,2})\s*[-/ .]\s*([A-Za-zÀ-ÿ]{3,10})\.?\s*[-/, .]\s*(\d{2,4})(?!\d)"
)
_TEXT_DATE_MDY: Final[re.Pattern[str]] = re.compile(
    r"([A-Za-zÀ-ÿ]{3,10})\.?\s*[-/ .]\s*(\d{1,2})\s*(?:st|nd|rd|th)?\s*[-/, .]\s*(\d{2,4})(?!\d)"
)

#: Time with optional seconds and meridiem.
_TIME: Final[re.Pattern[str]] = re.compile(
    r"(?<!\d)(\d{1,2})\s*[:.]\s*([0-5]\d)(?:\s*[:.]\s*([0-5]\d))?(?:\.\d+)?\s*([AaPp])\.?[Mm]\.?(?!\w)"
    r"|(?<!\d)(\d{1,2})\s*[:h]\s*([0-5]\d)(?:\s*[:.]\s*([0-5]\d))?(?:\s*(?:hrs?|hours?))?(?!\d)",
    re.IGNORECASE,
)

#: Two-digit years below this map to 20xx, at or above to 19xx. Receipts are
#: near-contemporary documents, so a wide 20xx window is correct.
_CENTURY_PIVOT: Final[int] = 70

#: Sanity window for a receipt date. Outside it the parse is reported but
#: flagged, since a 1970 receipt is almost certainly an OCR error.
_MIN_PLAUSIBLE_YEAR: Final[int] = 1990
_MAX_PLAUSIBLE_YEAR: Final[int] = 2100


class ParsedDate(NamedTuple):
    """Result of a date parse attempt.

    ``value`` is ``None`` whenever the text was ambiguous; ``raw`` always holds
    what was printed so a caller with locale knowledge can resolve it.
    """

    value: date | None
    raw: str
    ambiguous: bool = False
    #: Machine-readable notes: AMBIGUOUS_DATE_FORMAT, IMPLAUSIBLE_DATE...
    warnings: tuple[str, ...] = ()


class ParsedTime(NamedTuple):
    """Result of a time parse attempt."""

    value: time | None
    raw: str
    warnings: tuple[str, ...] = ()


def _expand_year(year: int) -> int:
    """Expand a two-digit year using the receipt-appropriate pivot."""
    if year >= 100:
        return year
    return 2000 + year if year < _CENTURY_PIVOT else 1900 + year


def _build(year: int, month: int, day: int, raw: str) -> ParsedDate:
    """Construct a validated date, reporting rather than raising on failure."""
    if year == 0:
        # "3/ 12.00" is the tail of a price, not a date in the year 2000. A
        # receipt from 1900 or 2000 is not a case worth supporting, and
        # accepting one turns every "OR 3/ 12.00" line into a transaction date.
        return ParsedDate(value=None, raw=raw, warnings=("INVALID_DATE",))
    year = _expand_year(year)
    try:
        value = date(year, month, day)
    except ValueError:
        return ParsedDate(value=None, raw=raw, warnings=("INVALID_DATE",))

    warnings: tuple[str, ...] = ()
    if not (_MIN_PLAUSIBLE_YEAR <= year <= _MAX_PLAUSIBLE_YEAR):
        warnings = ("IMPLAUSIBLE_DATE",)
    return ParsedDate(value=value, raw=raw, warnings=warnings)


def _generate_ocr_digit_variants(n: int) -> list[int]:
    """Generate plausible integer variants for an OCR-misread digit sequence, preserving priority order."""
    s = str(n)
    if len(s) > 2:
        return [n]

    replacements: dict[str, list[str]] = {
        "6": ["0", "3", "5", "8", "2"],
        "8": ["0", "3", "6", "9"],
        "3": ["3", "0", "2", "8", "5"],
        "5": ["0", "6", "3", "8"],
        "9": ["0", "4", "8"],
        "0": ["0", "6", "8"],
        "1": ["1", "0", "7", "4"],
        "2": ["2", "0", "3"],
        "4": ["4", "1", "9"],
        "7": ["1", "7", "2"],
    }

    variants: list[int] = []

    def _add(val: int) -> None:
        if 1 <= val <= 31 and val not in variants:
            variants.append(val)

    _add(n)
    if len(s) == 1:
        for r in replacements.get(s, []):
            _add(int(r))
    elif len(s) == 2:
        d1, d2 = s[0], s[1]
        for r1 in replacements.get(d1, []):
            with suppress(ValueError):
                _add(int(r1 + d2))
        for r2 in replacements.get(d2, []):
            with suppress(ValueError):
                _add(int(d1 + r2))

    return variants


def _repair_ocr_date(a: int, b: int, c: int, raw: str) -> ParsedDate | None:
    """Attempt OCR digit-confusion repairs when a date component is > 31 (e.g. 63/18/2026 or 36/16/2023)."""
    if a <= 31 and b <= 31:
        return None

    # Preserve original valid numbers first before generated variants
    a_candidates = ([a] if 1 <= a <= 31 else []) + [
        v for v in _generate_ocr_digit_variants(a) if v != a
    ]
    b_candidates = ([b] if 1 <= b <= 31 else []) + [
        v for v in _generate_ocr_digit_variants(b) if v != b
    ]

    if a > 31 and b <= 31:
        for cand_a in a_candidates:
            if 1 <= cand_a <= 12:
                res = _build(c, cand_a, b, raw)
                if res.value is not None:
                    repaired_raw = f"{cand_a:02d}/{b:02d}/{c}"
                    return ParsedDate(
                        value=res.value, raw=repaired_raw, warnings=("OCR_DIGIT_REPAIRED",)
                    )
            elif 1 <= cand_a <= 31 and b <= 12:
                res = _build(c, b, cand_a, raw)
                if res.value is not None:
                    repaired_raw = f"{b:02d}/{cand_a:02d}/{c}"
                    return ParsedDate(
                        value=res.value, raw=repaired_raw, warnings=("OCR_DIGIT_REPAIRED",)
                    )

    elif b > 31 and a <= 31:
        for cand_b in b_candidates:
            if 1 <= cand_b <= 12:
                res = _build(c, cand_b, a, raw)
                if res.value is not None:
                    repaired_raw = f"{cand_b:02d}/{a:02d}/{c}"
                    return ParsedDate(
                        value=res.value, raw=repaired_raw, warnings=("OCR_DIGIT_REPAIRED",)
                    )
            elif 1 <= cand_b <= 31 and a <= 12:
                res = _build(c, a, cand_b, raw)
                if res.value is not None:
                    repaired_raw = f"{a:02d}/{cand_b:02d}/{c}"
                    return ParsedDate(
                        value=res.value, raw=repaired_raw, warnings=("OCR_DIGIT_REPAIRED",)
                    )

    for cand_a in a_candidates:
        for cand_b in b_candidates:
            if 1 <= cand_a <= 12 and 1 <= cand_b <= 31:
                res = _build(c, cand_a, cand_b, raw)
                if res.value is not None:
                    repaired_raw = f"{cand_a:02d}/{cand_b:02d}/{c}"
                    return ParsedDate(
                        value=res.value, raw=repaired_raw, warnings=("OCR_DIGIT_REPAIRED",)
                    )
            if 1 <= cand_b <= 12 and 1 <= cand_a <= 31:
                res = _build(c, cand_b, cand_a, raw)
                if res.value is not None:
                    repaired_raw = f"{cand_b:02d}/{cand_a:02d}/{c}"
                    return ParsedDate(
                        value=res.value, raw=repaired_raw, warnings=("OCR_DIGIT_REPAIRED",)
                    )

    return None


def parse_date(text: str, *, date_order: DateOrder = "none") -> ParsedDate | None:
    """Parse the first date in ``text``.

    Args:
        text: Source text.
        date_order: Locale hint. ``"none"`` means never guess.

    Returns:
        A :class:`ParsedDate`, or ``None`` when ``text`` contains no date at all.
    """
    if not text:
        return None
    source = normalize_unicode(text)

    textual = _parse_textual(source)
    if textual is not None:
        return textual

    match = _NUMERIC_DATE.search(source)
    if match is None:
        return None

    raw = match.group(0).strip()
    a, b, c = (int(match.group(1)), int(match.group(2)), int(match.group(3)))

    # ISO order: a four-digit leading group can only be a year.
    if len(match.group(1)) == 4:
        res = _build(a, b, c, raw)
        if res.value is not None:
            return res
        repaired = _repair_ocr_date(b, c, a, raw)
        if repaired is not None:
            return repaired
        return res

    # A component above 12 can only be a day, which resolves the order.
    if a > 12 and b <= 12:
        res = _build(c, b, a, raw)
        if res.value is not None:
            return res
        repaired = _repair_ocr_date(a, b, c, raw)
        if repaired is not None:
            return repaired
        return res

    if b > 12 and a <= 12:
        res = _build(c, a, b, raw)
        if res.value is not None:
            return res
        repaired = _repair_ocr_date(a, b, c, raw)
        if repaired is not None:
            return repaired
        return res

    if a > 12 and b > 12:
        repaired = _repair_ocr_date(a, b, c, raw)
        if repaired is not None:
            return repaired
        return ParsedDate(value=None, raw=raw, warnings=("INVALID_DATE",))

    # Both components are 1-12: genuinely ambiguous. Only a declared locale
    # resolves it; otherwise the value stays null by design.
    if date_order == "DMY":
        return _build(c, b, a, raw)
    if date_order == "MDY":
        return _build(c, a, b, raw)

    return ParsedDate(value=None, raw=raw, ambiguous=True, warnings=("AMBIGUOUS_DATE_FORMAT",))


def _parse_textual(source: str) -> ParsedDate | None:
    """Parse a date containing a month name. Never ambiguous."""
    match = _TEXT_DATE_DMY.search(source)
    if match is not None:
        month = _MONTH_NAMES.get(match.group(2).lower().strip("."))
        if month is not None:
            return _build(int(match.group(3)), month, int(match.group(1)), match.group(0).strip())

    match = _TEXT_DATE_MDY.search(source)
    if match is not None:
        month = _MONTH_NAMES.get(match.group(1).lower().strip("."))
        if month is not None:
            return _build(int(match.group(3)), month, int(match.group(2)), match.group(0).strip())

    return None


def parse_time(text: str) -> ParsedTime | None:
    """Parse the first time-of-day in ``text``.

    Handles 12-hour forms with a meridiem and 24-hour forms. Returns ``None``
    when no time is present.
    """
    if not text:
        return None
    match = _TIME.search(normalize_unicode(text))
    if match is None:
        return None

    raw = match.group(0).strip()

    if match.group(1) is not None:  # 12-hour branch
        hour = int(match.group(1))
        minute = int(match.group(2))
        second = int(match.group(3) or 0)
        meridiem = (match.group(4) or "").lower()
        if hour == 12:
            hour = 0 if meridiem == "a" else 12
        elif meridiem == "p":
            hour += 12
    else:  # 24-hour branch
        hour = int(match.group(5))
        minute = int(match.group(6))
        second = int(match.group(7) or 0)

    try:
        return ParsedTime(value=time(hour, minute, second), raw=raw)
    except ValueError:
        return ParsedTime(value=None, raw=raw, warnings=("INVALID_TIME",))


def combine(parsed_date: date | None, parsed_time: time | None) -> datetime | None:
    """Combine date and time into a naive local datetime.

    Returns ``None`` unless both parts resolved: a datetime built from a date
    plus an assumed midnight would assert precision the receipt never carried.
    """
    if parsed_date is None or parsed_time is None:
        return None
    return datetime.combine(parsed_date, parsed_time)
