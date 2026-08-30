"""Date, time and currency normalisation tests.

Both modules share a governing rule: **never convert uncertainty into false
certainty**. These tests pin that rule down, because it is the one most easily
lost to a well-meaning "improvement".
"""

from __future__ import annotations

from datetime import date, time

import pytest

from app.normalization.currency import detect_currency
from app.normalization.dates import combine, parse_date, parse_time


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("25/08/2026", date(2026, 8, 25)),  # day > 12 resolves the order
        ("08/25/2026", date(2026, 8, 25)),  # month-first, likewise resolved
        ("2026-08-25", date(2026, 8, 25)),  # ISO
        ("25-Aug-2026", date(2026, 8, 25)),  # named month
        ("Aug 25, 2026", date(2026, 8, 25)),
        ("25.08.26", date(2026, 8, 25)),  # two-digit year
        ("DATE: 31/12/2025", date(2025, 12, 31)),
    ],
)
def test_unambiguous_dates_are_parsed(text: str, expected: date) -> None:
    parsed = parse_date(text)
    assert parsed is not None
    assert parsed.value == expected
    assert not parsed.ambiguous


@pytest.mark.parametrize("text", ["08/09/26", "03/04/2026", "01/02/26"])
def test_ambiguous_dates_are_not_guessed(text: str) -> None:
    """Both components are 1-12, so the order is unknowable without a locale."""
    parsed = parse_date(text)
    assert parsed is not None
    assert parsed.value is None
    assert parsed.ambiguous
    assert "AMBIGUOUS_DATE_FORMAT" in parsed.warnings
    assert parsed.raw, "the printed text must always be preserved"


def test_ocr_invalid_date_repair() -> None:
    """36/16/2023 and 63/18/2026 are repaired via OCR digit confusion to valid dates."""
    parsed1 = parse_date("36/16/2023")
    assert parsed1 is not None
    assert parsed1.value is not None
    assert parsed1.value.year == 2023

    parsed2 = parse_date("63/18/2026")
    assert parsed2 is not None
    assert parsed2.value == date(2026, 3, 18)


@pytest.mark.parametrize(
    ("order", "expected"),
    [("DMY", date(2026, 9, 8)), ("MDY", date(2026, 8, 9))],
)
def test_configured_locale_resolves_ambiguity(order: str, expected: date) -> None:
    parsed = parse_date("08/09/26", date_order=order)  # type: ignore[arg-type]
    assert parsed is not None
    assert parsed.value == expected


def test_invalid_calendar_date_is_reported_not_raised() -> None:
    parsed = parse_date("13/13/26")
    assert parsed is not None
    assert parsed.value is None
    assert "INVALID_DATE" in parsed.warnings


def test_missing_date_returns_none() -> None:
    """None means 'no date here', distinct from a ParsedDate with value None."""
    assert parse_date("THANK YOU FOR SHOPPING") is None


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("14:35", time(14, 35)),
        ("2:35 PM", time(14, 35)),
        ("12:00 AM", time(0, 0)),
        ("12:00 PM", time(12, 0)),
        ("09:15:42", time(9, 15, 42)),
    ],
)
def test_times_are_parsed(text: str, expected: time) -> None:
    parsed = parse_time(text)
    assert parsed is not None
    assert parsed.value == expected


def test_impossible_minutes_are_not_a_time_at_all() -> None:
    """ ":99" cannot be minutes, so nothing is reported as found."""
    assert parse_time("25:99") is None


def test_time_shaped_but_invalid_value_is_reported() -> None:
    """ "25:30" has a time's shape but no such hour exists."""
    parsed = parse_time("25:30")
    assert parsed is not None
    assert parsed.value is None
    assert "INVALID_TIME" in parsed.warnings


def test_amounts_are_never_parsed_as_times() -> None:
    """A dot separator would make every price on the receipt a candidate time."""
    assert parse_time("TOTAL 45.00") is None
    assert parse_time("Unleaded 30L 4.00") is None
    # A meridiem removes the ambiguity, so a dot is still accepted there.
    assert parse_time("8.45 PM") is not None


def test_combine_requires_both_parts() -> None:
    """A datetime built from a date plus assumed midnight asserts false precision."""
    assert combine(date(2026, 8, 25), None) is None
    assert combine(None, time(14, 35)) is None
    assert combine(date(2026, 8, 25), time(14, 35)) is not None


# ------------------------------------------------------------------ currency
def test_dollar_alone_is_never_assumed_to_be_usd() -> None:
    """The single most common currency bug, pinned."""
    detection = detect_currency("TOTAL $25.99")
    assert detection.code is None
    assert detection.symbol == "$"
    assert "AMBIGUOUS_CURRENCY" in detection.warnings


def test_iso_code_in_text_wins() -> None:
    detection = detect_currency("TOTAL 25.99 EUR")
    assert detection.code == "EUR"
    assert detection.confidence > 0.9


@pytest.mark.parametrize(
    ("text", "expected"),
    [("TOTAL €25.99", "EUR"), ("TOTAL ৳250", "BDT"), ("TOTAL ₹300", "INR")],
)
def test_unambiguous_symbols_resolve(text: str, expected: str) -> None:
    assert detect_currency(text).code == expected


def test_country_hint_resolves_ambiguous_symbol() -> None:
    detection = detect_currency("TOTAL $25.99", country_hint="CA")
    assert detection.code == "CAD"
    assert detection.reason == "symbol_plus_country_hint"


def test_locale_tax_term_resolves_ambiguous_symbol() -> None:
    detection = detect_currency("TOTAL $25.99 SALES TAX 2.10")
    assert detection.code == "USD"
    assert "tax_term" in detection.reason


def test_configured_default_is_low_confidence() -> None:
    """A configured default is an assumption, not evidence, and is scored so."""
    detection = detect_currency("TOTAL 25.99", default_currency="USD")
    assert detection.code == "USD"
    assert detection.confidence <= 0.4
    assert detection.reason == "configured_default"


def test_no_evidence_yields_no_currency() -> None:
    detection = detect_currency("TOTAL 25.99")
    assert detection.code is None
    assert "MISSING_CURRENCY" in detection.warnings


def test_mixed_currencies_are_flagged() -> None:
    detection = detect_currency("PRICE $10 AND €12")
    assert "MULTIPLE_CURRENCIES_DETECTED" in detection.warnings


def test_receipt_number_is_not_read_as_a_currency() -> None:
    """ "R-2026-00815" must not register as South African rand."""
    detection = detect_currency("Receipt No: R-2026-00815\nTOTAL 17.28")
    assert detection.code is None
