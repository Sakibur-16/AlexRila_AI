"""Monetary parsing tests.

The property that matters most: a document's decimal convention is inferred
once from all of its amounts, so ``1.234,50`` is never read as ``1.23``.
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from app.normalization.money import (
    SeparatorStyle,
    detect_separator_style,
    parse_all_amounts,
    parse_amount,
    parse_percentage,
    parse_quantity,
)
from app.schemas.common import quantize_money

_US_DOCUMENT = ["SUBTOTAL 1,234.50", "TAX 98.76", "TOTAL 1,333.26"]
_EU_DOCUMENT = ["ZWISCHENSUMME 1.234,50", "MWST 98,76", "GESAMT 1.333,26"]


def test_detects_dot_decimal_document() -> None:
    assert detect_separator_style(_US_DOCUMENT) is SeparatorStyle.DOT_DECIMAL


def test_detects_comma_decimal_document() -> None:
    assert detect_separator_style(_EU_DOCUMENT) is SeparatorStyle.COMMA_DECIMAL


def test_explicit_configuration_overrides_inference() -> None:
    """An operator who declares a locale is not second-guessed."""
    assert detect_separator_style(_US_DOCUMENT, configured="comma") is (
        SeparatorStyle.COMMA_DECIMAL
    )


@pytest.mark.parametrize(
    ("text", "style", "expected"),
    [
        ("TOTAL 25.99", SeparatorStyle.DOT_DECIMAL, "25.99"),
        ("SUBTOTAL 1,234.50", SeparatorStyle.DOT_DECIMAL, "1234.50"),
        ("GESAMT 1.234,50", SeparatorStyle.COMMA_DECIMAL, "1234.50"),
        ("MWST 98,76", SeparatorStyle.COMMA_DECIMAL, "98.76"),
        ("TOTAL 1 234,50", SeparatorStyle.COMMA_DECIMAL, "1234.50"),
        ("TOTAL 1'234.50", SeparatorStyle.DOT_DECIMAL, "1234.50"),
        ("TOTAL 5", SeparatorStyle.UNKNOWN, "5"),
    ],
)
def test_parses_locale_variants(text: str, style: SeparatorStyle, expected: str) -> None:
    parsed = parse_amount(text, style=style)
    assert parsed is not None
    assert parsed.value == Decimal(expected)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("DISCOUNT (2.50)", "-2.50"),
        ("REFUND -3.00", "-3.00"),
        ("ADJ 5.00-", "-5.00"),
    ],
)
def test_parses_negative_forms(text: str, expected: str) -> None:
    """Accounting parentheses and trailing minus both mean negative."""
    parsed = parse_amount(text)
    assert parsed is not None
    assert parsed.value == Decimal(expected)


def test_returns_none_when_no_amount_present() -> None:
    """Absence is None, never zero -- they mean different things."""
    assert parse_amount("THANK YOU FOR SHOPPING") is None
    assert parse_amount("") is None


def test_parses_every_amount_in_reading_order() -> None:
    amounts = parse_all_amounts("2 x 4.00   8.00")
    assert [a.value for a in amounts] == [Decimal("2"), Decimal("4.00"), Decimal("8.00")]


def test_repairs_ocr_confusion_while_parsing() -> None:
    parsed = parse_amount("TOTAL 1O.99")
    assert parsed is not None
    assert parsed.value == Decimal("10.99")


def test_ambiguous_three_digit_group_is_treated_as_grouping() -> None:
    """1,234 with no other evidence reads as 1234.

    Chosen deliberately: reading it as 1.234 would under-report by 1000x and
    the arithmetic check would not catch it, whereas the reverse error fails
    validation loudly.
    """
    parsed = parse_amount("TOTAL 1,234", style=SeparatorStyle.UNKNOWN)
    assert parsed is not None
    assert parsed.value == Decimal("1234")


@pytest.mark.parametrize(
    ("text", "expected"),
    [("VAT 20%", "20"), ("Tax @ 8.25 %", "8.25"), ("TOTAL 25.99", None)],
)
def test_parses_percentage(text: str, expected: str | None) -> None:
    result = parse_percentage(text)
    assert result == (Decimal(expected) if expected else None)


def test_parses_fractional_quantity() -> None:
    assert parse_quantity("1.5 kg") == Decimal("1.5")
    assert parse_quantity("no digits") is None


def test_money_quantization_is_half_up() -> None:
    """Retail rounding is HALF_UP; matching it avoids false mismatches."""
    assert quantize_money(Decimal("2.345")) == Decimal("2.35")
    assert quantize_money(Decimal("2.344")) == Decimal("2.34")


def test_decimal_arithmetic_has_no_float_error() -> None:
    """The reason money is Decimal throughout."""
    total = Decimal("0.1") + Decimal("0.2")
    assert total == Decimal("0.3")
    assert str(total) == "0.3"
