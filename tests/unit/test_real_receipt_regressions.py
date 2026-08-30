"""Regressions found on a real photographed receipt.

Every test here corresponds to a bug that survived the original suite and was
caught only by running an actual Target receipt through real Tesseract OCR.
Synthetic fixtures had not exercised these shapes.
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from app.extraction.items import _ANNOTATION
from app.extraction.merchant import _looks_like_address
from app.extraction.receipt import RuleBasedReceiptExtractor


@pytest.fixture
def extract(settings, ocr_result_factory):
    def _extract(lines: list[str]):
        return RuleBasedReceiptExtractor(settings).extract(ocr_result_factory(lines))

    return _extract


# --------------------------------------------------------- the critical one
def test_savings_footer_is_not_the_total(extract) -> None:
    """ "TOTAL SAVINGS THIS TRIP" must never win over the real total.

    The line carries both a TOTAL and a DISCOUNT label. Because the
    authoritative total is chosen as the bottom-most TOTAL line, this footer
    beat the real total and reported a $65.77 receipt as $0.88.
    """
    result = extract(
        [
            "TARGET",
            "MILK BF 3.79",
            "SUBTOTAL 61.52",
            "TOTAL 65.77",
            "VISA CHARGE 65.77",
            "TOTAL SAVINGS THIS TRIP",
            "0.88",
        ]
    )
    assert result.total.value == Decimal("65.77")


def test_change_and_tender_lines_are_not_the_total(extract) -> None:
    result = extract(["SHOP", "TOTAL 22.00", "TOTAL PAID BY CARD 22.00", "CHANGE 0.00"])
    assert result.total.value == Decimal("22.00")


# ------------------------------------------------------------ product codes
def test_leading_product_code_becomes_the_sku_not_a_price(extract) -> None:
    """A 9-digit DPCI parses as a valid amount and became a unit price."""
    result = extract(["TARGET", "270030028 MORNIF MEAT BF 3.79", "TOTAL 3.79"])
    item = result.items[0]
    assert item.sku == "270030028"
    assert item.total_price == Decimal("3.79")
    assert item.unit_price is None
    assert item.description == "MORNIF MEAT BF"


def test_quantity_times_product_code_keeps_the_quantity(extract) -> None:
    """ "2 x 284060377" looks like qty x unit price but the second is a code."""
    result = extract(["TARGET", "2 x 284060377 GG OATMILK BF 6.98", "TOTAL 6.98"])
    item = result.items[0]
    assert item.quantity == Decimal("2")
    assert item.sku == "284060377"
    assert item.total_price == Decimal("6.98")
    assert item.unit_price is None or item.unit_price < Decimal("100")


def test_short_numbers_are_still_prices(extract) -> None:
    """The product-code guard must not swallow ordinary two-price rows."""
    result = extract(["SHOP", "Bread 1.50 3.00", "TOTAL 3.00"])
    item = result.items[0]
    assert item.unit_price == Decimal("1.50")
    assert item.total_price == Decimal("3.00")


# -------------------------------------------------------------- annotations
@pytest.mark.parametrize(
    "line", ["Regular Price $1.69", "2 @ $1.25 ea", "WAS $5.00", "You Save $2.00"]
)
def test_promotional_annotations_are_not_items(line: str) -> None:
    assert _ANNOTATION.match(line)


@pytest.mark.parametrize("line", ["SAVOY CABBAGE $1.20", "WASABI PASTE $3.10", "SAVERS BF $2.00"])
def test_products_beginning_with_annotation_words_survive(line: str) -> None:
    """The word boundary that keeps SAVE from eating SAVOY was missing."""
    assert not _ANNOTATION.match(line)


def test_annotation_line_is_excluded_from_items(extract) -> None:
    result = extract(["SHOP", "SILK TF 2.50", "2 @ 1.25 ea", "Regular Price 1.69", "TOTAL 2.50"])
    assert [item.description for item in result.items] == ["SILK TF"]


# ------------------------------------------------------------------ address
@pytest.mark.parametrize(
    "line",
    ["5959 Poplar Ave", "123 Oak Street, Springfield", "88 Harbour Road", "10115 Berlin"],
)
def test_street_lines_are_recognised_as_addresses(line: str) -> None:
    """A trailing word boundary made the street-number branch unmatchable."""
    assert _looks_like_address(line)


@pytest.mark.parametrize("line", ["TARGET.", "MORNIF MEAT BF", "THANK YOU"])
def test_non_addresses_are_not_matched(line: str) -> None:
    assert not _looks_like_address(line)


def test_full_address_block_is_joined(extract) -> None:
    result = extract(
        ["TARGET", "Memphis East", "5959 Poplar Ave", "Memphis, Tennessee 38119", "TOTAL 5.00"]
    )
    address = result.merchant_address.value
    assert address is not None
    assert "5959 Poplar Ave" in address
    assert "Memphis, Tennessee 38119" in address


# ------------------------------------------------------------------ payment
def test_single_asterisk_card_mask(extract) -> None:
    """ "*5025 VISA CHARGE" -- one asterisk, not four."""
    result = extract(["SHOP", "TOTAL 65.77", "*5025 VISA CHARGE 65.77"])
    assert result.card_last4.value == "5025"


# --------------------------------------------------------------- validation
def test_informational_savings_does_not_break_the_total_identity(
    settings, ocr_result_factory
) -> None:
    """A savings summary is already reflected in item prices, not deducted.

    Subtracting it manufactured a TOTAL_MISMATCH on an arithmetically perfect
    receipt: 61.52 + 4.25 = 65.77 exactly.
    """
    from app.validation.engine import ValidationEngine

    ocr = ocr_result_factory(
        [
            "TARGET",
            "SUBTOTAL 61.52",
            "TAX 4.25",
            "TOTAL 65.77",
            "TOTAL SAVINGS THIS TRIP",
            "0.88",
        ]
    )
    extraction = RuleBasedReceiptExtractor(settings).extract(ocr)
    result = ValidationEngine(settings).validate(extraction, ocr=ocr)

    assert "TOTAL_MISMATCH" not in result.codes()
    # The discount is still reported -- it is real information.
    assert extraction.discount.value is not None


def test_genuine_deducted_discount_still_validates(settings, ocr_result_factory) -> None:
    """Trying both readings must not blind the check to real inconsistency."""
    from app.validation.engine import ValidationEngine

    ocr = ocr_result_factory(
        ["SHOP", "Item 20.00", "SUBTOTAL 20.00", "DISCOUNT 5.00", "TOTAL 15.00"]
    )
    extraction = RuleBasedReceiptExtractor(settings).extract(ocr)
    assert "TOTAL_MISMATCH" not in ValidationEngine(settings).validate(extraction, ocr=ocr).codes()


def test_real_mismatch_is_still_caught(settings, ocr_result_factory) -> None:
    from app.validation.engine import ValidationEngine

    ocr = ocr_result_factory(["SHOP", "SUBTOTAL 20.00", "TAX 2.00", "TOTAL 999.00"])
    extraction = RuleBasedReceiptExtractor(settings).extract(ocr)
    assert "TOTAL_MISMATCH" in ValidationEngine(settings).validate(extraction, ocr=ocr).codes()


# ------------------------------------------------------------ source hygiene
def test_no_control_characters_in_source() -> None:
    """A shell-mangled edit once wrote a literal backspace into a regex.

    ``\\b`` became 0x08, so the pattern silently matched nothing. Nothing in
    the package should contain a raw control character.
    """
    from pathlib import Path

    root = Path(__file__).resolve().parents[2] / "app"
    offenders = [
        str(path.relative_to(root))
        for path in root.rglob("*.py")
        if any(ch in path.read_text(encoding="utf-8") for ch in "\x07\x08\x0b\x0c")
    ]
    assert not offenders, f"control characters found in: {offenders}"


# ------------------------------------------------- unlabelled-total fallback
def test_cash_tendered_is_never_the_total(extract) -> None:
    """The fallback must not resurrect "largest number wins".

    With no TOTAL label, the biggest figure in the totals block is the cash
    presented. Reporting 20.00 for a 5.50 receipt is worse than reporting
    nothing.
    """
    result = extract(
        ["CORNER SHOP", "Tea 2.00", "Scone 3.50", "SUBTOTAL 5.50", "CASH 20.00", "CHANGE 14.50"]
    )
    assert result.total.value != Decimal("20.00")


def test_loyalty_balance_is_never_the_total(extract) -> None:
    """ "BALANCE" is a total label; "POINTS BALANCE" is a loyalty statement."""
    result = extract(["CORNER SHOP", "Tea 2.00", "SUBTOTAL 2.00", "POINTS BALANCE 9500"])
    assert result.total.value != Decimal("9500")


@pytest.mark.parametrize(
    "line", ["POINTS BALANCE 9500", "REWARDS BALANCE 120", "GIFT CARD BALANCE 25.00"]
)
def test_non_monetary_balances_are_recognised(line: str) -> None:
    from app.extraction.totals import _NON_MONETARY_BALANCE

    assert _NON_MONETARY_BALANCE.search(line)


@pytest.mark.parametrize("line", ["BALANCE DUE 12.00", "TOTAL 65.77", "POINTSETTIA PLANT 4.00"])
def test_real_balances_and_lookalikes_are_not_excluded(line: str) -> None:
    from app.extraction.totals import _NON_MONETARY_BALANCE

    assert not _NON_MONETARY_BALANCE.search(line)


def test_balance_due_is_still_a_valid_total(extract) -> None:
    """The exclusion must not break the legitimate BALANCE DUE label."""
    result = extract(["SHOP", "Item 12.00", "BALANCE DUE 12.00"])
    assert result.total.value == Decimal("12.00")


def test_labelled_total_still_wins_over_the_fallback(extract) -> None:
    result = extract(["SHOP", "Item 5.50", "TOTAL 5.50", "CASH 20.00", "CHANGE 14.50"])
    assert result.total.value == Decimal("5.50")
