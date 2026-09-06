"""Regressions from a real CVS pharmacy receipt.

Drug-store receipts are the messiest common format: register metadata dense
with long numbers, a loyalty card, per-item savings printed on the item line,
coupon rows, promotional continuations, and the date in the footer below all
of it. The original extractor returned twelve "items" for six, dated the
receipt to 2000 from a price fragment, and reported the loyalty card as the
payment card.

Every test here corresponds to one of those failures.
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from app.extraction.receipt import RuleBasedReceiptExtractor
from app.schemas.receipt import PaymentMethod


@pytest.fixture
def extract(settings, ocr_result_factory):
    def _extract(lines: list[str]):
        return RuleBasedReceiptExtractor(settings).extract(ocr_result_factory(lines))

    return _extract


# ------------------------------------------------------------ item counting
def test_register_metadata_is_not_an_item(extract) -> None:
    """ "REG#03 TRN#7524 CSHR#1416938 STR#6855" became a 6855.00 purchase."""
    result = extract(
        ["CVS/pharmacy", "REG#03 TRN#7524 CSHR#1416938 STR#6855", "Milk 2.50", "TOTAL 2.50"]
    )
    assert [i.description for i in result.items] == ["Milk"]


def test_loyalty_card_line_is_not_an_item(extract) -> None:
    result = extract(["CVS/pharmacy", "ExtraCare Card #: ********7080", "Milk 2.50", "TOTAL 2.50"])
    assert [i.description for i in result.items] == ["Milk"]


def test_unit_price_continuation_is_not_an_item(extract) -> None:
    """ "4.49 EACH OR 3/ 12.00" annotates the item above it."""
    result = extract(["CVS", "1 PNTNE PRO-V DMR CD 4.49", "4.49 EACH OR 3/ 12.00", "TOTAL 4.49"])
    assert [i.description for i in result.items] == ["PNTNE PRO-V DMR CD"]


def test_promotion_line_is_not_an_item(extract) -> None:
    result = extract(["CVS", "1 CPH AD EX 2.99", "BUY 1, GET 1 FOR 50% OFF", "TOTAL 2.99"])
    assert [i.description for i in result.items] == ["CPH AD EX"]


def test_coupon_rows_are_not_items(extract) -> None:
    result = extract(
        ["CVS", "1 SHAMPOO 4.49", "1 MFR COUPON 5.00 -", "1 CVS COUPON 2.00 -", "TOTAL 4.49"]
    )
    assert [i.description for i in result.items] == ["SHAMPOO"]


# --------------------------------------------------------------- item prices
def test_saved_suffix_is_not_the_line_total(extract) -> None:
    """ "4.49 SAVED .50" -- the rightmost number is the saving, not the price."""
    result = extract(["CVS", "1 PNTNE PRO-V DMR CD 12Z 4.49 SAVED .50", "TOTAL 4.49"])
    item = result.items[0]
    assert item.total_price == Decimal("4.49")
    assert item.discount == Decimal("0.50")


def test_percentage_saving_suffix_is_handled(extract) -> None:
    result = extract(["CVS", "1 CPH AD EX STGTH 2.99 SAVED 50% 3.00", "TOTAL 2.99"])
    assert result.items[0].total_price == Decimal("2.99")


def test_size_code_is_not_a_unit_price(extract) -> None:
    """ "12Z" reads as "122"; a receipt price always carries cents."""
    result = extract(["CVS", "1 PNTNE PRO-V DMR CD 122 4.49", "TOTAL 4.49"])
    item = result.items[0]
    assert item.total_price == Decimal("4.49")
    assert item.unit_price is None or item.unit_price < Decimal("100")


def test_genuine_two_price_row_still_works(extract) -> None:
    """The cents rule must not break an ordinary unit-price column."""
    result = extract(["SHOP", "Bread 1.50 3.00", "TOTAL 3.00"])
    item = result.items[0]
    assert item.unit_price == Decimal("1.50")
    assert item.total_price == Decimal("3.00")


# --------------------------------------------------------------------- date
def test_price_fragment_is_not_read_as_a_date(extract) -> None:
    """ "OR 3/ 12.00" parsed as 3/12/00 and dated the receipt to 2000."""
    result = extract(["CVS", "4.49 EACH OR 3/ 12.00", "Milk 2.50", "TOTAL 2.50"])
    assert result.transaction_date.value is None


def test_month_name_date_outranks_a_numeric_candidate(extract) -> None:
    """A month name is unambiguous, so it wins wherever it appears."""
    result = extract(
        [
            "CVS/pharmacy",
            "4.49 EACH OR 3/ 12.00",
            "Milk 2.50",
            "TOTAL 2.50",
            "MAY 28, 2017 9:41 AM",
        ]
    )
    assert result.transaction_date.value is not None
    assert result.transaction_date.value.isoformat() == "2017-05-28"


def test_year_zero_is_rejected(extract) -> None:
    from app.normalization.dates import parse_date

    parsed = parse_date("3/ 12.00")
    assert parsed is None or parsed.value is None


# ------------------------------------------------------------------ payment
def test_loyalty_number_is_not_a_payment_card(extract) -> None:
    """A store loyalty number reported as card_last4 is worse than nothing."""
    result = extract(["CVS/pharmacy", "ExtraCare Card #: ********7080", "TOTAL 3.05", "CASH 5.00"])
    assert result.card_last4.value is None


def test_loyalty_card_does_not_make_the_method_card(extract) -> None:
    result = extract(["CVS/pharmacy", "ExtraCare Card #: ********7080", "TOTAL 3.05", "CASH 5.00"])
    assert result.payment_method.value is PaymentMethod.CASH


def test_a_real_payment_card_is_still_captured(extract) -> None:
    """The loyalty exclusion must not blind the extractor to actual cards."""
    result = extract(["SHOP", "TOTAL 25.99", "VISA ****4321"])
    assert result.card_last4.value == "4321"
    assert result.payment_method.value is PaymentMethod.CARD


# ----------------------------------------------------------------- discount
def test_per_item_savings_do_not_become_a_receipt_discount(extract) -> None:
    """Summing item-line savings gave a 131.51 discount on a 1.98 subtotal."""
    result = extract(
        [
            "CVS",
            "1 PNTNE PRO-V DMR CD 4.49 SAVED .50",
            "1 PNTNE D MST RN SH 3.02 SAVED 1.97",
            "SUBTOTAL 1.98",
            "TOTAL 3.05",
        ]
    )
    discount = result.discount.value
    assert discount is None or discount.amount is None or discount.amount < Decimal("10")


def test_footer_savings_summary_is_not_a_discount(extract) -> None:
    result = extract(["CVS", "Milk 2.50", "SUBTOTAL 2.50", "TOTAL 2.50", "Today You Saved 26.74"])
    assert result.discount.value is None


def test_coupon_row_is_still_a_discount(extract) -> None:
    """Excluding item-level savings must not exclude real coupon rows."""
    result = extract(["CVS", "1 SHAMPOO 5.00", "1 MFR COUPON 2.00 -", "TOTAL 3.00"])
    assert result.discount.value is not None
    assert result.discount.value.amount == Decimal("2.00")
