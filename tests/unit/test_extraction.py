"""Field extraction tests.

Concentrated on the failure modes that cost real money: reading a subtotal as
the total, reading the largest number on the page as the total, and mangling
item rows.
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from app.extraction.items import sum_item_totals
from app.extraction.lexicon import LabelCategory, find_labels
from app.extraction.receipt import RuleBasedReceiptExtractor
from app.normalization.text import normalize_keyword_text
from app.schemas.receipt import PaymentMethod


@pytest.fixture
def extract(settings, ocr_result_factory):
    """Extract from a list of receipt lines."""

    def _extract(lines: list[str], **kwargs):
        extractor = RuleBasedReceiptExtractor(settings)
        return extractor.extract(ocr_result_factory(lines, **kwargs))

    return _extract


# ------------------------------------------------------------------ lexicon
@pytest.mark.parametrize(
    ("line", "category"),
    [
        ("SUBTOTAL 20.00", LabelCategory.SUBTOTAL),
        ("SUB TOTAL 20.00", LabelCategory.SUBTOTAL),
        ("NET TOTAL 20.00", LabelCategory.SUBTOTAL),
        ("TOTAL 25.99", LabelCategory.TOTAL),
        ("GRAND TOTAL 25.99", LabelCategory.TOTAL),
        ("AMOUNT DUE 25.99", LabelCategory.TOTAL),
        ("SALES TAX 2.10", LabelCategory.TAX),
        ("VAT 20% 4.00", LabelCategory.TAX),
        ("CHANGE 4.01", LabelCategory.CHANGE),
        ("DISCOUNT 2.00", LabelCategory.DISCOUNT),
        ("SERVICE CHARGE 1.50", LabelCategory.SERVICE_CHARGE),
    ],
)
def test_label_resolution(line: str, category: LabelCategory) -> None:
    """SUBTOTAL must never resolve to TOTAL even though it contains it."""
    labels = find_labels(normalize_keyword_text(line))
    assert labels
    assert labels[0].category is category


# -------------------------------------------------------------------- totals
def test_subtotal_and_total_are_distinguished(extract) -> None:
    result = extract(["SUBTOTAL 20.00", "TAX 2.00", "TOTAL 22.00"])
    assert result.subtotal.value == Decimal("20.00")
    assert result.total.value == Decimal("22.00")


def test_largest_number_is_not_taken_as_total(extract) -> None:
    """The tendered amount and a loyalty balance both exceed the total."""
    result = extract(
        [
            "LOYALTY POINTS BALANCE 98765",
            "TOTAL 22.00",
            "CASH 100.00",
            "CHANGE 78.00",
        ]
    )
    assert result.total.value == Decimal("22.00")
    assert result.amount_tendered.value == Decimal("100.00")
    assert result.change.value == Decimal("78.00")


def test_emphatic_total_beats_bare_total(extract) -> None:
    result = extract(["TOTAL 20.00", "TAX 2.00", "GRAND TOTAL 22.00"])
    assert result.total.value == Decimal("22.00")


def test_last_total_wins_when_several_appear(extract) -> None:
    """Receipts print running totals; the bottom-most is authoritative."""
    result = extract(["TOTAL 10.00", "TOTAL 22.00"])
    assert result.total.value == Decimal("22.00")


def test_tax_rate_is_not_read_as_the_amount(extract) -> None:
    result = extract(["SUBTOTAL 20.00", "VAT 20% 4.00", "TOTAL 24.00"])
    assert result.tax.value is not None
    assert result.tax.value.total == Decimal("4.00")
    assert result.tax.value.details[0].rate == Decimal("20")


def test_multiple_tax_lines_are_summed(extract) -> None:
    result = extract(["SUBTOTAL 100.00", "VAT 7% 3.50", "VAT 19% 9.50", "TOTAL 113.00"])
    assert result.tax.value is not None
    assert result.tax.value.total == Decimal("13.00")
    assert len(result.tax.value.details) == 2


def test_explicit_tax_total_overrides_the_sum(extract) -> None:
    result = extract(["VAT 7% 3.50", "VAT 19% 9.50", "TOTAL TAX 13.00", "TOTAL 113.00"])
    assert result.tax.value is not None
    assert result.tax.value.total == Decimal("13.00")


def test_discount_is_a_positive_magnitude(extract) -> None:
    """Printed as -2.00, stored as 2.00 so validation needs no sign guesswork."""
    result = extract(["SUBTOTAL 20.00", "DISCOUNT -2.00", "TOTAL 18.00"])
    assert result.discount.value is not None
    assert result.discount.value.amount == Decimal("2.00")


def test_missing_total_stays_none(extract) -> None:
    """No total on the receipt means null, never a substituted value."""
    result = extract(["MILK 2.50", "BREAD 3.00"])
    assert result.total.value is None


# --------------------------------------------------------------------- items
@pytest.mark.parametrize(
    ("line", "description", "quantity", "total"),
    [
        ("Milk                2.50", "Milk", None, "2.50"),
        ("Coffee x2           8.00", "Coffee", "2", "8.00"),
        ("2 Coffee            8.00", "Coffee", "2", "8.00"),
        ("Apples 1.2KG @ 3.00 3.60", "Apples", "1.2", "3.60"),
        ("Bread 1.50          3.00", "Bread", None, "3.00"),
    ],
)
def test_item_layout_variants(
    extract, line: str, description: str, quantity: str | None, total: str
) -> None:
    result = extract(["THE SHOP", line, "TOTAL 99.00"])
    assert len(result.items) == 1
    item = result.items[0]
    assert item.description == description
    assert item.total_price == Decimal(total)
    assert item.quantity == (Decimal(quantity) if quantity else None)


def test_quantity_times_unit_price_derives_the_line_total(extract) -> None:
    result = extract(["THE SHOP", "Coffee 2 x 4.00      8.00", "TOTAL 8.00"])
    item = result.items[0]
    assert item.quantity == Decimal("2")
    assert item.unit_price == Decimal("4.00")
    assert item.total_price == Decimal("8.00")


def test_description_only_line_pairs_with_next_line_price(extract) -> None:
    result = extract(["THE SHOP", "ESPRESSO", "4.00", "TOTAL 4.00"])
    assert len(result.items) == 1
    assert result.items[0].description == "ESPRESSO"
    assert result.items[0].total_price == Decimal("4.00")


def test_total_lines_are_not_treated_as_items(extract) -> None:
    result = extract(["THE SHOP", "Milk 2.50", "SUBTOTAL 2.50", "TAX 0.25", "TOTAL 2.75"])
    assert [item.description for item in result.items] == ["Milk"]


def test_sum_item_totals_returns_none_when_unpriced() -> None:
    """None and zero must stay distinguishable for the validator."""
    assert sum_item_totals(()) is None


# ------------------------------------------------------------------ merchant
def test_merchant_name_from_header(extract) -> None:
    result = extract(["GREEN VALLEY MARKET", "123 Oak Street", "TOTAL 5.00"])
    assert result.merchant_name.value == "GREEN VALLEY MARKET"


def test_address_is_not_mistaken_for_the_name(extract) -> None:
    result = extract(["GREEN VALLEY MARKET", "123 Oak Street, Springfield", "TOTAL 5.00"])
    assert result.merchant_name.value == "GREEN VALLEY MARKET"
    assert result.merchant_address.value is not None
    assert "Oak Street" in result.merchant_address.value


def test_greeting_is_not_the_merchant_name(extract) -> None:
    result = extract(["WELCOME TO", "GREEN VALLEY MARKET", "TOTAL 5.00"])
    assert result.merchant_name.value == "GREEN VALLEY MARKET"


def test_phone_requires_a_label_or_prefix(extract) -> None:
    """A bare digit run is more likely a transaction id than a phone number."""
    labelled = extract(["SHOP", "Tel: 555-0142", "TOTAL 5.00"])
    assert labelled.merchant_phone.value == "5550142"

    unlabelled = extract(["SHOP", "TXN 5550142", "TOTAL 5.00"])
    assert unlabelled.merchant_phone.value is None


def test_larger_header_text_wins_when_geometry_is_available(settings, ocr_result_factory) -> None:
    """POS software prints the store name larger than the address below it."""
    from app.schemas.ocr import OCRBox, OCRLine, OCRResult

    lines = (
        OCRLine(
            text="tiny legal name ltd",
            confidence=0.9,
            bbox=OCRBox(x=10, y=0, width=200, height=12),
        ),
        OCRLine(
            text="BIG STORE",
            confidence=0.9,
            bbox=OCRBox(x=10, y=20, width=200, height=40),
        ),
        OCRLine(
            text="TOTAL 5.00",
            confidence=0.9,
            bbox=OCRBox(x=10, y=80, width=200, height=12),
        ),
    )
    ocr = OCRResult(text="\n".join(x.text for x in lines), lines=lines, provider="fixture")
    result = RuleBasedReceiptExtractor(settings).extract(ocr)
    assert result.merchant_name.value == "BIG STORE"


# ------------------------------------------------------------------- payment
def test_card_tail_only_never_full_number(extract) -> None:
    result = extract(["SHOP", "TOTAL 5.00", "VISA ****4321"])
    assert result.card_last4.value == "4321"
    assert result.card_type.value == "VISA"
    assert result.payment_method.value is PaymentMethod.CARD


def test_masked_tail_is_not_read_as_an_amount(extract) -> None:
    """****4321 must never become a 4321.00 payment split."""
    result = extract(["SHOP", "TOTAL 5.00", "VISA ****4321", "CASH 10.00", "CHANGE 5.00"])
    assert result.payment_splits == ()


def test_genuine_split_payment_is_detected(extract) -> None:
    """Two instruments whose amounts reconcile to the total."""
    result = extract(["SHOP", "TOTAL 64.02", "CASH 34.02", "VISA ****9911 30.00"])
    assert len(result.payment_splits) == 2
    assert sum(s.amount for s in result.payment_splits) == Decimal("64.02")


def test_auth_code_extracted(extract) -> None:
    result = extract(["SHOP", "TOTAL 5.00", "AUTH CODE: 55A2B1"])
    assert result.authorization_code.value == "55A2B1"


# -------------------------------------------------------------- identifiers
def test_receipt_number_extracted(extract) -> None:

    result = extract(["SHOP", "Receipt No: R-2026-00815", "TOTAL 5.00"])
    assert result.receipt_number.value == "R-2026-00815"


def test_store_id_extracted(extract) -> None:
    result = extract(["SUPERMARKET", "Store #1234", "TOTAL 15.00"])
    assert result.merchant_store_id.value == "1234"


def test_merchant_name_cleans_greeting_prefix(extract) -> None:
    result = extract(["WELCOME TO COFFEE SHOP", "123 MAIN ST", "TOTAL 4.50"])
    assert result.merchant_name.value == "COFFEE SHOP"


def test_unlabelled_bottom_total_fallback(extract) -> None:
    result = extract(["BURGER BAR", "CHEESEBURGER 8.50", "FRIES 3.50", "12.00"])
    assert result.total.value == Decimal("12.00")


def test_every_field_carries_evidence(extract) -> None:
    """A value with no evidence cannot be produced by deterministic extraction."""
    result = extract(["GREEN VALLEY MARKET", "Milk 2.50", "TOTAL 2.50"])
    for path, field in result.field_map().items():
        if field.is_present and path != "currency":
            assert field.evidence, f"{path} has a value but no evidence"
