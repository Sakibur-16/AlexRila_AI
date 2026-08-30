"""Validation engine tests.

The behaviour being pinned: discrepancies are *reported*, never repaired.
A pipeline that silently rewrites a printed total to make the arithmetic work
destroys the only signal that something was misread.
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal

import pytest

from app.extraction.receipt import RuleBasedReceiptExtractor
from app.schemas.validation import IssueCode, IssueSeverity
from app.validation.engine import ValidationEngine


@pytest.fixture
def validate(settings, ocr_result_factory):
    """Extract and validate a list of receipt lines."""

    def _validate(lines: list[str], *, today: date | None = None):
        ocr = ocr_result_factory(lines)
        extraction = RuleBasedReceiptExtractor(settings).extract(ocr)
        result = ValidationEngine(settings).validate(extraction, ocr=ocr, today=today)
        return extraction, result

    return _validate


def _codes(result) -> set[str]:
    return set(result.codes())


# ------------------------------------------------------------------ arithmetic
def test_consistent_receipt_raises_no_financial_issue(validate) -> None:
    _, result = validate(
        [
            "THE SHOP",
            "Milk 2.50",
            "Bread 3.00",
            "SUBTOTAL 5.50",
            "TAX 0.50",
            "TOTAL 6.00",
            "Date: 2026-08-25",
        ],
        today=date(2026, 8, 26),
    )
    assert IssueCode.TOTAL_MISMATCH.value not in _codes(result)
    assert IssueCode.ITEM_ARITHMETIC_MISMATCH.value not in _codes(result)
    assert result.is_valid


def test_total_mismatch_is_reported_as_an_error(validate) -> None:
    extraction, result = validate(
        ["THE SHOP", "Hammer 12.00", "SUBTOTAL 12.00", "TAX 1.00", "TOTAL 99.99"]
    )
    assert IssueCode.TOTAL_MISMATCH.value in _codes(result)
    assert not result.is_valid
    # The printed value survives: it is reported, not corrected.
    assert extraction.total.value == Decimal("99.99")


def test_item_sum_mismatch_is_reported(validate) -> None:
    _, result = validate(["THE SHOP", "Milk 2.50", "Bread 3.00", "SUBTOTAL 99.00", "TOTAL 99.00"])
    assert IssueCode.ITEM_ARITHMETIC_MISMATCH.value in _codes(result)


def test_rounding_within_tolerance_is_accepted(validate) -> None:
    """Receipts round each line; an exact match is the exception."""
    _, result = validate(["THE SHOP", "Item 10.00", "SUBTOTAL 10.00", "TAX 0.83", "TOTAL 10.84"])
    assert IssueCode.TOTAL_MISMATCH.value not in _codes(result)


def test_tax_inclusive_pricing_does_not_report_a_mismatch(validate) -> None:
    """A VAT-inclusive subtotal already contains the tax; both readings tried."""
    _, result = validate(
        ["SHOP", "Item 120.00", "SUBTOTAL 120.00", "VAT 20% 20.00", "TOTAL 120.00"]
    )
    assert IssueCode.TOTAL_MISMATCH.value not in _codes(result)


def test_discount_is_subtracted_in_the_identity(validate) -> None:
    _, result = validate(
        [
            "SHOP",
            "Item 20.00",
            "SUBTOTAL 20.00",
            "DISCOUNT 5.00",
            "TAX 1.50",
            "TOTAL 16.50",
        ]
    )
    assert IssueCode.TOTAL_MISMATCH.value not in _codes(result)


def test_service_charge_and_tip_are_added(validate) -> None:
    _, result = validate(
        [
            "SHOP",
            "Meal 48.50",
            "SUBTOTAL 48.50",
            "SERVICE CHARGE 4.85",
            "VAT 20% 10.67",
            "TOTAL 64.02",
        ]
    )
    assert IssueCode.TOTAL_MISMATCH.value not in _codes(result)


def test_tender_minus_change_corroborates_the_total(validate) -> None:
    _, result = validate(["SHOP", "Item 6.00", "TOTAL 6.00", "CASH 20.00", "CHANGE 99.00"])
    assert IssueCode.TOTAL_MISMATCH.value in _codes(result)


def test_negative_total_is_an_error(validate) -> None:
    _, result = validate(["SHOP", "TOTAL -5.00"])
    assert IssueCode.NEGATIVE_TOTAL.value in _codes(result)
    assert not result.is_valid


# ------------------------------------------------------------------- semantic
def test_missing_fields_are_warnings_not_errors(validate) -> None:
    """An incomplete receipt is a valid outcome, reported through warnings."""
    _, result = validate(["Milk 2.50"])
    codes = _codes(result)
    assert IssueCode.MISSING_TOTAL.value in codes
    assert IssueCode.MISSING_DATE.value in codes
    assert IssueCode.MISSING_CURRENCY.value in codes
    assert result.is_valid, "missing data is not an error"


def test_ambiguous_date_surfaces_as_a_warning(validate) -> None:
    _, result = validate(["SHOP", "Date: 08/09/26", "TOTAL 5.00"])
    assert IssueCode.AMBIGUOUS_DATE_FORMAT.value in _codes(result)


def test_future_date_is_flagged(validate) -> None:
    _, result = validate(["SHOP", "Date: 2026-08-25", "TOTAL 5.00"], today=date(2020, 1, 1))
    assert IssueCode.FUTURE_DATE.value in _codes(result)


def test_ambiguous_currency_is_flagged(validate) -> None:
    _, result = validate(["SHOP", "TOTAL $5.00"])
    assert IssueCode.AMBIGUOUS_CURRENCY.value in _codes(result)


def test_issue_fields_use_schema_paths(validate) -> None:
    """Issue paths must map onto the receipt schema a consumer receives."""
    _, result = validate(["SHOP", "Item 10.00", "SUBTOTAL 10.00", "TOTAL 99.00"])
    paths = {issue.field for issue in result.all_issues if issue.field}
    assert "total" in paths or "subtotal" in paths


def test_duplicate_findings_are_collapsed(validate) -> None:
    """Two validators reaching the same conclusion report it once."""
    _, result = validate(["SHOP", "Item 6.00", "TOTAL 6.00", "CASH 20.00", "CHANGE 99.00"])
    mismatches = [issue for issue in result.all_issues if issue.code is IssueCode.TOTAL_MISMATCH]
    seen = {(issue.code, issue.field) for issue in mismatches}
    assert len(seen) == len(mismatches)


def test_errors_make_result_invalid_but_warnings_do_not(validate) -> None:
    _, warned = validate(["SHOP", "TOTAL $5.00"])
    assert warned.is_valid
    assert warned.warnings

    _, errored = validate(["SHOP", "Item 1.00", "SUBTOTAL 1.00", "TOTAL 500.00"])
    assert not errored.is_valid
    assert any(i.severity is IssueSeverity.ERROR for i in errored.all_issues)
