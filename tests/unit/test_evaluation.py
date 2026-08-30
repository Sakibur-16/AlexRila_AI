"""Evaluation harness tests.

A metrics harness that cannot detect a wrong answer is worse than none, since
it manufactures false confidence. These tests verify it actually catches the
errors it exists to catch.
"""

from __future__ import annotations

import pytest

from app.evaluation.metrics import (
    EvaluationSummary,
    compare_field,
    normalize_text,
    score_document,
    score_items,
)


def test_monetary_comparison_is_exact() -> None:
    """One cent out is wrong; partial credit would hide the errors that matter."""
    assert compare_field("total", "25.99", "25.99").correct
    assert not compare_field("total", "25.99", "25.98").correct


def test_monetary_comparison_ignores_representation() -> None:
    """25.9 and 25.90 are the same amount."""
    assert compare_field("total", "25.90", "25.9").correct


def test_text_comparison_folds_case_and_punctuation() -> None:
    assert compare_field("merchant.name", "GREEN VALLEY MARKET", "Green Valley Market").correct
    assert compare_field("merchant.name", "A&B Store", "A B Store").correct
    assert not compare_field("merchant.name", "GREEN VALLEY", "BLUE VALLEY").correct


def test_dates_are_compared_exactly() -> None:
    assert compare_field("transaction.date", "2026-08-25", "2026-08-25").correct
    assert not compare_field("transaction.date", "2026-08-25", "2026-09-08").correct


def test_shared_nulls_are_marked_and_excluded() -> None:
    """A dataset of sparse receipts must not inflate accuracy through nulls."""
    score = compare_field("merchant.phone", None, None)
    assert score.correct
    assert score.both_absent

    summary = EvaluationSummary(
        documents=[score_document("a", {"merchant": {}}, {"merchant": {}}, ["merchant.name"])]
    )
    assert summary.field_accuracy("merchant.name") == 1.0


def test_missing_value_counts_as_wrong() -> None:
    assert not compare_field("total", "25.99", None).correct
    assert not compare_field("total", None, "25.99").correct


# ---------------------------------------------------------------------- items
def test_identical_item_lists_score_perfectly() -> None:
    items = [
        {"description": "Milk", "total_price": "2.50"},
        {"description": "Bread", "total_price": "3.00"},
    ]
    assert score_items(items, list(items)) == (1.0, 1.0, 1.0)


def test_matching_name_with_wrong_price_is_not_a_match() -> None:
    """A right name with a wrong number is a wrong financial record."""
    expected = [{"description": "Milk", "total_price": "2.50"}]
    actual = [{"description": "Milk", "total_price": "9.99"}]
    precision, _recall, f1 = score_items(expected, actual)
    assert precision == 0.0
    assert f1 == 0.0


def test_missing_item_lowers_recall() -> None:
    expected = [
        {"description": "Milk", "total_price": "2.50"},
        {"description": "Bread", "total_price": "3.00"},
    ]
    actual = [{"description": "Milk", "total_price": "2.50"}]
    precision, recall, _ = score_items(expected, actual)
    assert precision == 1.0
    assert recall == 0.5


def test_spurious_item_lowers_precision() -> None:
    expected = [{"description": "Milk", "total_price": "2.50"}]
    actual = [
        {"description": "Milk", "total_price": "2.50"},
        {"description": "Invented", "total_price": "1.00"},
    ]
    precision, recall, _ = score_items(expected, actual)
    assert precision == 0.5
    assert recall == 1.0


def test_both_lists_empty_is_perfect() -> None:
    assert score_items([], []) == (1.0, 1.0, 1.0)


def test_one_list_empty_scores_zero() -> None:
    assert score_items([{"description": "Milk", "total_price": "1.00"}], []) == (0.0, 0.0, 0.0)


# ------------------------------------------------------------------- document
def test_document_exact_match_requires_everything() -> None:
    expected = {"total": "10.00", "items": [{"description": "X", "total_price": "10.00"}]}
    good = score_document("d", expected, dict(expected), ["total"])
    assert good.exact_match

    wrong = score_document("d", expected, {**expected, "total": "11.00"}, ["total"])
    assert not wrong.exact_match


def test_summary_reports_failures_with_detail() -> None:
    expected = {"total": "10.00", "items": []}
    actual = {"total": "11.00", "items": []}
    summary = EvaluationSummary(documents=[score_document("d1", expected, actual, ["total"])])

    payload = summary.to_dict()
    assert payload["exact_match_rate"] == 0.0
    assert payload["failures"][0]["document"] == "d1"
    assert payload["failures"][0]["fields"][0]["field"] == "total"


@pytest.mark.parametrize(
    ("value", "expected"),
    [("  Green   Valley  ", "green valley"), ("A&B!", "a b"), (None, None), ("", None)],
)
def test_text_normalisation(value: str | None, expected: str | None) -> None:
    assert normalize_text(value) == expected
