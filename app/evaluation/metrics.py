"""Extraction accuracy metrics.

Turns "the extraction looks about right" into a number that can be tracked
across changes. Without this, every tuning decision is an argument about
anecdotes.

Field types are scored differently on purpose:

* **Monetary fields** use exact decimal equality. A total that is one cent out
  is wrong, and partial credit would hide exactly the errors that matter.
* **Text fields** use a normalised comparison (case, whitespace and
  punctuation folded), because ``GREEN VALLEY MARKET`` and ``Green Valley
  Market`` are the same merchant and penalising the difference would measure
  formatting rather than understanding.
* **Item lists** are matched greedily by description similarity, then scored
  on price agreement, yielding precision, recall and F1.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from typing import Any

#: Fields compared as money.
MONETARY_FIELDS = frozenset(
    {"subtotal", "total", "tax.total", "discount.amount", "service_charge", "shipping", "tip"}
)

#: Fields compared as normalised text.
TEXT_FIELDS = frozenset(
    {
        "merchant.name",
        "merchant.address",
        "merchant.phone",
        "receipt_number",
        "currency",
        "payment.method",
        "payment.card_last4",
    }
)

#: Fields compared for exact equality (dates are ISO strings or None).
EXACT_FIELDS = frozenset({"transaction.date", "transaction.time"})

_PUNCTUATION = re.compile(r"[^\w\s]")

#: Minimum token overlap for two item descriptions to be considered the same.
_ITEM_MATCH_THRESHOLD = 0.5


@dataclass(frozen=True, slots=True)
class FieldScore:
    """Outcome for one field on one document."""

    field: str
    correct: bool
    expected: Any
    actual: Any
    #: True when both sides agree the field is absent. Counted separately so a
    #: dataset of sparse receipts cannot inflate accuracy through shared nulls.
    both_absent: bool = False


@dataclass
class DocumentScore:
    """All field outcomes for one document."""

    name: str
    fields: list[FieldScore] = field(default_factory=list)
    item_precision: float = 0.0
    item_recall: float = 0.0
    item_f1: float = 0.0
    expected_items: int = 0
    actual_items: int = 0

    @property
    def exact_match(self) -> bool:
        """True when every scored field matched and items scored perfectly."""
        return all(score.correct for score in self.fields) and self.item_f1 >= 1.0


@dataclass
class EvaluationSummary:
    """Aggregate accuracy across a dataset."""

    documents: list[DocumentScore] = field(default_factory=list)

    def field_accuracy(self, name: str) -> float:
        """Accuracy for one field, ignoring documents where both sides are null."""
        scored = [
            score
            for document in self.documents
            for score in document.fields
            if score.field == name and not score.both_absent
        ]
        if not scored:
            return 1.0
        return sum(1 for score in scored if score.correct) / len(scored)

    def all_field_accuracies(self) -> dict[str, float]:
        names = {score.field for document in self.documents for score in document.fields}
        return {name: self.field_accuracy(name) for name in sorted(names)}

    @property
    def exact_match_rate(self) -> float:
        if not self.documents:
            return 0.0
        return sum(1 for d in self.documents if d.exact_match) / len(self.documents)

    @property
    def item_f1(self) -> float:
        if not self.documents:
            return 0.0
        return sum(d.item_f1 for d in self.documents) / len(self.documents)

    def to_dict(self) -> dict[str, Any]:
        return {
            "documents": len(self.documents),
            "exact_match_rate": round(self.exact_match_rate, 4),
            "item_f1": round(self.item_f1, 4),
            "field_accuracy": {
                name: round(value, 4) for name, value in self.all_field_accuracies().items()
            },
            "failures": [
                {
                    "document": document.name,
                    "fields": [
                        {
                            "field": score.field,
                            "expected": score.expected,
                            "actual": score.actual,
                        }
                        for score in document.fields
                        if not score.correct
                    ],
                }
                for document in self.documents
                if not document.exact_match
            ],
        }


def normalize_text(value: Any) -> str | None:
    """Fold case, punctuation and whitespace for text comparison."""
    if value is None:
        return None
    collapsed = _PUNCTUATION.sub(" ", str(value)).lower()
    return " ".join(collapsed.split()) or None


def _as_decimal(value: Any) -> Decimal | None:
    if value is None:
        return None
    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError):
        return None


def _get(payload: Mapping[str, Any], path: str) -> Any:
    """Read a dotted path from a nested mapping, tolerating absent branches."""
    current: Any = payload
    for part in path.split("."):
        if not isinstance(current, Mapping):
            return None
        current = current.get(part)
        if current is None:
            return None
    return current


def compare_field(path: str, expected: Any, actual: Any) -> FieldScore:
    """Score one field using the comparison appropriate to its type."""
    both_absent = expected is None and actual is None

    if path in MONETARY_FIELDS:
        correct = _as_decimal(expected) == _as_decimal(actual)
    elif path in TEXT_FIELDS:
        correct = normalize_text(expected) == normalize_text(actual)
    elif path in EXACT_FIELDS:
        correct = expected == actual
    else:
        correct = normalize_text(expected) == normalize_text(actual)

    return FieldScore(
        field=path,
        correct=correct,
        expected=expected,
        actual=actual,
        both_absent=both_absent,
    )


def _description_similarity(left: str | None, right: str | None) -> float:
    """Token-overlap similarity (Jaccard) between two item descriptions."""
    left_tokens = set((normalize_text(left) or "").split())
    right_tokens = set((normalize_text(right) or "").split())
    if not left_tokens or not right_tokens:
        return 0.0
    intersection = left_tokens & right_tokens
    union = left_tokens | right_tokens
    return len(intersection) / len(union)


def score_items(
    expected: Sequence[Mapping[str, Any]], actual: Sequence[Mapping[str, Any]]
) -> tuple[float, float, float]:
    """Match item lists and return ``(precision, recall, f1)``.

    An actual item counts as a true positive only when its description matches
    an expected item *and* its total price agrees exactly. Matching the name
    but not the price is not a partial success -- it is a wrong number in a
    financial record.
    """
    if not expected and not actual:
        return 1.0, 1.0, 1.0
    if not expected or not actual:
        return 0.0, 0.0, 0.0

    unmatched = list(range(len(expected)))
    true_positives = 0

    for candidate in actual:
        best_index: int | None = None
        best_score = 0.0
        for index in unmatched:
            score = _description_similarity(
                candidate.get("description"), expected[index].get("description")
            )
            if score > best_score:
                best_score, best_index = score, index

        if best_index is None or best_score < _ITEM_MATCH_THRESHOLD:
            continue

        matched = expected[best_index]
        if _as_decimal(candidate.get("total_price")) == _as_decimal(matched.get("total_price")):
            true_positives += 1
        unmatched.remove(best_index)

    precision = true_positives / len(actual)
    recall = true_positives / len(expected)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


#: Fields reported by the evaluation harness by default.
DEFAULT_FIELDS: tuple[str, ...] = (
    "merchant.name",
    "transaction.date",
    "subtotal",
    "tax.total",
    "total",
    "currency",
    "receipt_number",
    "payment.method",
)


def score_document(
    name: str,
    expected: Mapping[str, Any],
    actual: Mapping[str, Any],
    fields: Sequence[str] = DEFAULT_FIELDS,
) -> DocumentScore:
    """Score one document against its expectation."""
    scores = [compare_field(path, _get(expected, path), _get(actual, path)) for path in fields]
    precision, recall, f1 = score_items(expected.get("items") or [], actual.get("items") or [])
    return DocumentScore(
        name=name,
        fields=scores,
        item_precision=precision,
        item_recall=recall,
        item_f1=f1,
        expected_items=len(expected.get("items") or []),
        actual_items=len(actual.get("items") or []),
    )
