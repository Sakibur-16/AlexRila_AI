"""Confidence scoring.

Confidence here is a *composed* quantity, not a number invented to look
plausible. Three independent signals feed every field score:

.. code-block:: text

    ocr_component        how legible the source text was (from the provider)
    extraction_component how strong the method that found it was (a prior)
    validation_multiplier whether the value survived arithmetic and semantic
                          checks

and they combine as::

    field_score = (w_ocr * ocr + w_ext * extraction) * validation_multiplier

with the weights configurable (``CONFIDENCE_WEIGHT_OCR`` /
``CONFIDENCE_WEIGHT_EXTRACTION``, normalised to sum to 1) and the method
priors defined once in :data:`~app.domain.evidence.METHOD_PRIOR`.

The multiplier is where consistency earns its keep. A total that satisfies
``subtotal - discount + tax == total`` has been independently corroborated by
figures the total extractor never touched, so it is scored *above* its raw
components; a total that fails is scored below. That is the difference between
a confidence number that means something and one that is decoration.

Absent fields score ``0.0``. They are excluded from the aggregate, because
averaging in zeros for fields a receipt never printed would make a perfectly
read simple receipt look worse than a badly read complex one.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from app.core.config import Settings
from app.domain.evidence import METHOD_PRIOR, ExtractedField, ExtractionMethod
from app.extraction.receipt import ExtractionResult
from app.schemas.receipt import ConfidenceReport
from app.schemas.validation import IssueSeverity, ValidationResult

#: Weight of each field in the overall score. Fields a consumer will act on
#: financially dominate; incidental metadata contributes little. Fields absent
#: from this map default to :data:`_DEFAULT_FIELD_WEIGHT`.
FIELD_WEIGHTS: Mapping[str, float] = {
    "total": 3.0,
    "currency": 2.0,
    "merchant.name": 2.0,
    "transaction.date": 2.0,
    "subtotal": 1.5,
    "tax.total": 1.5,
    "items": 2.5,
    "payment.method": 0.5,
    "payment.card_last4": 0.5,
    "receipt_number": 0.5,
    "merchant.address": 0.3,
    "merchant.phone": 0.3,
    "merchant.email": 0.3,
    "merchant.website": 0.2,
    "merchant.tax_id": 0.3,
    "merchant.registration_id": 0.2,
    "transaction.time": 0.5,
    "discount": 0.8,
    "service_charge": 0.5,
    "shipping": 0.5,
    "tip": 0.3,
}

_DEFAULT_FIELD_WEIGHT = 0.5

#: Applied to a field named by a validation ERROR.
_ERROR_MULTIPLIER = 0.55

#: Applied to a field named by a validation WARNING.
_WARNING_MULTIPLIER = 0.80

#: Applied to every financial field when the total identity *holds*. Passing an
#: independent arithmetic check is genuine corroborating evidence.
_ARITHMETIC_BONUS = 1.08

#: Fields the arithmetic bonus applies to.
_FINANCIAL_FIELDS = frozenset(
    {"total", "subtotal", "tax.total", "discount", "items", "service_charge", "shipping"}
)

#: Codes that indicate the arithmetic did *not* reconcile.
_ARITHMETIC_FAILURE_CODES = frozenset(
    {"TOTAL_MISMATCH", "SUBTOTAL_MISMATCH", "ITEM_ARITHMETIC_MISMATCH", "TAX_MISMATCH"}
)


@dataclass(frozen=True, slots=True)
class ScoredConfidence:
    """Confidence report plus the review decision derived from it."""

    report: ConfidenceReport
    review_required: bool
    review_reasons: tuple[str, ...]


class ConfidenceScorer:
    """Computes per-field and aggregate confidence."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    def score(
        self,
        extraction: ExtractionResult,
        validation: ValidationResult,
        *,
        ocr_confidence: float,
    ) -> ScoredConfidence:
        """Score every field and derive the aggregate.

        Args:
            extraction: Fields with their evidence.
            validation: Validation outcome, which modulates the scores.
            ocr_confidence: Document-level OCR confidence, used for fields
                whose evidence carries no per-span confidence.

        Returns:
            The report and whether the result warrants human review.
        """
        weight_ocr, weight_extraction = self._settings.confidence_weights()
        arithmetic_ok = self._arithmetic_reconciled(validation)
        penalised = self._penalty_map(validation)

        field_scores: dict[str, float] = {}
        extraction_components: list[float] = []

        for path, extracted in extraction.field_map().items():
            if not extracted.is_present:
                field_scores[path] = 0.0
                continue

            ocr_part = self._ocr_component(extracted, ocr_confidence)
            extraction_part = self._extraction_component(extracted)
            extraction_components.append(extraction_part)

            score = weight_ocr * ocr_part + weight_extraction * extraction_part
            score *= penalised.get(path, 1.0)
            if arithmetic_ok and path in _FINANCIAL_FIELDS:
                score *= _ARITHMETIC_BONUS

            field_scores[path] = _clamp(score)

        items_score = self._score_items(extraction, ocr_confidence, weight_ocr, weight_extraction)
        if items_score is not None:
            if arithmetic_ok:
                items_score *= _ARITHMETIC_BONUS
            items_score *= penalised.get("items", 1.0)
            field_scores["items"] = _clamp(items_score)
            extraction_components.append(items_score)

        overall = self._aggregate(field_scores)
        validation_multiplier = self._validation_multiplier(validation, arithmetic_ok)

        report = ConfidenceReport(
            overall=_clamp(overall),
            ocr=_clamp(ocr_confidence),
            extraction=_clamp(
                sum(extraction_components) / len(extraction_components)
                if extraction_components
                else 0.0
            ),
            validation=_clamp(validation_multiplier),
            fields=field_scores,
        )

        review_required, reasons = self._review_decision(report, validation)
        return ScoredConfidence(
            report=report, review_required=review_required, review_reasons=reasons
        )

    # ------------------------------------------------------------ components
    @staticmethod
    def _ocr_component(extracted: ExtractedField[Any], document_confidence: float) -> float:
        """OCR confidence for the span a value was read from.

        Uses the *minimum* across evidence items rather than the mean: a value
        assembled from two lines is only as trustworthy as the worse of them.
        """
        scores = [
            evidence.ocr_confidence
            for evidence in extracted.evidence
            if evidence.ocr_confidence is not None
        ]
        if not scores:
            return document_confidence
        return min(scores)

    @staticmethod
    def _extraction_component(extracted: ExtractedField[Any]) -> float:
        """Prior confidence in the method that produced the value.

        A field carrying its own explicit confidence (currency detection, which
        computes an evidence-graded score of its own) keeps it, blended with
        the method prior so neither signal is discarded.
        """
        method = extracted.method or ExtractionMethod.HEURISTIC
        prior = METHOD_PRIOR.get(method, 0.5)

        if extracted.confidence > 0:
            return (prior + extracted.confidence) / 2

        # Uncertainty flagged during normalisation is a direct statement that
        # the value is doubtful, so it caps the method prior.
        if extracted.warnings:
            return min(prior, 0.55)
        return prior

    def _score_items(
        self,
        extraction: ExtractionResult,
        ocr_confidence: float,
        weight_ocr: float,
        weight_extraction: float,
    ) -> float | None:
        """Score the item list as a whole.

        Combines the mean per-item score with a completeness factor derived
        from how many candidate lines the extractor had to skip -- a receipt
        where a third of the item region was unparseable should not report the
        same confidence as one where every row was read.
        """
        if not extraction.items:
            return None

        per_item: list[float] = []
        for index, method in enumerate(extraction.item_methods):
            evidence = (
                extraction.item_evidence[index] if index < len(extraction.item_evidence) else None
            )
            ocr_part = (
                evidence.ocr_confidence
                if evidence is not None and evidence.ocr_confidence is not None
                else ocr_confidence
            )
            extraction_part = METHOD_PRIOR.get(method, 0.5)
            per_item.append(weight_ocr * ocr_part + weight_extraction * extraction_part)

        if not per_item:
            return None

        mean = sum(per_item) / len(per_item)
        total_candidates = len(extraction.items) + extraction.skipped_item_lines
        completeness = len(extraction.items) / total_candidates if total_candidates else 1.0
        # Completeness is applied at half strength: a skipped line is often a
        # divider or a note rather than a missed item, so it should temper the
        # score rather than dominate it.
        return mean * (0.5 + 0.5 * completeness)

    # ------------------------------------------------------------- modifiers
    @staticmethod
    def _arithmetic_reconciled(validation: ValidationResult) -> bool:
        """Whether every arithmetic identity that could be checked held."""
        return not any(
            issue.code.value in _ARITHMETIC_FAILURE_CODES for issue in validation.all_issues
        )

    def _penalty_map(self, validation: ValidationResult) -> dict[str, float]:
        """Per-field multipliers derived from validation findings."""
        penalties: dict[str, float] = {}
        for issue in validation.all_issues:
            if issue.field is None:
                continue
            if issue.severity is IssueSeverity.ERROR:
                multiplier = min(_ERROR_MULTIPLIER, self._settings.confidence_validation_penalty)
            elif issue.severity is IssueSeverity.WARNING:
                multiplier = _WARNING_MULTIPLIER
            else:
                continue
            # An item-level issue also tempers the aggregate item score.
            key = "items" if issue.field.startswith("items[") else issue.field
            penalties[key] = min(penalties.get(key, 1.0), multiplier)
        return penalties

    @staticmethod
    def _validation_multiplier(validation: ValidationResult, arithmetic_ok: bool) -> float:
        """Document-level validation health, reported for transparency."""
        if validation.errors:
            return 0.4
        if not arithmetic_ok:
            return 0.6
        if validation.warnings:
            return 0.85
        return 1.0

    @staticmethod
    def _aggregate(field_scores: Mapping[str, float]) -> float:
        """Weighted mean over *present* fields only."""
        weighted = 0.0
        total_weight = 0.0
        for path, score in field_scores.items():
            if score <= 0.0:
                continue
            weight = FIELD_WEIGHTS.get(path, _DEFAULT_FIELD_WEIGHT)
            weighted += score * weight
            total_weight += weight
        if total_weight == 0.0:
            return 0.0
        return weighted / total_weight

    def _review_decision(
        self, report: ConfidenceReport, validation: ValidationResult
    ) -> tuple[bool, tuple[str, ...]]:
        """Decide whether a human should look at this result.

        Review is recommended when the aggregate is below the configured
        threshold, when validation raised an error, or when a *high-weight*
        field scored poorly -- a receipt whose total is uncertain needs review
        even if everything else was read perfectly.
        """
        reasons: list[str] = []

        if report.overall < self._settings.confidence_threshold:
            reasons.append("LOW_OVERALL_CONFIDENCE")
        if validation.errors:
            reasons.extend(sorted({issue.code.value for issue in validation.errors}))
        if report.ocr < self._settings.low_ocr_confidence_threshold:
            reasons.append("OCR_LOW_CONFIDENCE")

        for path in ("total", "currency", "transaction.date", "merchant.name"):
            score = report.fields.get(path, 0.0)
            if 0.0 < score < self._settings.review_field_confidence_threshold:
                reasons.append(f"LOW_FIELD_CONFIDENCE:{path}")

        return bool(reasons), tuple(dict.fromkeys(reasons))


def _clamp(value: float) -> float:
    """Constrain a score to ``[0, 1]``."""
    return max(0.0, min(1.0, value))
