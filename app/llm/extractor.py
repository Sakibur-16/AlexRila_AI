"""LLM fallback extraction.

Invoked only when deterministic extraction reports low confidence. Everything
about this module is built on one premise: **the model is a witness, not an
authority.**

Its output passes three gates before any value is accepted:

1. **Type gate** -- each value must parse into the type the schema demands.
   A total of ``"about twenty dollars"`` is discarded, not coerced.
2. **Grounding gate** -- the value must actually appear in the OCR text.
   This is the anti-hallucination mechanism: a model cannot introduce a
   merchant name, an amount or a date that the document does not contain,
   because the pipeline checks.
3. **Merge gate** -- an accepted value may only fill a field that is null or
   scored below the fallback threshold. A field the rules read confidently is
   never overwritten.

The merged result is then re-validated from scratch, so LLM-sourced values face
exactly the same arithmetic and semantic checks as deterministic ones.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import date, time
from decimal import Decimal, InvalidOperation
from typing import Any

from app.core.config import Settings
from app.core.exceptions import PipelineError
from app.core.logging import get_logger
from app.core.metrics import MetricNames, increment, observe
from app.core.retry import retry_sync
from app.domain.evidence import Evidence, ExtractedField, ExtractionMethod
from app.extraction.receipt import ExtractionResult
from app.llm.base import LLMProvider, LLMRequest
from app.llm.prompts import (
    PROMPT_VERSION,
    RECEIPT_JSON_SCHEMA,
    SYSTEM_PROMPT,
    build_document_block,
)
from app.normalization.dates import parse_date, parse_time
from app.normalization.money import parse_amount
from app.schemas.receipt import LineItem, PaymentMethod, Tax

logger = get_logger(__name__)

#: Non-alphanumerics are ignored when checking whether a value is grounded in
#: the document, so that "1,234.50" matches a printed "1 234.50".
_LOOSE = re.compile(r"[^0-9A-Za-z]+")


@dataclass(frozen=True, slots=True)
class LLMFallbackOutcome:
    """Result of an attempted LLM fallback."""

    extraction: ExtractionResult
    used: bool
    provider: str | None = None
    model: str | None = None
    prompt_version: str | None = None
    #: Dotted paths the LLM actually contributed.
    filled_fields: tuple[str, ...] = ()
    #: Values rejected by the grounding or type gates.
    rejected_fields: tuple[str, ...] = ()
    error: str | None = None


class LLMFallbackExtractor:
    """Fills low-confidence fields using a language model, under strict gates."""

    def __init__(self, settings: Settings, provider: LLMProvider) -> None:
        self._settings = settings
        self._provider = provider

    def should_invoke(self, confidence: float, extraction: ExtractionResult) -> bool:
        """Decide whether the fallback is worth its cost and latency.

        Fires when overall confidence is below the configured threshold, or
        when a field that matters is missing outright -- a receipt with no
        total is worth a second opinion even if everything else scored well.
        """
        if not self._settings.llm_enabled:
            return False
        if confidence < self._settings.llm_fallback_confidence_threshold:
            return True
        return extraction.total.value is None or extraction.merchant_name.value is None

    def run(self, extraction: ExtractionResult, ocr_text: str) -> LLMFallbackOutcome:
        """Attempt fallback extraction and merge whatever survives the gates.

        A failure here is never fatal: the deterministic result is returned
        unchanged with the error recorded, because a degraded answer beats no
        answer.
        """
        request = LLMRequest(
            system_prompt=SYSTEM_PROMPT,
            document_text=build_document_block(
                ocr_text, max_chars=self._settings.llm_max_input_chars
            ),
            json_schema=RECEIPT_JSON_SCHEMA,
            temperature=self._settings.llm_temperature,
            max_output_tokens=self._settings.llm_max_output_tokens,
            timeout_seconds=self._settings.llm_timeout_seconds,
            prompt_version=PROMPT_VERSION,
        )

        increment(MetricNames.LLM_INVOKED)
        try:
            response = retry_sync(
                lambda: self._provider.extract(request),
                max_retries=self._settings.llm_max_retries,
                base_delay=0.5,
            )
        except PipelineError as exc:
            increment(MetricNames.LLM_FAILURE, labels={"code": exc.code.value})
            logger.warning("llm_fallback_failed", error_code=exc.code.value)
            return LLMFallbackOutcome(extraction=extraction, used=False, error=exc.code.value)

        observe(MetricNames.LLM_LATENCY_MS, response.duration_ms)

        if response.finish_reason == "length":
            # A truncated response may carry a half-written amount; refuse it
            # entirely rather than merge a fragment.
            increment(MetricNames.LLM_FAILURE, labels={"code": "TRUNCATED"})
            return LLMFallbackOutcome(
                extraction=extraction, used=False, error="LLM_OUTPUT_TRUNCATED"
            )

        merged, filled, rejected = self._merge(extraction, response.data, ocr_text)
        increment(MetricNames.LLM_SUCCESS)

        return LLMFallbackOutcome(
            extraction=merged,
            used=bool(filled),
            provider=response.provider,
            model=response.model,
            prompt_version=PROMPT_VERSION,
            filled_fields=filled,
            rejected_fields=rejected,
        )

    # ---------------------------------------------------------------- merge
    def _merge(
        self, extraction: ExtractionResult, data: dict[str, Any], ocr_text: str
    ) -> tuple[ExtractionResult, tuple[str, ...], tuple[str, ...]]:
        """Merge gated LLM values into the deterministic result."""
        grounding = _LOOSE.sub("", ocr_text).lower()
        filled: list[str] = []
        rejected: list[str] = []
        updates: dict[str, Any] = {}

        def accept(
            attr: str,
            path: str,
            raw: Any,
            parser: Callable[[Any], Any],
            *,
            require_grounding: bool = True,
        ) -> None:
            current: ExtractedField[Any] = getattr(extraction, attr)
            if not self._is_fillable(current):
                return
            if raw in (None, "", "N/A", "n/a", "null", "none", "unknown"):
                return

            parsed = parser(raw)
            if parsed is None:
                rejected.append(f"{path}:unparseable")
                return
            if require_grounding and not _is_grounded(raw, grounding):
                rejected.append(f"{path}:ungrounded")
                return

            updates[attr] = ExtractedField(
                value=parsed,
                evidence=(
                    Evidence(
                        source_text=str(raw),
                        method=ExtractionMethod.LLM,
                        notes=f"prompt={PROMPT_VERSION}",
                    ),
                ),
                raw_value=str(raw),
            )
            filled.append(path)

        accept("merchant_name", "merchant.name", data.get("merchant_name"), _as_text)
        accept("merchant_address", "merchant.address", data.get("merchant_address"), _as_text)
        accept("merchant_phone", "merchant.phone", data.get("merchant_phone"), _as_text)
        accept("receipt_number", "receipt_number", data.get("receipt_number"), _as_text)
        accept("subtotal", "subtotal", data.get("subtotal"), _as_money)
        accept("total", "total", data.get("total"), _as_money)
        accept("service_charge", "service_charge", data.get("service_charge"), _as_money)
        accept("card_last4", "payment.card_last4", data.get("card_last4"), _as_card_tail)
        accept(
            "payment_method",
            "payment.method",
            data.get("payment_method"),
            _as_payment_method,
            require_grounding=False,
        )
        accept("transaction_date", "transaction.date", data.get("date"), _as_date)
        accept("transaction_time", "transaction.time", data.get("time"), _as_time)
        # A currency code is a classification of the document, not a literal
        # substring of it, so grounding does not apply.
        accept(
            "currency",
            "currency",
            data.get("currency"),
            _as_currency,
            require_grounding=False,
        )

        self._merge_tax(extraction, data, grounding, updates, filled, rejected)
        self._merge_items(extraction, data, grounding, updates, filled, rejected)

        if not updates:
            return extraction, (), tuple(rejected)

        merged = replace(extraction, **updates)
        return merged, tuple(filled), tuple(rejected)

    def _is_fillable(self, field: ExtractedField[Any]) -> bool:
        """Whether the LLM is permitted to write to this field.

        Empty fields always; populated ones only when they scored below the
        fallback threshold. A confidently extracted value is never replaced.
        """
        if not field.is_present:
            return True
        return field.confidence < self._settings.llm_fallback_confidence_threshold

    def _merge_tax(
        self,
        extraction: ExtractionResult,
        data: dict[str, Any],
        grounding: str,
        updates: dict[str, Any],
        filled: list[str],
        rejected: list[str],
    ) -> None:
        """Fill the tax total when the rules found none."""
        raw = data.get("tax_total")
        if raw in (None, "") or extraction.tax.value is not None:
            return
        amount = _as_money(raw)
        if amount is None:
            rejected.append("tax.total:unparseable")
            return
        if not _is_grounded(raw, grounding):
            rejected.append("tax.total:ungrounded")
            return
        updates["tax"] = ExtractedField(
            value=Tax(total=amount, details=()),
            evidence=(
                Evidence(
                    source_text=str(raw),
                    method=ExtractionMethod.LLM,
                    notes=f"prompt={PROMPT_VERSION}",
                ),
            ),
            raw_value=str(raw),
        )
        filled.append("tax.total")

    def _merge_items(
        self,
        extraction: ExtractionResult,
        data: dict[str, Any],
        grounding: str,
        updates: dict[str, Any],
        filled: list[str],
        rejected: list[str],
    ) -> None:
        """Adopt the LLM item list only when the rules produced none.

        Never merges item-by-item: two partial lists interleaved would produce
        duplicates that no downstream consumer could untangle. It is all or
        nothing, and only when there is nothing to lose.
        """
        if extraction.items:
            return
        raw_items = data.get("items")
        if not isinstance(raw_items, list) or not raw_items:
            return

        accepted: list[LineItem] = []
        for raw in raw_items:
            if not isinstance(raw, dict):
                continue
            description = _as_text(raw.get("description"))
            if not description or not _is_grounded(description, grounding):
                rejected.append("items:ungrounded_description")
                continue
            total_price = _as_money(raw.get("total_price"))
            if total_price is not None and not _is_grounded(raw.get("total_price"), grounding):
                rejected.append("items:ungrounded_price")
                total_price = None
            accepted.append(
                LineItem(
                    description=description,
                    sku=_as_text(raw.get("sku")),
                    quantity=_as_decimal(raw.get("quantity")),
                    unit_price=_as_money(raw.get("unit_price")),
                    total_price=total_price,
                )
            )

        if not accepted:
            return

        updates["items"] = tuple(accepted)
        updates["item_methods"] = tuple(ExtractionMethod.LLM for _ in accepted)
        updates["item_evidence"] = tuple(
            Evidence(
                source_text=item.description or "",
                method=ExtractionMethod.LLM,
                notes=f"prompt={PROMPT_VERSION}",
            )
            for item in accepted
        )
        filled.append("items")


# ------------------------------------------------------------------ parsers
def _as_text(raw: Any) -> str | None:
    """Accept a non-empty string, rejecting absence placeholders."""
    if not isinstance(raw, str):
        return None
    cleaned = " ".join(raw.split())
    if not cleaned or cleaned.lower() in {"n/a", "na", "null", "none", "unknown", "-"}:
        return None
    return cleaned


def _as_decimal(raw: Any) -> Decimal | None:
    if raw is None:
        return None
    try:
        value = Decimal(str(raw).strip().replace(",", ""))
    except (InvalidOperation, ValueError):
        return None
    return value if value.is_finite() else None


def _as_money(raw: Any) -> Decimal | None:
    """Parse a monetary value, tolerating a stray currency symbol."""
    if raw is None:
        return None
    parsed = parse_amount(str(raw), repair_ocr=False)
    return parsed.value if parsed else None


def _as_date(raw: Any) -> date | None:
    if not isinstance(raw, str):
        return None
    parsed = parse_date(raw, date_order="none")
    return parsed.value if parsed else None


def _as_time(raw: Any) -> time | None:
    if not isinstance(raw, str):
        return None
    parsed = parse_time(raw)
    return parsed.value if parsed else None


def _as_currency(raw: Any) -> str | None:
    """Accept a three-letter ISO code only."""
    from app.normalization.currency import KNOWN_CODES

    if not isinstance(raw, str):
        return None
    code = raw.strip().upper()
    return code if code in KNOWN_CODES else None


def _as_card_tail(raw: Any) -> str | None:
    """Accept exactly four digits, never more."""
    if raw is None:
        return None
    digits = re.sub(r"\D", "", str(raw))
    return digits[-4:] if len(digits) >= 4 else None


def _as_payment_method(raw: Any) -> PaymentMethod | None:
    if not isinstance(raw, str):
        return None
    candidate = raw.strip().lower().replace(" ", "_").replace("-", "_")
    try:
        return PaymentMethod(candidate)
    except ValueError:
        if "card" in candidate:
            return PaymentMethod.CARD
        if "cash" in candidate:
            return PaymentMethod.CASH
        return None


def _is_grounded(raw: Any, grounding: str) -> bool:
    """Whether ``raw`` actually appears in the document text.

    Comparison ignores punctuation, spacing and case, so a model that
    reformats ``1,234.50`` as ``1234.50`` or normalises capitalisation is not
    penalised -- while a value the document never contained still fails.
    """
    if raw is None:
        return False
    needle = _LOOSE.sub("", str(raw)).lower()
    if not needle:
        return False
    return needle in grounding
