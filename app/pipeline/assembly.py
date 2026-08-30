"""Projection from the internal extraction result to the public schema.

This is the boundary where evidence-carrying internal fields become the plain,
stable contract a backend consumes. Keeping it in one place means the public
schema can evolve (or gain a v2) without touching any extractor.

Two invariants are enforced here:

* **Absent stays absent.** A field with no value becomes ``null``. Nothing is
  substituted, defaulted or filled with a placeholder.
* **Raw text is preserved but sanitised.** OCR text is redacted of card-like
  sequences before it is returned or stored, and is omitted entirely when
  ``INCLUDE_RAW_OCR`` is off.
"""

from __future__ import annotations

from typing import Any

from app.core.config import Settings
from app.core.versions import PIPELINE_VERSION, SCHEMA_VERSION
from app.domain.evidence import ExtractedField
from app.extraction.receipt import ExtractionResult
from app.schemas.ocr import OCRResult
from app.schemas.receipt import (
    ConfidenceReport,
    FieldEvidence,
    Merchant,
    Payment,
    ProcessingMetadata,
    RawOCR,
    Receipt,
    ReviewInfo,
    ReviewStatus,
    Tax,
    Transaction,
)
from app.schemas.validation import ValidationResult
from app.security.redaction import redact_text


def build_receipt(
    extraction: ExtractionResult,
    *,
    settings: Settings,
    ocr: OCRResult | None,
    validation: ValidationResult,
    confidence: ConfidenceReport,
    processing: ProcessingMetadata | None = None,
    review_required: bool = False,
    review_reasons: tuple[str, ...] = (),
) -> Receipt:
    """Assemble the public receipt document."""
    merchant = Merchant(
        name=extraction.merchant_name.value,
        address=extraction.merchant_address.value,
        phone=extraction.merchant_phone.value,
        email=extraction.merchant_email.value,
        website=extraction.merchant_website.value,
        tax_id=extraction.merchant_tax_id.value,
        registration_id=extraction.merchant_registration_id.value,
        store_id=extraction.merchant_store_id.value,
        country=settings.default_country or None,
    )

    transaction = Transaction(
        transaction_id=extraction.receipt_number.value,
        date=extraction.transaction_date.value,
        time=extraction.transaction_time.value,
        datetime=extraction.transaction_datetime.value,
        raw_date=extraction.transaction_date.raw_value,
        raw_time=extraction.transaction_time.raw_value,
    )

    payment = Payment(
        method=extraction.payment_method.value,
        card_type=extraction.card_type.value,
        card_last4=extraction.card_last4.value,
        authorization_code=extraction.authorization_code.value,
        amount_paid=extraction.amount_tendered.value,
        change=extraction.change.value,
        splits=extraction.payment_splits,
    )

    return Receipt(
        schema_version=SCHEMA_VERSION,
        merchant=merchant,
        transaction=transaction,
        items=extraction.items,
        subtotal=extraction.subtotal.value,
        discount=extraction.discount.value,
        tax=extraction.tax.value or Tax(),
        shipping=extraction.shipping.value,
        service_charge=extraction.service_charge.value,
        tip=extraction.tip.value,
        rounding_adjustment=extraction.rounding.value,
        total=extraction.total.value,
        currency=extraction.currency.value,
        currency_symbol=extraction.currency_symbol.value,
        payment=payment,
        receipt_number=extraction.receipt_number.value,
        raw_ocr=_build_raw_ocr(ocr, settings),
        confidence=confidence,
        validation=validation,
        processing=processing,
        review=ReviewInfo(
            review_required=review_required,
            status=ReviewStatus.PENDING if review_required else ReviewStatus.NOT_REQUIRED,
            reasons=review_reasons,
        ),
        evidence=_build_evidence(extraction) if settings.include_evidence else {},
    )


def _build_raw_ocr(ocr: OCRResult | None, settings: Settings) -> RawOCR:
    """Package the recognition output for the response.

    Redaction runs here rather than at recognition time so that the in-process
    :class:`OCRResult` stays faithful for debugging and reprocessing, while
    nothing card-like ever crosses the API boundary.
    """
    if ocr is None:
        return RawOCR()

    text: str | None = ocr.text if settings.include_raw_ocr else None
    redacted = False
    if text is not None and settings.redact_sensitive_text:
        result = redact_text(text)
        text, redacted = result.text, result.redacted

    return RawOCR(
        text=text,
        line_count=len(ocr.lines),
        mean_confidence=ocr.mean_confidence or None,
        redacted=redacted,
    )


def _build_evidence(extraction: ExtractionResult) -> dict[str, FieldEvidence]:
    """Project internal provenance onto the public evidence map."""
    evidence: dict[str, FieldEvidence] = {}

    for path, extracted in extraction.field_map().items():
        projected = _project_evidence(extracted)
        if projected is not None:
            evidence[path] = projected

    for index, item_evidence in enumerate(extraction.item_evidence):
        evidence[f"items[{index}]"] = FieldEvidence(
            source_text=item_evidence.source_text,
            method=item_evidence.method.value,
            line_index=item_evidence.line_index,
            bbox=item_evidence.bbox.as_list() if item_evidence.bbox else None,
            ocr_confidence=item_evidence.ocr_confidence,
        )

    return evidence


def _project_evidence(extracted: ExtractedField[Any]) -> FieldEvidence | None:
    """Convert the primary evidence item into its public form."""
    if not extracted.is_present or not extracted.evidence:
        return None
    primary = extracted.evidence[0]
    return FieldEvidence(
        source_text=primary.source_text,
        method=primary.method.value,
        line_index=primary.line_index,
        bbox=primary.bbox.as_list() if primary.bbox else None,
        ocr_confidence=primary.ocr_confidence,
    )


def build_processing_metadata(
    *,
    request_id: str | None,
    ocr: OCRResult | None,
    preprocessing_applied: tuple[str, ...],
    stage_timings_ms: dict[str, float],
    total_ms: float,
    llm_used: bool = False,
    llm_provider: str | None = None,
    llm_model: str | None = None,
    prompt_version: str | None = None,
) -> ProcessingMetadata:
    """Build the diagnostics block reported with every result."""
    return ProcessingMetadata(
        request_id=request_id,
        pipeline_version=PIPELINE_VERSION,
        schema_version=SCHEMA_VERSION,
        ocr_provider=ocr.provider if ocr else None,
        ocr_model=ocr.model if ocr else None,
        ocr_languages=ocr.languages if ocr else (),
        preprocessing_applied=preprocessing_applied,
        llm_used=llm_used,
        llm_provider=llm_provider,
        llm_model=llm_model,
        prompt_version=prompt_version,
        processing_time_ms=total_ms,
        stage_timings_ms={k: round(v, 2) for k, v in stage_timings_ms.items()},
    )
