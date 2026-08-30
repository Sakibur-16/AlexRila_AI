"""End-to-end pipeline tests.

Exercises the real path -- input validation, decoding, quality assessment,
preprocessing, OCR, extraction, validation, confidence, assembly -- with OCR
replayed from a recorded fixture so the result is deterministic.
"""

from __future__ import annotations

from decimal import Decimal

import cv2
import numpy as np
import pytest

from app.core.exceptions import (
    ErrorCode,
    FileTooLargeError,
    InputValidationError,
    OCRError,
    PipelineError,
    ProviderUnavailableError,
    UnsupportedFileTypeError,
)
from app.domain.document import DocumentInput, DocumentType
from app.ocr.base import OCRProvider, OCRRequest
from app.pipeline.receipt_pipeline import ReceiptPipeline
from app.schemas.ocr import OCRResult
from app.schemas.validation import IssueCode


def _document(content: bytes, **kwargs) -> DocumentInput:
    return DocumentInput(
        content=content,
        filename=kwargs.pop("filename", "receipt.png"),
        media_type=kwargs.pop("media_type", "image/png"),
        document_type=DocumentType.RECEIPT,
        metadata=kwargs.pop("metadata", {}),
    )


def _run(pipeline: ReceiptPipeline, image: bytes, fixture: str):
    return pipeline.process(
        _document(image, metadata={"fixture": fixture}), request_id="test-request"
    )


# --------------------------------------------------------------- happy path
def test_clean_receipt_end_to_end(pipeline, receipt_image) -> None:
    result = _run(pipeline, receipt_image, "001_grocery_us")
    receipt = result.receipt

    assert receipt.merchant.name == "GREEN VALLEY MARKET"
    assert receipt.total == Decimal("17.28")
    assert receipt.subtotal == Decimal("17.10")
    assert receipt.currency == "USD"
    assert len(receipt.items) == 4
    assert receipt.validation.is_valid
    assert receipt.confidence.overall > 0.85


def test_processing_metadata_is_populated(pipeline, receipt_image) -> None:
    result = _run(pipeline, receipt_image, "001_grocery_us")
    processing = result.receipt.processing

    assert processing is not None
    assert processing.request_id == "test-request"
    assert processing.ocr_provider == "fixture"
    assert processing.pipeline_version
    assert processing.schema_version
    assert processing.processing_time_ms > 0
    # Every stage reports its own latency, so a regression is attributable.
    for stage in ("input_validation", "preprocessing", "ocr", "extraction", "validation"):
        assert stage in processing.stage_timings_ms


def test_preprocessing_is_conditional(pipeline, receipt_image) -> None:
    """Transforms are applied on measured need, not unconditionally."""
    result = _run(pipeline, receipt_image, "001_grocery_us")
    assert result.preprocessing is not None
    assert result.preprocessing.applied
    assert result.preprocessing.skipped, "some transforms should be skipped"


def test_quality_metrics_are_reported(pipeline, receipt_image) -> None:
    result = _run(pipeline, receipt_image, "001_grocery_us")
    assert result.quality is not None
    assert result.quality.width > 0
    assert result.quality.blur_score >= 0


# ------------------------------------------------------------ money handling
def test_money_serialises_as_strings(pipeline, receipt_image) -> None:
    """The contract: JSON strings, so JSON.parse cannot lose precision."""
    result = _run(pipeline, receipt_image, "001_grocery_us")
    payload = result.receipt.to_public_dict()

    assert payload["total"] == "17.28"
    assert isinstance(payload["total"], str)
    assert isinstance(payload["items"][0]["total_price"], str)


def test_european_decimals_are_parsed(pipeline, receipt_image) -> None:
    result = _run(pipeline, receipt_image, "002_eu_multi_vat")
    assert result.receipt.subtotal == Decimal("1242.00")
    assert result.receipt.total == Decimal("1333.00")


def test_multiple_tax_rates_are_captured(pipeline, receipt_image) -> None:
    result = _run(pipeline, receipt_image, "002_eu_multi_vat")
    assert result.receipt.tax.total == Decimal("91.00")
    assert len(result.receipt.tax.details) == 2


# ---------------------------------------------------------------- uncertainty
def test_uncertainty_is_reported_not_raised(pipeline, receipt_image) -> None:
    """A receipt that does not add up is a successful, warned result."""
    result = _run(pipeline, receipt_image, "005_total_mismatch")

    assert result.receipt.total == Decimal("99.99"), "printed value is preserved"
    assert not result.receipt.validation.is_valid
    assert IssueCode.TOTAL_MISMATCH.value in result.receipt.validation.codes()
    assert result.receipt.review.review_required


def test_ambiguous_date_stays_null_with_raw_preserved(pipeline, receipt_image) -> None:
    result = _run(pipeline, receipt_image, "003_ambiguous")
    transaction = result.receipt.transaction

    assert transaction.date is None
    assert transaction.raw_date == "08/09/26"
    assert IssueCode.AMBIGUOUS_DATE_FORMAT.value in result.receipt.validation.codes()


def test_ambiguous_currency_reports_symbol_only(pipeline, receipt_image) -> None:
    result = _run(pipeline, receipt_image, "003_ambiguous")
    assert result.receipt.currency is None
    assert result.receipt.currency_symbol == "$"


def test_ocr_glyph_confusion_is_repaired(pipeline, receipt_image) -> None:
    """4.OO becomes 4.00 while COFFEE stays COFFEE."""
    result = _run(pipeline, receipt_image, "004_ocr_confusion")

    assert result.receipt.total == Decimal("7.02")
    descriptions = [item.description for item in result.receipt.items]
    assert "COFFEE" in descriptions
    assert "SODA" in descriptions


def test_low_confidence_ocr_is_flagged(pipeline, receipt_image) -> None:
    result = _run(pipeline, receipt_image, "006_minimal_no_geometry")
    codes = result.receipt.validation.codes()
    assert IssueCode.OCR_LOW_CONFIDENCE.value in codes
    assert result.receipt.review.review_required


def test_geometry_free_provider_still_extracts(pipeline, receipt_image) -> None:
    """A provider reporting no bounding boxes degrades, it does not fail."""
    result = _run(pipeline, receipt_image, "006_minimal_no_geometry")
    assert result.receipt.total == Decimal("1.00")
    assert IssueCode.OCR_NO_GEOMETRY.value in result.receipt.validation.codes()


# -------------------------------------------------------------- no invention
def test_absent_fields_are_null_never_placeholders(pipeline, receipt_image) -> None:
    result = _run(pipeline, receipt_image, "006_minimal_no_geometry")
    payload = result.receipt.to_public_dict()

    assert payload["merchant"]["phone"] is None
    assert payload["merchant"]["email"] is None
    assert payload["transaction"]["date"] is None

    rendered = str(payload)
    for placeholder in ("N/A", "n/a", "unknown", "UNKNOWN", "TBD"):
        assert placeholder not in rendered


# ------------------------------------------------------------------ security
def test_full_card_number_is_redacted_from_raw_ocr(pipeline, receipt_image) -> None:
    result = _run(pipeline, receipt_image, "008_pan_redaction")
    raw = result.receipt.raw_ocr

    assert raw.text is not None
    assert "4111 1111 1111 1111" not in raw.text
    assert "4111111111111111" not in raw.text.replace(" ", "")
    assert raw.redacted


def test_card_field_holds_only_four_digits(pipeline, receipt_image) -> None:
    result = _run(pipeline, receipt_image, "008_pan_redaction")
    last4 = result.receipt.payment.card_last4
    assert last4 is None or len(last4) == 4


def test_instruction_text_on_a_receipt_is_just_text(pipeline, receipt_image) -> None:
    """Injection cannot alter deterministic extraction: there is no model here."""
    result = _run(pipeline, receipt_image, "009_prompt_injection")
    assert result.receipt.total == Decimal("50.00")
    assert result.receipt.merchant.name == "SNEAKY GOODS LTD"


def test_raw_ocr_can_be_withheld(make_settings, receipt_image) -> None:
    from app.ocr.factory import create_ocr_provider

    settings = make_settings(include_raw_ocr=False)
    pipeline = ReceiptPipeline(
        settings=settings, ocr_provider=create_ocr_provider("fixture", settings)
    )
    result = _run(pipeline, receipt_image, "001_grocery_us")

    assert result.receipt.raw_ocr.text is None
    assert result.receipt.raw_ocr.line_count > 0, "metadata survives, content does not"


def test_evidence_is_opt_in(make_settings, receipt_image) -> None:
    from app.ocr.factory import create_ocr_provider

    settings = make_settings(include_evidence=True)
    pipeline = ReceiptPipeline(
        settings=settings, ocr_provider=create_ocr_provider("fixture", settings)
    )
    result = _run(pipeline, receipt_image, "001_grocery_us")

    assert result.receipt.evidence
    assert "total" in result.receipt.evidence
    assert result.receipt.evidence["total"].source_text


# -------------------------------------------------------------- input errors
def test_rejects_non_image_content(pipeline) -> None:
    with pytest.raises(UnsupportedFileTypeError) as exc:
        pipeline.process(_document(b"this is definitely not an image"))
    assert exc.value.code is ErrorCode.UNSUPPORTED_FILE_TYPE


def test_rejects_empty_upload(pipeline) -> None:
    with pytest.raises(InputValidationError):
        pipeline.process(_document(b""))


def test_rejects_oversized_upload(make_settings, receipt_image) -> None:
    from app.ocr.factory import create_ocr_provider

    settings = make_settings(max_file_size_mb=0.001)
    pipeline = ReceiptPipeline(
        settings=settings, ocr_provider=create_ocr_provider("fixture", settings)
    )
    with pytest.raises(FileTooLargeError):
        pipeline.process(_document(receipt_image))


def test_rejects_undersized_image(pipeline) -> None:
    tiny = cv2.imencode(".png", np.full((40, 40, 3), 255, np.uint8))[1].tobytes()
    with pytest.raises(PipelineError) as exc:
        pipeline.process(_document(tiny))
    assert exc.value.code is ErrorCode.IMAGE_TOO_SMALL


def test_rejects_truncated_image(pipeline) -> None:
    """Valid magic bytes but undecodable content."""
    with pytest.raises(InputValidationError) as exc:
        pipeline.process(_document(b"\x89PNG\r\n\x1a\n" + b"\x00" * 200))
    assert exc.value.code is ErrorCode.INVALID_IMAGE


# ------------------------------------------------------------- OCR failures
class _FailingProvider(OCRProvider):
    """OCR provider that always raises a given error."""

    name = "failing"

    def __init__(self, error: Exception) -> None:
        self._error = error
        self.calls = 0

    def extract(self, request: OCRRequest) -> OCRResult:
        self.calls += 1
        raise self._error


def test_ocr_failure_surfaces_as_a_structured_error(settings, receipt_image) -> None:
    provider = _FailingProvider(OCRError("engine exploded"))
    pipeline = ReceiptPipeline(settings=settings, ocr_provider=provider)

    with pytest.raises(OCRError) as exc:
        pipeline.process(_document(receipt_image))
    assert exc.value.code is ErrorCode.OCR_FAILED


def test_non_retryable_failures_are_not_retried(settings, receipt_image) -> None:
    """Retrying a deterministic failure only multiplies the latency."""
    provider = _FailingProvider(OCRError("engine exploded"))
    pipeline = ReceiptPipeline(settings=settings, ocr_provider=provider)

    with pytest.raises(OCRError):
        pipeline.process(_document(receipt_image))
    assert provider.calls == 1


def test_transient_failures_are_retried(make_settings, receipt_image) -> None:
    settings = make_settings(ocr_max_retries=2, ocr_retry_base_delay_seconds=0.0)
    provider = _FailingProvider(ProviderUnavailableError("engine warming up"))
    pipeline = ReceiptPipeline(settings=settings, ocr_provider=provider)

    with pytest.raises(ProviderUnavailableError):
        pipeline.process(_document(receipt_image))
    assert provider.calls == 3, "one attempt plus two retries"


def test_empty_ocr_result_is_an_error(settings, receipt_image) -> None:
    class _EmptyProvider(OCRProvider):
        name = "empty"

        def extract(self, request: OCRRequest) -> OCRResult:
            return OCRResult(text="   ", lines=(), provider=self.name)

    pipeline = ReceiptPipeline(settings=settings, ocr_provider=_EmptyProvider())
    with pytest.raises(OCRError) as exc:
        pipeline.process(_document(receipt_image))
    assert exc.value.code is ErrorCode.OCR_EMPTY_RESULT


# ---------------------------------------------------------------- providers
def test_provider_is_selected_by_configuration(make_settings) -> None:
    """The pipeline never names a provider; configuration does."""
    from app.ocr.factory import available_providers, create_ocr_provider

    assert "fixture" in available_providers()
    assert "tesseract" in available_providers()

    settings = make_settings(ocr_provider="fixture")
    assert create_ocr_provider(settings=settings).name == "fixture"


def test_unknown_provider_fails_with_a_helpful_error(make_settings) -> None:
    from app.core.exceptions import ProviderNotRegisteredError
    from app.ocr.factory import create_ocr_provider

    settings = make_settings(ocr_provider="does_not_exist")
    with pytest.raises(ProviderNotRegisteredError) as exc:
        create_ocr_provider(settings=settings)
    assert "available" in exc.value.details
