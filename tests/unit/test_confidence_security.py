"""Confidence scoring and security-layer tests."""

from __future__ import annotations

import pytest

from app.confidence.scorer import ConfidenceScorer
from app.core.exceptions import (
    FileTooLargeError,
    InputValidationError,
    UnsupportedFileTypeError,
)
from app.extraction.receipt import RuleBasedReceiptExtractor
from app.security.files import detect_media_type, sanitize_filename, validate_upload
from app.security.redaction import extract_card_last4, mask_pan, redact_text
from app.validation.engine import ValidationEngine

_CONSISTENT = [
    "GREEN VALLEY MARKET",
    "Date: 2026-08-25",
    "Milk 2.50",
    "Bread 3.00",
    "SUBTOTAL 5.50",
    "TAX 0.50",
    "TOTAL 6.00 EUR",
]

_INCONSISTENT = [
    "GREEN VALLEY MARKET",
    "Date: 2026-08-25",
    "Milk 2.50",
    "Bread 3.00",
    "SUBTOTAL 5.50",
    "TAX 0.50",
    "TOTAL 600.00 EUR",
]


@pytest.fixture
def score(settings, ocr_result_factory):
    """Run extraction, validation and scoring over receipt lines."""

    def _score(lines: list[str], *, ocr_confidence: float = 0.95):
        ocr = ocr_result_factory(lines, confidence=ocr_confidence)
        extraction = RuleBasedReceiptExtractor(settings).extract(ocr)
        validation = ValidationEngine(settings).validate(extraction, ocr=ocr)
        return ConfidenceScorer(settings).score(
            extraction, validation, ocr_confidence=ocr.mean_confidence
        )

    return _score


# ----------------------------------------------------------------- confidence
def test_absent_fields_score_zero(score) -> None:
    result = score(["SHOP", "TOTAL 5.00"])
    assert result.report.fields["merchant.email"] == 0.0
    assert result.report.fields["shipping"] == 0.0


def test_absent_fields_do_not_drag_down_the_aggregate(score) -> None:
    """A simple receipt read perfectly must not score below a complex one."""
    result = score(_CONSISTENT)
    assert result.report.overall > 0.8


def test_reconciled_arithmetic_scores_above_inconsistent(score) -> None:
    """Independent corroboration is real evidence and is scored as such."""
    good = score(_CONSISTENT)
    bad = score(_INCONSISTENT)
    assert good.report.fields["total"] > bad.report.fields["total"]
    assert good.report.overall > bad.report.overall


def test_low_ocr_confidence_lowers_field_scores(score) -> None:
    clear = score(_CONSISTENT, ocr_confidence=0.98)
    murky = score(_CONSISTENT, ocr_confidence=0.35)
    assert murky.report.fields["total"] < clear.report.fields["total"]
    assert murky.report.ocr < clear.report.ocr


def test_keyword_anchored_beats_heuristic(score) -> None:
    """A labelled total is more trustworthy than a positionally guessed name."""
    result = score(_CONSISTENT)
    assert result.report.fields["total"] > result.report.fields["merchant.name"]


def test_validation_errors_trigger_review(score) -> None:
    result = score(_INCONSISTENT)
    assert result.review_required
    assert any("MISMATCH" in reason for reason in result.review_reasons)


def test_clean_receipt_reports_reasons_when_review_needed(score) -> None:
    """Every review recommendation must say why."""
    result = score(_INCONSISTENT)
    assert result.review_reasons


def test_all_scores_are_within_bounds(score) -> None:
    result = score(_INCONSISTENT)
    assert 0.0 <= result.report.overall <= 1.0
    for path, value in result.report.fields.items():
        assert 0.0 <= value <= 1.0, path


# ------------------------------------------------------------------- security
@pytest.mark.parametrize(
    ("content", "expected"),
    [
        (b"\xff\xd8\xff\xe0" + b"0" * 12, "image/jpeg"),
        (b"\x89PNG\r\n\x1a\n" + b"0" * 8, "image/png"),
        (b"II*\x00" + b"0" * 12, "image/tiff"),
        (b"RIFF____WEBP" + b"0" * 4, "image/webp"),
        (b"%PDF-1.7" + b"0" * 8, "application/pdf"),
        (b"not an image at all", None),
    ],
)
def test_media_type_detected_from_content(content: bytes, expected: str | None) -> None:
    """Detection reads magic bytes; the extension is never trusted."""
    assert detect_media_type(content) == expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("../../etc/passwd", "passwd"),
        (r"C:\Windows\System32\evil.png", "evil.png"),
        ("....//....//x.png", "x.png"),
        ("receipt.png", "receipt.png"),
        ("", None),
        ("...", None),
        (None, None),
    ],
)
def test_filename_sanitization(raw: str | None, expected: str | None) -> None:
    assert sanitize_filename(raw) == expected


def test_upload_rejects_disallowed_type(settings) -> None:
    with pytest.raises(UnsupportedFileTypeError):
        validate_upload(b"%PDF-1.7 not allowed", settings=settings)


def test_upload_rejects_unrecognized_content(settings) -> None:
    with pytest.raises(UnsupportedFileTypeError):
        validate_upload(b"plain text pretending to be a png", settings=settings)


def test_upload_rejects_empty(settings) -> None:
    with pytest.raises(InputValidationError):
        validate_upload(b"", settings=settings)


def test_upload_rejects_oversized(make_settings) -> None:
    settings = make_settings(max_file_size_mb=0.001)
    with pytest.raises(FileTooLargeError):
        validate_upload(b"\x89PNG\r\n\x1a\n" + b"0" * 5000, settings=settings)


def test_declared_type_mismatch_is_noted_not_fatal(settings) -> None:
    """Browsers mislabel uploads routinely; that is not a reason to refuse."""
    media_type, notes = validate_upload(
        b"\x89PNG\r\n\x1a\n" + b"0" * 8,
        settings=settings,
        declared_media_type="image/jpeg",
    )
    assert media_type == "image/png"
    assert any("mismatch" in note for note in notes)


# ------------------------------------------------------------------ redaction
def test_full_pan_is_masked() -> None:
    result = redact_text("VISA 4111 1111 1111 1111 APPROVED")
    assert "4111 1111 1111 1111" not in result.text
    assert result.text.endswith("1111 APPROVED")
    assert result.redacted


def test_already_masked_card_is_left_alone() -> None:
    result = redact_text("CARD ****1234 TOTAL 25.99")
    assert result.text == "CARD ****1234 TOTAL 25.99"
    assert not result.redacted


def test_short_identifiers_are_not_masked() -> None:
    """Over-redaction is preferred, but not at the cost of ordinary data."""
    result = redact_text("ORDER 12345 TOTAL 9.99 PHONE 5550142")
    assert result.text == "ORDER 12345 TOTAL 9.99 PHONE 5550142"


def test_cvv_is_masked() -> None:
    result = redact_text("CVV 123")
    assert "123" not in result.text


def test_mask_pan_keeps_last_four() -> None:
    assert mask_pan("4111111111111111") == "************1111"


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("CARD XXXX-XXXX-XXXX-9876", "9876"),
        ("VISA ****4321", "4321"),
        ("CASH PAYMENT", None),
    ],
)
def test_card_tail_extraction(text: str, expected: str | None) -> None:
    assert extract_card_last4(text) == expected
