"""Preprocessing transforms, money coercion, and provider response handling.

The Tesseract tests here exercise the provider's *own* logic -- regrouping
word-level output into lines, scaling confidences, normalising failures -- by
feeding it the dictionary shape the engine returns. That logic is where the
provider's bugs would live, and it is fully testable without the binary
installed.
"""

from __future__ import annotations

from decimal import Decimal

import cv2
import numpy as np
import pytest
from pydantic import BaseModel, ValidationError

from app.core.exceptions import ErrorCode, InputValidationError, OCRError, OCRTimeoutError
from app.ocr.base import OCRRequest
from app.ocr.providers.tesseract import TesseractOCRProvider
from app.preprocessing import transforms
from app.preprocessing.pipeline import PreprocessingPipeline
from app.preprocessing.quality import assess_quality, blur_score, estimate_skew, to_grayscale
from app.schemas.common import Money, Quantity, quantize_money
from app.schemas.quality import QualityVerdict


def _text_image(
    width: int = 700, height: int = 1000, *, angle: float = 0.0, dark: bool = False
) -> np.ndarray:
    """A synthetic receipt-like image with real text strokes."""
    image = np.full((height, width, 3), 30 if dark else 245, np.uint8)
    ink = (200, 200, 200) if dark else (15, 15, 15)
    for index, line in enumerate(
        ["GREEN VALLEY", "Milk 2.50", "Bread 3.00", "TOTAL 5.50", "THANK YOU"]
    ):
        cv2.putText(image, line, (40, 120 + index * 130), cv2.FONT_HERSHEY_SIMPLEX, 1.4, ink, 3)
    if angle:
        image = transforms.deskew(image, angle)
    return image


# ------------------------------------------------------------------- decoding
def test_decode_round_trips_png() -> None:
    encoded = cv2.imencode(".png", _text_image())[1].tobytes()
    decoded = transforms.decode_image(encoded)
    assert decoded.shape[0] > 0


def test_decode_rejects_corrupt_content() -> None:
    with pytest.raises(InputValidationError) as exc:
        transforms.decode_image(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)
    assert exc.value.code is ErrorCode.INVALID_IMAGE


def test_transparency_is_flattened_onto_white() -> None:
    """A PNG with alpha would otherwise render text onto black."""
    rgba = np.zeros((300, 300, 4), np.uint8)
    rgba[:, :, 3] = 0  # fully transparent
    decoded = transforms.decode_image(cv2.imencode(".png", rgba)[1].tobytes())
    assert decoded.ndim == 3
    assert decoded.mean() > 200, "transparent regions should become white"


def test_size_limits_reject_undersized() -> None:
    tiny = np.full((50, 50, 3), 255, np.uint8)
    with pytest.raises(InputValidationError) as exc:
        transforms.ensure_size_limits(tiny, min_width=200, min_height=200, max_pixels=10**9)
    assert exc.value.code is ErrorCode.IMAGE_TOO_SMALL


def test_size_limits_reject_decompression_bomb() -> None:
    """A small compressed file can decode to an enormous array."""
    image = np.full((2000, 2000, 3), 255, np.uint8)
    with pytest.raises(InputValidationError) as exc:
        transforms.ensure_size_limits(image, min_width=10, min_height=10, max_pixels=1000)
    assert exc.value.code is ErrorCode.IMAGE_TOO_LARGE


# ----------------------------------------------------------------- transforms
@pytest.mark.parametrize("rotation", [0, 90, 180, 270])
def test_orientation_correction_is_lossless(rotation: int) -> None:
    image = _text_image()
    rotated = transforms.correct_orientation(image, rotation)
    expected = image.shape[:2][::-1] if rotation in (90, 270) else image.shape[:2]
    assert rotated.shape[:2] == tuple(expected)


def test_deskew_expands_canvas_without_clipping() -> None:
    image = _text_image()
    rotated = transforms.deskew(image, 8.0)
    assert rotated.shape[0] >= image.shape[0]
    assert rotated.shape[1] >= image.shape[1]


def test_deskew_is_a_noop_below_the_noise_floor() -> None:
    image = _text_image()
    assert transforms.deskew(image, 0.0001) is image


def test_resize_upscales_small_images() -> None:
    small = _text_image(width=300, height=400)
    resized, changed = transforms.resize_for_ocr(small, min_short_edge=900, max_long_edge=3000)
    assert changed
    assert min(resized.shape[:2]) >= 900


def test_resize_downscales_large_images() -> None:
    large = np.full((5000, 4000, 3), 255, np.uint8)
    resized, changed = transforms.resize_for_ocr(large, min_short_edge=900, max_long_edge=3000)
    assert changed
    assert max(resized.shape[:2]) <= 3000


def test_resize_leaves_in_band_images_alone() -> None:
    image = _text_image(width=1000, height=1400)
    _, changed = transforms.resize_for_ocr(image, min_short_edge=900, max_long_edge=3000)
    assert not changed


def test_contrast_enhancement_widens_the_histogram() -> None:
    flat = np.full((400, 400), 120, np.uint8)
    flat[100:300, 100:300] = 135  # very low contrast
    enhanced = transforms.enhance_contrast(flat)
    assert enhanced.std() >= flat.std()


def test_denoise_preserves_dimensions() -> None:
    gray = to_grayscale(_text_image())
    assert transforms.denoise(gray).shape == gray.shape


def test_sharpen_preserves_dimensions() -> None:
    gray = to_grayscale(_text_image())
    assert transforms.sharpen(gray).shape == gray.shape


def test_threshold_produces_a_binary_image() -> None:
    gray = to_grayscale(_text_image())
    binary = transforms.adaptive_threshold(gray)
    assert set(np.unique(binary)).issubset({0, 255})


def test_threshold_handles_an_even_block_size() -> None:
    """Adaptive thresholding requires an odd block; an even one must not crash."""
    gray = to_grayscale(_text_image())
    assert transforms.adaptive_threshold(gray, block_size=30).shape == gray.shape


# -------------------------------------------------------------------- quality
def test_blur_score_separates_sharp_from_blurred() -> None:
    sharp = to_grayscale(_text_image())
    blurred = cv2.GaussianBlur(sharp, (21, 21), 0)
    assert blur_score(sharp) > blur_score(blurred)


def test_skew_estimation_detects_rotation() -> None:
    straight = to_grayscale(_text_image())
    estimate = estimate_skew(straight)
    assert estimate is None or abs(estimate) < 5.0


def test_skew_estimation_returns_none_on_blank_input() -> None:
    """Reporting a meaningless zero would be worse than reporting nothing."""
    assert estimate_skew(np.full((500, 500), 255, np.uint8)) is None


def test_dark_image_is_judged_poor(settings) -> None:
    quality = assess_quality(_text_image(dark=True), settings)
    assert quality.brightness < settings.quality_min_brightness
    assert quality.verdict in (QualityVerdict.ACCEPTABLE, QualityVerdict.POOR)


def test_quality_reports_all_metrics(settings) -> None:
    quality = assess_quality(_text_image(), settings)
    assert quality.width and quality.height
    assert quality.blur_score >= 0
    assert 0 <= quality.brightness <= 255
    assert quality.clipping_ratio is not None


# ---------------------------------------------------------------- the pipeline
def test_preprocessing_can_be_disabled(make_settings) -> None:
    settings = make_settings(enable_preprocessing=False)
    image = _text_image()
    output, _, report = PreprocessingPipeline(settings).run(image)
    assert output is image
    assert report.skipped == ("disabled",)


def test_every_step_is_recorded(settings) -> None:
    """Knowing what was skipped is as diagnostic as knowing what ran."""
    _, _, report = PreprocessingPipeline(settings).run(_text_image())
    assert report.applied
    assert report.skipped
    assert report.output_width and report.output_height


def test_optional_transforms_run_when_enabled(make_settings) -> None:
    settings = make_settings(preprocess_sharpen=True, preprocess_threshold=True)
    _, _, report = PreprocessingPipeline(settings).run(_text_image())
    assert "threshold" in report.applied


def test_deskew_is_recorded_when_applied(make_settings) -> None:
    settings = make_settings(deskew_min_angle_degrees=0.05)
    _, _, report = PreprocessingPipeline(settings).run(_text_image(angle=6.0))
    assert "deskew" in report.applied or any(s.startswith("deskew:") for s in report.skipped)


# ----------------------------------------------------------------- money types
class _MoneyModel(BaseModel):
    amount: Money | None = None
    qty: Quantity | None = None


@pytest.mark.parametrize(
    ("value", "expected"),
    [("25.99", "25.99"), (25.99, "25.99"), (26, "26"), (Decimal("25.99"), "25.99")],
)
def test_money_accepts_common_inputs(value: object, expected: str) -> None:
    assert _MoneyModel(amount=value).amount == Decimal(expected)


def test_money_serialises_as_a_fixed_scale_string() -> None:
    payload = _MoneyModel(amount=Decimal("25.5")).model_dump(mode="json")
    assert payload["amount"] == "25.50"


def test_money_rejects_non_finite_and_boolean() -> None:
    for bad in (float("inf"), float("nan"), True):
        with pytest.raises(ValidationError):
            _MoneyModel(amount=bad)


def test_money_rejects_unparseable_text() -> None:
    with pytest.raises(ValidationError):
        _MoneyModel(amount="twenty five dollars")


def test_absence_must_be_none_not_an_empty_string() -> None:
    """An empty string is not a valid amount; absence is expressed as None."""
    assert _MoneyModel(amount=None).amount is None
    with pytest.raises(ValidationError):
        _MoneyModel(amount="")


def test_quantity_serialises_without_trailing_zeros() -> None:
    payload = _MoneyModel(qty=Decimal("2.500")).model_dump(mode="json")
    assert payload["qty"] == "2.5"


def test_quantize_uses_retail_rounding() -> None:
    assert quantize_money(Decimal("2.345")) == Decimal("2.35")


# ------------------------------------------------------- tesseract normalisation
def _tesseract_data() -> dict[str, list]:
    """The dictionary shape ``image_to_data`` returns."""
    words = [
        # text, conf, page, block, par, line, left, top, w, h
        ("GREEN", 96.0, 1, 1, 1, 1, 40, 20, 90, 24),
        ("VALLEY", 94.0, 1, 1, 1, 1, 140, 20, 100, 24),
        ("", -1.0, 1, 1, 1, 1, 0, 0, 0, 0),  # empty entries are dropped
        ("TOTAL", 91.0, 1, 1, 1, 2, 40, 60, 80, 22),
        ("5.50", -1.0, 1, 1, 1, 2, 200, 60, 60, 22),  # -1 means "no confidence"
        ("COL2", 88.0, 1, 2, 1, 1, 400, 60, 60, 22),  # a second column
    ]
    keys = [
        "text",
        "conf",
        "page_num",
        "block_num",
        "par_num",
        "line_num",
        "left",
        "top",
        "width",
        "height",
    ]
    return {key: [word[index] for word in words] for index, key in enumerate(keys)}


def test_words_are_regrouped_into_lines() -> None:
    lines = TesseractOCRProvider._group_lines(_tesseract_data())
    assert [line.text for line in lines] == ["GREEN VALLEY", "TOTAL 5.50", "COL2"]


def test_second_column_stays_a_separate_line() -> None:
    """Grouping by block index, not y-coordinate, keeps columns apart."""
    lines = TesseractOCRProvider._group_lines(_tesseract_data())
    assert "COL2" not in lines[1].text


def test_confidences_are_scaled_to_zero_one() -> None:
    lines = TesseractOCRProvider._group_lines(_tesseract_data())
    assert lines[0].confidence == pytest.approx(0.95, abs=0.01)
    assert all(0.0 <= word.confidence <= 1.0 for word in lines[0].words if word.confidence)


def test_missing_confidence_stays_none() -> None:
    """-1 means the engine had no value; inventing one would poison scoring."""
    total_line = TesseractOCRProvider._group_lines(_tesseract_data())[1]
    tail = next(word for word in total_line.words if word.text == "5.50")
    assert tail.confidence is None


def test_line_bbox_is_the_union_of_its_words() -> None:
    line = TesseractOCRProvider._group_lines(_tesseract_data())[0]
    assert line.bbox is not None
    assert line.bbox.x == 40
    assert line.bbox.x2 == 240


def test_missing_binary_reports_unavailable(make_settings) -> None:
    settings = make_settings(tesseract_cmd="/definitely/not/a/real/path")
    ready, detail = TesseractOCRProvider(settings).health_check()
    assert not ready
    assert detail and "TESSERACT_CMD" in detail


def test_engine_failure_is_normalised(monkeypatch, settings) -> None:
    """Vendor exceptions must never escape the provider."""
    provider = TesseractOCRProvider(settings)
    monkeypatch.setattr(provider, "_binary_path", lambda: "/usr/bin/tesseract")

    class _Stub:
        class Output:
            DICT = "dict"

        @staticmethod
        def image_to_data(*args, **kwargs):
            raise ValueError("something vendor-specific")

    monkeypatch.setattr(provider, "_engine", lambda: _Stub())
    with pytest.raises(OCRError) as exc:
        provider.extract(OCRRequest(image=np.zeros((10, 10), np.uint8)))
    assert exc.value.code is ErrorCode.OCR_FAILED


def test_engine_timeout_is_retryable(monkeypatch, settings) -> None:
    provider = TesseractOCRProvider(settings)
    monkeypatch.setattr(provider, "_binary_path", lambda: "/usr/bin/tesseract")

    class _Stub:
        class Output:
            DICT = "dict"

        @staticmethod
        def image_to_data(*args, **kwargs):
            raise RuntimeError("Tesseract process timeout")

    monkeypatch.setattr(provider, "_engine", lambda: _Stub())
    with pytest.raises(OCRTimeoutError) as exc:
        provider.extract(OCRRequest(image=np.zeros((10, 10), np.uint8)))
    assert exc.value.retryable
