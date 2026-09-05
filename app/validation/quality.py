"""Image- and OCR-quality validation.

Turns the measurements taken before and during recognition into caller-visible
warnings. This is what stops the pipeline from returning confident-looking
data extracted from an unreadable photograph: if the image was too dark or the
engine was unsure, the response says so.
"""

from __future__ import annotations

from app.core.config import Settings
from app.normalization.text import contains_suspicious_characters
from app.schemas.ocr import OCRResult
from app.schemas.quality import ImageQuality
from app.schemas.validation import IssueCode, IssueSeverity, ValidationIssue

#: Below this many recognised characters a receipt is effectively unread.
_MIN_TEXT_LENGTH = 20

#: Below this many lines there is not enough structure to extract from.
_MIN_LINE_COUNT = 3


def _issue(
    code: IssueCode,
    severity: IssueSeverity,
    message: str,
    field: str | None = None,
    **context: object,
) -> ValidationIssue:
    return ValidationIssue(
        code=code,
        severity=severity,
        message=message,
        field=field,
        context={k: str(v) for k, v in context.items() if v is not None},
    )


def validate_image_quality(
    quality: ImageQuality | None, settings: Settings
) -> list[ValidationIssue]:
    """Report measured image problems that plausibly degraded recognition."""
    if quality is None:
        return []

    issues: list[ValidationIssue] = []

    if quality.width < settings.min_image_width or quality.height < settings.min_image_height:
        issues.append(
            _issue(
                IssueCode.IMAGE_LOW_RESOLUTION,
                IssueSeverity.WARNING,
                "Image resolution is below the recommended minimum.",
                context_width=quality.width,
                context_height=quality.height,
            )
        )

    if quality.blur_score < settings.quality_min_blur_score:
        issues.append(
            _issue(
                IssueCode.IMAGE_BLURRY,
                IssueSeverity.WARNING,
                "Image appears blurred; recognition accuracy is likely reduced.",
                blur_score=round(quality.blur_score, 2),
            )
        )

    if quality.brightness < settings.quality_min_brightness:
        issues.append(
            _issue(
                IssueCode.IMAGE_TOO_DARK,
                IssueSeverity.WARNING,
                "Image is underexposed.",
                brightness=round(quality.brightness, 1),
            )
        )
    elif quality.brightness > settings.quality_max_brightness:
        issues.append(
            _issue(
                IssueCode.IMAGE_TOO_BRIGHT,
                IssueSeverity.WARNING,
                "Image is overexposed.",
                brightness=round(quality.brightness, 1),
            )
        )

    if quality.contrast < settings.quality_min_contrast:
        issues.append(
            _issue(
                IssueCode.IMAGE_LOW_CONTRAST,
                IssueSeverity.WARNING,
                "Image contrast is low.",
                contrast=round(quality.contrast, 1),
            )
        )

    if (
        quality.skew_angle is not None
        and abs(quality.skew_angle) > settings.deskew_min_angle_degrees * 5
    ):
        issues.append(
            _issue(
                IssueCode.IMAGE_SKEWED,
                IssueSeverity.INFO,
                "Image was noticeably skewed; deskew correction was applied.",
                skew_angle=round(quality.skew_angle, 2),
            )
        )

    return issues


def validate_ocr_quality(ocr: OCRResult | None, settings: Settings) -> list[ValidationIssue]:
    """Report recognition-level problems."""
    if ocr is None:
        return []

    issues: list[ValidationIssue] = []
    mean = ocr.mean_confidence

    if not ocr.has_confidence:
        # The provider reported no confidence at all (vision models do not).
        # Reporting that as *low* confidence would flag every such document
        # for review and render the flag meaningless.
        issues.append(
            _issue(
                IssueCode.OCR_CONFIDENCE_UNAVAILABLE,
                IssueSeverity.INFO,
                "The OCR provider reported no confidence scores, so field "
                "confidence rests on extraction method and validation instead.",
            )
        )
    elif mean and mean < settings.low_ocr_confidence_threshold:
        issues.append(
            _issue(
                IssueCode.OCR_LOW_CONFIDENCE,
                IssueSeverity.WARNING,
                "OCR confidence is low; extracted values may be unreliable.",
                mean_confidence=round(mean, 3),
            )
        )

    text = ocr.text or ""
    if len(text.strip()) < _MIN_TEXT_LENGTH or len(ocr.lines) < _MIN_LINE_COUNT:
        issues.append(
            _issue(
                IssueCode.OCR_SPARSE_TEXT,
                IssueSeverity.WARNING,
                "Very little text was recognised in the document.",
                characters=len(text.strip()),
                lines=len(ocr.lines),
            )
        )

    if contains_suspicious_characters(text):
        issues.append(
            _issue(
                IssueCode.OCR_SUSPICIOUS_CHARACTERS,
                IssueSeverity.WARNING,
                "Recognised text contains unusual character patterns, which "
                "usually indicates the engine was guessing.",
            )
        )

    if not ocr.has_geometry:
        issues.append(
            _issue(
                IssueCode.OCR_NO_GEOMETRY,
                IssueSeverity.INFO,
                "The OCR provider reported no bounding boxes, so spatial "
                "extraction heuristics were unavailable.",
            )
        )

    return issues
