"""Conditional preprocessing pipeline.

The governing rule: **apply a transform only when the measured quality says it
will help.** Blindly running every enhancement degrades OCR on images that
were already fine, which is the most common way a preprocessing stage makes a
document pipeline worse rather than better.

Each decision below is therefore driven by a metric from
:mod:`app.preprocessing.quality`, and every applied *and skipped* step is
recorded in the :class:`~app.schemas.quality.PreprocessingReport` so that a bad
extraction can be traced to the image it was actually given.
"""

from __future__ import annotations

import time

import numpy as np

from app.core.config import Settings
from app.core.logging import get_logger
from app.preprocessing import transforms
from app.preprocessing.quality import assess_quality
from app.schemas.quality import ImageQuality, PreprocessingReport

logger = get_logger(__name__)


class PreprocessingPipeline:
    """Decides and applies image preparation for OCR."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    def run(self, image: np.ndarray) -> tuple[np.ndarray, ImageQuality, PreprocessingReport]:
        """Prepare ``image`` for recognition.

        Args:
            image: Decoded image, BGR or grayscale.

        Returns:
            ``(prepared_image, quality, report)``. ``quality`` describes the
            *input*, since that is what the caller needs to judge the result.
        """
        started = time.perf_counter()
        settings = self._settings

        quality = assess_quality(image, settings)
        applied: list[str] = []
        skipped: list[str] = []
        rotation = 0.0

        if not settings.enable_preprocessing:
            return (
                image,
                quality,
                PreprocessingReport(
                    applied=(),
                    skipped=("disabled",),
                    duration_ms=(time.perf_counter() - started) * 1000,
                    output_width=image.shape[1],
                    output_height=image.shape[0],
                ),
            )

        working = image

        # --- geometry first: every later metric is measured on straight text.
        if settings.preprocess_deskew:
            angle = quality.skew_angle
            if angle is None:
                skipped.append("deskew:no_estimate")
            elif abs(angle) < settings.deskew_min_angle_degrees:
                skipped.append("deskew:below_threshold")
            else:
                working = transforms.deskew(working, angle)
                rotation = angle
                applied.append("deskew")
        else:
            skipped.append("deskew:disabled")

        # --- resolution: engines have a working band; move into it.
        working, resized = transforms.resize_for_ocr(
            working,
            min_short_edge=settings.preprocess_min_short_edge,
            max_long_edge=settings.preprocess_max_long_edge,
        )
        (applied if resized else skipped).append("resize" if resized else "resize:not_needed")

        # --- grayscale: colour carries no information for text recognition.
        if settings.preprocess_grayscale:
            working = transforms.to_grayscale(working)
            applied.append("grayscale")
        else:
            skipped.append("grayscale:disabled")

        # --- denoise only when the image is soft: on a sharp image, non-local
        # means smooths glyph edges and costs accuracy.
        if settings.preprocess_denoise:
            if quality.blur_score < settings.quality_min_blur_score * 3:
                working = transforms.denoise(working)
                applied.append("denoise")
            else:
                skipped.append("denoise:image_already_sharp")
        else:
            skipped.append("denoise:disabled")

        # --- contrast only when it is actually low or exposure is off.
        if settings.preprocess_contrast:
            needs_contrast = (
                quality.contrast < settings.quality_min_contrast
                or quality.brightness < settings.quality_min_brightness
                or quality.brightness > settings.quality_max_brightness
            )
            if needs_contrast:
                working = transforms.enhance_contrast(working)
                applied.append("contrast")
            else:
                skipped.append("contrast:not_needed")
        else:
            skipped.append("contrast:disabled")

        # --- sharpen only a genuinely blurry image, and never after
        # thresholding, which would amplify binarisation artefacts.
        if settings.preprocess_sharpen:
            if quality.blur_score < settings.quality_min_blur_score:
                working = transforms.sharpen(working)
                applied.append("sharpen")
            else:
                skipped.append("sharpen:not_needed")
        else:
            skipped.append("sharpen:disabled")

        # --- binarise last, and only when explicitly enabled: it discards the
        # greyscale detail that modern engines are trained on.
        if settings.preprocess_threshold:
            working = transforms.adaptive_threshold(working)
            applied.append("threshold")
        else:
            skipped.append("threshold:disabled")

        report = PreprocessingReport(
            applied=tuple(applied),
            skipped=tuple(skipped),
            duration_ms=(time.perf_counter() - started) * 1000,
            output_width=int(working.shape[1]),
            output_height=int(working.shape[0]),
            rotation_applied_degrees=rotation,
        )

        logger.debug(
            "preprocessing_complete",
            applied=list(applied),
            verdict=quality.verdict.value,
            duration_ms=round(report.duration_ms, 2),
        )
        return working, quality, report
