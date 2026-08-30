"""Image quality assessment.

Metrics are computed once, before preprocessing, and drive two decisions:
which transforms are worth applying, and which warnings the caller receives.

The measurements are deliberately cheap (single-pass statistics on a
downscaled grayscale copy) because they run on every request. Absolute values
are meaningless in isolation -- they are compared against configured thresholds
so that operators can tune against their own traffic.
"""

from __future__ import annotations

import cv2
import numpy as np

from app.core.config import Settings
from app.schemas.quality import ImageQuality, QualityVerdict

#: Analysis is done at this width; quality statistics are scale sensitive, so
#: normalising the width keeps thresholds comparable across upload resolutions.
_ANALYSIS_WIDTH = 1000

#: Luminance values at or beyond these are treated as clipped.
_CLIP_LOW = 8
_CLIP_HIGH = 247


def to_grayscale(image: np.ndarray) -> np.ndarray:
    """Return a single-channel view of ``image``."""
    if image.ndim == 2:
        return image
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def _analysis_copy(gray: np.ndarray) -> np.ndarray:
    """Downscale to :data:`_ANALYSIS_WIDTH` for scale-stable statistics."""
    height, width = gray.shape[:2]
    if width <= _ANALYSIS_WIDTH:
        return gray
    scale = _ANALYSIS_WIDTH / width
    return cv2.resize(
        gray, (_ANALYSIS_WIDTH, max(1, int(height * scale))), interpolation=cv2.INTER_AREA
    )


def blur_score(gray: np.ndarray) -> float:
    """Variance of the Laplacian: a standard sharpness proxy.

    Low variance means few sharp edges, which on a document photo means blur.
    """
    return float(cv2.Laplacian(_analysis_copy(gray), cv2.CV_64F).var())


def estimate_skew(gray: np.ndarray, *, max_angle: float = 20.0) -> float | None:
    """Estimate text skew in degrees.

    Uses the minimum-area rectangle of thresholded foreground pixels, which is
    robust on receipts because their text forms one dominant elongated mass.
    Returns ``None`` when there is too little foreground to judge, rather than
    reporting a meaningless zero.
    """
    small = _analysis_copy(gray)
    inverted = cv2.threshold(small, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    coords = cv2.findNonZero(inverted)
    if coords is None or len(coords) < 50:
        return None

    angle = cv2.minAreaRect(coords)[-1]
    # OpenCV reports the angle in (0, 90]; map it to a signed deviation from
    # horizontal so that a 2-degree tilt does not read as an 88-degree one.
    if angle > 45:
        angle -= 90
    if abs(angle) > max_angle:
        return None
    return float(angle)


def assess_quality(image: np.ndarray, settings: Settings) -> ImageQuality:
    """Measure ``image`` and assign an overall verdict.

    The verdict is intentionally coarse: ``POOR`` means at least two
    independent metrics failed, which is a far better predictor of bad OCR
    than any single metric crossing a threshold.
    """
    gray = to_grayscale(image)
    height, width = gray.shape[:2]
    analysis = _analysis_copy(gray)

    brightness = float(analysis.mean())
    contrast = float(analysis.std())
    sharpness = blur_score(gray)
    skew = estimate_skew(gray, max_angle=settings.deskew_max_angle_degrees)

    total_pixels = analysis.size
    clipped = int(np.count_nonzero(analysis <= _CLIP_LOW)) + int(
        np.count_nonzero(analysis >= _CLIP_HIGH)
    )
    clipping_ratio = clipped / total_pixels if total_pixels else 0.0

    failures = 0
    if width < settings.min_image_width or height < settings.min_image_height:
        failures += 1
    if sharpness < settings.quality_min_blur_score:
        failures += 1
    if brightness < settings.quality_min_brightness or brightness > settings.quality_max_brightness:
        failures += 1
    if contrast < settings.quality_min_contrast:
        failures += 1

    if failures == 0:
        verdict = QualityVerdict.GOOD
    elif failures == 1:
        verdict = QualityVerdict.ACCEPTABLE
    else:
        verdict = QualityVerdict.POOR

    return ImageQuality(
        width=int(width),
        height=int(height),
        blur_score=sharpness,
        brightness=brightness,
        contrast=contrast,
        skew_angle=skew,
        clipping_ratio=clipping_ratio,
        verdict=verdict,
    )
