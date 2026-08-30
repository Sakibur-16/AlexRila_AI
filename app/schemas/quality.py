"""Image quality assessment contract.

Quality metrics are computed *before* OCR and are reported alongside the
result. They serve two purposes: they steer which preprocessing operations are
worth applying, and they let a consumer distinguish "this receipt is genuinely
missing a total" from "we could barely read this photo".
"""

from __future__ import annotations

import enum

from pydantic import BaseModel, ConfigDict, Field


class QualityVerdict(enum.StrEnum):
    """Overall readability judgement."""

    GOOD = "good"
    ACCEPTABLE = "acceptable"
    POOR = "poor"


class ImageQuality(BaseModel):
    """Measured properties of the submitted image.

    All scores are reported even when they pass their thresholds, so that
    operators can tune thresholds against real traffic rather than guesswork.
    """

    model_config = ConfigDict(frozen=True)

    width: int = Field(ge=0)
    height: int = Field(ge=0)
    #: Variance of the Laplacian. Higher is sharper; scale is resolution
    #: dependent, which is why it is compared against a configured threshold
    #: rather than an absolute standard.
    blur_score: float = Field(ge=0)
    #: Mean luminance, 0-255.
    brightness: float = Field(ge=0, le=255)
    #: Standard deviation of luminance; a proxy for contrast.
    contrast: float = Field(ge=0)
    #: Estimated skew in degrees; positive is counter-clockwise.
    skew_angle: float | None = None
    #: Fraction of pixels that are near-saturated at either end, 0-1.
    clipping_ratio: float | None = Field(default=None, ge=0, le=1)
    verdict: QualityVerdict = QualityVerdict.GOOD

    @property
    def megapixels(self) -> float:
        return (self.width * self.height) / 1_000_000


class PreprocessingReport(BaseModel):
    """What the preprocessing pipeline actually did.

    Recorded because preprocessing is conditional: knowing that deskew ran but
    denoise did not is essential when diagnosing a bad extraction.
    """

    model_config = ConfigDict(frozen=True)

    applied: tuple[str, ...] = ()
    skipped: tuple[str, ...] = ()
    duration_ms: float = Field(default=0.0, ge=0)
    #: Dimensions handed to OCR, after any resize.
    output_width: int | None = Field(default=None, ge=0)
    output_height: int | None = Field(default=None, ge=0)
    rotation_applied_degrees: float = 0.0
