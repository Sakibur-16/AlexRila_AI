"""The canonical pipeline result.

:class:`PipelineResult` is the single handoff between the AI pipeline and any
consumer -- HTTP API, batch job, evaluation harness. Nothing downstream of the
pipeline should reach into stage internals; if a consumer needs something, it
belongs on this object.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from app.schemas.ocr import OCRResult
from app.schemas.quality import ImageQuality, PreprocessingReport
from app.schemas.receipt import ConfidenceReport, ProcessingMetadata, Receipt
from app.schemas.validation import ValidationResult


class PipelineResult(BaseModel):
    """Everything one document produced.

    ``receipt`` already embeds validation, confidence and processing metadata
    because that is the contract a backend consumes. They are *also* exposed
    at the top level for internal consumers (evaluation, review tooling) that
    work with the result object rather than its JSON projection.
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    receipt: Receipt
    #: Full OCR output. Excluded from HTTP responses; retained in-process for
    #: evaluation, debugging and LLM reprocessing.
    ocr_result: OCRResult | None = Field(default=None, exclude=True)
    quality: ImageQuality | None = None
    preprocessing: PreprocessingReport | None = None
    validation: ValidationResult = Field(default_factory=ValidationResult)
    confidence: ConfidenceReport = Field(default_factory=ConfidenceReport)
    processing: ProcessingMetadata | None = None

    @property
    def succeeded(self) -> bool:
        """True when a result was produced, regardless of certainty.

        Uncertainty is not failure: a receipt extracted with warnings is a
        successful outcome that the caller must handle thoughtfully.
        """
        return self.receipt is not None

    @property
    def requires_review(self) -> bool:
        return self.receipt.review.review_required
