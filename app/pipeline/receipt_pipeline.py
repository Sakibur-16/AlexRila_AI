"""The receipt processing pipeline.

Single entry point from input bytes to :class:`PipelineResult`:

.. code-block:: text

    validate -> decode -> quality -> preprocess -> OCR -> extract
             -> validate -> score -> [LLM fallback -> re-validate -> re-score]
             -> assemble

Two properties are load-bearing.

**Uncertainty is not failure.** Only an unusable *input* or a total inability
to recognise text raises. A receipt that was read but does not add up returns
successfully with warnings, because that is a real answer a consumer can act on.

**Every stage is timed and counted.** Stage timings ride along in the response's
processing metadata, so a latency regression can be attributed to a stage
without reproducing it locally.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from app.confidence.scorer import ConfidenceScorer, ScoredConfidence
from app.core.config import Settings, get_settings
from app.core.exceptions import ErrorCode, OCRError, PipelineError
from app.core.logging import get_logger
from app.core.metrics import MetricNames, increment, observe
from app.core.retry import retry_sync
from app.domain.document import DocumentInput
from app.extraction.receipt import ExtractionResult, RuleBasedReceiptExtractor
from app.llm.extractor import LLMFallbackExtractor, LLMFallbackOutcome
from app.ocr.base import OCRProvider, OCRRequest
from app.pipeline.assembly import build_processing_metadata, build_receipt
from app.preprocessing.pipeline import PreprocessingPipeline
from app.preprocessing.transforms import decode_image, ensure_size_limits
from app.schemas.ocr import OCRResult
from app.schemas.pipeline import PipelineResult
from app.schemas.quality import ImageQuality, PreprocessingReport
from app.schemas.validation import ValidationResult
from app.security.files import validate_upload
from app.validation.engine import ValidationEngine

logger = get_logger(__name__)


@dataclass(slots=True)
class _StageTimer:
    """Accumulates per-stage timings in milliseconds."""

    timings: dict[str, float]

    def record(self, stage: str, started: float) -> None:
        self.timings[stage] = (time.perf_counter() - started) * 1000.0


class ReceiptPipeline:
    """Orchestrates every stage of receipt understanding.

    Constructed once per process and shared across requests. Its collaborators
    are injected, so a test can substitute an OCR provider without patching
    module globals.
    """

    def __init__(
        self,
        *,
        settings: Settings | None = None,
        ocr_provider: OCRProvider,
        llm_extractor: LLMFallbackExtractor | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._ocr = ocr_provider
        self._llm = llm_extractor
        self._preprocessor = PreprocessingPipeline(self._settings)
        self._extractor = RuleBasedReceiptExtractor(self._settings)
        self._validator = ValidationEngine(self._settings)
        self._scorer = ConfidenceScorer(self._settings)

    # ---------------------------------------------------------------- public
    def process(self, document: DocumentInput, *, request_id: str | None = None) -> PipelineResult:
        """Process one document end to end.

        Args:
            document: The submitted document.
            request_id: Correlation id echoed into processing metadata.

        Returns:
            The canonical result.

        Raises:
            PipelineError: The input was unusable or recognition failed
                outright. Uncertain results do not raise.
        """
        overall_started = time.perf_counter()
        timer = _StageTimer(timings={})
        increment(MetricNames.PIPELINE_REQUESTS)

        try:
            result = self._run(document, request_id, timer, overall_started)
        except PipelineError as exc:
            increment(MetricNames.PIPELINE_FAILURE, labels={"code": exc.code.value})
            observe(
                MetricNames.PIPELINE_LATENCY_MS,
                (time.perf_counter() - overall_started) * 1000.0,
                labels={"outcome": "failure"},
            )
            logger.warning(
                "pipeline_failed",
                error_code=exc.code.value,
                stage_timings_ms=timer.timings,
            )
            raise

        increment(MetricNames.PIPELINE_SUCCESS)
        observe(
            MetricNames.PIPELINE_LATENCY_MS,
            (time.perf_counter() - overall_started) * 1000.0,
            labels={"outcome": "success"},
        )
        return result

    # --------------------------------------------------------------- stages
    def _run(
        self,
        document: DocumentInput,
        request_id: str | None,
        timer: _StageTimer,
        overall_started: float,
    ) -> PipelineResult:
        settings = self._settings

        # --- input validation -------------------------------------------
        started = time.perf_counter()
        media_type, notes = validate_upload(
            document.content,
            settings=settings,
            declared_media_type=document.media_type,
            filename=document.filename,
        )
        image = decode_image(document.content)
        ensure_size_limits(
            image,
            min_width=settings.min_image_width,
            min_height=settings.min_image_height,
            max_pixels=settings.max_image_pixels,
        )
        timer.record("input_validation", started)

        # --- preprocessing ------------------------------------------------
        started = time.perf_counter()
        prepared, quality, preprocessing = self._preprocess(image)
        timer.record("preprocessing", started)
        observe(MetricNames.PREPROCESS_LATENCY_MS, preprocessing.duration_ms)

        # --- OCR ----------------------------------------------------------
        started = time.perf_counter()
        ocr = self._recognize(prepared, document)
        timer.record("ocr", started)
        observe(MetricNames.OCR_LATENCY_MS, timer.timings["ocr"])
        observe(MetricNames.OCR_CONFIDENCE, ocr.mean_confidence)

        # --- extraction ---------------------------------------------------
        started = time.perf_counter()
        extraction = self._extractor.extract(ocr)
        timer.record("extraction", started)
        observe(MetricNames.EXTRACTION_LATENCY_MS, timer.timings["extraction"])

        # --- validate and score -------------------------------------------
        started = time.perf_counter()
        validation = self._validator.validate(extraction, ocr=ocr, quality=quality)
        scored = self._scorer.score(
            extraction,
            validation,
            ocr_confidence=ocr.mean_confidence,
            ocr_confidence_measured=ocr.has_confidence,
        )
        timer.record("validation", started)
        observe(MetricNames.VALIDATION_LATENCY_MS, timer.timings["validation"])

        # --- optional LLM fallback ----------------------------------------
        llm_used = False
        llm_provider_name: str | None = None
        llm_model: str | None = None
        prompt_version: str | None = None

        if self._llm is not None and self._llm.should_invoke(scored.report.overall, extraction):
            started = time.perf_counter()
            extraction, validation, scored, outcome = self._run_llm_fallback(
                extraction, ocr, quality
            )
            timer.record("llm_fallback", started)
            llm_used = outcome.used
            llm_provider_name = outcome.provider
            llm_model = outcome.model
            prompt_version = outcome.prompt_version

        if scored.report.overall < settings.confidence_threshold:
            increment(MetricNames.LOW_CONFIDENCE)
        if scored.review_required:
            increment(MetricNames.REVIEW_REQUIRED)

        # --- assemble ------------------------------------------------------
        total_ms = (time.perf_counter() - overall_started) * 1000.0
        processing = build_processing_metadata(
            request_id=request_id,
            ocr=ocr,
            preprocessing_applied=preprocessing.applied,
            stage_timings_ms=timer.timings,
            total_ms=total_ms,
            llm_used=llm_used,
            llm_provider=llm_provider_name,
            llm_model=llm_model,
            prompt_version=prompt_version,
        )

        receipt = build_receipt(
            extraction,
            settings=settings,
            ocr=ocr,
            validation=validation,
            confidence=scored.report,
            processing=processing,
            review_required=scored.review_required,
            review_reasons=scored.review_reasons,
        )

        logger.info(
            "pipeline_complete",
            media_type=media_type,
            input_notes=notes or None,
            ocr_provider=ocr.provider,
            ocr_confidence=round(ocr.mean_confidence, 3),
            overall_confidence=round(scored.report.overall, 3),
            items=len(extraction.items),
            is_valid=validation.is_valid,
            warnings=len(validation.warnings),
            errors=len(validation.errors),
            review_required=scored.review_required,
            llm_used=llm_used,
            processing_time_ms=round(total_ms, 2),
        )

        return PipelineResult(
            receipt=receipt,
            ocr_result=ocr,
            quality=quality,
            preprocessing=preprocessing,
            validation=validation,
            confidence=scored.report,
            processing=processing,
        )

    def _preprocess(
        self, image: np.ndarray
    ) -> tuple[np.ndarray, ImageQuality, PreprocessingReport]:
        """Run preprocessing, degrading to the original image on failure.

        A preprocessing failure must not fail the request: the unprocessed
        image is very likely still recognisable, and returning a slightly worse
        result beats returning none.
        """
        try:
            return self._preprocessor.run(image)
        except Exception as exc:
            logger.warning("preprocessing_failed", error_type=type(exc).__name__)
            height, width = image.shape[:2]
            return (
                image,
                ImageQuality(
                    width=int(width),
                    height=int(height),
                    blur_score=0.0,
                    brightness=0.0,
                    contrast=0.0,
                ),
                PreprocessingReport(applied=(), skipped=("failed",)),
            )

    def _recognize(self, prepared: np.ndarray, document: DocumentInput) -> OCRResult:
        """Run OCR with the configured retry policy."""
        request = OCRRequest(
            image=prepared,
            languages=self._settings.ocr_languages,
            timeout_seconds=self._settings.ocr_timeout_seconds,
            original_bytes=document.content,
            hints={
                "document_type": document.document_type.value,
                **(document.metadata or {}),
            },
        )

        def on_retry(attempt: int, exc: BaseException) -> None:
            increment(
                MetricNames.OCR_RETRY,
                labels={"code": getattr(exc, "code", ErrorCode.OCR_FAILED).value},
            )
            del attempt

        try:
            result = retry_sync(
                lambda: self._ocr.extract(request),
                max_retries=self._settings.ocr_max_retries,
                base_delay=self._settings.ocr_retry_base_delay_seconds,
                on_retry=on_retry,
            )
        except PipelineError as exc:
            increment(MetricNames.OCR_FAILURE, labels={"code": exc.code.value})
            raise

        if result.is_empty:
            increment(MetricNames.OCR_FAILURE, labels={"code": ErrorCode.OCR_EMPTY_RESULT.value})
            raise OCRError(
                "No text could be recognised in the document.",
                code=ErrorCode.OCR_EMPTY_RESULT,
                details={"provider": result.provider},
            )

        increment(MetricNames.OCR_SUCCESS, labels={"provider": result.provider})
        return result

    def _run_llm_fallback(
        self,
        extraction: ExtractionResult,
        ocr: OCRResult,
        quality: ImageQuality | None,
    ) -> tuple[ExtractionResult, ValidationResult, ScoredConfidence, LLMFallbackOutcome]:
        """Run the LLM fallback and re-validate whatever it contributed.

        Re-validation is unconditional: LLM-sourced values are subject to
        exactly the same arithmetic and semantic checks as deterministic ones.
        If the merge made the result *worse*, the deterministic version is kept.
        """
        assert self._llm is not None
        baseline_confidence = None

        outcome = self._llm.run(extraction, ocr.text)
        if not outcome.used:
            validation = self._validator.validate(extraction, ocr=ocr, quality=quality)
            scored = self._scorer.score(
                extraction,
                validation,
                ocr_confidence=ocr.mean_confidence,
                ocr_confidence_measured=ocr.has_confidence,
            )
            return extraction, validation, scored, outcome

        baseline_validation = self._validator.validate(extraction, ocr=ocr, quality=quality)
        baseline_confidence = self._scorer.score(
            extraction,
            baseline_validation,
            ocr_confidence=ocr.mean_confidence,
            ocr_confidence_measured=ocr.has_confidence,
        )

        merged_validation = self._validator.validate(outcome.extraction, ocr=ocr, quality=quality)
        merged_confidence = self._scorer.score(
            outcome.extraction,
            merged_validation,
            ocr_confidence=ocr.mean_confidence,
            ocr_confidence_measured=ocr.has_confidence,
        )

        if merged_confidence.report.overall < baseline_confidence.report.overall:
            logger.info(
                "llm_fallback_discarded",
                reason="lower_confidence_after_merge",
                baseline=round(baseline_confidence.report.overall, 3),
                merged=round(merged_confidence.report.overall, 3),
            )
            return extraction, baseline_validation, baseline_confidence, outcome

        logger.info(
            "llm_fallback_applied",
            filled_fields=list(outcome.filled_fields),
            rejected_fields=list(outcome.rejected_fields) or None,
        )
        return outcome.extraction, merged_validation, merged_confidence, outcome
