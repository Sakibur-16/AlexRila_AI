"""Product-review generation endpoints."""

from __future__ import annotations

import time

from fastapi import APIRouter

from app.api.dependencies import RequestIdDep, SettingsDep
from app.core.config import Settings
from app.core.exceptions import ErrorCode, PipelineError
from app.reviews.generator import ReviewGenerator
from app.schemas.response import APIResponse, ResponseProcessing, WarningDetail
from app.schemas.review import (
    GeneratedReview,
    ReviewQuestions,
    ReviewQuestionsRequest,
    ReviewRequest,
)

router = APIRouter(prefix="/reviews", tags=["reviews"])


class ReviewResponse(APIResponse[GeneratedReview]):
    """Response for ``POST /api/v1/reviews/generate``."""


class QuestionsResponse(APIResponse[ReviewQuestions]):
    """Response for ``POST /api/v1/reviews/questions``."""


def _generator(settings: Settings) -> ReviewGenerator:
    """Build a generator, or explain why review generation is unavailable."""
    from app.llm.factory import create_llm_provider

    if not settings.review_enabled:
        raise PipelineError(
            "Review generation is disabled.",
            code=ErrorCode.LLM_UNAVAILABLE,
            http_status=503,
        )
    try:
        provider = create_llm_provider(settings=settings)
    except PipelineError:
        raise
    except Exception as exc:
        raise PipelineError(
            "No language model provider is configured.",
            code=ErrorCode.LLM_UNAVAILABLE,
            http_status=503,
            details={"error_type": type(exc).__name__},
        ) from exc

    ready, detail = provider.health_check()
    if not ready:
        raise PipelineError(
            detail or "The language model provider is not usable.",
            code=ErrorCode.LLM_UNAVAILABLE,
            http_status=503,
        )
    return ReviewGenerator(settings, provider)


@router.post(
    "/generate",
    response_model=ReviewResponse,
    summary="Generate a product review",
    responses={
        200: {"description": "Review generated. Check `data.ai_generated`."},
        422: {"description": "The request failed validation."},
        502: {"description": "The language model failed or returned unusable output."},
        503: {"description": "Review generation is disabled or unconfigured."},
    },
)
async def generate_review(
    settings: SettingsDep,
    request_id: RequestIdDep,
    payload: ReviewRequest,
) -> ReviewResponse:
    """Generate a product review.

    Supplying `answers` produces a review grounded in the reviewer's own
    stated experience. Omitting them makes the model invent the experience,
    and the response is flagged `ai_generated: true` with a `disclosure`
    string. **Publishing an ungenerated-from-experience review as genuine
    customer feedback is unlawful in several jurisdictions** -- label it, or
    supply answers.
    """
    started = time.perf_counter()
    review = _generator(settings).generate(payload)

    warnings: tuple[WarningDetail, ...] = ()
    if review.ai_generated:
        warnings = (
            WarningDetail(
                code="AI_GENERATED_CONTENT",
                message=review.disclosure,
                field="body",
            ),
        )

    return ReviewResponse(
        success=True,
        data=review,
        warnings=warnings,
        errors=(),
        processing=ResponseProcessing(
            request_id=request_id,
            processing_time_ms=round((time.perf_counter() - started) * 1000.0, 2),
        ),
    )


@router.post(
    "/questions",
    response_model=QuestionsResponse,
    summary="Suggest questions to ask a reviewer",
    responses={
        200: {"description": "Questions generated."},
        502: {"description": "The language model failed or returned unusable output."},
        503: {"description": "Review generation is disabled or unconfigured."},
    },
)
async def suggest_questions(
    settings: SettingsDep,
    request_id: RequestIdDep,
    payload: ReviewQuestionsRequest,
) -> QuestionsResponse:
    """Suggest questions to put to a reviewer about a product.

    Ask these in your UI, collect the answers, and post them to
    `/reviews/generate` as `answers`. That path produces a better review and
    one that is legally safe to publish.
    """
    started = time.perf_counter()
    questions = _generator(settings).suggest_questions(payload)

    return QuestionsResponse(
        success=True,
        data=questions,
        warnings=(),
        errors=(),
        processing=ResponseProcessing(
            request_id=request_id,
            processing_time_ms=round((time.perf_counter() - started) * 1000.0, 2),
        ),
    )
