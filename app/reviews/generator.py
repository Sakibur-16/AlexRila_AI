"""Review generation.

Sits on the same :class:`~app.llm.base.LLMProvider` abstraction the receipt
fallback uses, so switching model vendor is a configuration change here too.

Unlike receipt extraction there is nothing to validate arithmetically -- a
review is prose, and prose has no checkable identity. What *is* enforced:

* the model's JSON is parsed and repaired rather than trusted;
* the rating is clamped into 1-5 and, when the caller specified one, forced to
  match it, because a mismatch between the stars shown and the stars requested
  is a bug the caller cannot see;
* sentence count is measured and reported, and the body is trimmed when the
  model overruns;
* ``ai_generated`` is derived from whether real answers were supplied, never
  from anything the model says about itself.
"""

from __future__ import annotations

import json
import re
import time
from typing import Any

from app.core.config import Settings
from app.core.exceptions import LLMError, LLMInvalidOutputError
from app.core.logging import get_logger
from app.core.metrics import MetricNames, increment, observe
from app.core.retry import retry_sync
from app.llm.base import LLMProvider, LLMRequest
from app.reviews.prompts import (
    QUESTIONS_SYSTEM_PROMPT,
    REVIEW_PROMPT_VERSION,
    build_questions_user_prompt,
    build_review_user_prompt,
    system_prompt_for,
)
from app.schemas.review import (
    GeneratedReview,
    ReviewQuestions,
    ReviewQuestionsRequest,
    ReviewRequest,
)

logger = get_logger(__name__)

#: Sentence terminators. Deliberately simple: this counts and trims prose, and
#: a full sentence tokeniser would be a dependency for no practical gain.
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")

#: Shown alongside an ungrounded review. Consuming applications are expected to
#: surface this; publishing invented experience as genuine customer feedback is
#: unlawful in several jurisdictions.
_AI_DISCLOSURE = (
    "AI-generated sample content. No verified customer experience was supplied, "
    "so this must not be presented as a genuine customer review."
)
_GROUNDED_DISCLOSURE = (
    "Written by AI from answers supplied by the reviewer. The experience "
    "described comes from the reviewer, the wording from the model."
)

#: Response shape demanded of the model. Providers that support constrained
#: decoding enforce it; the output is validated on return regardless.
_REVIEW_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["title", "body", "rating"],
    "properties": {
        "title": {"type": "string"},
        "body": {"type": "string"},
        "rating": {"type": "integer", "minimum": 1, "maximum": 5},
    },
}

_QUESTIONS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["questions"],
    "properties": {
        "questions": {"type": "array", "items": {"type": "string"}},
    },
}

#: Rating assumed when neither the caller nor the model supplies a usable one.
#: Three is the only neutral choice; anything else invents a sentiment.
_NEUTRAL_RATING = 3


class ReviewGenerator:
    """Generates product reviews and reviewer questions."""

    def __init__(self, settings: Settings, provider: LLMProvider) -> None:
        self._settings = settings
        self._provider = provider

    # --------------------------------------------------------------- public
    def generate(self, request: ReviewRequest) -> GeneratedReview:
        """Generate a review.

        Raises:
            LLMError: The provider failed.
            LLMInvalidOutputError: The response was not usable JSON.
        """
        started = time.perf_counter()
        increment(MetricNames.LLM_INVOKED, labels={"task": "review"})

        llm_request = LLMRequest(
            system_prompt=system_prompt_for(request),
            document_text=build_review_user_prompt(request),
            json_schema=_REVIEW_SCHEMA,
            temperature=self._settings.review_temperature,
            max_output_tokens=self._settings.review_max_output_tokens,
            timeout_seconds=self._settings.llm_timeout_seconds,
            prompt_version=REVIEW_PROMPT_VERSION,
        )

        response = self._call(llm_request, task="review")
        data = response.data

        title = _as_text(data.get("title")) or f"Review of {request.product_name}"
        body = _as_text(data.get("body"))
        if not body:
            raise LLMInvalidOutputError(
                "The model returned no review text.",
                details={"provider": response.provider},
            )

        body = _trim_to_sentences(body, request.sentences)
        rating = _resolve_rating(request.rating, data.get("rating"))
        duration_ms = (time.perf_counter() - started) * 1000.0
        observe(MetricNames.LLM_LATENCY_MS, duration_ms, labels={"task": "review"})
        increment(MetricNames.LLM_SUCCESS, labels={"task": "review"})

        grounded = request.is_grounded
        logger.info(
            "review_generated",
            grounded=grounded,
            rating=rating,
            sentences=_count_sentences(body),
            duration_ms=round(duration_ms, 2),
        )

        return GeneratedReview(
            title=title,
            body=body,
            rating=rating,
            sentence_count=_count_sentences(body),
            # Derived from the request, never from the model's self-report.
            ai_generated=not grounded,
            disclosure=_GROUNDED_DISCLOSURE if grounded else _AI_DISCLOSURE,
            model=response.model,
            provider=response.provider,
            prompt_version=REVIEW_PROMPT_VERSION,
            generation_time_ms=duration_ms,
        )

    def suggest_questions(self, request: ReviewQuestionsRequest) -> ReviewQuestions:
        """Propose questions to put to a reviewer.

        Answering these and posting them to :meth:`generate` is what turns an
        AI-written review into a grounded one.
        """
        increment(MetricNames.LLM_INVOKED, labels={"task": "questions"})

        llm_request = LLMRequest(
            system_prompt=QUESTIONS_SYSTEM_PROMPT,
            document_text=build_questions_user_prompt(
                request.product_name, request.category, request.count, request.language
            ),
            json_schema=_QUESTIONS_SCHEMA,
            temperature=self._settings.review_temperature,
            max_output_tokens=self._settings.review_max_output_tokens,
            timeout_seconds=self._settings.llm_timeout_seconds,
            prompt_version=REVIEW_PROMPT_VERSION,
        )

        response = self._call(llm_request, task="questions")
        raw = response.data.get("questions")
        questions = [
            text for text in (_as_text(item) for item in raw if isinstance(raw, list)) if text
        ][: request.count]

        if not questions:
            raise LLMInvalidOutputError(
                "The model returned no questions.",
                details={"provider": response.provider},
            )

        increment(MetricNames.LLM_SUCCESS, labels={"task": "questions"})
        return ReviewQuestions(
            product_name=request.product_name,
            questions=tuple(questions),
            model=response.model,
            provider=response.provider,
        )

    # -------------------------------------------------------------- helpers
    def _call(self, request: LLMRequest, *, task: str) -> Any:
        """Invoke the provider with the configured retry policy."""
        try:
            return retry_sync(
                lambda: self._provider.extract(request),
                max_retries=self._settings.llm_max_retries,
                base_delay=0.5,
            )
        except LLMError as exc:
            increment(MetricNames.LLM_FAILURE, labels={"task": task, "code": exc.code.value})
            logger.warning("review_generation_failed", task=task, error_code=exc.code.value)
            raise


def _as_text(value: Any) -> str | None:
    """Accept a non-empty string, collapsing whitespace."""
    if not isinstance(value, str):
        return None
    cleaned = " ".join(value.split())
    return cleaned or None


def _count_sentences(text: str) -> int:
    return len([part for part in _SENTENCE_END.split(text.strip()) if part.strip()])


def _trim_to_sentences(text: str, limit: int) -> str:
    """Cut the body back to ``limit`` sentences.

    Models overrun a stated sentence count often enough that enforcing it here
    is cheaper than re-prompting, and truncating whole sentences keeps the
    prose readable.
    """
    parts = [part for part in _SENTENCE_END.split(text.strip()) if part.strip()]
    if len(parts) <= limit:
        return text.strip()
    return " ".join(parts[:limit]).strip()


def _resolve_rating(requested: int | None, produced: Any) -> int:
    """Settle on the star rating.

    A caller-specified rating always wins: if the UI shows four stars and the
    payload says three, the caller has no way to know which is right.
    """
    if requested is not None:
        return requested
    try:
        value = int(produced)
    except (TypeError, ValueError):
        return _NEUTRAL_RATING
    return max(1, min(5, value))


def parse_json_object(content: str) -> dict[str, Any]:
    """Parse a JSON object from model output, tolerating a markdown fence."""
    stripped = content.strip()
    if stripped.startswith("```"):
        stripped = stripped.split("\n", 1)[-1]
        if stripped.endswith("```"):
            stripped = stripped.rsplit("```", 1)[0]
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError as exc:
        raise LLMInvalidOutputError("Model output was not valid JSON.") from exc
    if not isinstance(parsed, dict):
        raise LLMInvalidOutputError("Model output was not a JSON object.")
    return parsed
