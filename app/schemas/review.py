"""Product-review contracts.

A second, independent capability alongside receipt extraction. It shares the
LLM provider abstraction, the response envelope and the error taxonomy, but
nothing else -- a review has no OCR, no arithmetic to validate and no
document to cite.

One field here carries legal weight rather than technical: ``ai_generated``.
Presenting a fabricated review as genuine customer feedback is prohibited by
the FTC in the United States and by comparable consumer-protection rules
elsewhere. The flag is always present and always accurate so that the
consuming application can label the content, and so that "we didn't know"
is never available as an explanation.
"""

from __future__ import annotations

import enum
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field

from app.core.versions import SCHEMA_VERSION

#: Sentence bounds for the generated body. Five reads as a natural review;
#: fewer feels curt, more drifts into marketing copy.
MIN_SENTENCES = 4
MAX_SENTENCES = 6
DEFAULT_SENTENCES = 5


class ReviewTone(enum.StrEnum):
    """Voice the review should adopt."""

    BALANCED = "balanced"
    POSITIVE = "positive"
    CRITICAL = "critical"
    ENTHUSIASTIC = "enthusiastic"
    PROFESSIONAL = "professional"


class ReviewAnswer(BaseModel):
    """One question and the user's own answer to it.

    Supplying these is what makes a review *grounded* -- written from a real
    person's stated experience rather than invented from the product name.
    """

    model_config = ConfigDict(frozen=True)

    question: Annotated[str, Field(min_length=1, max_length=500)]
    answer: Annotated[str, Field(min_length=1, max_length=2000)]


class ReviewRequest(BaseModel):
    """A request to generate a product review."""

    model_config = ConfigDict(frozen=True)

    product_name: Annotated[str, Field(min_length=1, max_length=200)]
    category: Annotated[str | None, Field(default=None, max_length=100)] = None
    description: Annotated[str | None, Field(default=None, max_length=2000)] = None
    #: Attributes worth mentioning: brand, price, colour, size, material.
    attributes: dict[str, str] = Field(default_factory=dict)

    #: The user's own answers. When present the review is grounded in them and
    #: ``ai_generated`` is false; when absent the model invents the experience
    #: and the response says so.
    answers: tuple[ReviewAnswer, ...] = ()

    #: Star rating the user wants reflected, 1-5. When omitted the model infers
    #: one from the answers, or from the requested tone.
    rating: Annotated[int | None, Field(default=None, ge=1, le=5)] = None

    tone: ReviewTone = ReviewTone.BALANCED
    sentences: Annotated[int, Field(ge=MIN_SENTENCES, le=MAX_SENTENCES)] = DEFAULT_SENTENCES
    language: Annotated[str, Field(min_length=2, max_length=32)] = "English"

    @property
    def is_grounded(self) -> bool:
        """Whether the review will be based on real user input."""
        return bool(self.answers)


class GeneratedReview(BaseModel):
    """A generated product review."""

    model_config = ConfigDict(frozen=True)

    schema_version: str = SCHEMA_VERSION

    title: str
    body: str
    #: 1-5. Echoes the requested rating when one was given, otherwise inferred.
    rating: Annotated[int, Field(ge=1, le=5)]
    sentence_count: int = Field(ge=0)

    #: **True when no user answers were supplied**, meaning the experience
    #: described is invented rather than reported. Publishing such a review as
    #: genuine customer feedback is unlawful in several jurisdictions; label it.
    ai_generated: bool
    #: Human-readable statement of the above, suitable for surfacing in a UI.
    disclosure: str

    #: Diagnostics: which model and prompt produced this.
    model: str | None = None
    provider: str | None = None
    prompt_version: str | None = None
    generation_time_ms: float = Field(default=0.0, ge=0)


class ReviewQuestionsRequest(BaseModel):
    """A request for questions to put to a reviewer."""

    model_config = ConfigDict(frozen=True)

    product_name: Annotated[str, Field(min_length=1, max_length=200)]
    category: Annotated[str | None, Field(default=None, max_length=100)] = None
    count: Annotated[int, Field(ge=1, le=10)] = 5
    language: Annotated[str, Field(min_length=2, max_length=32)] = "English"


class ReviewQuestions(BaseModel):
    """Questions to put to a reviewer.

    Answering these and posting them back to the generate endpoint produces a
    grounded review, which is both better writing and legally safe to publish.
    """

    model_config = ConfigDict(frozen=True)

    product_name: str
    questions: tuple[str, ...]
    model: str | None = None
    provider: str | None = None
