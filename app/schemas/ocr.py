"""Unified OCR representation.

Every provider -- local engine, cloud API, vision LLM -- normalises its native
response into these models. Downstream code depends only on this module, which
is what makes providers swappable.

Providers vary in what they can report. Rather than inventing values to fill
gaps, optional fields stay ``None`` and consumers degrade gracefully: an engine
that reports no bounding boxes still produces usable extraction, only with
spatial heuristics disabled and slightly lower confidence.
"""

from __future__ import annotations

import enum
from typing import Annotated, Any, Self

from pydantic import BaseModel, ConfigDict, Field, computed_field, model_validator

from app.domain.evidence import BoundingBox

Confidence = Annotated[float, Field(ge=0.0, le=1.0)]


class TextOrientation(int, enum.Enum):
    """Detected page rotation, in degrees clockwise."""

    ROTATE_0 = 0
    ROTATE_90 = 90
    ROTATE_180 = 180
    ROTATE_270 = 270


class OCRBox(BaseModel):
    """Serialisable bounding box, ``[x1, y1, x2, y2]`` on the wire."""

    model_config = ConfigDict(frozen=True)

    x: int = Field(ge=0)
    y: int = Field(ge=0)
    width: int = Field(ge=0)
    height: int = Field(ge=0)
    page: int = Field(default=0, ge=0)

    @property
    def x2(self) -> int:
        return self.x + self.width

    @property
    def y2(self) -> int:
        return self.y + self.height

    def to_domain(self) -> BoundingBox:
        return BoundingBox(x=self.x, y=self.y, width=self.width, height=self.height, page=self.page)


class OCRWord(BaseModel):
    """A single recognised token."""

    model_config = ConfigDict(frozen=True)

    text: str
    confidence: Confidence | None = None
    bbox: OCRBox | None = None


class OCRLine(BaseModel):
    """A recognised text line -- the unit receipt extraction reasons about.

    Receipts are line-oriented documents: a label and its value almost always
    share a line, and item rows are lines. Word-level detail is retained for
    per-token confidence and for splitting label from value by x-position.
    """

    model_config = ConfigDict(frozen=True)

    text: str
    confidence: Confidence | None = None
    bbox: OCRBox | None = None
    words: tuple[OCRWord, ...] = ()
    page: int = Field(default=0, ge=0)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def effective_confidence(self) -> float:
        """Line confidence, falling back to the mean of its words.

        Defaults to ``0.5`` when a provider reports no confidence at all --
        an explicit "unknown", not an optimistic assumption.
        """
        if self.confidence is not None:
            return self.confidence
        scored = [w.confidence for w in self.words if w.confidence is not None]
        if scored:
            return sum(scored) / len(scored)
        return 0.5

    @property
    def is_blank(self) -> bool:
        return not self.text.strip()


class OCRPage(BaseModel):
    """One page of a document."""

    model_config = ConfigDict(frozen=True)

    page_number: int = Field(default=0, ge=0)
    width: int | None = Field(default=None, ge=0)
    height: int | None = Field(default=None, ge=0)
    orientation: TextOrientation = TextOrientation.ROTATE_0
    lines: tuple[OCRLine, ...] = ()


class OCRResult(BaseModel):
    """Provider-independent OCR output.

    This is the sole handoff between the OCR layer and everything downstream.
    It is immutable: normalisation produces *new* views rather than mutating
    it, so the original recognition result survives for audit and reprocessing.
    """

    model_config = ConfigDict(frozen=True)

    #: Full document text with lines joined by newlines, exactly as recognised.
    text: str
    lines: tuple[OCRLine, ...] = ()
    pages: tuple[OCRPage, ...] = ()

    provider: str
    #: Engine/model identifier, e.g. "tesseract-5.3.0" or a cloud model name.
    model: str | None = None
    #: Language codes the engine was configured with or detected.
    languages: tuple[str, ...] = ()
    duration_ms: float | None = Field(default=None, ge=0)
    #: Provider-specific extras. Deliberately quarantined here so that no
    #: provider-specific key leaks into the receipt contract.
    provider_metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _ensure_text_present(self) -> Self:
        """Guarantee ``text`` and ``lines`` describe the same content.

        A provider that only fills one of the two would otherwise silently
        halve the information available downstream.
        """
        if not self.text and self.lines:
            object.__setattr__(self, "text", "\n".join(line.text for line in self.lines))
        return self

    @computed_field  # type: ignore[prop-decorator]
    @property
    def mean_confidence(self) -> float:
        """Length-weighted mean line confidence.

        Weighting by text length stops a stray one-character line from
        dominating the document-level score.
        """
        weighted = 0.0
        total_weight = 0.0
        for line in self.lines:
            if line.is_blank:
                continue
            weight = float(len(line.text.strip()))
            weighted += line.effective_confidence * weight
            total_weight += weight
        if total_weight == 0.0:
            return 0.0
        return weighted / total_weight

    @property
    def is_empty(self) -> bool:
        """True when no usable text was recognised."""
        return not self.text.strip()

    @property
    def has_geometry(self) -> bool:
        """Whether spatial reasoning is available for this result."""
        return any(line.bbox is not None for line in self.lines)

    def line_at(self, index: int) -> OCRLine | None:
        """Return the line at ``index``, or ``None`` if out of range."""
        if 0 <= index < len(self.lines):
            return self.lines[index]
        return None
