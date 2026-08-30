"""Evidence-carrying field values.

Every value the pipeline produces is wrapped in an :class:`ExtractedField`,
which pairs the value with *where it came from* and *how it was obtained*.
This is the mechanism that makes hallucination structurally difficult: a field
with no evidence cannot be produced by the deterministic extractors at all, and
LLM-produced fields must cite a source line that actually exists in the OCR
text before they are accepted.

The public API can render these as plain values (default) or with evidence
attached (``include_evidence``), so the internal model stays rich without
forcing every consumer to deal with it.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any


class ExtractionMethod(enum.StrEnum):
    """How a value was obtained.

    The ordering of the associated priors in :data:`METHOD_PRIOR` encodes a
    deliberate belief: an explicitly labelled match ("TOTAL 12.34") is far more
    trustworthy than a positional guess ("the last number on the receipt").
    """

    #: Matched an explicit keyword label plus a value on the same line.
    KEYWORD_ANCHORED = "keyword_anchored"
    #: Matched a strict regular expression (phone, tax id, card mask).
    REGEX = "regex"
    #: Keyword on one line, value on a neighbouring line (spatial association).
    SPATIAL = "spatial"
    #: Derived from other extracted values (e.g. subtotal from item sum).
    DERIVED = "derived"
    #: Positional or structural heuristic with no explicit label.
    HEURISTIC = "heuristic"
    #: Produced by the LLM fallback layer.
    LLM = "llm"
    #: Supplied by configuration (e.g. default currency), not read from the
    #: document. Deliberately low-confidence.
    CONFIGURED = "configured"


#: Prior confidence in each extraction method, before OCR quality is folded in.
#: Documented in ``docs/confidence.md``; these are the only "magic numbers" in
#: the confidence system and they live in exactly one place.
METHOD_PRIOR: dict[ExtractionMethod, float] = {
    ExtractionMethod.KEYWORD_ANCHORED: 0.95,
    ExtractionMethod.REGEX: 0.92,
    ExtractionMethod.SPATIAL: 0.80,
    ExtractionMethod.DERIVED: 0.75,
    ExtractionMethod.HEURISTIC: 0.60,
    ExtractionMethod.LLM: 0.70,
    ExtractionMethod.CONFIGURED: 0.40,
}


@dataclass(frozen=True, slots=True)
class BoundingBox:
    """Axis-aligned box in source-image pixel coordinates.

    Coordinates refer to the image *as submitted to OCR* (i.e. after
    preprocessing). ``page`` is zero-based.
    """

    x: int
    y: int
    width: int
    height: int
    page: int = 0

    @property
    def x2(self) -> int:
        return self.x + self.width

    @property
    def y2(self) -> int:
        return self.y + self.height

    def as_list(self) -> list[int]:
        """Return ``[x1, y1, x2, y2]``, the conventional wire format."""
        return [self.x, self.y, self.x2, self.y2]

    def union(self, other: BoundingBox) -> BoundingBox:
        """Return the smallest box containing both. Pages must match."""
        x1 = min(self.x, other.x)
        y1 = min(self.y, other.y)
        x2 = max(self.x2, other.x2)
        y2 = max(self.y2, other.y2)
        return BoundingBox(x=x1, y=y1, width=x2 - x1, height=y2 - y1, page=self.page)


@dataclass(frozen=True, slots=True)
class Evidence:
    """Provenance for a single extracted value.

    Attributes:
        source_text: The exact OCR text the value was read from, unmodified.
        line_index: Index into :attr:`~app.schemas.ocr.OCRResult.lines`.
        bbox: Location on the page, when the provider reported one.
        ocr_confidence: Provider confidence for the source span, ``0..1``.
        method: How the value was derived from ``source_text``.
        notes: Free-form diagnostic detail (which pattern matched, etc.).
    """

    source_text: str
    line_index: int | None = None
    bbox: BoundingBox | None = None
    ocr_confidence: float | None = None
    method: ExtractionMethod = ExtractionMethod.HEURISTIC
    notes: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_text": self.source_text,
            "line_index": self.line_index,
            "bbox": self.bbox.as_list() if self.bbox else None,
            "ocr_confidence": self.ocr_confidence,
            "method": self.method.value,
            "notes": self.notes,
        }


@dataclass(frozen=True, slots=True)
class ExtractedField[T]:
    """A value plus its provenance and confidence.

    ``value`` may be ``None``: an absent field is a legitimate, meaningful
    result. Absent fields carry confidence ``0.0`` and no evidence, which is
    what distinguishes "not on the receipt" from "read with low certainty".
    """

    value: T | None
    confidence: float = 0.0
    evidence: tuple[Evidence, ...] = field(default_factory=tuple)
    #: Text as it appeared before normalisation, when normalisation occurred.
    raw_value: str | None = None
    #: Machine-readable notes about uncertainty (e.g. AMBIGUOUS_DATE_FORMAT).
    warnings: tuple[str, ...] = field(default_factory=tuple)

    @property
    def is_present(self) -> bool:
        return self.value is not None

    @property
    def method(self) -> ExtractionMethod | None:
        """Method of the primary (first) evidence item, if any."""
        return self.evidence[0].method if self.evidence else None

    def with_confidence(self, confidence: float) -> ExtractedField[T]:
        """Return a copy with confidence clamped into ``[0, 1]``."""
        return ExtractedField(
            value=self.value,
            confidence=max(0.0, min(1.0, confidence)),
            evidence=self.evidence,
            raw_value=self.raw_value,
            warnings=self.warnings,
        )

    def with_warnings(self, *codes: str) -> ExtractedField[T]:
        """Return a copy with additional warning codes appended (deduplicated)."""
        merged = list(self.warnings)
        for code in codes:
            if code not in merged:
                merged.append(code)
        return ExtractedField(
            value=self.value,
            confidence=self.confidence,
            evidence=self.evidence,
            raw_value=self.raw_value,
            warnings=tuple(merged),
        )

    @classmethod
    def absent(cls, *warnings: str) -> ExtractedField[T]:
        """Return an explicitly empty field. Never a placeholder value."""
        return cls(value=None, confidence=0.0, warnings=tuple(warnings))
