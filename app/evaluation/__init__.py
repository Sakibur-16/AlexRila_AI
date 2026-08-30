"""Extraction accuracy measurement.

Makes the pipeline measurable rather than judged by inspection. See
``docs/testing.md``.
"""

from __future__ import annotations

from app.evaluation.metrics import (
    DocumentScore,
    EvaluationSummary,
    FieldScore,
    score_document,
    score_items,
)

__all__ = [
    "DocumentScore",
    "EvaluationSummary",
    "FieldScore",
    "score_document",
    "score_items",
]
