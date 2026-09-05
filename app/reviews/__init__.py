"""Product-review generation.

A second capability alongside receipt extraction, sharing the LLM provider
abstraction, response envelope and error taxonomy but nothing else.
"""

from __future__ import annotations

from app.reviews.generator import ReviewGenerator

__all__ = ["ReviewGenerator"]
