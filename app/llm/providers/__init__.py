"""Bundled LLM providers.

Importing this package registers every bundled provider with the factory.
See ``docs/ocr-providers.md`` for the pattern (it is identical for LLMs).
"""

from __future__ import annotations

from app.llm.providers import openai_compatible

__all__ = ["openai_compatible"]
