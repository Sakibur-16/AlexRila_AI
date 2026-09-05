"""Bundled OCR providers.

Importing this package registers every bundled provider with the factory.
Adding a provider means adding a module here and importing it below -- see
``docs/ocr-providers.md``.
"""

from __future__ import annotations

from app.ocr.providers import fixture, openai_vision, tesseract

__all__ = ["fixture", "openai_vision", "tesseract"]
