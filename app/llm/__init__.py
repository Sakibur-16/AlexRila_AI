"""Optional LLM extraction layer.

Import :func:`~app.llm.factory.create_llm_provider` rather than a concrete
provider class. Importing this package registers the bundled providers.
"""

from __future__ import annotations

from app.llm import providers
from app.llm.base import LLMProvider, LLMRequest, LLMResponse
from app.llm.extractor import LLMFallbackExtractor, LLMFallbackOutcome
from app.llm.factory import available_llm_providers, create_llm_provider

__all__ = [
    "LLMFallbackExtractor",
    "LLMFallbackOutcome",
    "LLMProvider",
    "LLMRequest",
    "LLMResponse",
    "available_llm_providers",
    "create_llm_provider",
]
