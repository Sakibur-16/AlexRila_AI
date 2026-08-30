"""LLM provider registry and factory.

Same pattern as :mod:`app.ocr.factory`: configuration names a provider, the
factory resolves it, and no pipeline code ever names a vendor.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from typing import TypeVar

from app.core.config import Settings, get_settings
from app.core.exceptions import ProviderNotRegisteredError
from app.core.logging import get_logger
from app.llm.base import LLMProvider

logger = get_logger(__name__)

ProviderFactory = Callable[[Settings], LLMProvider]

_registry: dict[str, ProviderFactory] = {}
_instances: dict[str, LLMProvider] = {}
_lock = threading.Lock()

P = TypeVar("P", bound=LLMProvider)


def register_llm_provider(name: str) -> Callable[[type[P]], type[P]]:
    """Register an LLM provider class under ``name``."""
    key = name.strip().lower()

    def decorator(cls: type[P]) -> type[P]:
        with _lock:
            _registry[key] = cls
        return cls

    return decorator


def available_llm_providers() -> tuple[str, ...]:
    """Names of all registered LLM providers, sorted."""
    with _lock:
        return tuple(sorted(_registry))


def create_llm_provider(name: str | None = None, settings: Settings | None = None) -> LLMProvider:
    """Return the provider named by ``name`` or by ``LLM_PROVIDER``.

    Raises:
        ProviderNotRegisteredError: The name is unknown.
    """
    settings = settings or get_settings()
    key = (name or settings.llm_provider).strip().lower()

    with _lock:
        cached = _instances.get(key)
        if cached is not None:
            return cached
        factory = _registry.get(key)

    if factory is None:
        raise ProviderNotRegisteredError(
            f"LLM provider {key!r} is not registered.",
            details={"requested": key, "available": list(available_llm_providers())},
        )

    provider = factory(settings)
    with _lock:
        _instances[key] = provider
    logger.info("llm_provider_created", provider=key)
    return provider


def reset_llm_provider_cache() -> None:
    """Drop cached provider instances. Intended for tests."""
    with _lock:
        _instances.clear()
