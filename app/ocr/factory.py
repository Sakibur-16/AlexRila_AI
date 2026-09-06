"""OCR provider registry and factory.

Provider selection is configuration-driven. Application code calls
:func:`create_ocr_provider` and never names a concrete class, which is what
lets a deployment switch engines with an environment variable.

Providers are registered by decorating the class. Registration is
import-triggered, so :mod:`app.ocr.providers` imports every bundled provider;
a plugin can register its own by importing this module and applying the
decorator.

Instances are cached per configuration signature because construction can be
expensive (client setup, model loading) and providers are required to be
thread-safe for :meth:`~app.ocr.base.OCRProvider.extract`.
"""

from __future__ import annotations

import hashlib
import threading
from collections.abc import Callable
from typing import TypeVar

from app.core.config import Settings, get_settings
from app.core.exceptions import ProviderNotRegisteredError
from app.core.logging import get_logger
from app.ocr.base import OCRProvider

logger = get_logger(__name__)

ProviderFactory = Callable[[Settings], OCRProvider]

_registry: dict[str, ProviderFactory] = {}
_instances: dict[tuple[str, tuple[str, ...]], OCRProvider] = {}
_lock = threading.Lock()

P = TypeVar("P", bound=OCRProvider)


def register_provider(name: str) -> Callable[[type[P]], type[P]]:
    """Register a provider class under ``name``.

    Used as a decorator on a class whose ``__init__`` accepts ``Settings``::

        @register_provider("tesseract")
        class TesseractOCRProvider(OCRProvider):
            def __init__(self, settings: Settings) -> None: ...

    For a provider needing custom construction, use
    :func:`register_provider_factory` instead.
    """
    key = name.strip().lower()

    def decorator(cls: type[P]) -> type[P]:
        with _lock:
            _registry[key] = cls
        return cls

    return decorator


def register_provider_factory(name: str, factory: ProviderFactory) -> None:
    """Register a provider built by an arbitrary callable.

    The imperative counterpart of :func:`register_provider`, for providers
    whose construction needs more than ``cls(settings)``.
    """
    with _lock:
        _registry[name.strip().lower()] = factory


def available_providers() -> tuple[str, ...]:
    """Names of all registered providers, sorted."""
    with _lock:
        return tuple(sorted(_registry))


def create_ocr_provider(name: str | None = None, settings: Settings | None = None) -> OCRProvider:
    """Return the provider named by ``name`` or by ``OCR_PROVIDER``.

    Raises:
        ProviderNotRegisteredError: No provider is registered under the name.
            The message lists what *is* registered, which turns a typo in an
            environment variable into an immediately actionable error.
    """
    settings = settings or get_settings()
    key = (name or settings.ocr_provider).strip().lower()

    cache_key = (key, _config_signature(settings))
    with _lock:
        cached = _instances.get(cache_key)
        if cached is not None:
            return cached

    if "+" in key:
        from app.ocr.providers.fallback import FallbackOCRProvider

        parts = [part.strip() for part in key.split("+") if part.strip()]
        if len(parts) >= 2:
            primary = create_ocr_provider(parts[0], settings)
            secondary = create_ocr_provider(parts[1], settings)
            fallback_provider = FallbackOCRProvider(primary, secondary)
            with _lock:
                _instances[cache_key] = fallback_provider
            logger.info("ocr_provider_created", provider=key)
            return fallback_provider

    with _lock:
        factory = _registry.get(key)

    if factory is None:
        raise ProviderNotRegisteredError(
            f"OCR provider {key!r} is not registered.",
            details={"requested": key, "available": list(available_providers())},
        )

    provider = factory(settings)
    with _lock:
        _instances[cache_key] = provider
    logger.info("ocr_provider_created", provider=key)
    return provider


def _config_signature(settings: Settings) -> tuple[str, ...]:
    """Settings that, when changed, must produce a fresh provider instance.

    Every provider-affecting setting belongs here. Omitting one means a cached
    instance built from stale configuration is silently handed back -- which
    looked exactly like "the API key I just set is being ignored".

    The API key is hashed rather than stored: this tuple is a cache key held
    for the process lifetime, and a secret has no business living in one.
    """
    vision_key = settings.vision_api_key.get_secret_value()
    return (
        ",".join(settings.ocr_languages),
        settings.tesseract_cmd,
        settings.tesseract_tessdata_dir,
        str(settings.tesseract_psm),
        str(settings.tesseract_oem),
        settings.fixture_ocr_dir,
        settings.vision_model,
        settings.vision_base_url,
        hashlib.sha256(vision_key.encode()).hexdigest() if vision_key else "",
    )


def reset_provider_cache() -> None:
    """Drop cached provider instances. Intended for tests."""
    with _lock:
        _instances.clear()
