"""OCR abstraction layer.

Import :func:`~app.ocr.factory.create_ocr_provider` rather than a concrete
provider class. Importing this package registers the bundled providers.
"""

from __future__ import annotations

from app.ocr import providers
from app.ocr.base import OCRProvider, OCRRequest
from app.ocr.factory import (
    available_providers,
    create_ocr_provider,
    register_provider,
    register_provider_factory,
)

__all__ = [
    "OCRProvider",
    "OCRRequest",
    "available_providers",
    "create_ocr_provider",
    "register_provider",
    "register_provider_factory",
]
