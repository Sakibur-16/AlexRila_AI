"""OCR provider interface.

Implementing :class:`OCRProvider` is the only thing required to add a new
recognition backend -- local engine, cloud API, or vision model. No other
module in the codebase may branch on which provider is active.

Providers receive an already-decoded, already-preprocessed image so that image
handling is not reimplemented per provider. They return a
:class:`~app.schemas.ocr.OCRResult` and raise
:class:`~app.core.exceptions.PipelineError` subclasses on failure -- never
provider-native exceptions, which would leak vendor detail into callers.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np

from app.schemas.ocr import OCRResult


@dataclass(frozen=True, slots=True)
class OCRRequest:
    """One recognition request.

    Attributes:
        image: Preprocessed image as an ``H x W`` (grayscale) or ``H x W x 3``
            (BGR) uint8 array.
        languages: Language codes in priority order.
        timeout_seconds: Upper bound on recognition time.
        original_bytes: The undecoded upload. Cloud providers and vision
            models often prefer the original encoding; local engines ignore it.
        hints: Non-authoritative context (document type, locale) a provider may
            use. Never required for correctness.
    """

    image: np.ndarray
    languages: tuple[str, ...] = ("eng",)
    timeout_seconds: float = 30.0
    original_bytes: bytes | None = None
    hints: dict[str, Any] | None = None


class OCRProvider(ABC):
    """A text-recognition backend."""

    #: Registry key used in ``OCR_PROVIDER``. Must be unique and stable.
    name: str = "unnamed"

    @abstractmethod
    def extract(self, request: OCRRequest) -> OCRResult:
        """Recognise text in ``request.image``.

        Args:
            request: The recognition request.

        Returns:
            A normalised result. Providers that cannot report geometry or
            confidence leave those fields ``None`` rather than fabricating
            values.

        Raises:
            OCRError: Recognition failed.
            OCRTimeoutError: Recognition exceeded ``timeout_seconds``.
            ProviderUnavailableError: The backend is not usable right now.
        """

    def health_check(self) -> tuple[bool, str | None]:
        """Report whether the provider can currently service requests.

        Used by ``/ready`` so a deployment missing an engine binary or an API
        credential fails readiness instead of failing every request.

        Returns:
            ``(ready, detail)``. ``detail`` explains a negative result and must
            not contain credentials.
        """
        return True, None

    def describe(self) -> dict[str, Any]:
        """Return non-sensitive provider metadata for logs and ``/version``."""
        return {"provider": self.name}
