"""Replay provider for recorded OCR output.

This provider does **not** invent OCR results. It loads recognition output
that was captured from a real engine (or hand-authored to represent a specific
edge case) and replays it verbatim. That distinction matters: it is test
infrastructure that makes the deterministic stages -- normalisation,
extraction, validation, confidence -- reproducible in CI without a native
binary, and it is what golden tests run against.

It refuses to run in a production environment, so a misconfigured deployment
fails loudly instead of silently serving canned data.

Fixture selection order:
1. ``request.hints["fixture"]`` -- an explicit name.
2. A SHA-256 content hash of the original upload, matching ``<hash>.ocr.json``.
3. ``default.ocr.json``.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from app.core.config import Settings
from app.core.exceptions import ErrorCode, OCRError, ProviderUnavailableError
from app.core.logging import get_logger
from app.ocr.base import OCRProvider, OCRRequest
from app.ocr.factory import register_provider
from app.schemas.ocr import OCRBox, OCRLine, OCRPage, OCRResult, OCRWord

logger = get_logger(__name__)

_SUFFIX = ".ocr.json"


@register_provider("fixture")
class FixtureOCRProvider(OCRProvider):
    """Serves pre-recorded :class:`OCRResult` payloads from disk."""

    name = "fixture"

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._root = Path(settings.fixture_ocr_dir)

    def health_check(self) -> tuple[bool, str | None]:
        if self._settings.is_production:
            return False, "Fixture OCR provider must not be used in production."
        if not self._root.exists():
            return False, f"Fixture directory does not exist: {self._root}"
        return True, None

    def describe(self) -> dict[str, Any]:
        return {
            "provider": self.name,
            "model": "recorded",
            "root": str(self._root),
            "fixtures": self.available_fixtures(),
        }

    def available_fixtures(self) -> list[str]:
        """Names of every recorded fixture, searching one level deep."""
        names = {path.name.removesuffix(_SUFFIX) for path in self._root.glob(f"*{_SUFFIX}")}
        names.update(path.name.removesuffix(_SUFFIX) for path in self._root.glob(f"*/*{_SUFFIX}"))
        return sorted(names)

    def extract(self, request: OCRRequest) -> OCRResult:
        if self._settings.is_production:
            raise ProviderUnavailableError(
                "Fixture OCR provider is disabled in production.",
                details={"provider": self.name},
            )

        path = self._resolve(request)
        if path is None:
            # Naming what *is* available turns a dead end into a next step.
            # This is the error a developer hits when Swagger's placeholder
            # "string" is left in the fixture field, so it has to be actionable.
            available = self.available_fixtures()
            requested = (request.hints or {}).get("fixture")
            raise OCRError(
                (
                    f"No recorded OCR fixture named {requested!r}. "
                    f"Available: {', '.join(available) or 'none'}."
                    if requested
                    else (
                        "The fixture OCR provider needs a fixture name. Send one in "
                        f"the 'fixture' form field. Available: {', '.join(available) or 'none'}."
                    )
                ),
                code=ErrorCode.OCR_EMPTY_RESULT,
                details={
                    "provider": self.name,
                    "root": str(self._root),
                    "requested": requested,
                    "available": available,
                },
            )

        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise OCRError(
                "Recorded OCR fixture could not be read.",
                details={"provider": self.name, "error_type": type(exc).__name__},
            ) from exc

        return self._to_result(payload)

    # -------------------------------------------------------------- helpers
    def _resolve(self, request: OCRRequest) -> Path | None:
        """Find the fixture file for this request, or ``None``."""
        hints = request.hints or {}
        name = hints.get("fixture")
        if name:
            # Resolve strictly inside the fixture root: the hint may originate
            # from a request parameter in development tooling.
            candidate = (self._root / f"{Path(str(name)).name}{_SUFFIX}").resolve()
            if candidate.is_file() and self._root.resolve() in candidate.parents:
                return candidate
            nested = self._find_nested(Path(str(name)).name)
            if nested is not None:
                return nested

        if request.original_bytes:
            digest = hashlib.sha256(request.original_bytes).hexdigest()
            by_hash = self._find_nested(digest)
            if by_hash is not None:
                return by_hash

        default = self._root / f"default{_SUFFIX}"
        return default if default.is_file() else None

    def _find_nested(self, stem: str) -> Path | None:
        """Search the fixture tree for ``<stem>.ocr.json`` (one level deep)."""
        direct = self._root / f"{stem}{_SUFFIX}"
        if direct.is_file():
            return direct
        for child in sorted(self._root.glob(f"*/{stem}{_SUFFIX}")):
            return child
        for child in sorted(self._root.glob(f"{stem}/*{_SUFFIX}")):
            return child
        return None

    def _to_result(self, payload: dict[str, Any]) -> OCRResult:
        """Rehydrate a recorded payload into the unified schema.

        Accepts a compact authoring form where a line is just
        ``{"text": ..., "confidence": ...}``; geometry is optional so that a
        fixture exercising a geometry-less provider is easy to write.
        """
        lines: list[OCRLine] = []
        for raw in payload.get("lines", []):
            bbox = _box(raw.get("bbox"), raw.get("page", 0))
            words = tuple(
                OCRWord(
                    text=word["text"],
                    confidence=word.get("confidence"),
                    bbox=_box(word.get("bbox"), raw.get("page", 0)),
                )
                for word in raw.get("words", [])
            )
            lines.append(
                OCRLine(
                    text=raw["text"],
                    confidence=raw.get("confidence"),
                    bbox=bbox,
                    words=words,
                    page=int(raw.get("page", 0)),
                )
            )

        text = payload.get("text") or "\n".join(line.text for line in lines)
        page = OCRPage(
            page_number=0,
            width=payload.get("width"),
            height=payload.get("height"),
            lines=tuple(lines),
        )
        return OCRResult(
            text=text,
            lines=tuple(lines),
            pages=(page,),
            provider=self.name,
            model=payload.get("model", "recorded"),
            languages=tuple(payload.get("languages", ("eng",))),
            provider_metadata={"source": payload.get("source", "fixture")},
        )


def _box(raw: Any, page: int) -> OCRBox | None:
    """Parse ``[x1, y1, x2, y2]`` into an :class:`OCRBox`."""
    if not raw or len(raw) != 4:
        return None
    x1, y1, x2, y2 = (int(v) for v in raw)
    return OCRBox(x=x1, y=y1, width=max(0, x2 - x1), height=max(0, y2 - y1), page=int(page))
