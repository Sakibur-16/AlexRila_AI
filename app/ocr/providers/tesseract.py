"""Tesseract OCR provider.

Tesseract is the default because it runs locally: no per-request cost, no data
leaving the host, and no external dependency in the request path -- which
matters for a document type as sensitive as a receipt.

The provider uses ``image_to_data`` rather than ``image_to_string`` because the
word-level geometry and per-word confidence it returns are what make spatial
extraction and honest confidence scoring possible. Words are grouped back into
lines using Tesseract's own block/paragraph/line indices instead of clustering
by y-coordinate, which is more reliable on multi-column receipt footers.
"""

from __future__ import annotations

import shutil
from typing import Any

from app.core.config import Settings
from app.core.exceptions import ErrorCode, OCRError, OCRTimeoutError, ProviderUnavailableError
from app.core.logging import get_logger
from app.ocr.base import OCRProvider, OCRRequest
from app.ocr.factory import register_provider
from app.schemas.ocr import OCRBox, OCRLine, OCRPage, OCRResult, OCRWord, TextOrientation

logger = get_logger(__name__)

#: Tesseract reports -1 for entries it has no confidence value for.
_NO_CONFIDENCE = -1

#: Tesseract confidences are 0-100; the unified schema uses 0-1.
_CONFIDENCE_SCALE = 100.0


@register_provider("tesseract")
class TesseractOCRProvider(OCRProvider):
    """Local recognition via the Tesseract engine."""

    name = "tesseract"

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._languages = settings.ocr_languages or ("eng",)
        self._version: str | None = None
        self._pytesseract: Any = None

    # ------------------------------------------------------------- plumbing
    def _engine(self) -> Any:
        """Import and configure pytesseract lazily.

        Deferring the import keeps the package importable (and the rest of the
        test suite runnable) on hosts without Tesseract installed.
        """
        if self._pytesseract is not None:
            return self._pytesseract
        try:
            import pytesseract
        except ImportError as exc:  # pragma: no cover - packaging failure
            raise ProviderUnavailableError(
                "pytesseract is not installed.",
                details={"provider": self.name},
            ) from exc

        if self._settings.tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = self._settings.tesseract_cmd

        self._pytesseract = pytesseract
        return pytesseract

    def _binary_path(self) -> str | None:
        """Resolve the tesseract executable, honouring configuration first."""
        configured = self._settings.tesseract_cmd
        if configured:
            return configured if shutil.which(configured) or _exists(configured) else None
        return shutil.which("tesseract")

    def _config_string(self) -> str:
        """Build the tesseract CLI config string from settings."""
        parts = [f"--oem {self._settings.tesseract_oem}", f"--psm {self._settings.tesseract_psm}"]
        if self._settings.tesseract_tessdata_dir:
            parts.append(f'--tessdata-dir "{self._settings.tesseract_tessdata_dir}"')
        # Preserve inter-word spacing: receipts encode label/value association
        # in whitespace, and collapsing it loses that signal.
        parts.append("-c preserve_interword_spaces=1")
        return " ".join(parts)

    # --------------------------------------------------------------- public
    def health_check(self) -> tuple[bool, str | None]:
        try:
            self._engine()
        except ProviderUnavailableError as exc:
            return False, exc.message
        if self._binary_path() is None:
            return False, (
                "Tesseract binary not found on PATH. Install tesseract-ocr or set TESSERACT_CMD."
            )
        try:
            self._version = str(self._pytesseract.get_tesseract_version())
        except Exception as exc:
            return False, f"Tesseract is not executable: {type(exc).__name__}"
        return True, None

    def describe(self) -> dict[str, Any]:
        return {
            "provider": self.name,
            "model": self._model_id(),
            "languages": list(self._languages),
            "psm": self._settings.tesseract_psm,
            "oem": self._settings.tesseract_oem,
        }

    def extract(self, request: OCRRequest) -> OCRResult:
        pytesseract = self._engine()

        if self._binary_path() is None:
            raise ProviderUnavailableError(
                "Tesseract binary not found. Install tesseract-ocr or set TESSERACT_CMD.",
                details={"provider": self.name},
            )

        languages = request.languages or self._languages
        lang_arg = "+".join(languages)

        try:
            data = pytesseract.image_to_data(
                request.image,
                lang=lang_arg,
                config=self._config_string(),
                output_type=pytesseract.Output.DICT,
                timeout=request.timeout_seconds,
            )
        except RuntimeError as exc:
            # pytesseract raises RuntimeError("Tesseract process timeout") when
            # its own timeout fires.
            if "timeout" in str(exc).lower():
                raise OCRTimeoutError(
                    "OCR timed out.",
                    details={
                        "provider": self.name,
                        "timeout_seconds": request.timeout_seconds,
                    },
                ) from exc
            raise OCRError(
                "OCR engine failed.",
                details={"provider": self.name, "error_type": type(exc).__name__},
            ) from exc
        except Exception as exc:
            raise OCRError(
                "OCR engine failed.",
                details={"provider": self.name, "error_type": type(exc).__name__},
            ) from exc

        lines = self._group_lines(data)
        height, width = request.image.shape[:2]
        page = OCRPage(
            page_number=0,
            width=int(width),
            height=int(height),
            orientation=TextOrientation.ROTATE_0,
            lines=lines,
        )

        result = OCRResult(
            text="\n".join(line.text for line in lines),
            lines=lines,
            pages=(page,),
            provider=self.name,
            model=self._model_id(),
            languages=tuple(languages),
            provider_metadata={"psm": self._settings.tesseract_psm},
        )

        if result.is_empty:
            raise OCRError(
                "OCR produced no readable text.",
                code=ErrorCode.OCR_EMPTY_RESULT,
                details={"provider": self.name},
            )
        return result

    # -------------------------------------------------------------- helpers
    def _model_id(self) -> str:
        if self._version is None:
            try:
                self._version = str(self._engine().get_tesseract_version())
            except Exception:
                self._version = "unknown"
        return f"tesseract-{self._version}"

    @staticmethod
    def _group_lines(data: dict[str, list[Any]]) -> tuple[OCRLine, ...]:
        """Reassemble word-level output into lines.

        Grouping uses Tesseract's structural indices (page/block/paragraph/
        line) rather than y-coordinate clustering, so two columns printed at
        the same height stay in their own lines.
        """
        count = len(data.get("text", []))
        grouped: dict[tuple[int, int, int, int], list[int]] = {}
        order: list[tuple[int, int, int, int]] = []

        for i in range(count):
            text = str(data["text"][i]).strip()
            if not text:
                continue
            key = (
                int(data["page_num"][i]),
                int(data["block_num"][i]),
                int(data["par_num"][i]),
                int(data["line_num"][i]),
            )
            if key not in grouped:
                grouped[key] = []
                order.append(key)
            grouped[key].append(i)

        lines: list[OCRLine] = []
        for key in order:
            indices = grouped[key]
            words: list[OCRWord] = []
            confidences: list[float] = []
            x1 = y1 = 10**9
            x2 = y2 = -(10**9)

            for i in indices:
                text = str(data["text"][i]).strip()
                raw_conf = float(data["conf"][i])
                confidence = (
                    None
                    if raw_conf == _NO_CONFIDENCE
                    else max(0.0, min(1.0, raw_conf / _CONFIDENCE_SCALE))
                )
                if confidence is not None:
                    confidences.append(confidence)

                wx, wy = int(data["left"][i]), int(data["top"][i])
                ww, wh = int(data["width"][i]), int(data["height"][i])
                x1, y1 = min(x1, wx), min(y1, wy)
                x2, y2 = max(x2, wx + ww), max(y2, wy + wh)

                words.append(
                    OCRWord(
                        text=text,
                        confidence=confidence,
                        bbox=OCRBox(
                            x=wx, y=wy, width=ww, height=wh, page=key[0] - 1 if key[0] > 0 else 0
                        ),
                    )
                )

            page_index = max(0, key[0] - 1)
            lines.append(
                OCRLine(
                    text=" ".join(word.text for word in words),
                    confidence=(sum(confidences) / len(confidences)) if confidences else None,
                    bbox=OCRBox(
                        x=max(0, x1),
                        y=max(0, y1),
                        width=max(0, x2 - x1),
                        height=max(0, y2 - y1),
                        page=page_index,
                    ),
                    words=tuple(words),
                    page=page_index,
                )
            )
        return tuple(lines)


def _exists(path: str) -> bool:
    """Whether ``path`` names an existing file (absolute binary paths)."""
    from pathlib import Path

    try:
        return Path(path).is_file()
    except OSError:  # pragma: no cover - defensive
        return False
