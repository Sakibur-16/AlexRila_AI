"""Shared test fixtures.

Environment is pinned to ``test`` before any application module is imported so
that a developer's local ``.env`` cannot change what CI asserts.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "receipts"

# Set before importing application modules: Settings reads the environment at
# construction and is cached for the process.
os.environ.setdefault("APP_ENV", "test")
os.environ.setdefault("OCR_PROVIDER", "fixture")
os.environ.setdefault("FIXTURE_OCR_DIR", str(FIXTURE_ROOT))
os.environ.setdefault("LOG_LEVEL", "CRITICAL")
os.environ.setdefault("LLM_ENABLED", "false")

from app.core.config import Settings, reset_settings_cache  # noqa: E402
from app.core.metrics import InMemoryMetricsSink, set_metrics_sink  # noqa: E402
from app.ocr.factory import create_ocr_provider, reset_provider_cache  # noqa: E402
from app.pipeline.receipt_pipeline import ReceiptPipeline  # noqa: E402
from app.schemas.ocr import OCRLine, OCRResult  # noqa: E402


@pytest.fixture(autouse=True)
def _isolate_caches():
    """Reset process-wide caches between tests.

    Settings, provider instances and metrics are all process-scoped by design.
    Clearing them per test keeps a test that overrides configuration from
    leaking into the next one.
    """
    reset_settings_cache()
    reset_provider_cache()
    set_metrics_sink(InMemoryMetricsSink())
    yield
    reset_settings_cache()
    reset_provider_cache()


@pytest.fixture
def settings() -> Settings:
    """Default test settings, isolated from any local ``.env``."""
    return Settings(
        _env_file=None,
        app_env="test",
        ocr_provider="fixture",
        fixture_ocr_dir=str(FIXTURE_ROOT),
        log_level="CRITICAL",
    )


@pytest.fixture
def make_settings():
    """Build a Settings instance with specific overrides.

    Overrides are passed directly rather than through the environment, so a
    test states exactly what it depends on.
    """

    def _make(**overrides: object) -> Settings:
        base = {
            "app_env": "test",
            "ocr_provider": "fixture",
            "fixture_ocr_dir": str(FIXTURE_ROOT),
            "log_level": "CRITICAL",
        }
        base.update(overrides)
        # _env_file=None keeps the suite hermetic: a developer's .env must
        # never change what CI asserts.
        return Settings(_env_file=None, **base)  # type: ignore[arg-type]

    return _make


@pytest.fixture
def ocr_result_factory():
    """Build an :class:`OCRResult` from plain text lines."""

    def _make(
        lines: list[str],
        *,
        confidence: float = 0.95,
        provider: str = "fixture",
        with_geometry: bool = False,
    ) -> OCRResult:
        from app.schemas.ocr import OCRBox

        ocr_lines = []
        y = 0
        for text in lines:
            bbox = (
                OCRBox(x=10, y=y, width=max(60, len(text) * 10), height=24)
                if with_geometry
                else None
            )
            ocr_lines.append(OCRLine(text=text, confidence=confidence, bbox=bbox))
            y += 30
        return OCRResult(
            text="\n".join(lines),
            lines=tuple(ocr_lines),
            provider=provider,
            model="test",
            languages=("eng",),
        )

    return _make


@pytest.fixture
def pipeline(settings: Settings) -> ReceiptPipeline:
    """A pipeline backed by the fixture OCR provider."""
    return ReceiptPipeline(
        settings=settings,
        ocr_provider=create_ocr_provider("fixture", settings),
        llm_extractor=None,
    )


@pytest.fixture
def receipt_image() -> bytes:
    """A minimal valid PNG large enough to pass input validation.

    Its content is irrelevant: the fixture OCR provider replays recorded text
    rather than reading the image. It exists so the full input-validation and
    preprocessing path still runs.
    """
    import cv2
    import numpy as np

    image = np.full((900, 600, 3), 245, dtype=np.uint8)
    cv2.putText(image, "RECEIPT", (60, 400), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (20, 20, 20), 3)
    ok, buffer = cv2.imencode(".png", image)
    assert ok
    return bytes(buffer.tobytes())


@pytest.fixture
def fixture_names() -> list[str]:
    """Every recorded OCR fixture, sorted."""
    return sorted(path.name.removesuffix(".ocr.json") for path in FIXTURE_ROOT.glob("*.ocr.json"))
