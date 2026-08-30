"""Environment-variable parsing.

These exist because ``OCR_LANGUAGES=eng`` -- the exact syntax documented in
``.env.example`` -- crashed the application at import. pydantic-settings treats
a tuple field as a complex type and runs ``json.loads`` on the raw environment
value before any field validator sees it, so a bare word raised
``JSONDecodeError`` during ``Settings()`` construction.

The bug survived the original test suite because every test built settings with
constructor keyword arguments, never through the environment. These tests take
the path a real deployment takes.
"""

from __future__ import annotations

import pytest

from app.core.config import Settings


@pytest.fixture
def env(monkeypatch):
    """Set environment variables with no .env file interfering."""

    def _set(**values: str) -> Settings:
        for key, value in values.items():
            monkeypatch.setenv(key, value)
        return Settings(_env_file=None)

    return _set


def test_single_language_from_env(env) -> None:
    """The documented `OCR_LANGUAGES=eng` must not be a startup crash."""
    assert env(OCR_LANGUAGES="eng").ocr_languages == ("eng",)


def test_multiple_languages_from_env(env) -> None:
    assert env(OCR_LANGUAGES="eng,deu,fra").ocr_languages == ("eng", "deu", "fra")


def test_whitespace_around_separators_is_trimmed(env) -> None:
    assert env(OCR_LANGUAGES="eng , deu ,fra ").ocr_languages == ("eng", "deu", "fra")


def test_mime_types_from_env(env) -> None:
    settings = env(ALLOWED_MIME_TYPES="image/png,image/jpeg")
    assert settings.allowed_mime_types == ("image/png", "image/jpeg")


def test_cors_origins_from_env(env) -> None:
    settings = env(CORS_ALLOW_ORIGINS="https://a.example,https://b.example")
    assert settings.cors_allow_origins == ("https://a.example", "https://b.example")


def test_empty_csv_yields_empty_tuple(env) -> None:
    assert env(CORS_ALLOW_ORIGINS="").cors_allow_origins == ()


def test_windows_path_with_backslashes_survives(env) -> None:
    """A Windows TESSERACT_CMD must not be mangled by escape processing."""
    path = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    assert env(TESSERACT_CMD=path).tesseract_cmd == path


def test_scalar_settings_from_env(env) -> None:
    settings = env(
        OCR_PROVIDER="fixture",
        MAX_FILE_SIZE_MB="5",
        CONFIDENCE_THRESHOLD="0.7",
        LLM_ENABLED="true",
        TOTAL_TOLERANCE="0.02",
    )
    assert settings.ocr_provider == "fixture"
    assert settings.max_file_size_mb == 5
    assert settings.confidence_threshold == 0.7
    assert settings.llm_enabled is True
    assert str(settings.total_tolerance) == "0.02"


def test_secret_is_not_exposed_by_repr(env) -> None:
    settings = env(LLM_API_KEY="super-secret-value")
    assert "super-secret-value" not in repr(settings)
    assert settings.llm_api_key.get_secret_value() == "super-secret-value"


def test_env_example_parses_as_a_whole(tmp_path, monkeypatch) -> None:
    """Every line of the shipped .env.example must produce valid settings.

    This is the file operators copy verbatim, so a value that cannot be parsed
    is a startup crash for whoever follows the documentation.
    """
    from pathlib import Path

    example = Path(__file__).resolve().parents[2] / ".env.example"
    target = tmp_path / ".env"
    target.write_text(example.read_text(encoding="utf-8"), encoding="utf-8")

    monkeypatch.chdir(tmp_path)
    settings = Settings(_env_file=str(target))

    assert settings.ocr_languages
    assert settings.allowed_mime_types
    assert settings.max_file_size_mb > 0
