"""Application configuration.

All configuration is environment-driven (12-factor). Nothing here has a
hardcoded credential, path or model name; every default is a safe local
development default. Settings are read once and cached -- import
:func:`get_settings` rather than instantiating :class:`Settings` directly so
that a single immutable instance is shared (no global mutable state).
"""

from __future__ import annotations

from decimal import Decimal
from functools import lru_cache
from typing import Annotated, Literal

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

from app.core.versions import PIPELINE_VERSION

Environment = Literal["development", "staging", "production", "test"]


class Settings(BaseSettings):
    """Runtime configuration, populated from environment variables / ``.env``."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        frozen=True,
    )

    # ------------------------------------------------------------------ app
    app_env: Environment = "development"
    app_name: str = "receipt-ocr"
    app_version: str = PIPELINE_VERSION
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    log_format: Literal["json", "console"] = "json"
    #: Echo ``details`` of structured errors in API responses. Keep off in
    #: production: details are safe-by-construction but noisy for clients.
    debug_errors: bool = False
    cors_allow_origins: Annotated[tuple[str, ...], NoDecode] = ()
    #: Bind address used when the module is run directly. A container or
    #: process manager normally supplies these on the uvicorn command line
    #: instead, which is why they are not referenced anywhere else.
    host: str = "127.0.0.1"
    port: Annotated[int, Field(ge=1, le=65535)] = 8000
    reload: bool = False

    # ------------------------------------------------------------------ ocr
    ocr_provider: str = "tesseract"
    ocr_timeout_seconds: Annotated[float, Field(gt=0, le=300)] = 30.0
    ocr_max_retries: Annotated[int, Field(ge=0, le=5)] = 2
    ocr_retry_base_delay_seconds: Annotated[float, Field(ge=0, le=10)] = 0.25
    #: Languages passed to the OCR engine, in priority order. Tesseract expects
    #: ISO 639-2/T codes: eng, ben, hin, ara, spa, fra.
    ocr_languages: Annotated[tuple[str, ...], NoDecode] = ("eng",)
    #: Path to the tesseract binary. Empty means "resolve from PATH".
    tesseract_cmd: str = ""
    #: Directory holding tesseract language data. Empty means engine default.
    tesseract_tessdata_dir: str = ""
    #: Page segmentation mode. 6 == "assume a single uniform block of text",
    #: the best general-purpose default for receipts.
    tesseract_psm: Annotated[int, Field(ge=0, le=13)] = 6
    tesseract_oem: Annotated[int, Field(ge=0, le=3)] = 3
    #: Directory of recorded OCR payloads used by the "fixture" provider.
    fixture_ocr_dir: str = "tests/fixtures"

    # --- vision-model OCR (OCR_PROVIDER=openai_vision) -------------------
    #: Multimodal model used to transcribe receipts. Model names move faster
    #: than code does, which is why this is configuration and not a constant.
    #: Verify the current name with `python scripts/check_llm.py` before a
    #: deploy; a retired name fails at the first request, not at startup.
    vision_model: str = "gpt-5.1-mini"
    #: Override for an OpenAI-compatible gateway or a self-hosted endpoint.
    #: Empty means the OpenAI API.
    vision_base_url: str = ""
    vision_api_key: SecretStr = SecretStr("")
    vision_max_output_tokens: Annotated[int, Field(ge=256, le=16_000)] = 4096

    # -------------------------------------------------------------- uploads
    max_file_size_mb: Annotated[float, Field(gt=0, le=100)] = 10.0
    min_image_width: Annotated[int, Field(ge=1)] = 200
    min_image_height: Annotated[int, Field(ge=1)] = 200
    max_image_pixels: Annotated[int, Field(ge=1)] = 40_000_000
    allowed_mime_types: Annotated[tuple[str, ...], NoDecode] = (
        "image/jpeg",
        "image/png",
        "image/webp",
        "image/tiff",
        "image/bmp",
    )

    # --------------------------------------------------------- preprocessing
    enable_preprocessing: bool = True
    preprocess_grayscale: bool = True
    preprocess_deskew: bool = True
    preprocess_denoise: bool = True
    preprocess_contrast: bool = True
    preprocess_sharpen: bool = False
    preprocess_threshold: bool = False
    #: Upscale when the shorter edge is below this many pixels.
    preprocess_min_short_edge: Annotated[int, Field(ge=100)] = 900
    preprocess_max_long_edge: Annotated[int, Field(ge=500)] = 3000
    #: Skew beyond this angle (degrees) is corrected; below it, left alone.
    deskew_min_angle_degrees: Annotated[float, Field(ge=0, le=45)] = 0.4
    deskew_max_angle_degrees: Annotated[float, Field(ge=0, le=45)] = 20.0

    # -------------------------------------------------------- quality gates
    quality_min_blur_score: Annotated[float, Field(ge=0)] = 60.0
    quality_min_brightness: Annotated[float, Field(ge=0, le=255)] = 45.0
    quality_max_brightness: Annotated[float, Field(ge=0, le=255)] = 225.0
    quality_min_contrast: Annotated[float, Field(ge=0)] = 28.0

    # --------------------------------------------------------------- locale
    #: Preferred day/month order when a date is otherwise ambiguous. "none"
    #: means never guess -- ambiguous dates stay null with a warning.
    date_order: Literal["DMY", "MDY", "none"] = "none"
    #: Preferred decimal separator hint; "auto" infers per document.
    decimal_separator: Literal["auto", "dot", "comma"] = "auto"
    default_currency: str = ""
    #: ISO 3166-1 alpha-2 hint used to disambiguate shared symbols such as "$".
    default_country: str = ""

    # ----------------------------------------------------------- confidence
    confidence_threshold: Annotated[float, Field(ge=0, le=1)] = 0.85
    low_ocr_confidence_threshold: Annotated[float, Field(ge=0, le=1)] = 0.60
    #: Weights blending OCR confidence and extraction-method confidence into a
    #: field score. Normalised at use. See docs/confidence.md.
    confidence_weight_ocr: Annotated[float, Field(ge=0, le=1)] = 0.45
    confidence_weight_extraction: Annotated[float, Field(ge=0, le=1)] = 0.55
    #: Multiplier applied to affected fields when financial validation fails.
    confidence_validation_penalty: Annotated[float, Field(ge=0, le=1)] = 0.65
    #: An individual field triggers human review below this score. Deliberately
    #: lower than CONFIDENCE_THRESHOLD: a field read by a sound heuristic scores
    #: below the aggregate bar by design, and flagging every such receipt would
    #: make the review queue meaningless.
    review_field_confidence_threshold: Annotated[float, Field(ge=0, le=1)] = 0.65

    # ----------------------------------------------------------- validation
    #: Absolute tolerance, in major currency units, for arithmetic checks.
    total_tolerance: Decimal = Decimal("0.05")
    #: Relative tolerance added on top of the absolute one for large totals.
    total_tolerance_ratio: Annotated[float, Field(ge=0, le=0.1)] = 0.01

    # ------------------------------------------------------------------ llm
    llm_enabled: bool = False
    llm_provider: str = "openai_compatible"
    llm_model: str = ""
    llm_base_url: str = ""
    llm_api_key: SecretStr = SecretStr("")
    llm_timeout_seconds: Annotated[float, Field(gt=0, le=300)] = 45.0
    llm_max_retries: Annotated[int, Field(ge=0, le=5)] = 1
    llm_temperature: Annotated[float, Field(ge=0, le=2)] = 0.0
    llm_max_output_tokens: Annotated[int, Field(ge=256, le=32_000)] = 4096
    #: Only invoke the LLM when deterministic confidence falls below this.
    llm_fallback_confidence_threshold: Annotated[float, Field(ge=0, le=1)] = 0.75
    #: Truncate OCR text sent to the LLM, bounding cost and injection surface.
    llm_max_input_chars: Annotated[int, Field(ge=500, le=100_000)] = 20_000

    # ---------------------------------------------------- review generation
    #: Reviews read as prose, so a little sampling variety helps. Extraction
    #: stays at 0 -- there the goal is reproducibility, not readability.
    review_enabled: bool = True
    review_temperature: Annotated[float, Field(ge=0, le=2)] = 0.7
    review_max_output_tokens: Annotated[int, Field(ge=128, le=8_000)] = 1024

    # -------------------------------------------------------------- privacy
    #: Include full OCR text in the API response.
    include_raw_ocr: bool = True
    #: Include per-field evidence (source text + bbox) in the API response.
    include_evidence: bool = False
    #: Persist uploaded images / OCR payloads. Off by default (privacy first).
    enable_raw_ocr_storage: bool = False
    storage_dir: str = "data/output"
    #: Redact card-number-like sequences from stored and returned OCR text.
    redact_sensitive_text: bool = True
    #: Emit document content in logs. Keep off in production.
    log_document_content: bool = False

    # --------------------------------------------------------------- limits
    #: Shared secret clients must send as X-API-Key. Empty disables
    #: authentication entirely, which is the default so that local
    #: development and internal deployments are unaffected.
    api_key: SecretStr = SecretStr("")
    rate_limit_enabled: bool = False
    rate_limit_requests_per_minute: Annotated[int, Field(ge=1)] = 60

    @field_validator("ocr_languages", "allowed_mime_types", "cors_allow_origins", mode="before")
    @classmethod
    def _split_csv(cls, value: object) -> object:
        """Accept comma-separated strings for tuple-valued settings.

        These fields are annotated ``NoDecode`` because pydantic-settings
        otherwise treats a tuple as a complex type and runs ``json.loads`` on
        the raw environment value before any validator sees it -- which makes
        the documented ``OCR_LANGUAGES=eng`` a startup crash rather than a
        setting. With ``NoDecode`` the raw string arrives here intact.
        """
        if isinstance(value, str):
            return tuple(part.strip() for part in value.split(",") if part.strip())
        return value

    @field_validator("default_currency")
    @classmethod
    def _upper_currency(cls, value: str) -> str:
        return value.strip().upper()

    @field_validator("default_country")
    @classmethod
    def _upper_country(cls, value: str) -> str:
        return value.strip().upper()

    @property
    def max_file_size_bytes(self) -> int:
        return int(self.max_file_size_mb * 1024 * 1024)

    @property
    def is_production(self) -> bool:
        return self.app_env == "production"

    def confidence_weights(self) -> tuple[float, float]:
        """Return normalised ``(ocr, extraction)`` weights.

        Normalising defensively means a misconfigured pair cannot silently
        inflate or deflate every score in the system.
        """
        total = self.confidence_weight_ocr + self.confidence_weight_extraction
        if total <= 0:
            return (0.5, 0.5)
        return (
            self.confidence_weight_ocr / total,
            self.confidence_weight_extraction / total,
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings singleton."""
    return Settings()


def reset_settings_cache() -> None:
    """Clear the settings cache. Intended for tests only."""
    get_settings.cache_clear()
