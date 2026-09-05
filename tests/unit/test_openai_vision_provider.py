"""Vision-model OCR provider tests.

The provider is exercised against a stubbed HTTP layer, so no network call or
API key is involved. What matters here is not that a model can read a receipt
-- that is the model's job -- but that this provider behaves like every other
one at the seams: normalised errors, no fabricated confidence or geometry, and
no vendor detail escaping into the result.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from app.core.exceptions import ErrorCode, OCRError, OCRTimeoutError, ProviderUnavailableError
from app.ocr.base import OCRRequest
from app.ocr.providers.openai_vision import OpenAIVisionOCRProvider

RECEIPT_TEXT = "GREEN VALLEY MARKET\nMilk 2.50\nSUBTOTAL 2.50\nTOTAL 2.50"


class _StubResponse:
    def __init__(self, status_code: int = 200, payload: Any = None) -> None:
        self.status_code = status_code
        self._payload = payload

    def json(self) -> Any:
        if self._payload is None:
            raise ValueError("no body")
        return self._payload


def _completion(content: str) -> dict[str, Any]:
    return {"choices": [{"message": {"content": content}}], "model": "stub-vision"}


class _StubClient:
    """Stands in for the httpx client, recording what was sent."""

    def __init__(self, response: Any = None, error: Exception | None = None) -> None:
        self._response = response
        self._error = error
        self.last_payload: dict[str, Any] | None = None

    def post(self, _url: str, json: dict[str, Any], **_kwargs: Any) -> Any:
        self.last_payload = json
        if self._error is not None:
            raise self._error
        return self._response


@pytest.fixture
def configured(make_settings):
    return make_settings(vision_api_key="sk-test", vision_model="stub-vision")


@pytest.fixture
def image() -> np.ndarray:
    return np.full((400, 300, 3), 240, np.uint8)


def _provider(settings, client: _StubClient) -> OpenAIVisionOCRProvider:
    provider = OpenAIVisionOCRProvider(settings)
    provider._http = lambda: client  # type: ignore[method-assign]
    return provider


# --------------------------------------------------------------- readiness
def test_missing_key_is_reported_not_raised(make_settings) -> None:
    ready, detail = OpenAIVisionOCRProvider(make_settings()).health_check()
    assert not ready
    assert detail and "VISION_API_KEY" in detail


def test_missing_model_is_reported(make_settings) -> None:
    settings = make_settings(vision_api_key="sk-test", vision_model="")
    ready, detail = OpenAIVisionOCRProvider(settings).health_check()
    assert not ready
    assert detail and "VISION_MODEL" in detail


def test_health_detail_never_contains_the_key(make_settings) -> None:
    settings = make_settings(vision_api_key="sk-super-secret", vision_model="")
    _, detail = OpenAIVisionOCRProvider(settings).health_check()
    assert detail is not None
    assert "sk-super-secret" not in detail


def test_describe_does_not_leak_the_key(configured) -> None:
    described = OpenAIVisionOCRProvider(configured).describe()
    assert "sk-test" not in str(described)
    assert described["model"] == "stub-vision"


def test_unconfigured_provider_refuses_to_run(make_settings, image) -> None:
    provider = OpenAIVisionOCRProvider(make_settings())
    with pytest.raises(ProviderUnavailableError):
        provider.extract(OCRRequest(image=image))


# ------------------------------------------------------------ transcription
def test_transcription_becomes_ocr_lines(configured, image) -> None:
    client = _StubClient(_StubResponse(200, _completion(RECEIPT_TEXT)))
    result = _provider(configured, client).extract(OCRRequest(image=image))

    assert [line.text for line in result.lines] == RECEIPT_TEXT.splitlines()
    assert result.provider == "openai_vision"
    assert result.text == RECEIPT_TEXT


def test_confidence_and_geometry_stay_none(configured, image) -> None:
    """Fabricating either would corrupt every downstream score."""
    client = _StubClient(_StubResponse(200, _completion(RECEIPT_TEXT)))
    result = _provider(configured, client).extract(OCRRequest(image=image))

    assert all(line.confidence is None for line in result.lines)
    assert all(line.bbox is None for line in result.lines)
    assert not result.has_geometry


def test_blank_lines_are_dropped(configured, image) -> None:
    client = _StubClient(_StubResponse(200, _completion("A\n\n   \nB")))
    result = _provider(configured, client).extract(OCRRequest(image=image))
    assert [line.text for line in result.lines] == ["A", "B"]


def test_markdown_fence_is_stripped(configured, image) -> None:
    """Models wrap output in a fence despite being told not to."""
    client = _StubClient(_StubResponse(200, _completion(f"```\n{RECEIPT_TEXT}\n```")))
    result = _provider(configured, client).extract(OCRRequest(image=image))
    assert "```" not in result.text
    assert result.lines[0].text == "GREEN VALLEY MARKET"


def test_empty_transcription_is_an_error(configured, image) -> None:
    client = _StubClient(_StubResponse(200, _completion("   ")))
    with pytest.raises(OCRError) as exc:
        _provider(configured, client).extract(OCRRequest(image=image))
    assert exc.value.code is ErrorCode.OCR_EMPTY_RESULT


# ------------------------------------------------------------------ request
def test_request_is_deterministic_and_high_detail(configured, image) -> None:
    client = _StubClient(_StubResponse(200, _completion(RECEIPT_TEXT)))
    _provider(configured, client).extract(OCRRequest(image=image))

    payload = client.last_payload
    assert payload is not None
    assert payload["temperature"] == 0, "transcription must be reproducible"
    assert payload["model"] == "stub-vision"

    image_part = payload["messages"][1]["content"][1]
    assert image_part["image_url"]["detail"] == "high"
    assert image_part["image_url"]["url"].startswith("data:image/jpeg;base64,")


def test_prompt_states_the_untrusted_data_contract(configured, image) -> None:
    """Receipt text can carry instruction-like content."""
    client = _StubClient(_StubResponse(200, _completion(RECEIPT_TEXT)))
    _provider(configured, client).extract(OCRRequest(image=image))

    system = client.last_payload["messages"][0]["content"].lower()  # type: ignore[index]
    assert "never instructions" in system
    assert "do not correct" in system


def test_oversized_images_are_downscaled(configured) -> None:
    """Cost rises with resolution well past the point accuracy stops improving."""
    client = _StubClient(_StubResponse(200, _completion(RECEIPT_TEXT)))
    huge = np.full((4000, 3000, 3), 240, np.uint8)
    _provider(configured, client).extract(OCRRequest(image=huge))

    encoded = client.last_payload["messages"][1]["content"][1]["image_url"]["url"]  # type: ignore[index]
    # A 4000px image encoded at full size would be far larger than this.
    assert len(encoded) < 2_000_000


# ------------------------------------------------------------------- errors
def test_rate_limit_is_retryable(configured, image) -> None:
    """429 is transient; it must map to an error the retry policy retries."""
    client = _StubClient(_StubResponse(429, {}))
    with pytest.raises(ProviderUnavailableError) as exc:
        _provider(configured, client).extract(OCRRequest(image=image))
    assert exc.value.retryable


def test_server_error_is_not_retryable_and_hides_the_body(configured, image) -> None:
    client = _StubClient(_StubResponse(500, {"error": "internal detail"}))
    with pytest.raises(OCRError) as exc:
        _provider(configured, client).extract(OCRRequest(image=image))
    assert exc.value.code is ErrorCode.OCR_FAILED
    assert "internal detail" not in str(exc.value.details)


def test_timeout_maps_to_ocr_timeout(configured, image) -> None:
    import httpx

    client = _StubClient(error=httpx.TimeoutException("timed out"))
    with pytest.raises(OCRTimeoutError) as exc:
        _provider(configured, client).extract(OCRRequest(image=image))
    assert exc.value.retryable


def test_transport_error_is_normalised(configured, image) -> None:
    """A vendor exception must never escape the provider."""
    import httpx

    client = _StubClient(error=httpx.ConnectError("refused"))
    with pytest.raises(OCRError):
        _provider(configured, client).extract(OCRRequest(image=image))


def test_malformed_response_is_normalised(configured, image) -> None:
    client = _StubClient(_StubResponse(200, {"unexpected": "shape"}))
    with pytest.raises(OCRError):
        _provider(configured, client).extract(OCRRequest(image=image))


# -------------------------------------------------- the reason for the layer
def test_downstream_validation_still_catches_a_hallucinated_total(
    configured, image, settings
) -> None:
    """The whole argument for keeping the deterministic layer.

    A vision model can transcribe a total it could not actually read. The
    arithmetic check is what turns that into a visible warning instead of a
    confident wrong answer.
    """
    from app.extraction.receipt import RuleBasedReceiptExtractor
    from app.validation.engine import ValidationEngine

    hallucinated = "SHOP\nItem 2.00\nSUBTOTAL 2.00\nTAX 0.20\nTOTAL 999.00"
    client = _StubClient(_StubResponse(200, _completion(hallucinated)))
    ocr = _provider(configured, client).extract(OCRRequest(image=image))

    extraction = RuleBasedReceiptExtractor(settings).extract(ocr)
    result = ValidationEngine(settings).validate(extraction, ocr=ocr)

    assert "TOTAL_MISMATCH" in result.codes()
    assert not result.is_valid


# --------------------------------------------------- retired / wrong model
def test_unknown_model_names_itself_and_the_fix(configured, image) -> None:
    """Model names change on the provider's schedule, so this failure is likely.

    A bare "HTTP 404" would send the reader hunting through logs; the message
    has to name the configured model and how to find a valid one.
    """
    body = {"error": {"message": "The model `stub-vision` does not exist"}}
    client = _StubClient(_StubResponse(404, body))

    with pytest.raises(ProviderUnavailableError) as exc:
        _provider(configured, client).extract(OCRRequest(image=image))

    message = exc.value.message
    assert "stub-vision" in message, "must name the model that failed"
    assert "check_llm" in message, "must point at how to find a valid one"
    assert exc.value.details["configured_model"] == "stub-vision"
    # Retryable so a transient provider blip is not treated as fatal config.
    assert exc.value.retryable


def test_no_access_to_a_model_is_treated_the_same(configured, image) -> None:
    body = {"error": {"message": "You do not have access to model gpt-x"}}
    client = _StubClient(_StubResponse(403, body))
    # 403 is not in the model-error branch, so this stays a generic failure.
    with pytest.raises(OCRError):
        _provider(configured, client).extract(OCRRequest(image=image))


def test_other_400s_are_not_misreported_as_a_model_problem(configured, image) -> None:
    """A malformed request must not be blamed on the model name."""
    body = {"error": {"message": "Invalid value for 'temperature'"}}
    client = _StubClient(_StubResponse(400, body))

    with pytest.raises(OCRError) as exc:
        _provider(configured, client).extract(OCRRequest(image=image))
    assert "not available to this API key" not in exc.value.message
