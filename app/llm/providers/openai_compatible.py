"""OpenAI-compatible chat-completions provider.

Many inference services -- hosted and self-hosted alike -- expose the
``/v1/chat/completions`` shape. Implementing that one wire format against a
configurable ``LLM_BASE_URL`` therefore covers a wide range of deployments
without the pipeline knowing which is in use.

The base URL and model name are configuration; nothing is hardcoded. The API
key is held as a :class:`~pydantic.SecretStr` and is only ever unwrapped into
an Authorization header -- never logged, never echoed in an error.
"""

from __future__ import annotations

import json
from typing import Any

from app.core.config import Settings
from app.core.exceptions import ErrorCode, LLMError, LLMInvalidOutputError, ProviderUnavailableError
from app.core.logging import get_logger
from app.llm.base import LLMProvider, LLMRequest, LLMResponse
from app.llm.factory import register_llm_provider

logger = get_logger(__name__)

_JSON_MEDIA_TYPE = "application/json"


def _error_identity(response: Any) -> str:
    """The provider's machine-readable error type/code, or ""."""
    try:
        error = response.json().get("error", {})
    except (ValueError, AttributeError):
        return ""
    return str(error.get("type") or error.get("code") or "")


def _error_message(response: Any) -> str:
    """The provider's own message, which names the fix precisely."""
    try:
        return str(response.json().get("error", {}).get("message", ""))[:300]
    except (ValueError, AttributeError):
        return ""


@register_llm_provider("openai_compatible")
class OpenAICompatibleProvider(LLMProvider):
    """Structured extraction over an OpenAI-compatible chat endpoint."""

    name = "openai_compatible"

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._client: Any = None

    # -------------------------------------------------------------- plumbing
    def _http(self) -> Any:
        """Build the HTTP client lazily and reuse the connection pool."""
        if self._client is not None:
            return self._client
        try:
            import httpx
        except ImportError as exc:  # pragma: no cover - packaging failure
            raise ProviderUnavailableError(
                "httpx is required for the openai_compatible LLM provider.",
                details={"provider": self.name},
            ) from exc

        base_url = self._settings.llm_base_url.rstrip("/")
        if not base_url:
            raise ProviderUnavailableError(
                "LLM_BASE_URL is not configured.",
                details={"provider": self.name},
            )

        self._client = httpx.Client(
            base_url=base_url,
            timeout=self._settings.llm_timeout_seconds,
            headers={"Content-Type": _JSON_MEDIA_TYPE},
        )
        return self._client

    def _auth_headers(self) -> dict[str, str]:
        """Authorization header, or empty for an unauthenticated endpoint."""
        key = self._settings.llm_api_key.get_secret_value()
        return {"Authorization": f"Bearer {key}"} if key else {}

    # ---------------------------------------------------------------- public
    def health_check(self) -> tuple[bool, str | None]:
        if not self._settings.llm_base_url:
            return False, "LLM_BASE_URL is not configured."
        if not self._settings.llm_model:
            return False, "LLM_MODEL is not configured."
        return True, None

    def describe(self) -> dict[str, Any]:
        return {
            "provider": self.name,
            "model": self._settings.llm_model,
            "base_url_configured": bool(self._settings.llm_base_url),
        }

    def extract(self, request: LLMRequest) -> LLMResponse:
        import httpx

        client = self._http()
        model = self._settings.llm_model
        if not model:
            raise ProviderUnavailableError(
                "LLM_MODEL is not configured.", details={"provider": self.name}
            )

        payload = {
            "model": model,
            "temperature": request.temperature,
            "max_tokens": request.max_output_tokens,
            # Constrained decoding where the endpoint supports it. Endpoints
            # that ignore it still receive the schema in the system prompt, and
            # the response is validated on return regardless.
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "receipt_extraction",
                    "strict": True,
                    "schema": request.json_schema,
                },
            },
            "messages": [
                {"role": "system", "content": request.system_prompt},
                {"role": "user", "content": request.document_text},
            ],
        }

        started = _now_ms()
        try:
            response = client.post(
                "/chat/completions",
                json=payload,
                headers=self._auth_headers(),
                timeout=request.timeout_seconds,
            )
        except httpx.TimeoutException as exc:
            raise LLMError(
                "LLM request timed out.",
                code=ErrorCode.LLM_TIMEOUT,
                details={"provider": self.name},
            ) from exc
        except httpx.HTTPError as exc:
            raise LLMError(
                "LLM request failed.",
                details={"provider": self.name, "error_type": type(exc).__name__},
            ) from exc

        duration_ms = _now_ms() - started

        if response.status_code == 429:
            # A 429 covers both transient rate limiting and a permanently
            # exhausted credit balance. Retrying the latter can never succeed,
            # and calling it "rate limited" points at the wrong fix.
            error = _error_identity(response)
            if "quota" in error or "credit" in error:
                raise LLMError(
                    _error_message(response) or "The provider account has no remaining credit.",
                    code=ErrorCode.PROVIDER_QUOTA_EXHAUSTED,
                    details={"provider": self.name, "status_code": 429},
                )
            raise LLMError(
                "LLM provider rate limit reached.",
                code=ErrorCode.LLM_TIMEOUT,  # retryable
                details={"provider": self.name, "status_code": 429},
            )

        if response.status_code >= 400:
            # The body may echo request content; only the status is surfaced.
            raise LLMError(
                "LLM provider returned an error response.",
                details={"provider": self.name, "status_code": response.status_code},
            )

        try:
            body = response.json()
            choice = body["choices"][0]
            content = choice["message"]["content"]
            data = json.loads(content)
        except (KeyError, IndexError, ValueError, TypeError) as exc:
            raise LLMInvalidOutputError(
                "LLM response was not valid structured JSON.",
                details={"provider": self.name, "error_type": type(exc).__name__},
            ) from exc

        if not isinstance(data, dict):
            raise LLMInvalidOutputError(
                "LLM response was not a JSON object.",
                details={"provider": self.name},
            )

        usage = body.get("usage") or {}
        return LLMResponse(
            data=data,
            model=body.get("model", model),
            provider=self.name,
            duration_ms=duration_ms,
            input_tokens=usage.get("prompt_tokens"),
            output_tokens=usage.get("completion_tokens"),
            finish_reason=choice.get("finish_reason"),
        )


def _now_ms() -> float:
    import time

    return time.perf_counter() * 1000.0
