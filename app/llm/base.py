"""LLM provider interface.

Mirrors the OCR abstraction: one interface, configuration-driven selection, no
vendor names anywhere in pipeline code.

The interface is deliberately narrow -- a single structured-extraction call.
The LLM is used here for one purpose (reading fields out of text that rules
could not confidently parse), so a broad chat abstraction would be surface area
without benefit.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from app.core.versions import PROMPT_VERSION


@dataclass(frozen=True, slots=True)
class LLMRequest:
    """One structured-extraction request.

    Attributes:
        system_prompt: Instructions and the untrusted-data contract.
        document_text: OCR text, treated strictly as data.
        json_schema: Schema the response must satisfy. Providers that support
            constrained decoding enforce it; others receive it in the prompt
            and the result is validated on return either way.
        temperature: Sampling temperature. Extraction uses 0.
        max_output_tokens: Response cap.
        timeout_seconds: Wall-clock bound.
        prompt_version: Recorded in processing metadata for traceability.
    """

    system_prompt: str
    document_text: str
    json_schema: dict[str, Any]
    temperature: float = 0.0
    max_output_tokens: int = 4096
    timeout_seconds: float = 45.0
    prompt_version: str = PROMPT_VERSION


@dataclass(frozen=True, slots=True)
class LLMResponse:
    """A structured-extraction response.

    ``data`` is the parsed JSON object. It has *not* been validated against
    business rules -- that is the caller's job, and it is never skipped.
    """

    data: dict[str, Any]
    model: str
    provider: str
    duration_ms: float
    #: Token usage when the provider reports it, for cost observability.
    input_tokens: int | None = None
    output_tokens: int | None = None
    #: Reason generation stopped; a truncated response must not be trusted.
    finish_reason: str | None = None


class LLMProvider(ABC):
    """A structured-extraction backend."""

    #: Registry key used in ``LLM_PROVIDER``.
    name: str = "unnamed"

    @abstractmethod
    def extract(self, request: LLMRequest) -> LLMResponse:
        """Run structured extraction.

        Raises:
            LLMError: The call failed.
            LLMTimeoutError: The call exceeded its timeout.
            LLMInvalidOutputError: The response was not parseable JSON.
        """

    def health_check(self) -> tuple[bool, str | None]:
        """Whether the provider is configured and reachable.

        Must never expose a credential in its detail string.
        """
        return True, None

    def describe(self) -> dict[str, Any]:
        """Non-sensitive provider metadata."""
        return {"provider": self.name}
