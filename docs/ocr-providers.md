# OCR providers

## The contract

One interface. Implement it, register it, select it with `OCR_PROVIDER`.
No other module in the codebase branches on which provider is active — that is
the whole point of the abstraction.

```python
class OCRProvider(ABC):
    name: str

    @abstractmethod
    def extract(self, request: OCRRequest) -> OCRResult: ...

    def health_check(self) -> tuple[bool, str | None]:
        return True, None

    def describe(self) -> dict[str, Any]:
        return {"provider": self.name}
```

### What a provider receives

```python
@dataclass(frozen=True, slots=True)
class OCRRequest:
    image: np.ndarray               # already decoded and preprocessed
    languages: tuple[str, ...]
    timeout_seconds: float
    original_bytes: bytes | None    # for APIs that want the original encoding
    hints: dict[str, Any] | None    # non-authoritative context
```

Image handling is done for you, so it is not reimplemented per provider.
`original_bytes` exists because cloud APIs and vision models usually prefer the
original file to a re-encoded array.

### What a provider must return

An `OCRResult`. Two rules matter:

**Never fabricate what you cannot measure.** If your engine reports no
confidence, leave it `None` — a hardcoded `0.99` would poison every confidence
score downstream. If it reports no geometry, leave `bbox` as `None`; the
pipeline degrades to non-spatial heuristics and raises `OCR_NO_GEOMETRY`.

**Never leak vendor detail into the result.** Provider-specific extras belong
in `provider_metadata`, which is quarantined from the receipt contract.

### How a provider must fail

Raise `PipelineError` subclasses — never a vendor exception, which would leak
implementation detail into callers and bypass the retry policy:

| Situation | Raise | Retried? |
|---|---|---|
| Recognition failed | `OCRError` | no |
| Timed out | `OCRTimeoutError` | **yes** |
| Engine or credentials unavailable | `ProviderUnavailableError` | **yes** |
| Nothing readable | `OCRError(code=OCR_EMPTY_RESULT)` | no |

Retry eligibility is decided solely by `RETRYABLE_ERROR_CODES`. A bad image or
a rejected credential fails identically on every attempt, so retrying it only
multiplies latency and amplifies load during an incident.

---

## Bundled providers

### `tesseract` (default)

Local, free, private — no data leaves the host and there is no external
dependency in the request path, which matters for a document as sensitive as a
receipt.

Uses `image_to_data` rather than `image_to_string`, because the word-level
geometry and per-word confidence it returns are what make spatial extraction
and honest confidence scoring possible. Words are regrouped into lines using
Tesseract's own block/paragraph/line indices rather than y-coordinate
clustering, which is more reliable on multi-column footers.

Configuration: `TESSERACT_CMD`, `TESSERACT_TESSDATA_DIR`, `TESSERACT_PSM`
(default 6 = uniform text block, the best general default for receipts),
`TESSERACT_OEM`.

### `fixture`

Replays **recorded** OCR output from disk. It does not generate anything — it
is test infrastructure that makes the deterministic stages reproducible in CI
without a native binary, and it is what golden tests run against.

It refuses to run when `APP_ENV=production`, so a misconfigured deployment
fails loudly instead of silently serving canned data.

---

## Adding a provider

```python
# app/ocr/providers/my_engine.py
from typing import Any

from app.core.config import Settings
from app.core.exceptions import OCRError, OCRTimeoutError
from app.ocr.base import OCRProvider, OCRRequest
from app.ocr.factory import register_provider
from app.schemas.ocr import OCRBox, OCRLine, OCRResult


@register_provider("my_engine")
class MyEngineProvider(OCRProvider):
    """One-line description of the engine."""

    name = "my_engine"

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._client: Any = None      # build lazily; startup must not need it

    def health_check(self) -> tuple[bool, str | None]:
        if not self._settings.my_engine_api_key.get_secret_value():
            # Report the reason, never the credential.
            return False, "MY_ENGINE_API_KEY is not configured."
        return True, None

    def extract(self, request: OCRRequest) -> OCRResult:
        try:
            native = self._call(request)
        except TimeoutError as exc:
            raise OCRTimeoutError(
                "OCR timed out.", details={"provider": self.name}
            ) from exc
        except Exception as exc:
            raise OCRError(
                "OCR engine failed.",
                details={"provider": self.name, "error_type": type(exc).__name__},
            ) from exc

        lines = tuple(
            OCRLine(
                text=block.text,
                confidence=block.score,          # None when unavailable
                bbox=OCRBox(x=..., y=..., width=..., height=...),
            )
            for block in native.blocks
        )
        return OCRResult(
            text="\n".join(line.text for line in lines),
            lines=lines,
            provider=self.name,
            model=native.model_version,
            languages=request.languages,
            provider_metadata={"native_request_id": native.id},
        )
```

Then:

1. Import it in `app/ocr/providers/__init__.py` — registration is
   import-triggered.
2. Add credentials to `Settings` as `SecretStr`, and to `.env.example` as an
   empty placeholder.
3. Set `OCR_PROVIDER=my_engine`.

For a provider needing construction beyond `cls(settings)`, use
`register_provider_factory(name, factory)` instead of the decorator.

### Checklist

- [ ] Failures raise `PipelineError` subclasses only
- [ ] Timeouts raise `OCRTimeoutError` so the retry policy applies
- [ ] `health_check` gives a reason and never exposes a credential
- [ ] `None` wherever confidence or geometry is unavailable — nothing invented
- [ ] Vendor extras confined to `provider_metadata`
- [ ] `extract()` is thread-safe (instances are shared across requests)
- [ ] A recorded fixture and a golden expectation are added

---

## A note on vision LLMs

A vision model can be an OCR provider: implement `extract()`, send
`request.original_bytes`, and normalise the response into `OCRResult`.

Two cautions. It will not report per-word geometry or calibrated confidence, so
leave both `None` rather than inventing them — the pipeline handles that
honestly. And a vision model *is* susceptible to instructions printed on the
document, so everything in
[extraction.md](extraction.md#prompt-injection) applies to it as an OCR
provider, not merely as an extractor.
