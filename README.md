# Receipt Intelligence Pipeline

Production receipt OCR and structured extraction. Submit a receipt image,
receive validated JSON with per-field confidence, machine-readable validation
findings and full provenance.

The OCR engine, the extraction strategy and the optional LLM layer are all
implementation details behind a stable contract. A backend consuming this
service never needs to know which of them is in use.

```
POST /api/v1/receipts/extract   (multipart: image=<file>)
  -> { success, data, warnings, errors, processing }
```

---

## Three things to know before you integrate

**1. Monetary values are JSON strings, not numbers.**

```json
{ "total": "17.28", "subtotal": "17.10" }
```

Every mainstream JSON parser turns numbers into IEEE-754 doubles, and
`0.1 + 0.2 !== 0.3` is exactly the class of one-cent discrepancy a financial
pipeline exists to prevent. Parse these into your own decimal type
(`BigDecimal`, `decimal.Decimal`, `Decimal.js`) — never into a float.

**2. `success: true` means "processed", not "certain".**

A receipt that was read but does not add up returns `200` with
`success: true`, a `TOTAL_MISMATCH` warning, a lowered confidence score and
`review.review_required = true`. Uncertainty is data, not a crash. Only an
unusable input or an OCR failure returns `success: false`.

**3. Absent means `null`.**

Never `"N/A"`, never `""`, never `0`. If a receipt has no phone number, the
field is `null`. The pipeline never invents a value to fill a gap, and it never
converts an uncertain reading into a confident one — an ambiguous date such as
`08/09/26` comes back as `date: null` with `raw_date: "08/09/26"` and an
`AMBIGUOUS_DATE_FORMAT` warning, because guessing would be wrong half the time.

---

## Architecture

```
                        Receipt image
                              |
                    [ input validation ]        magic-byte type detection,
                              |                 size and dimension limits
                    [ image quality  ]          blur, brightness, contrast, skew
                              |
                    [ preprocessing  ]          conditional: only transforms the
                              |                 measurements say will help
                    [ OCR abstraction ]  <----- OCR_PROVIDER selects the engine
                              |
                    +---------+---------+
                    |         |         |
               Tesseract   Cloud OCR  Vision LLM      (one interface, any engine)
                    +---------+---------+
                              |
                      Unified OCRResult             text + lines + words +
                              |                     boxes + confidence
                    [ normalisation  ]              context-aware glyph repair,
                              |                     locale-aware money, dates
                    [ structure detect ]            header / items / totals /
                              |                     payment / footer
                    [ extraction     ]              rules first; LLM only on
                              |                     low confidence
                    [ validation     ]              schema, financial, semantic
                              |
                    [ confidence     ]              per-field + aggregate
                              |
                      Versioned JSON
                              |
                       Backend service
```

Each stage is a module with one job, and the stage boundaries are typed. See
[docs/architecture.md](docs/architecture.md).

### Why deterministic extraction first

Almost every receipt field has *shape*: a label (`TOTAL 17.28`), a pattern
(a phone number, a card mask), or a position (the merchant name is the largest
text in the header). Rules exploit that shape at zero marginal cost, with
reproducible output and no possibility of invention. The LLM layer exists for
the residue that genuinely needs semantic judgement — and it is off by default.

---

## Installation

Requires **Python 3.12+** and the **Tesseract** binary.

> **`pip install` does not install the OCR engine.** The `pytesseract`
> dependency is only a thin Python wrapper that shells out to a `tesseract`
> executable. The engine itself is a *system* package and must be installed
> separately — except in Docker, where the image installs it for you.
>
> If it is missing, the service still starts, logs `ocr_provider_not_ready`,
> and `/ready` returns **503** so an orchestrator withholds traffic. It fails
> visibly rather than silently.

```bash
# macOS
brew install tesseract

# Debian / Ubuntu
sudo apt-get install tesseract-ocr tesseract-ocr-eng

# Windows
winget install UB-Mannheim.TesseractOCR
```

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
cp .env.example .env
```

Verify the engine is visible to the application:

```bash
python -c "from app.ocr import create_ocr_provider; print(create_ocr_provider('tesseract').health_check())"
# (True, None)
```

`(False, 'Tesseract binary not found...')` means the binary is not on `PATH` —
set `TESSERACT_CMD` to its absolute path.

---

## Running

```bash
python -m app.main                        # development server, honours HOST/PORT/RELOAD
make run                                  # same, with reload forced on
uvicorn app.main:app --reload --port 8000 # explicit, what production uses
```

`python app/main.py` works too. Note that a bare `import app.main` only builds
the ASGI app object -- the server starts from the `main()` entrypoint or from
uvicorn.

- Interactive docs: <http://localhost:8000/docs>
- OpenAPI schema: <http://localhost:8000/openapi.json>

### Docker

```bash
docker build -t receipt-ocr .
docker run --rm -p 8000:8000 --env-file .env receipt-ocr
```

The image installs Tesseract, runs as an unprivileged user, bakes in no
secrets, and declares a health check. For development with live reload:

```bash
docker compose up --build
```

Add languages by installing more packs in the `runtime` stage
(`tesseract-ocr-deu`, `-fra`, `-spa`, `-ben`, `-hin`, `-ara`) and widening
`OCR_LANGUAGES`.

---

## API

### `POST /api/v1/receipts/extract`

| Field | Type | Notes |
|---|---|---|
| `image` | file | **required.** JPEG, PNG, WebP, TIFF or BMP |
| `fixture` | string | development only; replays a recorded OCR fixture |

Optional `X-Request-ID` header is echoed back and appears in every log line for
the request.

```bash
curl -X POST http://localhost:8000/api/v1/receipts/extract \
  -H "X-Request-ID: trace-abc-123" \
  -F "image=@receipt.jpg"
```

<details>
<summary><b>Example response</b> (abridged)</summary>

```json
{
  "success": true,
  "data": {
    "schema_version": "1.0",
    "document_type": "receipt",
    "merchant": {
      "name": "GREEN VALLEY MARKET",
      "address": "123 Oak Street, Springfield",
      "phone": "5550142",
      "email": null,
      "tax_id": null
    },
    "transaction": {
      "date": "2026-08-25",
      "time": "14:35:00",
      "datetime": "2026-08-25T14:35:00",
      "raw_date": "08/25/2026",
      "transaction_id": "R-2026-00815"
    },
    "items": [
      {
        "description": "Coffee",
        "quantity": "2",
        "unit_price": "4.00",
        "total_price": "8.00",
        "sku": null
      }
    ],
    "subtotal": "17.10",
    "discount": { "description": "DISCOUNT", "amount": "1.10", "rate": null },
    "tax": {
      "total": "1.28",
      "details": [{ "name": "SALES TAX", "rate": "8", "amount": "1.28" }]
    },
    "total": "17.28",
    "currency": "USD",
    "currency_symbol": "$",
    "payment": {
      "method": "card",
      "card_type": "VISA",
      "card_last4": "4321",
      "amount_paid": "20.00",
      "change": "2.72"
    },
    "confidence": {
      "overall": 0.94,
      "ocr": 0.94,
      "extraction": 0.89,
      "validation": 1.0,
      "fields": { "total": 1.0, "merchant.name": 0.75, "transaction.date": 0.95 }
    },
    "validation": { "is_valid": true, "warnings": [], "errors": [] },
    "review": { "review_required": false, "status": "not_required", "reasons": [] },
    "raw_ocr": { "text": "GREEN VALLEY MARKET\n...", "line_count": 22, "redacted": false },
    "processing": {
      "request_id": "trace-abc-123",
      "pipeline_version": "1.0.0",
      "schema_version": "1.0",
      "ocr_provider": "tesseract",
      "processing_time_ms": 812.4
    }
  },
  "warnings": [],
  "errors": [],
  "processing": { "request_id": "trace-abc-123", "api_version": "v1" }
}
```

</details>

### Failure

```json
{
  "success": false,
  "data": null,
  "warnings": [],
  "errors": [{ "code": "UNSUPPORTED_FILE_TYPE", "message": "File content does not match any supported image format." }],
  "processing": { "request_id": "...", "pipeline_version": "1.0.0" }
}
```

| Status | Codes |
|---|---|
| 400 | `EMPTY_FILE`, `INVALID_IMAGE` |
| 413 | `IMAGE_TOO_LARGE` |
| 415 | `UNSUPPORTED_FILE_TYPE` |
| 422 | `IMAGE_TOO_SMALL`, request validation |
| 502 | `OCR_FAILED` |
| 503 | `PROVIDER_UNAVAILABLE` |
| 504 | `OCR_TIMEOUT` |

Full contract and every warning code: [docs/api.md](docs/api.md).

### Operational endpoints

| Endpoint | Purpose |
|---|---|
| `GET /health` | Liveness. Touches no dependency, so a slow engine cannot cause a restart loop. |
| `GET /ready` | Readiness. Checks the OCR provider; returns 503 when it cannot serve. |
| `GET /version` | API, pipeline and schema versions. |
| `GET /metrics` | Metrics snapshot (development; bind a real sink in production). |

---

## Deploying

The image is self-contained -- Tesseract included, nothing to install on the
host. Every push to `main` publishes it to GitHub Container Registry:

```bash
docker pull ghcr.io/sakibur-16/alexrila_ai:latest
docker run -d -p 8000:8000 -e API_KEY=<secret> ghcr.io/sakibur-16/alexrila_ai:latest
```

Full operator guide, configuration table and troubleshooting:
**[DEPLOY.md](DEPLOY.md)**.

## Handing this to a backend developer

Everything needed to integrate is generated into `contract/`, so the service
does not have to be running to code against it:

```bash
make contract        # or: python scripts/export_openapi.py
```

| File | Use |
|---|---|
| `contract/openapi.json` | Import into Postman/Insomnia, or generate a typed client |
| `contract/sample-response.json` | A complete success envelope, produced by the real pipeline |
| `contract/sample-error.json` | The failure envelope |

Point them at [docs/api.md](docs/api.md) for the full contract and
[docs/deployment.md](docs/deployment.md) for deployment. The three things they
must not miss are at the top of this file: **money is a string**, **`success:
true` does not mean certain**, and **absent is `null`**.

## Configuration

Everything is environment-driven; see [`.env.example`](.env.example) for the
annotated full list. The settings that most change behaviour:

| Variable | Default | Effect |
|---|---|---|
| `OCR_PROVIDER` | `tesseract` | Which engine runs. `fixture` replays recorded OCR and refuses to run in production. |
| `OCR_LANGUAGES` | `eng` | Comma-separated ISO 639-2/T codes. |
| `DATE_ORDER` | `none` | `none` never guesses ambiguous dates. Set `DMY`/`MDY` only if you know the source locale. |
| `DEFAULT_COUNTRY` | *(empty)* | Resolves shared symbols: `$` + `CA` → `CAD`. |
| `CONFIDENCE_THRESHOLD` | `0.85` | Below this, `review_required` is set. |
| `TOTAL_TOLERANCE` | `0.05` | Absolute rounding tolerance for arithmetic checks. |
| `LLM_ENABLED` | `false` | Enables the fallback layer. |
| `INCLUDE_RAW_OCR` | `true` | Return full OCR text. |
| `INCLUDE_EVIDENCE` | `false` | Return per-field source text and bounding boxes. |
| `HOST` / `PORT` | `127.0.0.1` / `8000` | Bind address for the development server only. |
| `REDACT_SENSITIVE_TEXT` | `true` | Mask card-like sequences before returning or storing. |

---

## Adding an OCR provider

Implement one interface and register it. No other module changes.

```python
# app/ocr/providers/my_engine.py
from app.ocr.base import OCRProvider, OCRRequest
from app.ocr.factory import register_provider
from app.schemas.ocr import OCRResult

@register_provider("my_engine")
class MyEngineProvider(OCRProvider):
    name = "my_engine"

    def __init__(self, settings): ...

    def extract(self, request: OCRRequest) -> OCRResult:
        # Convert your engine's native response into OCRResult.
        # Leave confidence/bbox as None where your engine reports nothing —
        # never fabricate them.
        ...

    def health_check(self) -> tuple[bool, str | None]:
        return True, None
```

Import it in `app/ocr/providers/__init__.py`, then set `OCR_PROVIDER=my_engine`.
Full walkthrough, including cloud providers and the failure contract:
[docs/ocr-providers.md](docs/ocr-providers.md).

## Adding an LLM provider

Same pattern against `app/llm/base.LLMProvider`, registered with
`@register_llm_provider`. The bundled `openai_compatible` provider works with
any `/v1/chat/completions` endpoint via `LLM_BASE_URL`. Nothing a model returns
is trusted — see [docs/extraction.md](docs/extraction.md#llm-fallback).

---

## Testing

```bash
make test           # everything (286 tests)
make test-unit      # normalisation, extraction, validation, confidence, security
make test-golden    # full-output regression against committed expectations
make evaluate       # per-field accuracy report
make check          # lint + types + tests + golden drift (what CI runs)
```

Golden tests replay recorded OCR fixtures, so the deterministic stages are
tested reproducibly without a native engine. Nine fixtures cover a clean
receipt, European decimals with multiple VAT rates, ambiguous dates and
currency, OCR glyph confusion, arithmetic that does not reconcile, a
geometry-free provider, service charges and split payments, an unmasked card
number, and instruction-like text printed on the document.

Changing extraction behaviour changes a golden file. That diff is the review
signal — regenerate with `make golden` and justify it in the same commit.
Details: [docs/testing.md](docs/testing.md).

---

## Security and privacy

- **Type detection by content**, never by extension or client `Content-Type`.
- **Uploads stay in memory** — never written to disk, so path traversal and
  temp-file cleanup are not concerns. Size is enforced while streaming.
- **Card data**: only `card_last4` is ever captured. Full PANs and labelled
  CVV/PIN values are masked out of OCR text before it is returned, stored or
  sent to any external provider. Full numbers are never stored.
- **Logs** carry no secrets (key-pattern scrubbing) and no document content
  unless `LOG_DOCUMENT_CONTENT` is explicitly enabled.
- **Errors** return a stable code and a safe message. Stack traces, paths and
  provider error text never reach a client.
- **Nothing is persisted** by default. `ENABLE_RAW_OCR_STORAGE=false`.
- **Prompt injection**: receipt text is treated as untrusted data. See
  [docs/extraction.md](docs/extraction.md#prompt-injection).

More: [docs/deployment.md](docs/deployment.md#security).

---

## Known limitations

Stated plainly, because a document-AI system that claims none is not being
honest:

- **Accuracy is unquantified on real-world receipts.** The evaluation harness
  currently measures *drift from a committed baseline*, not accuracy against
  independently labelled ground truth. The fixtures are synthetic. Before
  trusting a number, label a real dataset and run `scripts/evaluate.py`
  against it.
- **PDF input is not implemented.** The architecture accommodates it (the
  ingest path is document-typed, `OCRPage` is multi-page aware), but no
  rasterisation step exists yet.
- **Non-Latin scripts are architecturally supported, not validated.** The
  lexicon carries some German, French, Spanish and Italian terms; Bangla,
  Hindi and Arabic need vocabulary and, for Arabic, right-to-left handling.
- **Handwriting is not supported.** Tesseract does not read it.
- **Merchant name uses a positional heuristic** and is consequently the
  lowest-confidence major field. Geometry improves it when the provider
  reports bounding boxes.
- **Perspective correction is not implemented.** Deskew handles rotation, not
  a photograph taken at an angle.
- **Rate limiting is configured but not enforced** in-process; put it at the
  gateway.
- **Processing is synchronous.** Fine at receipt sizes; a batch API would need
  the job endpoints sketched in [docs/roadmap.md](docs/roadmap.md).

## Roadmap

[docs/roadmap.md](docs/roadmap.md) — ordered by value, with PDF support,
a labelled evaluation set, perspective correction and human review at the top.

## Documentation

| | |
|---|---|
| [architecture.md](docs/architecture.md) | Stage-by-stage design and the reasoning behind it |
| [extraction.md](docs/extraction.md) | How each field is found; adding vocabulary |
| [ocr-providers.md](docs/ocr-providers.md) | The provider contract; adding an engine |
| [validation.md](docs/validation.md) | Every rule and issue code |
| [confidence.md](docs/confidence.md) | The scoring formula, in full |
| [api.md](docs/api.md) | Complete HTTP contract |
| [testing.md](docs/testing.md) | Test strategy, fixtures, evaluation |
| [deployment.md](docs/deployment.md) | Docker, scaling, observability, security |
| [roadmap.md](docs/roadmap.md) | What is next and why |
