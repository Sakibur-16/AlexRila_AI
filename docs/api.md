# API reference

Base path: `/api/v1`. Operational endpoints are unversioned, because probes
must not move when the API does.

## The envelope

Every response, success or failure, has the same five keys:

```json
{
  "success": true,
  "data": { },
  "warnings": [ ],
  "errors": [ ],
  "processing": { "request_id": "...", "api_version": "v1",
                  "pipeline_version": "1.0.0", "schema_version": "1.0",
                  "processing_time_ms": 812.4 }
}
```

`success` answers **"did processing complete?"**, not "is the result certain?".
A receipt read with warnings is `success: true`. Only an unusable input or an
OCR failure is `success: false`, and then `data` is `null`.

---

## `POST /api/v1/receipts/extract`

`multipart/form-data`:

| Field | Type | Required | Notes |
|---|---|---|---|
| `image` | file | yes | JPEG, PNG, WebP, TIFF, BMP |
| `fixture` | string | no | Development only. Ignored when `APP_ENV=production`. |

Optional `X-Request-ID` request header is sanitised, echoed in the response
header and body, and attached to every log line for the request. Supply your
own to correlate traces across services.

```bash
curl -X POST http://localhost:8000/api/v1/receipts/extract \
  -H "X-Request-ID: trace-abc-123" \
  -F "image=@receipt.jpg"
```

---

## The receipt document

`data` is a `Receipt`. Field by field:

### Top level

| Field | Type | Notes |
|---|---|---|
| `schema_version` | string | `"1.0"`. Check it. |
| `document_type` | string | `"receipt"` |
| `subtotal` | string \| null | **money** |
| `total` | string \| null | **money** |
| `shipping`, `service_charge`, `tip`, `rounding_adjustment` | string \| null | **money** |
| `currency` | string \| null | ISO-4217, `null` when undetermined |
| `currency_symbol` | string \| null | As printed — evidence for the code |
| `receipt_number` | string \| null | |
| `notes` | string \| null | |

### `merchant`

`name`, `address`, `phone`, `email`, `website`, `tax_id`, `registration_id`,
`store_id`, `country` — all `string | null`.

### `transaction`

| Field | Type | Notes |
|---|---|---|
| `date` | string \| null | ISO `YYYY-MM-DD`. **`null` when ambiguous.** |
| `time` | string \| null | ISO `HH:MM:SS` |
| `datetime` | string \| null | Present only when both parts resolved |
| `raw_date`, `raw_time` | string \| null | Always what was printed |
| `transaction_id`, `cashier`, `register_id`, `timezone` | string \| null | |

**Always read `raw_date` when `date` is `null`.** That combination means "a
date was found but could not be resolved", which is different from "no date".

### `items[]`

`description`, `sku`, `unit`, `category` (string \| null); `quantity` (string \|
null, decimal); `unit_price`, `total_price`, `discount`, `tax` (**money**);
`tax_rate` (string \| null); `line_index` (int \| null).

### `tax`

`total` (**money**), `inclusive` (bool \| null), and `details[]` of
`{ name, rate, amount, taxable_amount }`.

### `discount`

`{ description, amount, rate }` or `null`. `amount` is a **positive
magnitude**, even though receipts print discounts negatively.

### `payment`

`method` (enum: `cash`, `card`, `credit_card`, `debit_card`, `mobile`,
`voucher`, `bank_transfer`, `check`, `other`), `card_type`, `card_last4`
(exactly 4 digits or `null` — **never more**), `authorization_code`,
`amount_paid`, `change`, `splits[]`.

### `confidence`

`overall`, `ocr`, `extraction`, `validation` (floats 0–1) and `fields`, a map
of dotted path to score. See [confidence.md](confidence.md).

### `validation`

`is_valid` (bool) plus `errors[]`, `warnings[]`, `infos[]` of
`{ code, severity, message, field, context }`. See
[validation.md](validation.md).

### `review`

`review_required` (bool), `status` (`not_required` | `pending` | `in_review` |
`approved` | `rejected`), `reasons[]`.

### `raw_ocr`

`text` (null when `INCLUDE_RAW_OCR=false`), `line_count`, `mean_confidence`,
`redacted` (true when card-like sequences were masked).

### `evidence`

Present only when `INCLUDE_EVIDENCE=true`: dotted path to
`{ source_text, method, line_index, bbox, ocr_confidence }`.

### `processing`

`request_id`, `pipeline_version`, `schema_version`, `ocr_provider`,
`ocr_model`, `ocr_languages`, `preprocessing_applied`, `llm_used`,
`llm_provider`, `llm_model`, `prompt_version`, `processing_time_ms`,
`stage_timings_ms`.

---

## Errors

```json
{
  "success": false,
  "data": null,
  "warnings": [],
  "errors": [{ "code": "IMAGE_TOO_LARGE",
               "message": "File exceeds the 10 MB limit.",
               "field": null, "details": null }],
  "processing": { "request_id": "..." }
}
```

| Status | Codes |
|---|---|
| 400 | `EMPTY_FILE`, `INVALID_IMAGE` |
| 413 | `IMAGE_TOO_LARGE`, `REQUEST_TOO_LARGE` |
| 415 | `UNSUPPORTED_FILE_TYPE` |
| 422 | `IMAGE_TOO_SMALL`, request validation |
| 429 | `RATE_LIMITED` |
| 500 | `INTERNAL_ERROR`, `EXTRACTION_FAILED` |
| 502 | `OCR_FAILED`, `LLM_FAILED` |
| 503 | `PROVIDER_UNAVAILABLE` |
| 504 | `OCR_TIMEOUT` |

`details` is populated only when `DEBUG_ERRORS=true`. Stack traces, file paths
and provider error text never reach a client — quote the `request_id` to find
the full record in the logs.

---

## Operational endpoints

### `GET /health`

Liveness. Touches no dependency, so a slow engine can never cause a restart
loop. Always `200` while the process is alive.

### `GET /ready`

Readiness. Checks the OCR provider (and the LLM provider when enabled) and
returns `503` when it cannot serve.

```json
{ "ready": true,
  "components": [{ "name": "ocr:tesseract", "ready": true, "detail": null }] }
```

The LLM is an optional enhancement — its absence does not make an instance
unready, because the deterministic path works without it.

### `GET /version`

API, pipeline and schema versions, plus the active OCR provider.

### `GET /metrics`

A metrics snapshot for development. In production, bind a real sink with
`set_metrics_sink()` and scrape that instead.

---

## Integration guidance

**Parse money as decimal.**

```typescript
import { Decimal } from "decimal.js";
const total = data.total ? new Decimal(data.total) : null;   // never Number()
```

**Branch on outcome, not on status alone.**

```typescript
if (!body.success)                         return handleFailure(body.errors);
if (body.data.review.review_required)      return queueForReview(body.data);
if (body.data.confidence.overall < 0.9)    return flagForAudit(body.data);
return autoApprove(body.data);
```

**Treat `null` as information.** `merchant.phone === null` means the receipt
did not print one — not that extraction failed.

**Pin `schema_version`.** Minor bumps add optional fields; a major bump may
retype or remove one.

---

## Versioning

| Version | Changes when |
|---|---|
| `api_version` | The envelope changes incompatibly → new URL prefix |
| `schema_version` | The receipt document shape changes |
| `pipeline_version` | Extraction *behaviour* changes, even if the shape does not |

They move independently, and all three appear in every response. A
`pipeline_version` bump with an unchanged `schema_version` means the same
fields may now hold different values — worth noticing.

---

## Not implemented

Async job endpoints (`POST /receipts/jobs`, `GET /receipts/jobs/{id}`) are
sketched in [roadmap.md](roadmap.md) but deliberately not built: at receipt
sizes, synchronous processing is well within a normal HTTP timeout, and the
job store, worker pool and polling contract would be complexity without a
current justification.
