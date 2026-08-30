# Architecture

## The governing idea

The system is **not** `image -> OCR -> JSON`. It is a sequence of stages, each
with one responsibility and a typed boundary, arranged so that the two things
most likely to change — the OCR engine and the extraction strategy — can be
replaced without touching anything else.

```
DocumentInput (bytes)
      |
      v
[ input validation ]   security/files.py        magic bytes, size, MIME allow-list
      |
      v
[ decode + limits  ]   preprocessing/transforms.py
      |
      v
[ quality assess   ]   preprocessing/quality.py  blur, brightness, contrast, skew
      |
      v
[ preprocessing    ]   preprocessing/pipeline.py conditional transforms
      |
      v
[ OCR              ]   ocr/base.py + providers/  ---> OCRResult
      |
      v
[ context build    ]   extraction/context.py     line views, sections, locale
      |
      v
[ extraction       ]   extraction/*.py           ---> ExtractionResult
      |
      v
[ validation       ]   validation/*.py           ---> ValidationResult
      |
      v
[ confidence       ]   confidence/scorer.py      ---> ConfidenceReport
      |
      v
[ LLM fallback ?   ]   llm/extractor.py          only when confidence is low
      |
      v
[ assembly         ]   pipeline/assembly.py      ---> Receipt (public schema)
      |
      v
PipelineResult
```

---

## Layers

### `app/core` — cross-cutting concerns

`config` (environment-driven settings, frozen and cached), `exceptions` (the
`ErrorCode` taxonomy), `logging` (structlog with secret scrubbing and a
document-content gate), `metrics` (a sink interface with a bounded in-memory
default), `retry` (exponential backoff with full jitter, applied only to
transient errors), `versions` (three independent version numbers).

### `app/domain` — document abstractions

`DocumentInput` and `DocumentType` are deliberately *not* receipt-specific.
Everything upstream of extraction speaks in documents, so adding invoices later
means adding an extractor, not reworking ingest.

`evidence.py` holds the most important internal type:

```python
@dataclass(frozen=True, slots=True)
class ExtractedField[T]:
    value: T | None
    confidence: float
    evidence: tuple[Evidence, ...]     # source text, line index, bbox, method
    raw_value: str | None              # text before normalisation
    warnings: tuple[str, ...]
```

Every extractor returns these, never bare values. That is what makes §37
field-level traceability real rather than aspirational, and it is the
mechanism by which hallucination is structurally difficult: a deterministic
extractor cannot produce a value without citing the line it came from.

### `app/schemas` — contracts

`ocr.py` is the unified OCR representation every provider normalises into.
`receipt.py` is the versioned public document. `common.py` defines `Money`
(Decimal in Python, string on the wire — see the README). `validation.py`,
`quality.py`, `pipeline.py` and `response.py` complete the surface.

All models are `frozen=True`. A result that is passed between stages cannot be
mutated by one of them.

### `app/preprocessing` — image preparation

**The rule: apply a transform only when the measurements say it will help.**

Blindly running every enhancement is the most common way a preprocessing stage
makes a document pipeline *worse*. Modern OCR engines are trained on
photographs of documents; aggressive binarisation and denoising destroy the
anti-aliased glyph edges they rely on.

So each transform is gated:

| Transform | Applied when |
|---|---|
| deskew | measured skew exceeds `DESKEW_MIN_ANGLE_DEGREES` |
| resize | the short edge is outside the engine's working band |
| grayscale | always (colour carries no textual information) |
| denoise | the image is soft (`blur_score < threshold * 3`) |
| contrast (CLAHE) | contrast is low or exposure is off |
| sharpen | the image is genuinely blurry — **off by default** |
| threshold | explicitly enabled — **off by default** |

Every applied *and skipped* step is recorded in the `PreprocessingReport`, so a
bad extraction can always be traced to the image OCR actually saw.

### `app/ocr` — engine abstraction

One ABC, one factory, configuration-driven selection. Application code calls
`create_ocr_provider(settings=...)` and never names a class. See
[ocr-providers.md](ocr-providers.md).

Providers report `None` for anything they cannot measure. Downstream code
degrades: without bounding boxes, spatial heuristics are skipped and an
`OCR_NO_GEOMETRY` info issue is raised — the pipeline still works.

### `app/normalization` — making text parseable

- **`text.py`** — context-aware glyph repair. `O -> 0` is applied only inside
  tokens that already look numeric *and* either start with a digit or carry a
  decimal separator. `COFFEE`, `LOSS`, `B12`, `1ST` and `A1B2C3` survive
  untouched; `1O.99` and `2S.OO` are repaired. Raw text is never mutated.
- **`money.py`** — the decimal convention is inferred **per document**, from
  every amount on it, not per number. Reading `1.234,50` as `1.23` is a
  1000× error that arithmetic validation would then "confirm" against an
  equally misparsed total.
- **`dates.py`** — never guesses. A date resolves only when self-disambiguating
  or when `DATE_ORDER` declares a locale.
- **`currency.py`** — evidence-ranked. `$` alone never means USD.
- **`numbers.py`** — phone, email, URL, tax id, receipt id, card scheme.

### `app/extraction` — finding fields

`context.py` computes the shared document view once: cleaned lines, a
keyword-matching projection, parsed amounts, and section assignment
(header / metadata / items / totals / payment / footer).

`lexicon.py` holds all label vocabulary in one place, with a priority ordering
that guarantees `SUBTOTAL` can never satisfy a request for `TOTAL` — the single
most damaging failure mode in receipt parsing.

Then one module per field group: `merchant`, `dates`, `items`, `totals`,
`payment`. `receipt.py` orchestrates them into an `ExtractionResult`.

### `app/validation` — three independent checks

`quality` explains why extraction may have gone wrong; `semantic` checks
individual values; `financial` checks the relationships between them. All three
run — a single response tells a consumer everything questionable about a
document. See [validation.md](validation.md).

### `app/confidence` — composed scoring

Three signals per field: OCR legibility, extraction-method strength, and
validation outcome. Documented in full in [confidence.md](confidence.md).

### `app/llm` — optional semantic fallback

Off by default. Fires only on low confidence. Can only fill fields that are
null or low-confidence. Output must parse, must be grounded in the OCR text,
and is re-validated afterwards — and if the merge *lowers* confidence, it is
discarded. See [extraction.md](extraction.md#llm-fallback).

### `app/pipeline` — orchestration

`receipt_pipeline.py` runs the stages, times each one, records metrics and
handles failure. `assembly.py` projects the evidence-carrying internal result
into the public schema — the one place where the contract is constructed.

---

## Design decisions and their reasons

### Money is a string on the wire

The most consequential decision in the contract. JSON numbers are doubles in
every mainstream parser. A pipeline that validates totals to the cent must not
discard that precision at serialisation. Documented prominently because it is
the one genuinely surprising thing about the API.

### Uncertainty is data, not failure

Three distinct outcomes, and conflating them is a design error:

| Outcome | Shape |
|---|---|
| Extracted cleanly | `success: true`, no warnings, high confidence |
| Extracted with doubt | `success: true`, warnings, `review_required` |
| Could not process | `success: false`, structured error |

### Evidence internally, plain values publicly

The internal model carries provenance for everything. The public schema returns
plain values by default and evidence on request (`INCLUDE_EVIDENCE`). The rich
model does not force complexity on consumers who do not need it, and the
capability is there when human review or auditing arrives.

### Sections are hints, not constraints

Every extractor prefers its own section but falls back to the whole document.
Receipts do not reliably follow a layout — and a footer note containing the
word "total" should not be able to strand the real items outside the search
region. (It could, once; there is a fallback and a golden fixture for it now.)

### Three independent version numbers

`API_VERSION` (URL contract), `SCHEMA_VERSION` (document shape),
`PIPELINE_VERSION` (extraction behaviour). Output *values* can change without
the shape changing; that must be expressible, and it must be visible in the
response.

---

## Concurrency

The pipeline and its collaborators are built once at startup and shared. They
hold no per-document state — all of that lives in the `ReceiptContext`, which
is created per request. Provider instances must be thread-safe for `extract()`;
the bundled ones are.

`InMemoryMetricsSink` is lock-guarded and bounded, so a long-running process
cannot grow without limit through it.

---

## Extending to other document types

The abstractions are already generic where it matters:

| Reusable as-is | Receipt-specific |
|---|---|
| `DocumentInput`, `DocumentType` | `Receipt` schema |
| Input validation, preprocessing | `extraction/lexicon.py` |
| The whole OCR layer | `extraction/*.py` |
| `ExtractedField`, `Evidence` | `validation/financial.py` |
| Confidence scoring | |
| Response envelopes | |

An invoice extractor means: an `Invoice` schema, an invoice lexicon, invoice
extractors, and a `DocumentType` branch in the pipeline. Ingest, OCR,
normalisation, confidence and the API are untouched.

Future RAG (embed the structured JSON, store, retrieve) attaches after
assembly and is deliberately not built — nothing in the current design
obstructs it, and nothing depends on it.
