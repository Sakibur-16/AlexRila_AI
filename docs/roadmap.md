# Roadmap

Ordered by value, not by ease. Each item states what exists today and what
would have to change — no item here is speculative architecture for its own
sake.

---

## 1. A labelled evaluation dataset

**The highest-value item by a wide margin.**

Today `scripts/evaluate.py` compares pipeline output against expectations that
the pipeline itself generated. That is a genuine regression signal and it is
not an accuracy measurement. Every tuning decision — confidence priors,
tolerances, thresholds — is currently made on reasoning rather than evidence.

What is needed: 200–500 real receipts, hand-labelled, spanning the merchants,
languages and photo quality of actual traffic. The harness already computes
per-field accuracy, item F1 and exact-match rate against arbitrary
expectations, so the tooling is done; the data is not.

Until this exists, the honest claim about accuracy is *unknown*.

## 2. Confidence calibration

Once (1) exists: plot accuracy against reported confidence and fit
`METHOD_PRIOR` and the field weights to real outcomes. The scores are currently
calibrated by construction — a well-founded ranking, not a probability. A
consumer that automates above 0.90 deserves to know what 0.90 actually means.

## 3. PDF input

The architecture accommodates it — `DocumentInput` is format-agnostic,
`OCRPage` is multi-page aware, and the magic-byte sniffer already recognises
`%PDF-`. What is missing is rasterisation (`pypdfium2`), a page loop, and
per-page results merged into one document. Native-text PDFs should skip OCR
entirely and read the embedded text layer, which is both faster and perfectly
accurate.

## 4. Perspective correction

Deskew handles rotation. It does not handle a receipt photographed at an angle,
which is how most phone photos arrive. Contour detection plus a four-point
transform is well-understood; the work is doing it *conditionally*, since
applying it to an already-flat scan makes things worse. It belongs in
`preprocessing/transforms.py` behind a quality gate like every other transform.

## 5. Human-review workflow

The data model is ready: `review_required`, `status`, `reasons`, and
per-field evidence with source text and bounding boxes. What does not exist is
the workflow — a queue, a correction UI, and a store for corrections.

The valuable part is the feedback loop: corrections become labelled data, which
feeds (1) and (2). Reviewing without capturing the corrections wastes most of
the effort.

## 6. Wider language coverage

The lexicon carries English thoroughly and some German, French, Spanish and
Italian. Bangla, Hindi and Arabic need vocabulary; Arabic additionally needs
right-to-left handling in the label/value association logic, which currently
assumes values sit to the right of labels (it does try left as a fallback, but
that is a fallback, not proper RTL support).

Each language also needs its Tesseract pack in the image and a fixture.

## 7. A cloud OCR provider

For the cases local OCR handles poorly — heavy skew, poor lighting, unusual
fonts. The interface is ready; the work is one provider implementation plus a
policy for *when* to escalate. The natural trigger already exists: low
confidence, the same signal that gates the LLM fallback.

This is a privacy decision as much as a technical one. Receipts would leave the
host, and that should be an explicit, configurable choice.

## 8. Batch and async processing

```
POST /api/v1/receipts/jobs      -> { job_id }
GET  /api/v1/receipts/jobs/{id} -> { status, result }
```

Deliberately not built. At receipt sizes, synchronous processing finishes well
within a normal HTTP timeout, and a job store, worker pool and polling contract
would be complexity without a current justification. It becomes worthwhile with
multi-page PDFs or bulk import — not before.

## 9. Document types beyond receipts

Invoices are the natural next one. What is reusable as-is: ingest,
preprocessing, the whole OCR layer, normalisation, confidence, the response
envelope. What is new: an `Invoice` schema, an invoice lexicon (purchase order
numbers, payment terms, line-item tax breakdowns), invoice extractors, and a
`DocumentType` branch in the pipeline.

The abstractions were named generically for this reason, so the work is
additive rather than a refactor.

## 10. RAG over extracted receipts

Structured JSON → embeddings → vector store → question answering over spending
history.

Explicitly **not** built, and nothing in the current design depends on it. It
attaches cleanly after assembly if it is ever wanted. Building it now would add
a vector database, an embedding provider and a retrieval layer to a system
whose job is to turn an image into JSON — that is the kind of complexity that
makes a pipeline harder to operate without making it better at its actual task.

---

## Deliberately out of scope

| | Why |
|---|---|
| Authentication | Belongs at the gateway |
| A web UI | This is a service |
| Receipt image storage | A privacy liability with no current requirement |
| Fine-tuning an OCR model | Enormous effort; a cloud provider is the cheaper answer to the same problem |
| Multi-tenancy | The service is stateless; run separate deployments |

---

## Contributing

Whatever the change:

1. Add a fixture covering the case, if it is a behaviour change.
2. Run `make check` — lint, types, tests, golden drift.
3. If goldens changed, read the diff and justify it in the commit message.
4. Bump `PIPELINE_VERSION` for any behaviour change; bump `SCHEMA_VERSION` only
   for a contract change.
5. Update the relevant document in `docs/`. Documentation that drifts from the
   implementation is worse than none.
