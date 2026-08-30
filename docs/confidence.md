# Confidence scoring

Confidence here is a **composed** quantity. Nothing in this document is a
number chosen because it looked plausible; every constant has a stated reason
and lives in exactly one place.

## The formula

For each field that has a value:

```
field_score = (w_ocr x ocr_component + w_ext x extraction_component)
              x validation_multiplier
              x arithmetic_bonus          (financial fields only, when checks pass)
```

clamped to `[0, 1]`.

### `ocr_component` — how legible was the source?

The provider's confidence for the span the value was read from. When a value
draws on several evidence items, the **minimum** is used: a value assembled
from two lines is only as trustworthy as the worse of them.

When a provider reports no confidence at all, the document-level mean is used —
and `OCRLine.effective_confidence` falls back to `0.5`, an explicit "unknown"
rather than an optimistic assumption.

### `extraction_component` — how strong was the method?

Prior confidence in *how* the value was found, from
`app/domain/evidence.py::METHOD_PRIOR`:

| Method | Prior | Meaning |
|---|---:|---|
| `KEYWORD_ANCHORED` | 0.95 | An explicit label and a value on the same line: `TOTAL 17.28` |
| `REGEX` | 0.92 | A strict pattern matched: phone, tax id, card mask |
| `SPATIAL` | 0.80 | Label on one line, value on the next — a real layout, but inferred |
| `DERIVED` | 0.75 | Computed from other extracted values (a summed tax total) |
| `LLM` | 0.70 | Produced by the model, after passing the grounding gate |
| `HEURISTIC` | 0.60 | Position or structure, with no label: the merchant name |
| `CONFIGURED` | 0.40 | From configuration, not from the document at all |

The ordering encodes a belief worth stating: **an explicitly labelled match is
far more trustworthy than a positional guess.** A merchant name found by "first
substantial header line" genuinely deserves to score lower than a total found
next to the word `TOTAL`.

Two adjustments:

- A field that computed its own evidence-graded score (currency detection does)
  has it averaged with the prior, so neither signal is discarded.
- A field carrying a normalisation warning is capped at `0.55` — an explicit
  statement of doubt outranks the method prior.

### `validation_multiplier` — did it survive checking?

| Situation | Multiplier |
|---|---:|
| A validation `ERROR` names this field | 0.55 |
| A validation `WARNING` names this field | 0.80 |
| Neither | 1.00 |

Item-level issues (`items[2].quantity`) also temper the aggregate `items` score.

### `arithmetic_bonus` — was it independently corroborated?

`1.08`, applied to financial fields when every checkable arithmetic identity
holds.

This is where the score earns its keep. A total satisfying
`subtotal - discount + tax == total` has been corroborated by figures the total
extractor never touched. That is genuine independent evidence, and it is the
difference between a confidence number that means something and one that is
decoration.

## Item scoring

```
items_score = mean(per_item_score) x (0.5 + 0.5 x completeness)
completeness = extracted_items / (extracted_items + skipped_lines)
```

Completeness applies at half strength: a skipped line in the item region is
often a divider or a note rather than a missed item, so it should temper the
score, not dominate it.

## Aggregate

A weighted mean over **present fields only**:

| Field | Weight |
|---|---:|
| `total` | 3.0 |
| `items` | 2.5 |
| `currency`, `merchant.name`, `transaction.date` | 2.0 |
| `subtotal`, `tax.total` | 1.5 |
| `discount` | 0.8 |
| `payment.*`, `receipt_number`, `transaction.time` | 0.5 |
| `merchant.address/phone/email/tax_id` | 0.2–0.3 |

**Absent fields score `0.0` and are excluded from the aggregate.** Averaging in
zeros for fields a receipt never printed would make a perfectly read corner-shop
receipt score worse than a badly read supermarket one — the opposite of useful.

## Reported values

```json
"confidence": {
  "overall": 0.94,      // weighted mean over present fields
  "ocr": 0.94,          // document-level OCR confidence
  "extraction": 0.89,   // mean extraction-method confidence
  "validation": 1.0,    // document-level validation health
  "fields": { "total": 1.0, "merchant.name": 0.75 }
}
```

`validation` is reported separately for transparency: `0.4` with errors, `0.6`
when arithmetic did not reconcile, `0.85` with warnings, `1.0` clean.

## Human review

`review.review_required` is set when any of:

- `overall < CONFIDENCE_THRESHOLD` (0.85)
- validation raised an `ERROR`
- `ocr < LOW_OCR_CONFIDENCE_THRESHOLD` (0.60)
- a high-weight field (`total`, `currency`, `transaction.date`,
  `merchant.name`) scores below `REVIEW_FIELD_CONFIDENCE_THRESHOLD` (0.65)

`review.reasons` always says which. The per-field threshold is deliberately
**lower** than the aggregate one: a field read by a sound heuristic scores
below the aggregate bar by design, and flagging every receipt on that basis
would make the review queue meaningless. That was observed in practice —
before the thresholds were separated, every fixture demanded review because
the merchant-name heuristic scores 0.75.

## Interpreting a score

| Range | Reading |
|---|---|
| ≥ 0.90 | Labelled matches, legible scan, arithmetic reconciles. Safe to automate. |
| 0.75–0.90 | Sound, with some heuristic fields. Automate; sample for audit. |
| 0.60–0.75 | Degraded OCR or unlabelled fields. Route to review if the amount matters. |
| < 0.60 | Do not act on unreviewed. |

A caveat worth stating: these are **calibrated by construction, not
empirically**. The priors encode reasonable beliefs about extraction methods,
but they have not been fitted against a labelled corpus. Doing so is the
highest-value item on the roadmap — until then, treat the scores as a
well-founded ranking rather than a probability.

## Tuning

`CONFIDENCE_WEIGHT_OCR` / `CONFIDENCE_WEIGHT_EXTRACTION` are normalised at use,
so a misconfigured pair cannot silently inflate every score in the system.

To recalibrate: label a real dataset, run `scripts/evaluate.py`, and plot
accuracy against reported confidence. Any threshold change is a behaviour
change — bump `PIPELINE_VERSION` and regenerate goldens.
