# Validation

Validation answers a different question from extraction: not "what does the
receipt say" but "does what it says make sense". It never repairs anything.

## Severity

| Severity | Meaning | Effect |
|---|---|---|
| `INFO` | Worth knowing; extraction is sound | none |
| `WARNING` | The value may be wrong; check if the amount matters | field confidence x 0.80 |
| `ERROR` | The document is internally inconsistent or impossible | `is_valid: false`, x 0.55, review required |

**None of these is a failed request.** All arrive inside `success: true`.
A document that does not add up is a real answer a consumer must handle, not a
crash.

## Why discrepancies are reported, never corrected

If `subtotal - discount + tax != total`, one of those numbers was misread. The
system does not know which. Rewriting the printed total to make the arithmetic
work would produce a receipt that looks perfect and is wrong — and would
destroy the only signal that anything went astray. The printed value is
preserved and the discrepancy is reported.

## Financial checks

```
sum(item totals)                                   ~= subtotal
subtotal - discount + tax + service + shipping
        + tip + rounding                           ~= total
tendered - change                                  ~= total
sum(tax detail lines)                              ~= tax total
```

`~=` is a tolerance comparison. Receipts round each line independently, so an
exact match is the exception:

```
tolerance = max(TOTAL_TOLERANCE, |amount| x TOTAL_TOLERANCE_RATIO)
```

The proportional component matters: a flat 5-cent allowance is reasonable on a
20-unit receipt and absurdly tight on a 5,000-unit one.

Three details worth stating:

- **Tax-inclusive pricing is tried too.** European VAT receipts print a
  subtotal that already contains the tax, so the additive identity does not
  hold. Both readings are tested and only a failure of *both* is reported —
  otherwise every VAT receipt would be flagged.
- **The tender check is independent corroboration.** `tendered - change` uses
  figures the total extractor never touched. When it agrees, that is real
  evidence; when it disagrees it is a warning rather than an error, because
  partial and split tenders break the identity legitimately.
- **Item sums fall back to the total** when no subtotal was printed, but only
  when there is no tax to account for — otherwise the comparison is meaningless.

## Codes

### Image quality
`IMAGE_LOW_RESOLUTION`, `IMAGE_BLURRY`, `IMAGE_TOO_DARK`, `IMAGE_TOO_BRIGHT`,
`IMAGE_LOW_CONTRAST`, `IMAGE_SKEWED`

### OCR quality
`OCR_LOW_CONFIDENCE`, `OCR_SPARSE_TEXT`, `OCR_SUSPICIOUS_CHARACTERS`,
`OCR_NO_GEOMETRY`

### Completeness
`MISSING_MERCHANT_NAME`, `MISSING_TOTAL`, `MISSING_DATE`, `MISSING_CURRENCY`,
`MISSING_ITEMS`, `MISSING_SUBTOTAL`

### Financial

| Code | Severity | Raised when |
|---|---|---|
| `TOTAL_MISMATCH` | ERROR | The total identity fails under both tax readings |
| `SUBTOTAL_MISMATCH` | WARNING | Item sum disagrees with the total (no subtotal printed) |
| `ITEM_ARITHMETIC_MISMATCH` | WARNING | Item sum disagrees with the subtotal |
| `TAX_MISMATCH` | WARNING | Tax detail lines do not sum to the tax total |
| `SUBTOTAL_EXCEEDS_TOTAL` | WARNING | Subtotal above total with no discount found |
| `NEGATIVE_TOTAL` | ERROR | Total below zero (a refund, or a misread) |
| `NEGATIVE_SUBTOTAL` | ERROR | Subtotal below zero |
| `DISCOUNT_EXCEEDS_SUBTOTAL` | ERROR | Discount larger than the subtotal |

### Semantic
`INVALID_QUANTITY` (ERROR), `NEGATIVE_ITEM_PRICE`, `INVALID_CURRENCY`,
`AMBIGUOUS_CURRENCY`, `MULTIPLE_CURRENCIES_DETECTED`, `AMBIGUOUS_DATE_FORMAT`,
`INVALID_DATE`, `FUTURE_DATE`, `IMPLAUSIBLE_DATE`, `INVALID_TIME`

### Confidence and privacy
`LOW_FIELD_CONFIDENCE`, `LOW_OVERALL_CONFIDENCE`, `REVIEW_RECOMMENDED`,
`SENSITIVE_DATA_REDACTED`

### LLM
`LLM_FALLBACK_USED`, `LLM_OUTPUT_REJECTED`, `LLM_UNAVAILABLE`

Codes are part of the public contract: new ones are added, existing ones are
never renamed.

## Field paths

Every issue names the field it concerns, using the same dotted paths as the
receipt schema and the confidence map:

```json
{
  "code": "TOTAL_MISMATCH",
  "severity": "error",
  "field": "total",
  "message": "Calculated total does not match the total printed on the receipt.",
  "context": { "calculated": "17.82", "reported": "99.99", "difference": "82.17" }
}
```

`items[2].quantity` addresses a specific item. `context` carries the numbers
behind the finding, as strings, so a consumer can render the discrepancy
without recomputing it.

## Deduplication

Two validators can legitimately reach the same conclusion by different routes —
the tender check and the identity check both report `TOTAL_MISMATCH`. Findings
sharing a code and field are collapsed to the highest severity, so a consumer
sees each problem once.

## Adding a rule

1. Add the code to `IssueCode` (append; never rename).
2. Implement the check in `validation/financial.py`, `semantic.py` or
   `quality.py`, returning `list[ValidationIssue]`.
3. Choose severity by asking: is the document *impossible*, or merely *suspect*?
4. Add a test in `tests/unit/test_validation.py`.
5. Regenerate goldens — a new rule changes output, which is a behaviour change.
