# Test fixtures

Each fixture is a recorded OCR payload replayed by the `fixture` OCR provider,
paired with the structured output the pipeline is expected to produce.

```
receipts/
  <name>.ocr.json       recorded OCR output (input to the pipeline)
  <name>.expected.json  expected extraction (golden output)
```

## Why recorded OCR rather than images

Golden tests must isolate the deterministic stages -- normalisation,
extraction, validation, confidence -- from the OCR engine. Recording the
recognition output means:

* CI needs no native engine binary;
* an engine upgrade cannot silently rewrite every golden file;
* an edge case that is hard to photograph (a specific glyph confusion, a
  multi-currency receipt) can simply be authored.

These payloads are **synthetic**: fictional merchants, addresses and card
tails, hand-authored to represent real layouts. No customer receipt, and no
copyrighted material, is committed here. Add fixtures the same way.

## OCR payload format

```json
{
  "text": "optional; derived from lines when omitted",
  "lines": [
    {"text": "TOTAL 25.99", "confidence": 0.97, "bbox": [10, 500, 300, 530]}
  ],
  "languages": ["eng"],
  "width": 800,
  "height": 1400
}
```

`confidence` and `bbox` are optional. Omitting `bbox` on every line exercises
the geometry-free path taken by providers that report no layout.

## Regenerating expected output

```
python scripts/generate_golden.py --fixture <name>
```

Review the diff before committing: a change to an expected file is a change to
the product's behaviour and must be justified in the same commit.
