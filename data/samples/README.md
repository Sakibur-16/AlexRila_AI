# Sample receipts

Synthetic receipt images for manual testing. Fictional merchants and card
tails only — no real customer data.

| File | Contents |
|---|---|
| `receipt_grocery_us.png` | US grocery receipt: 3 items, 8% sales tax, VISA tail. Arithmetic reconciles (13.50 + 1.08 = 14.58). |

Upload one to `POST /api/v1/receipts/extract` with the `fixture` field left
empty to exercise the real OCR path.
