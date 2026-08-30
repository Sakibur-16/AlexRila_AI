"""Line-item extraction.

Item rows are the least standardised part of a receipt. These are all real
layouts, and all are handled here::

    Milk                    2.50      description + total
    Coffee x2               8.00      trailing quantity
    2 Coffee                8.00      leading quantity
    2 x 4.00                8.00      quantity x unit price + total
    Bread              1.50  3.00     quantity implied, unit + total
    COFFEE                            description-only, price on next line
        4.00
    Apples 1.2kg @ 3.00     3.60      weighted item

The strategy is to classify each candidate line by *how many amounts it
carries and what precedes them*, rather than by fixed column positions --
which vary between every point-of-sale system in existence.

Nothing is invented: a row with no price yields an item with
``total_price=None``, and a line that cannot be interpreted as an item is
skipped rather than forced into one.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from decimal import Decimal

from app.domain.evidence import Evidence, ExtractionMethod
from app.extraction.context import LineView, ReceiptContext
from app.extraction.lexicon import LabelCategory
from app.normalization.money import ParsedAmount, parse_all_amounts
from app.schemas.receipt import LineItem

#: "2 x 4.00", "2 @ 4.00", "2x4.00" -- quantity and unit price together.
_QTY_AT_PRICE = re.compile(
    r"(?<![\w.])(\d{1,4}(?:[.,]\d{1,3})?)\s*(?:X|@|\*)\s*(\d+(?:[.,]\d{1,3})?)",
    re.IGNORECASE,
)

#: A leading quantity: "2 COFFEE", "3  BREAD".
_LEADING_QTY = re.compile(r"^\s*(\d{1,3})\s+(?=[A-Za-z])")

#: A quantity marker attached to the description: "COFFEE X2", "COFFEE x 2".
#: Not anchored to end-of-line, because the line total follows it.
_MARKED_QTY = re.compile(r"(?<=[A-Za-z\s])[Xx]\s*(\d{1,3})(?![\d.,])")

#: A weighted or counted quantity with its unit: "1.2KG", "0.85 kg", "3 PCS".
_WEIGHT = re.compile(
    r"(?<![\w.])(\d+(?:[.,]\d{1,3})?)\s*(KG|G|GM|LB|OZ|L|ML|EA|PCS?|UNIT)\b",
    re.IGNORECASE,
)

#: The "@ unit-price" that follows a weight: "1.2KG @ 3.00".
_AT_PRICE = re.compile(r"@\s*(\d+(?:[.,]\d{1,3})?)")

#: A SKU / product code printed alongside the description.
_SKU = re.compile(r"(?<![\w])((?=[A-Z0-9\-]{5,20}$)[A-Z0-9]{2,}[A-Z0-9\-]*)")

#: A leading product code: a long run of digits with no decimal separator, at
#: the start of the line. Target-style receipts print a 9-digit DPCI before the
#: description, and it parses as a perfectly good "amount" -- which is how a
#: 270030028 unit price on a $3.79 item comes about. Captured as the SKU and
#: removed before prices are read.
_LEADING_PRODUCT_CODE = re.compile(r"^\s*(\d{6,14})(?![\d.,])\s+(?=\S)")

#: A bare integer of at least this many digits is a product code, not a price.
_PRODUCT_CODE_MIN_DIGITS = 6

#: The unit price of a line cannot plausibly exceed its total by this factor.
#: A guard against any remaining identifier that slips through as a price.
_MAX_UNIT_PRICE_RATIO = 100

#: Per-item discount markers.
_ITEM_DISCOUNT = re.compile(r"\b(DISCOUNT|DISC|PROMO|COUPON|SAVE|OFF)\b", re.IGNORECASE)

#: Text that means the line is structural, not an item.
_NON_ITEM = re.compile(
    r"^\s*(?:ITEM|DESCRIPTION|QTY|QUANTITY|PRICE|AMOUNT|UNIT|TOTAL|NO\.?)"
    r"(?:\s+(?:ITEM|DESCRIPTION|QTY|QUANTITY|PRICE|AMOUNT|UNIT|TOTAL))*\s*$",
    re.IGNORECASE,
)

#: Promotional annotations printed beneath an item. They carry a price but
#: describe the item above rather than a purchase of their own, so counting
#: them double-bills the receipt.
_ANNOTATION = re.compile(
    r"^\s*(?:REGULAR\s+PRICE|REG\.?\s+PRICE|WAS\b|YOU\s+(?:SAVE|PAY)|SAVE\b"
    r"|MSRP|LIST\s+PRICE|ORIG(?:INAL)?\.?\s+PRICE|MEMBER\s+PRICE|PRICE\s+EACH"
    r"|\d+\s*@\s*\S+\s*(?:EA|EACH)?\s*$)",
    re.IGNORECASE,
)

#: Labels that disqualify a line from being an item.
_DISQUALIFYING = (
    LabelCategory.SUBTOTAL,
    LabelCategory.TOTAL,
    LabelCategory.TAX,
    LabelCategory.CHANGE,
    LabelCategory.TENDERED,
    LabelCategory.SERVICE_CHARGE,
    LabelCategory.SHIPPING,
    LabelCategory.ITEM_COUNT,
    LabelCategory.DATE,
    LabelCategory.TIME,
    LabelCategory.RECEIPT_ID,
    LabelCategory.MERCHANT_CONTACT,
    LabelCategory.FOOTER,
)

#: A description shorter than this is more likely OCR noise than a product.
_MIN_DESCRIPTION_LENGTH = 2

#: Beyond this, a "quantity" is a quantity no longer -- it is a year, a code
#: or a mis-parsed price.
_MAX_PLAUSIBLE_QUANTITY = Decimal("1000")


@dataclass(frozen=True, slots=True)
class ItemsResult:
    """Extracted items plus the evidence supporting the set as a whole."""

    items: tuple[LineItem, ...]
    #: Per-item extraction confidence priors, aligned with ``items``.
    methods: tuple[ExtractionMethod, ...]
    evidence: tuple[Evidence, ...]
    #: Lines inside the item region that could not be interpreted.
    skipped_lines: int = 0


def extract_items(context: ReceiptContext) -> ItemsResult:
    """Extract line items from the item region of ``context``.

    Falls back to scanning the whole document when section detection found no
    item region, which happens on receipts whose totals block is unlabelled.
    """
    result = _scan_region(_item_region(context), context)
    if not result.items:
        # Section detection can misplace the items boundary -- a footer note
        # containing the word "total", for instance, drags the totals block
        # upwards and strands the real items below it. Rather than return an
        # empty list, retry across every line that is not clearly structural.
        fallback = _scan_region(_fallback_region(context), context)
        if fallback.items:
            return fallback
    return result


def _scan_region(region: list[LineView], context: ReceiptContext) -> ItemsResult:
    """Interpret the lines of one candidate item region."""
    items: list[LineItem] = []
    methods: list[ExtractionMethod] = []
    evidence: list[Evidence] = []
    skipped = 0

    index = 0
    while index < len(region):
        line = region[index]

        if _is_structural(line):
            index += 1
            continue

        parsed = _parse_item_line(line, context)

        if parsed is None:
            # A description-only line may be completed by a price on the line
            # below -- a common narrow-receipt layout.
            continuation = _try_continuation(region, index)
            if continuation is not None:
                item, method, ev, consumed = continuation
                items.append(item)
                methods.append(method)
                evidence.append(ev)
                index += consumed
                continue
            if line.normalized.strip():
                skipped += 1
            index += 1
            continue

        item, method, ev = parsed
        items.append(item)
        methods.append(method)
        evidence.append(ev)
        index += 1

    return ItemsResult(
        items=tuple(items),
        methods=tuple(methods),
        evidence=tuple(evidence),
        skipped_lines=skipped,
    )


def _item_region(context: ReceiptContext) -> list[LineView]:
    """The item region identified by section detection."""
    return [
        line for line in context.lines if context.items_start <= line.index < context.items_end
    ] or _fallback_region(context)


def _fallback_region(context: ReceiptContext) -> list[LineView]:
    """Every line that is not clearly a header, total or footer line.

    Skips the first line, which is the merchant name on essentially every
    receipt and would otherwise be read as an item whenever it happens to sit
    beside a number.
    """
    return [
        line for line in context.lines[1:] if not line.has(*_DISQUALIFYING) and not line.is_blank
    ]


def _is_structural(line: LineView) -> bool:
    """Whether the line is a divider, column header or otherwise not an item."""
    if line.is_blank or line.is_divider:
        return True
    if _NON_ITEM.match(line.normalized) or _ANNOTATION.match(line.normalized):
        return True
    return line.has(*_DISQUALIFYING)


def _parse_item_line(
    line: LineView, context: ReceiptContext
) -> tuple[LineItem, ExtractionMethod, Evidence] | None:
    """Interpret a single line as an item, or return ``None``.

    The number of amounts on the line drives the interpretation:

    * **0** -- not an item on its own; may be completed by a continuation.
    * **1** -- description plus line total.
    * **2** -- either ``qty x unit`` (when a multiplication marker is present)
      or ``unit total`` (when it is not).
    * **3+** -- ``qty unit total``; the rightmost is the line total, which is
      the one invariant across point-of-sale layouts.
    """
    if not line.amounts:
        return None

    text = line.normalized
    quantity: Decimal | None = None
    unit: str | None = None
    unit_price: Decimal | None = None
    method = ExtractionMethod.HEURISTIC

    # Consume quantity markers *before* looking for prices. Doing it in this
    # order is what keeps the "1" of "Milk 1L" out of the price list and the
    # "L" out of the description -- parsing prices first would leave both
    # fragments behind.
    working = text
    sku: str | None = None

    # Strip a leading product code before anything else reads it as money.
    code = _LEADING_PRODUCT_CODE.match(working)
    if code is not None:
        sku = code.group(1)
        working = _blank(working, code.span(1))

    qty_at_price = _QTY_AT_PRICE.search(working)
    weight = _WEIGHT.search(working)

    if qty_at_price is not None and _is_product_code(qty_at_price.group(2)):
        # "2 x 284060377 GG OATMILK" has the exact shape of quantity-times-unit-
        # price, but the second number is a product code. The quantity is still
        # real, so keep it and record the code as the SKU rather than pricing
        # the line at 284 million.
        quantity = _to_decimal(qty_at_price.group(1))
        sku = sku or qty_at_price.group(2)
        working = _blank(working, qty_at_price.span())
        method = ExtractionMethod.REGEX
    elif qty_at_price is not None:
        quantity = _to_decimal(qty_at_price.group(1))
        unit_price = _to_decimal(qty_at_price.group(2))
        working = _blank(working, qty_at_price.span())
        method = ExtractionMethod.REGEX
    elif weight is not None:
        quantity = _to_decimal(weight.group(1))
        unit = weight.group(2).upper()
        working = _blank(working, weight.span())
        at_price = _AT_PRICE.search(working)
        if at_price is not None:
            unit_price = _to_decimal(at_price.group(1))
            working = _blank(working, at_price.span())
        method = ExtractionMethod.REGEX
    else:
        marked = _MARKED_QTY.search(working)
        leading = _LEADING_QTY.match(working)
        if marked is not None:
            quantity = _to_decimal(marked.group(1))
            working = _blank(working, marked.span())
            method = ExtractionMethod.REGEX
        elif leading is not None:
            quantity = _to_decimal(leading.group(1))
            working = _blank(working, leading.span(1))
            method = ExtractionMethod.REGEX

    prices = parse_all_amounts(working, style=context.separator_style, repair_ocr=False)
    if not prices:
        return None

    total_price: Decimal | None = prices[-1].value
    if unit_price is None and len(prices) >= 2:
        unit_price = prices[-2].value

    description = _extract_description(working, prices)
    if description is None or len(description) < _MIN_DESCRIPTION_LENGTH:
        return None

    if quantity is not None and (quantity <= 0 or quantity > _MAX_PLAUSIBLE_QUANTITY):
        quantity = None

    # A "unit price" equal to the line total is not a separate figure, it is
    # the same number read twice from a single-quantity row.
    if (
        unit_price is not None
        and total_price is not None
        and unit_price == total_price
        and (quantity is None or quantity == 1)
    ):
        unit_price = None

    # Complete the quantity/unit-price/total triple only when two members are
    # known -- never invent a value from one.
    if quantity is not None and unit_price is None and total_price is not None and quantity > 0:
        candidate = (total_price / quantity).quantize(Decimal("0.01"))
        if candidate > 0:
            unit_price = candidate

    if total_price is None:
        return None

    discount = None
    if _ITEM_DISCOUNT.search(text) and total_price < 0:
        discount = abs(total_price)

    evidence = line.evidence(method, notes=f"prices={len(prices)}")
    item = LineItem(
        description=description,
        sku=sku or _extract_sku(description),
        quantity=quantity,
        unit=unit,
        unit_price=unit_price,
        total_price=total_price,
        discount=discount,
        line_index=line.index,
    )
    return item, method, evidence


def _try_continuation(
    region: list[LineView], index: int
) -> tuple[LineItem, ExtractionMethod, Evidence, int] | None:
    """Join a description-only line with a price on the following line.

    Only applies when the description line carries no amount at all and the
    next line carries nothing *but* an amount -- a narrow pattern, chosen so
    that unrelated adjacent lines are not fused into a fictional item.
    """
    line = region[index]
    if line.amounts or not line.normalized.strip() or index + 1 >= len(region):
        return None

    following = region[index + 1]
    if not following.amounts or len(following.amounts) > 1:
        return None
    remainder = following.normalized.replace(following.amounts[0].raw, "").strip()
    if remainder:
        return None

    description = line.normalized.strip()
    if len(description) < _MIN_DESCRIPTION_LENGTH:
        return None

    evidence = Evidence(
        source_text=f"{line.raw} / {following.raw}",
        line_index=line.index,
        bbox=line.bbox,
        ocr_confidence=min(line.confidence, following.confidence),
        method=ExtractionMethod.SPATIAL,
        notes="description_and_price_on_adjacent_lines",
    )
    item = LineItem(
        description=description,
        sku=_extract_sku(description),
        total_price=following.amounts[0].value,
        line_index=line.index,
    )
    return item, ExtractionMethod.SPATIAL, evidence, 2


def _is_product_code(token: str) -> bool:
    """Whether a numeric token is an identifier rather than a price.

    ``2 x 284060377 GG OATMILK`` has the exact shape of "quantity times unit
    price", but 284060377 is a product code. A price on a receipt carries a
    decimal separator or is small; a long bare integer is neither.
    """
    cleaned = token.strip()
    if any(sep in cleaned for sep in ".,"):
        return False
    return cleaned.isdigit() and len(cleaned) >= _PRODUCT_CODE_MIN_DIGITS


def _blank(text: str, span: tuple[int, int]) -> str:
    """Replace ``span`` with spaces, preserving every other character offset.

    Blanking rather than deleting keeps later matches aligned with the
    original string, so several markers can be consumed in sequence without
    each one invalidating the next one's offsets.
    """
    start, end = span
    return text[:start] + " " * (end - start) + text[end:]


def _extract_description(text: str, amounts: list[ParsedAmount]) -> str | None:
    """Strip prices and decoration, leaving the product description.

    ``text`` has already had its quantity markers blanked by the caller, so
    only prices and punctuation remain to remove.
    """
    cleaned = text
    for amount in amounts:
        cleaned = cleaned.replace(amount.raw, " ", 1)

    cleaned = re.sub(r"[@*]", " ", cleaned)
    cleaned = re.sub(r"[.\-_*=]{2,}", " ", cleaned)
    cleaned = re.sub(r"[$€£¥₹৳]", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" .,-:*#")

    if not cleaned or not any(char.isalpha() for char in cleaned):
        return None
    return cleaned


def _extract_sku(description: str) -> str | None:
    """Pull a product code out of a description, when one is clearly present."""
    match = _SKU.search(description.strip())
    if match is None:
        return None
    candidate = match.group(1)
    if not any(char.isdigit() for char in candidate):
        return None
    return candidate


def _to_decimal(raw: str) -> Decimal | None:
    try:
        return Decimal(raw.replace(",", "."))
    except (ArithmeticError, ValueError):
        return None


def sum_item_totals(items: tuple[LineItem, ...]) -> Decimal | None:
    """Sum line totals, or ``None`` when no item carries one.

    Returning ``None`` rather than zero keeps "no priced items" distinguishable
    from "items totalling zero" in the financial validator.
    """
    priced = [item.total_price for item in items if item.total_price is not None]
    if not priced:
        return None
    return sum(priced, Decimal("0"))
