"""Receipt vocabulary.

All label keywords live here rather than being scattered through the
extractors, for three reasons: adding a language means editing one file; the
priority ordering that separates ``SUBTOTAL`` from ``TOTAL`` is visible in one
place; and the same vocabulary can be reused by the structure detector and the
evaluation harness.

**Matching order matters.** ``SUBTOTAL`` contains ``TOTAL``, and ``TOTAL``
appears inside ``GRAND TOTAL``. Longest-first matching within a category, and
explicit exclusion of narrower categories, is what stops a subtotal from being
read as the total -- the single most damaging failure mode in receipt parsing.
"""

from __future__ import annotations

import enum
import re
from typing import Final, NamedTuple


class LabelCategory(enum.StrEnum):
    """Semantic role of a label found on a receipt line."""

    SUBTOTAL = "subtotal"
    TOTAL = "total"
    TAX = "tax"
    DISCOUNT = "discount"
    SERVICE_CHARGE = "service_charge"
    SHIPPING = "shipping"
    TIP = "tip"
    ROUNDING = "rounding"
    CHANGE = "change"
    TENDERED = "tendered"
    PAYMENT_METHOD = "payment_method"
    ITEM_COUNT = "item_count"
    DATE = "date"
    TIME = "time"
    RECEIPT_ID = "receipt_id"
    MERCHANT_CONTACT = "merchant_contact"
    FOOTER = "footer"


class LabelMatch(NamedTuple):
    """A vocabulary hit on a line."""

    category: LabelCategory
    #: The lexicon entry that matched, uppercased.
    keyword: str
    start: int
    end: int
    #: True for a label that is an exact standalone line, e.g. a bare "TOTAL".
    exact_line: bool = False


#: Keyword sets per category. English is complete; other languages carry the
#: high-frequency terms. Entries are matched case-insensitively on whole words.
#:
#: Ordering within a tuple is irrelevant -- matching sorts by length -- but
#: the *category* order in :data:`_CATEGORY_PRIORITY` is significant.
KEYWORDS: Final[dict[LabelCategory, tuple[str, ...]]] = {
    LabelCategory.SUBTOTAL: (
        "SUBTOTAL",
        "SUB TOTAL",
        "SUB-TOTAL",
        "SUBTOT",
        "SUB TOT",
        "NET TOTAL",
        "NET AMOUNT",
        "GOODS TOTAL",
        "ITEMS TOTAL",
        "MERCHANDISE",
        "ZWISCHENSUMME",
        "SOUS-TOTAL",
        "SOUS TOTAL",
        "SUBTOTAAL",
        "SUBTOTALE",
        "IMPORTE",
        "SUBTOTAL NETO",
        "DELSUMMA",
    ),
    LabelCategory.TOTAL: (
        "GRAND TOTAL",
        "TOTAL DUE",
        "AMOUNT DUE",
        "BALANCE DUE",
        "TOTAL AMOUNT",
        "TOTAL PAYABLE",
        "NET PAYABLE",
        "AMOUNT PAYABLE",
        "TO PAY",
        "TOTAL",
        "BALANCE",
        "TOTAL SALE",
        "SALE TOTAL",
        "ORDER TOTAL",
        "INVOICE TOTAL",
        "BAL DUE",
        "G.TOTAL",
        "G TOTAL",
        "NET AMT",
        "TOTAL RS",
        "TOTAL USD",
        "TOTALL",
        "TO TAL",
        "FINAL TOTAL",
        "TOTAL COST",
        "GESAMT",
        "GESAMTBETRAG",
        "SUMME",
        "MONTANT",
        "MONTANT TOTAL",
        "TOTALE",
        "TOTAAL",
        "IMPORTE TOTAL",
        "SUMA",
        "ИТОГО",
        "合計",
    ),
    LabelCategory.TAX: (
        "SALES TAX",
        "STATE TAX",
        "LOCAL TAX",
        "TOTAL TAX",
        "TAX TOTAL",
        "VAT",
        "V.A.T",
        "GST",
        "HST",
        "PST",
        "QST",
        "CGST",
        "SGST",
        "IGST",
        "TVA",
        "MWST",
        "IVA",
        "BTW",
        "MOMS",
        "TAX",
        "TAXES",
        "SERVICE TAX",
        "GST/HST",
        "CONSUMPTION TAX",
        "MVA",
    ),
    LabelCategory.DISCOUNT: (
        "DISCOUNT",
        "DISCOUNTS",
        "TOTAL DISCOUNT",
        "SAVINGS",
        "YOU SAVED",
        "YOU SAVE",
        "PROMO",
        "PROMOTION",
        "COUPON",
        "REBATE",
        "MARKDOWN",
        "OFFER",
        "REDUCTION",
        "RABATT",
        "REMISE",
        "DESCUENTO",
        "SCONTO",
        "LOYALTY DISCOUNT",
        "MEMBER DISCOUNT",
        "STAFF DISCOUNT",
    ),
    LabelCategory.SERVICE_CHARGE: (
        "SERVICE CHARGE",
        "SERVICE CHG",
        "SERVICE FEE",
        "SVC CHARGE",
        "SERVICE",
        "GRATUITY INCLUDED",
        "COVER CHARGE",
        "SERVICEGEBUHR",
        "FRAIS DE SERVICE",
    ),
    LabelCategory.SHIPPING: (
        "SHIPPING",
        "DELIVERY",
        "DELIVERY FEE",
        "DELIVERY CHARGE",
        "FREIGHT",
        "POSTAGE",
        "SHIPPING & HANDLING",
        "S&H",
        "VERSAND",
        "LIVRAISON",
    ),
    LabelCategory.TIP: (
        "TIP",
        "TIPS",
        "GRATUITY",
        "TRINKGELD",
        "POURBOIRE",
    ),
    LabelCategory.ROUNDING: (
        "ROUNDING",
        "ROUND OFF",
        "ROUNDED",
        "ROUND-OFF",
        "ADJUSTMENT",
        "RUNDUNG",
        "ARRONDI",
    ),
    LabelCategory.CHANGE: (
        "CHANGE",
        "CHANGE DUE",
        "CHG DUE",
        "RETURN",
        "WECHSELGELD",
        "MONNAIE",
        "CAMBIO",
    ),
    LabelCategory.TENDERED: (
        "CASH TENDERED",
        "AMOUNT TENDERED",
        "TENDERED",
        "TENDER",
        "CASH PAID",
        "AMOUNT PAID",
        "PAID",
        "PAYMENT",
        "CASH",
    ),
    LabelCategory.PAYMENT_METHOD: (
        "CASH",
        "CARD",
        "CREDIT",
        "CREDIT CARD",
        "DEBIT",
        "DEBIT CARD",
        "VISA",
        "MASTERCARD",
        "MASTER CARD",
        "AMEX",
        "AMERICAN EXPRESS",
        "DISCOVER",
        "MAESTRO",
        "JCB",
        "UNIONPAY",
        "RUPAY",
        "CONTACTLESS",
        "APPLE PAY",
        "GOOGLE PAY",
        "SAMSUNG PAY",
        "PAYPAL",
        "BKASH",
        "NAGAD",
        "UPI",
        "PAYTM",
        "MOBILE PAY",
        "SWISH",
        "GIFT CARD",
        "VOUCHER",
        "BANK TRANSFER",
        "CHEQUE",
        "CHECK",
        "EFTPOS",
        "IDEAL",
        "ALIPAY",
        "WECHAT PAY",
    ),
    LabelCategory.ITEM_COUNT: (
        "ITEM COUNT",
        "ITEMS SOLD",
        "TOTAL ITEMS",
        "NO OF ITEMS",
        "NUMBER OF ITEMS",
        "QTY SOLD",
        "ARTIKEL",
    ),
    LabelCategory.DATE: (
        "DATE",
        "DATUM",
        "FECHA",
        "DATA",
        "TRANSACTION DATE",
        "SALE DATE",
        "PURCHASE DATE",
        "ISSUED",
    ),
    LabelCategory.TIME: (
        "TIME",
        "ZEIT",
        "HORA",
        "HEURE",
        "TRANSACTION TIME",
    ),
    LabelCategory.RECEIPT_ID: (
        "RECEIPT NO",
        "RECEIPT NUMBER",
        "RECEIPT #",
        "RECEIPT",
        "RCPT #",
        "RCPT NO",
        "INVOICE NO",
        "INVOICE NUMBER",
        "INVOICE #",
        "INVOICE",
        "INV #",
        "INV NO",
        "BILL NO",
        "BILL #",
        "ORDER NO",
        "ORDER NUMBER",
        "ORDER #",
        "TRANSACTION NO",
        "TRANSACTION ID",
        "TXN ID",
        "TXN #",
        "TRANS #",
        "TRANS NO",
        "TRX",
        "REF NO",
        "REF #",
        "REFERENCE",
        "TICKET NO",
        "TICKET #",
        "CHECK NO",
        "CHECK #",
        "DOC NO",
        "DOC #",
        "STORE/TRANS",
        "BELEG NR",
        "BELEG-NR",
        "BELEG",
        "RECHNUNG",
        "QUITTUNG",
        "KASSENBON",
        "FACTURE",
        "FACTURA",
        "RECIBO",
        "NOTA",
        "FATTURA",
        "SCONTRINO",
    ),
    LabelCategory.MERCHANT_CONTACT: (
        "TEL",
        "TELEPHONE",
        "PHONE",
        "MOBILE",
        "FAX",
        "EMAIL",
        "E-MAIL",
        "WEBSITE",
        "WWW",
        "CONTACT",
    ),
    LabelCategory.FOOTER: (
        "THANK YOU",
        "THANKS FOR SHOPPING",
        "PLEASE COME AGAIN",
        "HAVE A NICE DAY",
        "CUSTOMER COPY",
        "MERCHANT COPY",
        "RETURN POLICY",
        "NO REFUND",
        "EXCHANGE WITHIN",
        "VISIT US",
        "FOLLOW US",
        "SURVEY",
        "VIELEN DANK",
        "MERCI",
        "GRACIAS",
        "GRAZIE",
    ),
}

#: Category resolution order when several match the same span. More specific
#: categories come first: a line reading "SUBTOTAL" must resolve to SUBTOTAL,
#: never to TOTAL, even though "TOTAL" is a substring of it.
_CATEGORY_PRIORITY: Final[tuple[LabelCategory, ...]] = (
    LabelCategory.SUBTOTAL,
    LabelCategory.SERVICE_CHARGE,
    LabelCategory.SHIPPING,
    LabelCategory.DISCOUNT,
    LabelCategory.ROUNDING,
    LabelCategory.TAX,
    LabelCategory.TIP,
    LabelCategory.CHANGE,
    LabelCategory.TENDERED,
    LabelCategory.ITEM_COUNT,
    LabelCategory.TOTAL,
    LabelCategory.RECEIPT_ID,
    LabelCategory.DATE,
    LabelCategory.TIME,
    LabelCategory.PAYMENT_METHOD,
    LabelCategory.MERCHANT_CONTACT,
    LabelCategory.FOOTER,
)


def _compile(keywords: tuple[str, ...]) -> re.Pattern[str]:
    """Compile a longest-first alternation with word boundaries.

    Sorting by descending length is what makes ``GRAND TOTAL`` win over
    ``TOTAL`` within a category. ``\\b`` is applied only on sides that begin or
    end with a word character, so entries such as ``RECEIPT #`` still match.
    """
    parts = []
    for keyword in sorted(set(keywords), key=len, reverse=True):
        escaped = re.escape(keyword).replace(r"\ ", r"[\s.\-]*")
        prefix = r"\b" if keyword[0].isalnum() else ""
        suffix = r"\b" if keyword[-1].isalnum() else ""
        parts.append(f"{prefix}{escaped}{suffix}")
    return re.compile("|".join(parts), re.IGNORECASE | re.UNICODE)


_PATTERNS: Final[dict[LabelCategory, re.Pattern[str]]] = {
    category: _compile(keywords) for category, keywords in KEYWORDS.items()
}


def find_labels(text: str) -> list[LabelMatch]:
    """Find every category label in ``text``.

    Overlapping matches are resolved by :data:`_CATEGORY_PRIORITY` and then by
    match length, so each span is attributed to exactly one category.

    Args:
        text: A line, ideally already passed through
            :func:`~app.normalization.text.normalize_keyword_text`.

    Returns:
        Non-overlapping matches ordered by position in the line.
    """
    if not text:
        return []

    candidates: list[tuple[int, int, LabelCategory, str]] = []
    for priority, category in enumerate(_CATEGORY_PRIORITY):
        pattern = _PATTERNS.get(category)
        if pattern is None:
            continue
        for match in pattern.finditer(text):
            candidates.append((priority, match.start(), category, match.group(0)))

    # Resolve overlaps: prefer higher category priority, then the longer match.
    chosen: list[LabelMatch] = []
    occupied: list[tuple[int, int]] = []
    for _priority, start, category, keyword in sorted(
        candidates, key=lambda c: (c[0], -len(c[3]), c[1])
    ):
        end = start + len(keyword)
        if any(start < o_end and end > o_start for o_start, o_end in occupied):
            continue
        occupied.append((start, end))
        chosen.append(
            LabelMatch(
                category=category,
                keyword=keyword.upper().strip(),
                start=start,
                end=end,
                exact_line=text.strip().upper() == keyword.upper().strip(),
            )
        )

    return sorted(chosen, key=lambda m: m.start)
