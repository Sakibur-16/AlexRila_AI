"""Currency detection.

``$`` is shared by more than twenty currencies, ``£`` by several, and ``kr`` by
four Nordic ones. Mapping ``$ -> USD`` unconditionally is the single most
common correctness bug in receipt parsers, so detection here is evidence-based
and ranked:

1. An explicit ISO-4217 code printed on the receipt (``USD``, ``EUR``) --
   unambiguous, highest confidence.
2. A symbol that maps to exactly one currency (``€``, ``৳``, ``₹``).
3. An ambiguous symbol *plus* corroborating context -- a configured country, a
   locale-specific tax word (``GST``/``VAT``/``TVA``), or a country name.
4. An ambiguous symbol with no context: the symbol is reported in
   ``currency_symbol``, ``currency`` stays ``None``, and
   ``AMBIGUOUS_CURRENCY`` is raised.

A configured ``DEFAULT_CURRENCY`` is a last resort and is marked as
:attr:`~app.domain.evidence.ExtractionMethod.CONFIGURED`, which caps its
confidence -- it is an operator's assumption, not something the document said.
"""

from __future__ import annotations

import re
from typing import Final, NamedTuple

from app.normalization.text import normalize_unicode

#: Symbols that identify exactly one currency.
UNAMBIGUOUS_SYMBOLS: Final[dict[str, str]] = {
    "€": "EUR",
    "£": "GBP",
    "₹": "INR",
    "৳": "BDT",
    "₩": "KRW",
    "₪": "ILS",
    "₺": "TRY",
    "฿": "THB",
    "₦": "NGN",
    "₴": "UAH",
    "₫": "VND",
    "₱": "PHP",
    "₲": "PYG",
    "₡": "CRC",
    "﷼": "SAR",
    "zł": "PLN",
    "Kč": "CZK",
    "Ft": "HUF",
    "R$": "BRL",
}

#: Symbols shared by several currencies, with the candidates they may denote.
AMBIGUOUS_SYMBOLS: Final[dict[str, tuple[str, ...]]] = {
    "$": ("USD", "CAD", "AUD", "NZD", "SGD", "HKD", "MXN", "BRL", "ARS", "CLP", "TWD"),
    "¥": ("JPY", "CNY"),
    "kr": ("SEK", "NOK", "DKK", "ISK"),
    "R": ("ZAR",),
    "Rs": ("INR", "PKR", "LKR", "NPR"),
    "RM": ("MYR",),
}

#: ISO-4217 codes this pipeline recognises when printed literally.
KNOWN_CODES: Final[frozenset[str]] = frozenset(
    {
        "USD",
        "EUR",
        "GBP",
        "JPY",
        "CNY",
        "INR",
        "BDT",
        "PKR",
        "LKR",
        "NPR",
        "CAD",
        "AUD",
        "NZD",
        "SGD",
        "HKD",
        "MYR",
        "THB",
        "PHP",
        "IDR",
        "VND",
        "KRW",
        "TWD",
        "AED",
        "SAR",
        "QAR",
        "KWD",
        "BHD",
        "OMR",
        "ILS",
        "TRY",
        "ZAR",
        "NGN",
        "KES",
        "EGP",
        "MAD",
        "GHS",
        "SEK",
        "NOK",
        "DKK",
        "ISK",
        "PLN",
        "CZK",
        "HUF",
        "RON",
        "BGN",
        "HRK",
        "RUB",
        "UAH",
        "CHF",
        "MXN",
        "BRL",
        "ARS",
        "CLP",
        "COP",
        "PEN",
        "UYU",
        "CRC",
        "PYG",
    }
)

#: Country hint -> currency. Used to resolve an ambiguous symbol.
COUNTRY_CURRENCY: Final[dict[str, str]] = {
    "US": "USD",
    "CA": "CAD",
    "AU": "AUD",
    "NZ": "NZD",
    "SG": "SGD",
    "HK": "HKD",
    "MX": "MXN",
    "BR": "BRL",
    "GB": "GBP",
    "IE": "EUR",
    "DE": "EUR",
    "FR": "EUR",
    "ES": "EUR",
    "IT": "EUR",
    "NL": "EUR",
    "IN": "INR",
    "BD": "BDT",
    "PK": "PKR",
    "LK": "LKR",
    "NP": "NPR",
    "JP": "JPY",
    "CN": "CNY",
    "SE": "SEK",
    "NO": "NOK",
    "DK": "DKK",
    "ZA": "ZAR",
    "AE": "AED",
    "SA": "SAR",
    "TR": "TRY",
    "CH": "CHF",
    "MY": "MYR",
    "TH": "THB",
    "PH": "PHP",
    "ID": "IDR",
    "VN": "VND",
    "KR": "KRW",
    "TW": "TWD",
    "PL": "PLN",
    "CZ": "CZK",
    "HU": "HUF",
}

#: Locale-specific tax vocabulary that corroborates an ambiguous symbol.
#: Only entries that genuinely narrow the field are listed -- "VAT" spans too
#: many countries to be useful, so it is deliberately absent.
TAX_WORD_HINTS: Final[dict[str, tuple[str, ...]]] = {
    "GST": ("AUD", "NZD", "CAD", "SGD", "INR"),
    "HST": ("CAD",),
    "PST": ("CAD",),
    "QST": ("CAD",),
    "TVA": ("EUR", "CHF"),
    "MWST": ("EUR", "CHF"),
    "IVA": ("EUR", "MXN", "ARS", "CLP"),
    "BTW": ("EUR",),
    "MOMS": ("SEK", "DKK", "NOK"),
    "SALES TAX": ("USD",),
    "STATE TAX": ("USD",),
}

_CODE_PATTERN: Final[re.Pattern[str]] = re.compile(r"\b([A-Z]{3})\b")

#: Symbols ordered longest-first so "R$" is matched before "R".
_SYMBOL_ORDER: Final[tuple[str, ...]] = tuple(
    sorted(
        list(UNAMBIGUOUS_SYMBOLS) + list(AMBIGUOUS_SYMBOLS),
        key=len,
        reverse=True,
    )
)


class CurrencyDetection(NamedTuple):
    """Outcome of currency detection.

    ``code`` is ``None`` when nothing could be concluded confidently -- the
    symbol is still reported so a consumer can decide for itself.
    """

    code: str | None
    symbol: str | None
    confidence: float
    #: How the conclusion was reached, for evidence notes.
    reason: str
    warnings: tuple[str, ...] = ()
    #: Every distinct currency indicator seen, for multi-currency detection.
    candidates: tuple[str, ...] = ()


def find_symbols(text: str) -> list[str]:
    """Return every currency symbol present in ``text``, in first-seen order."""
    source = normalize_unicode(text)
    found: list[str] = []
    for symbol in _SYMBOL_ORDER:
        if symbol in source and symbol not in found:
            # An alphabetic symbol ("R", "kr", "Rs", "RM") must stand as its
            # own token *and* sit directly against an amount. Without the
            # second condition a receipt number like "R-2026-00815" would
            # register as South African rand.
            if symbol.isalpha() and not re.search(
                rf"(?<![A-Za-z]){re.escape(symbol)}\.?\s?\d", source
            ):
                continue
            found.append(symbol)
    return found


def find_codes(text: str) -> list[str]:
    """Return ISO-4217 codes printed literally in ``text``."""
    source = normalize_unicode(text).upper()
    return [code for code in dict.fromkeys(_CODE_PATTERN.findall(source)) if code in KNOWN_CODES]


def detect_currency(
    text: str,
    *,
    country_hint: str = "",
    default_currency: str = "",
) -> CurrencyDetection:
    """Determine the receipt's currency from its full text.

    Args:
        text: Complete OCR text of the document.
        country_hint: ISO 3166-1 alpha-2 country from configuration.
        default_currency: Operator-configured fallback ISO code.

    Returns:
        A :class:`CurrencyDetection`. Confidence reflects the strength of the
        evidence, not a fixed per-source constant.
    """
    if not text:
        return CurrencyDetection(
            code=default_currency or None,
            symbol=None,
            confidence=0.30 if default_currency else 0.0,
            reason="configured_default" if default_currency else "no_evidence",
            warnings=() if default_currency else ("MISSING_CURRENCY",),
        )

    codes = find_codes(text)
    symbols = find_symbols(text)
    candidates = tuple(dict.fromkeys(codes + symbols))

    warnings: list[str] = []
    # Two distinct indicators of any kind -- two ISO codes, two symbols, or a
    # symbol alongside a code for a different currency -- mean the document
    # mixes currencies (or OCR invented one). Either way the caller must know.
    distinct_indicators = len(set(codes)) + len(set(symbols))
    # One code alongside its own symbol ("EUR" and "€") is a single currency
    # stated twice, not a mixed-currency document.
    agrees_with_itself = (
        len(codes) == 1 and len(symbols) == 1 and UNAMBIGUOUS_SYMBOLS.get(symbols[0]) == codes[0]
    )
    if distinct_indicators > 1 and not agrees_with_itself:
        warnings.append("MULTIPLE_CURRENCIES_DETECTED")

    # 1. An explicit ISO code is definitive.
    if codes:
        return CurrencyDetection(
            code=codes[0],
            symbol=symbols[0] if symbols else None,
            confidence=0.97,
            reason="iso_code_in_text",
            warnings=tuple(warnings),
            candidates=candidates,
        )

    # 2. A symbol that can only mean one currency.
    for symbol in symbols:
        if symbol in UNAMBIGUOUS_SYMBOLS:
            return CurrencyDetection(
                code=UNAMBIGUOUS_SYMBOLS[symbol],
                symbol=symbol,
                confidence=0.92,
                reason="unambiguous_symbol",
                warnings=tuple(warnings),
                candidates=candidates,
            )

    # 3. An ambiguous symbol, resolved only if context corroborates.
    for symbol in symbols:
        options = AMBIGUOUS_SYMBOLS.get(symbol)
        if not options:
            continue
        if len(options) == 1:
            return CurrencyDetection(
                code=options[0],
                symbol=symbol,
                confidence=0.88,
                reason="single_candidate_symbol",
                warnings=tuple(warnings),
                candidates=candidates,
            )

        resolved = _resolve_with_context(options, text, country_hint, default_currency)
        if resolved is not None:
            code, reason, confidence = resolved
            return CurrencyDetection(
                code=code,
                symbol=symbol,
                confidence=confidence,
                reason=reason,
                warnings=tuple(warnings),
                candidates=candidates,
            )

        warnings.append("AMBIGUOUS_CURRENCY")
        return CurrencyDetection(
            code=None,
            symbol=symbol,
            confidence=0.0,
            reason="ambiguous_symbol_no_context",
            warnings=tuple(warnings),
            candidates=candidates,
        )

    # 4. No indicator at all.
    if default_currency:
        return CurrencyDetection(
            code=default_currency,
            symbol=None,
            confidence=0.30,
            reason="configured_default",
            warnings=tuple(warnings),
            candidates=candidates,
        )

    warnings.append("MISSING_CURRENCY")
    return CurrencyDetection(
        code=None,
        symbol=None,
        confidence=0.0,
        reason="no_evidence",
        warnings=tuple(warnings),
        candidates=candidates,
    )


def _resolve_with_context(
    options: tuple[str, ...],
    text: str,
    country_hint: str,
    default_currency: str,
) -> tuple[str, str, float] | None:
    """Narrow ambiguous symbol candidates using surrounding evidence.

    Returns ``(code, reason, confidence)`` or ``None`` when nothing corroborates.
    """
    upper = normalize_unicode(text).upper()

    # A configured country is the strongest contextual signal: the operator
    # knows where these receipts come from.
    if country_hint:
        mapped = COUNTRY_CURRENCY.get(country_hint.upper())
        if mapped and mapped in options:
            return mapped, "symbol_plus_country_hint", 0.90

    # Locale-specific tax vocabulary printed on the receipt itself.
    for word, implied in TAX_WORD_HINTS.items():
        if re.search(rf"\b{re.escape(word)}\b", upper):
            overlap = [code for code in options if code in implied]
            if len(overlap) == 1:
                return overlap[0], f"symbol_plus_tax_term:{word}", 0.85

    # A configured default that is one of the candidates is weak but coherent
    # corroboration.
    if default_currency and default_currency in options:
        return default_currency, "symbol_plus_configured_default", 0.70

    return None
