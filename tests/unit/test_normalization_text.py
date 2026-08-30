"""OCR text normalisation tests.

The central property under test: glyph confusion repair must fix numbers
without ever corrupting words. Each case in :data:`_PRESERVED` is a real word
or code that a naive global ``O -> 0`` substitution would destroy.
"""

from __future__ import annotations

import pytest

from app.normalization.text import (
    clean_line,
    contains_suspicious_characters,
    is_numeric_context,
    normalize_keyword_text,
    normalize_numeric_tokens,
    normalize_unicode,
    repair_word,
)

#: (input, expected) pairs where a numeric token must be repaired.
_REPAIRED = [
    ("TOTAL 1O.99", "TOTAL 10.99"),
    ("SUBTOTAL 2S.OO", "SUBTOTAL 25.00"),
    ("CASH 1OO.00", "CASH 100.00"),
    ("$1O.99", "$10.99"),
    ("2 x 4.OO 8.00", "2 x 4.00 8.00"),
    ("O.99", "0.99"),
    ("MILK 1L 2.SO", "MILK 1L 2.50"),
    ("TAX l.25", "TAX 1.25"),
]

#: Text that must survive normalisation byte for byte.
_PRESERVED = [
    "COFFEE",
    "LOSS",
    "BOSS SODA",
    "VITAMIN B12",
    "SKU A1B2C3",
    "1ST FLOOR",
    "5OZ CAN",
    "ID 12345",
    "SOAP",
    "OIL",
    "SALSA",
    "BISCUITS",
]


@pytest.mark.parametrize(("source", "expected"), _REPAIRED)
def test_numeric_tokens_are_repaired(source: str, expected: str) -> None:
    result, corrections = normalize_numeric_tokens(source)
    assert result == expected
    assert corrections, "a repair should be recorded for evidence"


@pytest.mark.parametrize("source", _PRESERVED)
def test_words_are_never_corrupted(source: str) -> None:
    """A word must pass through untouched -- this is the anti-corruption rule."""
    result, corrections = normalize_numeric_tokens(source)
    assert result == source
    assert corrections == []


@pytest.mark.parametrize(
    ("token", "expected"),
    [
        ("1O.99", True),
        ("12345", True),
        ("2S.OO", True),
        ("COFFEE", False),
        ("LOSS", False),
        ("B12", False),
        ("A1B2C3", False),
        ("", False),
        ("OO", False),
    ],
)
def test_numeric_context_detection(token: str, expected: bool) -> None:
    assert is_numeric_context(token) is expected


def test_keyword_text_repairs_digits_in_labels() -> None:
    """A label misread as T0TAL must still match the TOTAL vocabulary."""
    assert normalize_keyword_text("T0TAL: ....") == "TOTAL"
    assert normalize_keyword_text("SUBT0TAL") == "SUBTOTAL"


def test_keyword_text_collapses_letter_spacing() -> None:
    assert normalize_keyword_text("S U B T O T A L") == "SUBTOTAL"


def test_repair_word_leaves_mixed_codes_alone() -> None:
    assert repair_word("H2O") == "H2O"
    assert repair_word("A1B2") == "A1B2"
    assert repair_word("T0TAL") == "TOTAL"


def test_clean_line_collapses_decoration() -> None:
    assert clean_line("  MARKET   ------------  ") == "MARKET ---"
    assert clean_line("") == ""


def test_normalize_unicode_folds_punctuation() -> None:
    assert normalize_unicode("1 234,50") == "1 234,50"
    assert normalize_unicode("“QUOTED”") == '"QUOTED"'


def test_suspicious_characters_detected() -> None:
    assert contains_suspicious_characters("TOTAL �� 25.99")
    assert contains_suspicious_characters("~~^^{}[]<>|\\@@@@@")
    assert not contains_suspicious_characters("TOTAL $25.99 (VAT 20%)")


def test_raw_text_is_not_mutated_in_place() -> None:
    """Normalisation returns new strings; the original stays available."""
    source = "TOTAL 1O.99"
    normalized, _ = normalize_numeric_tokens(source)
    assert source == "TOTAL 1O.99"
    assert normalized != source
