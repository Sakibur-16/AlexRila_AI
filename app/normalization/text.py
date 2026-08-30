"""OCR text normalisation.

The central rule of this module: **character confusion corrections are applied
only inside numeric contexts, never globally.**

A global ``O -> 0`` substitution turns ``COFFEE`` into ``C0FFEE`` and
``TOTAL`` into ``T0TAL``, destroying the very keywords extraction depends on.
So corrections here are gated twice: a token must be *predominantly numeric*
already, and the correction must produce a token that parses as a number.
Anything else is left exactly as recognised.

Raw OCR text is never mutated in place. These functions return new strings and
the original :class:`~app.schemas.ocr.OCRResult` remains available for audit.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Final

#: Glyph pairs OCR engines genuinely confuse, mapped letter -> digit.
#: Restricted to unambiguous, well-attested confusions; ``Z``/``2`` and
#: ``G``/``6`` are omitted because they misfire far more often than they help.
DIGIT_CONFUSIONS: Final[dict[str, str]] = {
    "O": "0",
    "o": "0",
    "D": "0",
    "Q": "0",
    "I": "1",
    "l": "1",
    "|": "1",
    "!": "1",
    "S": "5",
    "s": "5",
    "B": "8",
}

#: The reverse mapping, for repairing digits inside alphabetic words.
LETTER_CONFUSIONS: Final[dict[str, str]] = {
    "0": "O",
    "1": "I",
    "5": "S",
    "8": "B",
}

#: Unicode variants that should be folded to ASCII before any parsing.
_PUNCTUATION_FOLD: Final[dict[int, str]] = {
    ord("‘"): "'",
    ord("’"): "'",
    ord("“"): '"',
    ord("”"): '"',
    ord("–"): "-",
    ord("—"): "-",
    ord("−"): "-",  # minus sign
    ord(" "): " ",  # non-breaking space
    ord("•"): "*",
    ord("，"): ",",
    ord("．"): ".",
}

_WHITESPACE = re.compile(r"[ \t​‌‍]+")
_REPEATED_PUNCT = re.compile(r"([.\-=_*~])\1{3,}")

#: A token that is plausibly a number: at least one digit, and composed only of
#: digits, confusable glyphs, separators and currency-adjacent punctuation.
_NUMERIC_CANDIDATE = re.compile(
    r"^[\-+(]?[0-9OoDQIl|!SsB.,'’\s]*[0-9][0-9OoDQIl|!SsB.,'’\s]*[)%]?$"
)

#: Digits and separators only -- an already-clean number needs no repair.
_CLEAN_NUMBER = re.compile(r"^[\-+(]?[\d.,'\s]+[)%]?$")


def normalize_unicode(text: str) -> str:
    """Fold Unicode punctuation and compose characters canonically.

    NFKC is used so that full-width digits and ligatures produced by some
    engines become their ASCII equivalents before any pattern matching runs.
    """
    if not text:
        return text
    folded = unicodedata.normalize("NFKC", text)
    return folded.translate(_PUNCTUATION_FOLD)


def clean_line(text: str) -> str:
    """Normalise whitespace and decorative runs in a single OCR line.

    Collapses runs of spaces, trims, and shortens long runs of separator
    characters (the ``-----`` dividers receipts are full of) so they do not
    look like content. Does **not** touch alphanumerics.
    """
    if not text:
        return ""
    cleaned = normalize_unicode(text)
    cleaned = _REPEATED_PUNCT.sub(r"\1\1\1", cleaned)
    cleaned = _WHITESPACE.sub(" ", cleaned)
    return cleaned.strip()


def is_numeric_context(token: str) -> bool:
    """Whether ``token`` looks like a number that OCR may have corrupted.

    Requires an existing digit: a purely alphabetic token such as ``LOSS`` is
    never treated as a number, which is exactly what stops ``S -> 5`` from
    corrupting words.
    """
    if not token:
        return False
    stripped = token.strip()
    if not any(char.isdigit() for char in stripped):
        return False
    if not _NUMERIC_CANDIDATE.match(stripped):
        return False

    # Guard against alphanumeric identifiers (SKU "A1B2C3", ref "INV0012B"):
    # every letter present must be a known confusable glyph.
    letters = [c for c in stripped if c.isalpha()]
    if not all(c in DIGIT_CONFUSIONS for c in letters):
        return False

    # Structural guard against product codes made of confusable letters, such
    # as a vitamin "B12": a real amount either starts with a digit or carries a
    # decimal/thousands separator. "B12" satisfies neither, "2S.OO" satisfies
    # both, and "1O.99" satisfies the first.
    head = stripped.lstrip("-+(")
    starts_numerically = bool(head) and (head[0].isdigit() or head[0] in ".,")
    has_separator = any(c in ".," for c in stripped)
    return starts_numerically or has_separator


def repair_numeric_token(token: str) -> tuple[str, bool]:
    """Apply digit confusions to a token *already known* to be numeric.

    Returns:
        ``(repaired, changed)``. When the token contains no confusable glyph
        it is returned untouched with ``changed=False``.
    """
    if not token or _CLEAN_NUMBER.match(token.strip()):
        return token, False
    repaired = "".join(DIGIT_CONFUSIONS.get(char, char) for char in token)
    return repaired, repaired != token


def normalize_numeric_tokens(text: str) -> tuple[str, list[str]]:
    """Repair confusable glyphs in the numeric tokens of ``text``.

    Non-numeric tokens pass through verbatim -- this is the function that
    guarantees ``COFFEE`` survives while ``1O.99`` becomes ``10.99``.

    Returns:
        ``(normalised_text, corrections)`` where each correction is rendered
        ``"before->after"`` for logging and evidence notes.
    """
    if not text:
        return text, []

    corrections: list[str] = []
    parts = re.split(r"(\s+)", text)
    output: list[str] = []

    for part in parts:
        if not part.strip():
            output.append(part)
            continue
        core, prefix, suffix = _strip_affixes(part)
        if is_numeric_context(core):
            repaired, changed = repair_numeric_token(core)
            if changed:
                corrections.append(f"{core}->{repaired}")
            output.append(prefix + repaired + suffix)
        else:
            output.append(part)

    return "".join(output), corrections


def _strip_affixes(token: str) -> tuple[str, str, str]:
    """Split leading/trailing currency and punctuation off a token.

    ``"$1O.99,"`` becomes ``("1O.99", "$", ",")`` so that the numeric test
    examines only the numeric part.
    """
    leading = 0
    while leading < len(token) and token[leading] in "$€£¥₹৳(:":
        leading += 1
    trailing = len(token)
    # Trailing separators are punctuation, not part of the amount: a receipt
    # never prints a value as "12." so nothing of value is lost by stripping.
    while trailing > leading and token[trailing - 1] in ".,;:)*":
        trailing -= 1
    return token[leading:trailing], token[:leading], token[trailing:]


def repair_word(token: str) -> str:
    """Repair digits that OCR substituted into an otherwise alphabetic word.

    The mirror of :func:`repair_numeric_token`: ``T0TAL`` becomes ``TOTAL``.
    Applied only when letters clearly dominate, so ``H2O`` and product codes
    are left alone.
    """
    if not token or len(token) < 3:
        return token
    letters = sum(1 for c in token if c.isalpha())
    digits = sum(1 for c in token if c.isdigit())
    if digits == 0 or letters < digits * 2:
        return token
    return "".join(LETTER_CONFUSIONS.get(char, char) for char in token)


def normalize_keyword_text(text: str) -> str:
    """Aggressively normalise a line for *keyword matching only*.

    Uppercases, repairs digit-for-letter substitutions and strips punctuation
    so that ``T0TAL:``, ``T O T A L`` and ``Total .....`` all match the same
    lexicon entry. The result is never stored or returned -- it exists purely
    as a matching key, which is why it can be this destructive.
    """
    if not text:
        return ""
    cleaned = normalize_unicode(text).upper()
    repaired = " ".join(repair_word(token) for token in cleaned.split())
    # Collapse letter-spaced headings ("T O T A L") into a solid word.
    if re.fullmatch(r"(?:[A-Z] ){2,}[A-Z]", repaired):
        repaired = repaired.replace(" ", "")
    return re.sub(r"[^\w\s%]", " ", repaired).strip()


def strip_leading_label(text: str, label: str) -> str:
    """Remove ``label`` and any separator from the start of ``text``."""
    pattern = re.compile(rf"^\s*{re.escape(label)}\s*[:#=.\-]*\s*", re.IGNORECASE)
    return pattern.sub("", text, count=1).strip()


def contains_suspicious_characters(text: str) -> bool:
    """Detect recognition noise that suggests an unreliable read.

    Looks for replacement characters and dense runs of symbol glyphs, both of
    which indicate the engine was guessing at unreadable regions.
    """
    if not text:
        return False
    if "�" in text:
        return True
    # Receipts are full of legitimate decoration: "=====" separators, "*****"
    # emphasis, "---" rules. Counting those as noise would flag almost every
    # receipt, so only genuinely unusual glyphs contribute.
    symbols = sum(
        1 for c in text if not c.isalnum() and not c.isspace() and c not in ".,:-/$%()*#&+'=~_[]"
    )
    return symbols > max(8, len(text) * 0.15)
