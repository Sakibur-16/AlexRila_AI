"""Extraction context.

Computed once per document and shared by every extractor. It exists so that
the expensive, document-wide analyses -- keyword normalisation, label
indexing, separator-style detection, section segmentation -- happen a single
time rather than once per field, and so that every extractor sees the *same*
view of the document.

It also owns the raw/normalised distinction: :attr:`LineView.raw` is exactly
what OCR produced and is what evidence cites, while :attr:`LineView.normalized`
and :attr:`LineView.keyword_text` are derived views used only for matching.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from dataclasses import dataclass, field

from app.core.config import Settings
from app.domain.evidence import BoundingBox, Evidence, ExtractionMethod
from app.extraction.lexicon import LabelCategory, LabelMatch, find_labels
from app.normalization import money as money_mod
from app.normalization.money import ParsedAmount, SeparatorStyle
from app.normalization.text import clean_line, normalize_keyword_text, normalize_numeric_tokens
from app.schemas.ocr import OCRLine, OCRResult
from app.schemas.receipt import ReceiptSection

#: A line of only separator characters, used as a section boundary hint.
_DIVIDER = re.compile(r"^[\s\-=_*~.#]{3,}$")


@dataclass(frozen=True, slots=True)
class LineView:
    """One OCR line with its derived representations.

    Attributes:
        index: Position in the document.
        raw: Text exactly as recognised. Cited by all evidence.
        normalized: Whitespace-cleaned with numeric tokens repaired. Used for
            amount parsing.
        keyword_text: Aggressively uppercased/repaired form used *only* for
            label matching, never surfaced.
        labels: Vocabulary matches found on this line.
        amounts: Every monetary amount on the line, in reading order.
        confidence: OCR confidence for the line.
        bbox: Geometry, when the provider supplied it.
        section: Layout region assigned by the structure detector.
        corrections: OCR repairs applied, for evidence notes.
    """

    index: int
    raw: str
    normalized: str
    keyword_text: str
    labels: tuple[LabelMatch, ...]
    amounts: tuple[ParsedAmount, ...]
    confidence: float
    bbox: BoundingBox | None = None
    section: ReceiptSection = ReceiptSection.UNKNOWN
    corrections: tuple[str, ...] = ()

    @property
    def is_blank(self) -> bool:
        return not self.raw.strip()

    @property
    def is_divider(self) -> bool:
        return bool(_DIVIDER.match(self.raw.strip()))

    @property
    def has_amount(self) -> bool:
        return bool(self.amounts)

    @property
    def has_monetary_amount(self) -> bool:
        """Whether the line carries an amount with a fractional part.

        A bare integer is not evidence of money: house numbers, phone
        fragments and quantities all parse as integers. Requiring a decimal
        separator is what keeps "123 Oak Street" in the merchant header
        instead of starting the items block.
        """
        return any("." in amount.raw or "," in amount.raw for amount in self.amounts)

    def label_of(self, *categories: LabelCategory) -> LabelMatch | None:
        """First label on this line belonging to any of ``categories``."""
        wanted = set(categories)
        for label in self.labels:
            if label.category in wanted:
                return label
        return None

    def has(self, *categories: LabelCategory) -> bool:
        return self.label_of(*categories) is not None

    @property
    def offsets_aligned(self) -> bool:
        """Whether ``keyword_text`` offsets are valid in ``normalized``.

        The keyword view is built with length-preserving substitutions, so the
        two normally align character for character. The one exception is the
        collapse of letter-spaced headings ("T O T A L"), which shortens the
        string -- and that only happens on lines carrying no amounts.
        """
        return len(self.keyword_text) == len(self.normalized)

    def text_after(self, label: LabelMatch) -> str:
        """Text following ``label``, bounded by the next label on the line.

        Reads from ``normalized`` rather than ``keyword_text`` because the
        keyword view replaces punctuation with spaces -- which would turn
        ``25.99`` into ``25 99`` and ``INV-0012`` into ``INV 0012``.
        """
        source = self.normalized if self.offsets_aligned else self.keyword_text
        end = len(source)
        for other in self.labels:
            if other.start >= label.end:
                end = min(end, other.start)
        return _strip_separators(source[label.end : end])

    def evidence(self, method: ExtractionMethod, notes: str | None = None) -> Evidence:
        """Build an :class:`Evidence` record citing this line."""
        return Evidence(
            source_text=self.raw,
            line_index=self.index,
            bbox=self.bbox,
            ocr_confidence=self.confidence,
            method=method,
            notes=notes,
        )


@dataclass
class ReceiptContext:
    """Document-wide state shared by every extractor."""

    ocr: OCRResult
    settings: Settings
    lines: list[LineView] = field(default_factory=list)
    separator_style: SeparatorStyle = SeparatorStyle.UNKNOWN
    #: Index of the first line judged to belong to the items region.
    items_start: int = 0
    #: Index just past the last items line; equals the first totals line.
    items_end: int = 0
    #: All OCR glyph repairs applied across the document, for diagnostics.
    corrections: list[str] = field(default_factory=list)

    # ------------------------------------------------------------ traversal
    def __iter__(self) -> Iterator[LineView]:
        return iter(self.lines)

    def __len__(self) -> int:
        return len(self.lines)

    def line(self, index: int) -> LineView | None:
        if 0 <= index < len(self.lines):
            return self.lines[index]
        return None

    def in_section(self, *sections: ReceiptSection) -> list[LineView]:
        return [line for line in self.lines if line.section in sections]

    def find_all(self, *categories: LabelCategory) -> list[LineView]:
        """Every line carrying a label of any of ``categories``."""
        return [line for line in self.lines if line.has(*categories)]

    def find_last(self, *categories: LabelCategory) -> LineView | None:
        """Last line carrying such a label.

        Last rather than first because receipts print running totals: the
        final ``TOTAL`` is the authoritative one when several appear.
        """
        matches = self.find_all(*categories)
        return matches[-1] if matches else None

    @property
    def has_geometry(self) -> bool:
        return any(line.bbox is not None for line in self.lines)

    @property
    def full_text(self) -> str:
        return "\n".join(line.raw for line in self.lines)

    # ------------------------------------------------------- value location
    def value_for_label(
        self, line: LineView, label: LabelMatch, *, allow_next_line: bool = True
    ) -> tuple[ParsedAmount, Evidence] | None:
        """Find the amount belonging to ``label``.

        Resolution order mirrors how receipts are laid out:

        1. An amount to the right of the label on the same line -- the normal
           case, and the highest-confidence one.
        2. An amount to the *left*, for right-to-left layouts and for the
           occasional ``12.34 TOTAL``.
        3. The next non-blank line, when the label sits alone -- a real layout
           on narrow thermal receipts, scored lower as a spatial inference.

        Returns:
            ``(amount, evidence)``, or ``None`` when no amount can be
            associated. Never falls back to "some amount somewhere".
        """
        same_line = self._amount_after_label(line, label)
        if same_line is not None:
            return same_line, line.evidence(
                ExtractionMethod.KEYWORD_ANCHORED, notes=f"label={label.keyword}"
            )

        if line.amounts:
            return line.amounts[-1], line.evidence(
                ExtractionMethod.KEYWORD_ANCHORED, notes=f"label={label.keyword}:left_of_label"
            )

        if allow_next_line:
            following = self._next_content_line(line.index)
            if following is not None and following.amounts and not following.labels:
                return following.amounts[0], following.evidence(
                    ExtractionMethod.SPATIAL,
                    notes=f"label={label.keyword}@line{line.index}",
                )

        return None

    def _amount_after_label(self, line: LineView, label: LabelMatch) -> ParsedAmount | None:
        """Parse the amount belonging to ``label`` on the same line.

        The search region ends at the next label, so ``TOTAL 25.99 CASH 30.00``
        attributes 25.99 to TOTAL rather than reaching past it. Percentage
        rates are removed first, because ``VAT 20% 4.00`` prints the rate and
        the amount side by side and only the latter is the value.

        The rightmost remaining amount is taken: receipts print modifiers
        (quantity, rate, taxable base) to the left of the value, never to the
        right.
        """
        tail = line.text_after(label)
        if not tail:
            return None
        tail = _PERCENTAGE_RATE.sub(" ", tail)
        amounts = money_mod.parse_all_amounts(tail, style=self.separator_style, repair_ocr=False)
        return amounts[-1] if amounts else None

    def _next_content_line(self, index: int) -> LineView | None:
        """The next line that carries content (not blank, not a divider)."""
        for candidate in self.lines[index + 1 : index + 4]:
            if candidate.is_blank or candidate.is_divider:
                continue
            return candidate
        return None


def _strip_separators(text: str) -> str:
    """Trim label/value separators without eating a leading minus sign.

    ``TOTAL - 5.00`` uses a dash as a separator, while ``TOTAL -5.00`` uses it
    as a sign on a refund. Stripping both identically would silently turn every
    negative amount positive, so a dash is only removed when whitespace follows
    it.
    """
    stripped = text.strip().strip(" :#=.")
    while stripped.startswith("-") and stripped[1:2].isspace():
        stripped = stripped[1:].lstrip()
    return stripped.rstrip(" :#=.-")


#: A percentage rate printed alongside an amount ("VAT 20% 4.00"). Removed
#: before amount parsing so the rate is never mistaken for the value.
_PERCENTAGE_RATE = re.compile(r"\d{1,2}(?:[.,]\d{1,3})?\s*%")


def build_context(ocr: OCRResult, settings: Settings) -> ReceiptContext:
    """Analyse an OCR result into a :class:`ReceiptContext`.

    Runs in two passes: the first builds line views and detects the document's
    decimal convention, the second re-parses amounts under that convention and
    assigns sections. The two passes are necessary because separator style is a
    document-level property that cannot be known while reading the first line.
    """
    raw_lines = list(ocr.lines) or _synthesize_lines(ocr)

    cleaned = [clean_line(line.text) for line in raw_lines]
    separator_style = money_mod.detect_separator_style(
        cleaned, configured=settings.decimal_separator
    )

    views: list[LineView] = []
    all_corrections: list[str] = []

    for index, (ocr_line, text) in enumerate(zip(raw_lines, cleaned, strict=True)):
        normalized, corrections = normalize_numeric_tokens(text)
        all_corrections.extend(corrections)
        keyword_text = normalize_keyword_text(text)
        views.append(
            LineView(
                index=index,
                raw=ocr_line.text,
                normalized=normalized,
                keyword_text=keyword_text,
                labels=tuple(find_labels(keyword_text)),
                amounts=tuple(
                    money_mod.parse_all_amounts(normalized, style=separator_style, repair_ocr=False)
                ),
                confidence=ocr_line.effective_confidence,
                bbox=ocr_line.bbox.to_domain() if ocr_line.bbox else None,
                corrections=tuple(corrections),
            )
        )

    context = ReceiptContext(
        ocr=ocr,
        settings=settings,
        lines=views,
        separator_style=separator_style,
        corrections=all_corrections,
    )
    _assign_sections(context)
    return context


def _synthesize_lines(ocr: OCRResult) -> list[OCRLine]:
    """Build line objects from flat text.

    A provider that returns only a text blob still has to work; it simply
    yields no geometry and a document-level confidence per line.
    """
    return [
        OCRLine(text=text, confidence=ocr.mean_confidence or None) for text in ocr.text.splitlines()
    ]


def _is_price_only(line: LineView | None) -> bool:
    """Whether a line consists of a single monetary amount and nothing else."""
    if line is None or not line.has_monetary_amount or len(line.amounts) != 1:
        return False
    remainder = line.normalized.replace(line.amounts[0].raw, "", 1)
    return not remainder.strip(" 	$€£¥₹৳.,:-")


def _assign_sections(context: ReceiptContext) -> None:
    """Segment the document into layout regions.

    The totals block is located first because it is the most reliably
    identifiable region -- it is where the subtotal/tax/total labels cluster.
    Everything above it and below the header is treated as the items region,
    and everything after the payment labels is the footer.

    Sections are *hints*, not constraints: extractors prefer their own section
    but fall back to the whole document, because receipts do not always follow
    this layout.
    """
    lines = context.lines
    if not lines:
        return

    totals_categories = (
        LabelCategory.SUBTOTAL,
        LabelCategory.TOTAL,
        LabelCategory.TAX,
        LabelCategory.SERVICE_CHARGE,
        LabelCategory.DISCOUNT,
        LabelCategory.SHIPPING,
    )
    totals_indices = [i for i, line in enumerate(lines) if line.has(*totals_categories)]
    payment_indices = [
        i
        for i, line in enumerate(lines)
        if line.has(LabelCategory.PAYMENT_METHOD, LabelCategory.CHANGE, LabelCategory.TENDERED)
    ]

    totals_start = min(totals_indices) if totals_indices else len(lines)
    if not totals_indices and len(lines) >= 3 and _is_price_only(lines[-1]):
        totals_start = len(lines) - 1

    totals_end = max(totals_indices) + 1 if totals_indices else len(lines)
    payment_start = (
        min(i for i in payment_indices if i >= totals_start)
        if any(i >= totals_start for i in payment_indices)
        else totals_end
    )

    # The header is the merchant block: the leading lines before any amount or
    # metadata label appears. Capped so a receipt with no items does not
    # classify itself entirely as header.
    header_end = 0
    for index, line in enumerate(lines[: min(12, len(lines))]):
        if index > 0 and line.is_divider:
            break
        if line.has_monetary_amount or line.has(
            LabelCategory.DATE, LabelCategory.TIME, LabelCategory.RECEIPT_ID
        ):
            break
        # A line whose successor is nothing but a price is an item description
        # on a narrow receipt, not part of the merchant block. Without this the
        # header would swallow the first item of any receipt that prints
        # descriptions and prices on separate lines.
        if index > 0 and _is_price_only(lines[index + 1] if index + 1 < len(lines) else None):
            break
        header_end = index + 1

    metadata_end = header_end
    for index in range(header_end, min(header_end + 8, totals_start)):
        line = lines[index]
        if line.has(LabelCategory.DATE, LabelCategory.TIME, LabelCategory.RECEIPT_ID):
            metadata_end = index + 1

    context.items_start = metadata_end
    context.items_end = totals_start

    updated: list[LineView] = []
    for index, line in enumerate(lines):
        if index < header_end:
            section = ReceiptSection.MERCHANT
        elif index < metadata_end:
            section = ReceiptSection.METADATA
        elif index < totals_start:
            section = ReceiptSection.ITEMS
        elif index < payment_start:
            section = ReceiptSection.TOTALS
        elif index < max(payment_start + 6, totals_end):
            section = ReceiptSection.PAYMENT
        else:
            section = ReceiptSection.FOOTER
        updated.append(
            LineView(
                index=line.index,
                raw=line.raw,
                normalized=line.normalized,
                keyword_text=line.keyword_text,
                labels=line.labels,
                amounts=line.amounts,
                confidence=line.confidence,
                bbox=line.bbox,
                section=section,
                corrections=line.corrections,
            )
        )
    context.lines = updated
