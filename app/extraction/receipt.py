"""Rule-based receipt extraction.

Orchestrates the field extractors and assembles the internal
:class:`ExtractionResult` -- the evidence-carrying form of a receipt that the
validation and confidence layers consume before it is projected into the
public :class:`~app.schemas.receipt.Receipt` schema.

Deterministic extraction is deliberately the primary path. Every field here
has *shape* -- a label, a pattern, a position -- and rules exploit that shape
at zero marginal cost, with reproducible output and no possibility of
invention. The LLM layer exists for the residue that genuinely needs semantic
judgement, and it is invoked only when this layer reports low confidence.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date as date_type
from datetime import datetime as datetime_type
from datetime import time as time_type
from decimal import Decimal
from typing import Any

from app.core.config import Settings
from app.domain.evidence import ExtractedField, ExtractionMethod
from app.extraction.context import ReceiptContext, build_context
from app.extraction.dates import extract_datetime
from app.extraction.items import ItemsResult, extract_items
from app.extraction.lexicon import LabelCategory
from app.extraction.merchant import extract_merchant
from app.extraction.payment import extract_payment
from app.extraction.totals import TotalsResult, extract_totals
from app.normalization.currency import detect_currency
from app.normalization.numbers import extract_receipt_number
from app.schemas.ocr import OCRResult
from app.schemas.receipt import Discount, PaymentMethod, PaymentSplit, Tax


@dataclass(slots=True)
class ExtractionResult:
    """Receipt fields with their evidence, before schema projection.

    Every attribute is an :class:`ExtractedField`, so provenance and
    per-field warnings survive all the way to the confidence layer. The public
    schema is produced from this by
    :func:`~app.pipeline.assembly.build_receipt`.
    """

    # merchant
    merchant_name: ExtractedField[str] = field(default_factory=ExtractedField.absent)
    merchant_address: ExtractedField[str] = field(default_factory=ExtractedField.absent)
    merchant_phone: ExtractedField[str] = field(default_factory=ExtractedField.absent)
    merchant_email: ExtractedField[str] = field(default_factory=ExtractedField.absent)
    merchant_website: ExtractedField[str] = field(default_factory=ExtractedField.absent)
    merchant_tax_id: ExtractedField[str] = field(default_factory=ExtractedField.absent)
    merchant_registration_id: ExtractedField[str] = field(default_factory=ExtractedField.absent)
    merchant_store_id: ExtractedField[str] = field(default_factory=ExtractedField.absent)

    # transaction
    transaction_date: ExtractedField[date_type] = field(default_factory=ExtractedField.absent)
    transaction_time: ExtractedField[time_type] = field(default_factory=ExtractedField.absent)
    transaction_datetime: ExtractedField[datetime_type] = field(
        default_factory=ExtractedField.absent
    )
    receipt_number: ExtractedField[str] = field(default_factory=ExtractedField.absent)

    # money
    items: tuple[Any, ...] = ()
    item_methods: tuple[ExtractionMethod, ...] = ()
    item_evidence: tuple[Any, ...] = ()
    subtotal: ExtractedField[Decimal] = field(default_factory=ExtractedField.absent)
    total: ExtractedField[Decimal] = field(default_factory=ExtractedField.absent)
    tax: ExtractedField[Tax] = field(default_factory=ExtractedField.absent)
    discount: ExtractedField[Discount] = field(default_factory=ExtractedField.absent)
    service_charge: ExtractedField[Decimal] = field(default_factory=ExtractedField.absent)
    shipping: ExtractedField[Decimal] = field(default_factory=ExtractedField.absent)
    tip: ExtractedField[Decimal] = field(default_factory=ExtractedField.absent)
    rounding: ExtractedField[Decimal] = field(default_factory=ExtractedField.absent)

    # currency
    currency: ExtractedField[str] = field(default_factory=ExtractedField.absent)
    currency_symbol: ExtractedField[str] = field(default_factory=ExtractedField.absent)

    # payment
    payment_method: ExtractedField[PaymentMethod] = field(default_factory=ExtractedField.absent)
    card_type: ExtractedField[str] = field(default_factory=ExtractedField.absent)
    card_last4: ExtractedField[str] = field(default_factory=ExtractedField.absent)
    authorization_code: ExtractedField[str] = field(default_factory=ExtractedField.absent)
    amount_tendered: ExtractedField[Decimal] = field(default_factory=ExtractedField.absent)
    change: ExtractedField[Decimal] = field(default_factory=ExtractedField.absent)
    payment_splits: tuple[PaymentSplit, ...] = ()

    #: The context the fields were read from. Retained for the validation and
    #: confidence layers, and for LLM grounding.
    context: ReceiptContext | None = None
    #: Item lines the extractor could not interpret.
    skipped_item_lines: int = 0

    def field_map(self) -> dict[str, ExtractedField[Any]]:
        """Return fields keyed by their dotted path in the public schema.

        This mapping is the join between internal fields, validation issue
        paths and confidence keys, so all three vocabularies stay aligned.
        """
        return {
            "merchant.name": self.merchant_name,
            "merchant.address": self.merchant_address,
            "merchant.phone": self.merchant_phone,
            "merchant.email": self.merchant_email,
            "merchant.website": self.merchant_website,
            "merchant.tax_id": self.merchant_tax_id,
            "merchant.registration_id": self.merchant_registration_id,
            "merchant.store_id": self.merchant_store_id,
            "transaction.date": self.transaction_date,
            "transaction.time": self.transaction_time,
            "receipt_number": self.receipt_number,
            "subtotal": self.subtotal,
            "total": self.total,
            "tax.total": self.tax,
            "discount": self.discount,
            "service_charge": self.service_charge,
            "shipping": self.shipping,
            "tip": self.tip,
            "currency": self.currency,
            "payment.method": self.payment_method,
            "payment.card_last4": self.card_last4,
        }


class RuleBasedReceiptExtractor:
    """Deterministic receipt extraction.

    Stateless with respect to documents: all per-document state lives in the
    :class:`~app.extraction.context.ReceiptContext`, so one instance is safe to
    share across concurrent requests.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    def extract(self, ocr: OCRResult) -> ExtractionResult:
        """Extract every supported field from an OCR result."""
        context = build_context(ocr, self._settings)

        merchant = extract_merchant(context)
        timing = extract_datetime(context)
        totals: TotalsResult = extract_totals(context)
        items: ItemsResult = extract_items(context)
        payment = extract_payment(
            context,
            tendered=totals.tendered,
            change=totals.change,
            total=totals.total,
        )
        currency, symbol = self._extract_currency(context)

        return ExtractionResult(
            merchant_name=merchant.name,
            merchant_address=merchant.address,
            merchant_phone=merchant.phone,
            merchant_email=merchant.email,
            merchant_website=merchant.website,
            merchant_tax_id=merchant.tax_id,
            merchant_registration_id=merchant.registration_id,
            merchant_store_id=merchant.store_id,
            transaction_date=timing.date,
            transaction_time=timing.time,
            transaction_datetime=timing.datetime,
            receipt_number=self._extract_receipt_number(context),
            items=items.items,
            item_methods=items.methods,
            item_evidence=items.evidence,
            skipped_item_lines=items.skipped_lines,
            subtotal=totals.subtotal,
            total=totals.total,
            tax=totals.tax,
            discount=totals.discount,
            service_charge=totals.service_charge,
            shipping=totals.shipping,
            tip=totals.tip,
            rounding=totals.rounding,
            currency=currency,
            currency_symbol=symbol,
            payment_method=payment.method,
            card_type=payment.card_type,
            card_last4=payment.card_last4,
            authorization_code=payment.authorization_code,
            amount_tendered=totals.tendered,
            change=totals.change,
            payment_splits=payment.splits,
            context=context,
        )

    # ------------------------------------------------------------- helpers
    def _extract_currency(
        self, context: ReceiptContext
    ) -> tuple[ExtractedField[str], ExtractedField[str]]:
        """Detect the currency across the whole document.

        Currency is a document-level property: a symbol printed once in the
        header applies to every amount below it, so detection runs on the full
        text rather than per line.
        """
        detection = detect_currency(
            context.full_text,
            country_hint=self._settings.default_country,
            default_currency=self._settings.default_currency,
        )

        anchor = self._currency_anchor_line(context, detection.symbol)
        method = (
            ExtractionMethod.CONFIGURED
            if detection.reason == "configured_default"
            else ExtractionMethod.REGEX
        )
        evidence = (anchor.evidence(method, notes=detection.reason),) if anchor is not None else ()

        code_field: ExtractedField[str] = ExtractedField(
            value=detection.code,
            confidence=detection.confidence,
            evidence=evidence,
            raw_value=detection.symbol,
            warnings=detection.warnings,
        )
        symbol_field: ExtractedField[str] = ExtractedField(
            value=detection.symbol,
            confidence=detection.confidence,
            evidence=evidence,
        )
        return code_field, symbol_field

    @staticmethod
    def _currency_anchor_line(context: ReceiptContext, symbol: str | None) -> Any:
        """First line containing the detected currency indicator, for evidence.

        Falls back to the total line, then the first line, so that a currency
        established by configuration rather than by a printed symbol still
        cites where on the document the conclusion applies.
        """
        if symbol:
            for line in context.lines:
                if symbol in line.normalized:
                    return line
        totals_line = context.find_last(LabelCategory.TOTAL)
        if totals_line is not None:
            return totals_line
        return context.lines[0] if context.lines else None

    @staticmethod
    def _extract_receipt_number(context: ReceiptContext) -> ExtractedField[str]:
        """Extract the receipt / transaction identifier.

        Searched from the top: the identifier is printed in the metadata block,
        and footer text can contain similar-looking codes (loyalty ids, survey
        codes) that must not win.
        """
        for line in context.lines:
            value = extract_receipt_number(line.normalized)
            if value:
                return ExtractedField(
                    value=value,
                    evidence=(
                        line.evidence(
                            ExtractionMethod.KEYWORD_ANCHORED, notes="labelled_identifier"
                        ),
                    ),
                    raw_value=line.raw,
                )
        return ExtractedField.absent()
