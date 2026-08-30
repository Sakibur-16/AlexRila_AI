"""The receipt document contract, version 1.0.

This is what a backend developer consumes. Its stability matters more than any
internal convenience, so three rules govern changes to it:

1. **Never rename or retype a field** within a schema major version.
2. **New fields are optional and default to ``None``/empty**, so old consumers
   keep working -- that is what makes the version "backward-compatible
   evolution" rather than a break.
3. **Absent means ``null``.** Never ``"N/A"``, never ``""``, never ``0``. A
   consumer must be able to distinguish "not printed on the receipt" from
   "printed as zero".

Monetary values serialise as JSON *strings* -- see ``app.schemas.common``.
"""

from __future__ import annotations

import enum
from datetime import date as date_type
from datetime import datetime as datetime_type
from datetime import time as time_type
from decimal import Decimal
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from app.core.versions import SCHEMA_VERSION
from app.domain.document import DocumentType
from app.schemas.common import ConfidenceScore, Money, Quantity
from app.schemas.validation import ValidationResult


class PaymentMethod(enum.StrEnum):
    """Normalised payment instrument. ``OTHER`` when recognised but unmapped."""

    CASH = "cash"
    CARD = "card"
    CREDIT_CARD = "credit_card"
    DEBIT_CARD = "debit_card"
    MOBILE = "mobile"
    VOUCHER = "voucher"
    BANK_TRANSFER = "bank_transfer"
    CHECK = "check"
    OTHER = "other"


class ReceiptSection(enum.StrEnum):
    """Layout regions the structure detector assigns lines to."""

    HEADER = "header"
    MERCHANT = "merchant"
    METADATA = "metadata"
    ITEMS = "items"
    TOTALS = "totals"
    PAYMENT = "payment"
    FOOTER = "footer"
    UNKNOWN = "unknown"


class ReviewStatus(enum.StrEnum):
    """Human-in-the-loop state.

    The pipeline only ever sets ``PENDING`` or ``NOT_REQUIRED``; the remaining
    values exist so a downstream review system can round-trip through this
    schema without needing its own parallel contract.
    """

    NOT_REQUIRED = "not_required"
    PENDING = "pending"
    IN_REVIEW = "in_review"
    APPROVED = "approved"
    REJECTED = "rejected"


class FieldEvidence(BaseModel):
    """Public projection of internal provenance.

    Included only when ``include_evidence`` is enabled, because it roughly
    doubles payload size and echoes raw document text.
    """

    model_config = ConfigDict(frozen=True)

    source_text: str
    method: str
    line_index: int | None = None
    #: ``[x1, y1, x2, y2]`` in preprocessed-image pixels.
    bbox: list[int] | None = None
    ocr_confidence: ConfidenceScore | None = None


class Merchant(BaseModel):
    """Who issued the receipt."""

    model_config = ConfigDict(frozen=True)

    name: str | None = None
    address: str | None = None
    phone: str | None = None
    email: str | None = None
    website: str | None = None
    #: VAT / GST / sales-tax registration number as printed.
    tax_id: str | None = None
    #: Company or business registration identifier.
    registration_id: str | None = None
    #: Branch or store identifier, when distinct from the merchant.
    store_id: str | None = None
    #: ISO 3166-1 alpha-2, when confidently determined.
    country: str | None = None


class Transaction(BaseModel):
    """When the purchase happened, and its identifiers.

    ``date`` and ``time`` are ``null`` whenever parsing was ambiguous;
    ``raw_date`` / ``raw_time`` always preserve what was printed so a consumer
    with extra context can resolve the ambiguity itself.
    """

    model_config = ConfigDict(frozen=True)

    transaction_id: str | None = None
    #: ISO-8601 calendar date.
    date: date_type | None = None
    #: ISO-8601 local time, no timezone (receipts rarely print one).
    time: time_type | None = None
    #: Combined local datetime, present only when both parts were resolved.
    datetime: datetime_type | None = None
    raw_date: str | None = None
    raw_time: str | None = None
    #: IANA timezone if the receipt stated one. Usually ``None``.
    timezone: str | None = None
    cashier: str | None = None
    register_id: str | None = None


class TaxDetail(BaseModel):
    """One tax line. Receipts routinely print several at different rates."""

    model_config = ConfigDict(frozen=True)

    #: Label as printed: "VAT", "GST", "Sales Tax", "TVA"...
    name: str | None = None
    #: Percentage rate, e.g. ``Decimal("20.0")`` for 20%.
    rate: Decimal | None = None
    amount: Money | None = None
    #: The amount this tax was levied on, when printed.
    taxable_amount: Money | None = None


class Tax(BaseModel):
    """Aggregate tax information."""

    model_config = ConfigDict(frozen=True)

    total: Money | None = None
    details: tuple[TaxDetail, ...] = ()
    #: True when the printed total already includes tax (VAT-style pricing).
    inclusive: bool | None = None


class Discount(BaseModel):
    """A price reduction. Amounts are positive magnitudes, not negatives."""

    model_config = ConfigDict(frozen=True)

    description: str | None = None
    amount: Money | None = None
    rate: Decimal | None = None


class LineItem(BaseModel):
    """A purchased line.

    Every field is optional: receipts omit quantities for single items, omit
    unit prices when quantity is 1, and frequently omit SKUs entirely.
    """

    model_config = ConfigDict(frozen=True)

    description: str | None = None
    sku: str | None = None
    quantity: Quantity | None = None
    #: Unit of measure as printed: "kg", "L", "ea".
    unit: str | None = None
    unit_price: Money | None = None
    total_price: Money | None = None
    discount: Money | None = None
    tax: Money | None = None
    tax_rate: Decimal | None = None
    #: Merchant-assigned category, when the receipt prints one.
    category: str | None = None
    #: Index of the OCR line this item was read from.
    line_index: int | None = None
    evidence: FieldEvidence | None = None


class Payment(BaseModel):
    """How the receipt was settled.

    Deliberately narrow: only a masked card tail is ever captured. Full PANs,
    CVVs and PINs are redacted upstream and never populated here.
    """

    model_config = ConfigDict(frozen=True)

    method: PaymentMethod | None = None
    #: Card scheme as printed: VISA, MASTERCARD, AMEX...
    card_type: str | None = None
    #: Last four digits only.
    card_last4: str | None = Field(default=None, pattern=r"^\d{4}$")
    authorization_code: str | None = None
    amount_paid: Money | None = None
    change: Money | None = None
    #: Populated when a receipt is settled by more than one instrument.
    splits: tuple[PaymentSplit, ...] = ()


class PaymentSplit(BaseModel):
    """One instrument in a split payment."""

    model_config = ConfigDict(frozen=True)

    method: PaymentMethod | None = None
    amount: Money | None = None
    card_last4: str | None = Field(default=None, pattern=r"^\d{4}$")


class RawOCR(BaseModel):
    """Preserved recognition output.

    Kept distinct from the extracted fields so that reprocessing, auditing and
    human review always have the original to work from. Redaction is applied
    here when enabled -- the only transformation this text ever receives.
    """

    model_config = ConfigDict(frozen=True)

    text: str | None = None
    line_count: int = Field(default=0, ge=0)
    mean_confidence: ConfidenceScore | None = None
    #: True when card-like sequences were masked out of ``text``.
    redacted: bool = False


class ConfidenceReport(BaseModel):
    """Per-field and aggregate confidence.

    ``fields`` is keyed by the same dotted paths used by validation issues.
    See ``docs/confidence.md`` for how each score is computed.
    """

    model_config = ConfigDict(frozen=True)

    #: Weighted aggregate across the fields that matter for a receipt.
    overall: ConfidenceScore = 0.0
    #: Document-level OCR confidence, independent of extraction.
    ocr: ConfidenceScore = 0.0
    #: Mean extraction-method confidence across populated fields.
    extraction: ConfidenceScore = 0.0
    #: Multiplier in ``[0, 1]`` reflecting arithmetic/semantic consistency.
    validation: ConfidenceScore = 0.0
    fields: dict[str, ConfidenceScore] = Field(default_factory=dict)


class ProcessingMetadata(BaseModel):
    """How this result was produced. Diagnostics, not business data."""

    model_config = ConfigDict(frozen=True)

    request_id: str | None = None
    pipeline_version: str
    schema_version: str = SCHEMA_VERSION
    ocr_provider: str | None = None
    ocr_model: str | None = None
    ocr_languages: tuple[str, ...] = ()
    preprocessing_applied: tuple[str, ...] = ()
    llm_used: bool = False
    llm_provider: str | None = None
    llm_model: str | None = None
    prompt_version: str | None = None
    processing_time_ms: float = Field(default=0.0, ge=0)
    stage_timings_ms: dict[str, float] = Field(default_factory=dict)


class ReviewInfo(BaseModel):
    """Human-in-the-loop routing hints.

    The pipeline decides *whether* review is warranted; it does not implement
    review. A downstream system can consume these fields directly.
    """

    model_config = ConfigDict(frozen=True)

    review_required: bool = False
    status: ReviewStatus = ReviewStatus.NOT_REQUIRED
    #: Issue codes that triggered the recommendation.
    reasons: tuple[str, ...] = ()


class Receipt(BaseModel):
    """A structured receipt.

    Field order here is the order in serialised JSON, chosen so that the
    fields a human scans first appear first.
    """

    model_config = ConfigDict(frozen=True)

    schema_version: str = SCHEMA_VERSION
    document_type: DocumentType = DocumentType.RECEIPT

    merchant: Merchant = Field(default_factory=Merchant)
    transaction: Transaction = Field(default_factory=Transaction)

    items: tuple[LineItem, ...] = ()

    subtotal: Money | None = None
    discount: Discount | None = None
    tax: Tax = Field(default_factory=Tax)
    shipping: Money | None = None
    service_charge: Money | None = None
    tip: Money | None = None
    rounding_adjustment: Money | None = None
    total: Money | None = None

    #: ISO-4217 code. ``None`` when it could not be determined confidently.
    currency: str | None = None
    #: The symbol or code as printed, e.g. "$" or "Rs." Retained because it is
    #: evidence for the ISO code and is sometimes all that was printed.
    currency_symbol: str | None = None

    payment: Payment = Field(default_factory=Payment)
    receipt_number: str | None = None
    #: Free-text footer content (loyalty messages, return policy).
    notes: str | None = None

    raw_ocr: RawOCR = Field(default_factory=RawOCR)
    confidence: ConfidenceReport = Field(default_factory=ConfidenceReport)
    validation: ValidationResult = Field(default_factory=ValidationResult)
    processing: ProcessingMetadata | None = None
    review: ReviewInfo = Field(default_factory=ReviewInfo)

    #: Per-field provenance, keyed by dotted path. Populated only when
    #: evidence output is enabled.
    evidence: dict[str, FieldEvidence] = Field(default_factory=dict)

    def to_public_dict(self) -> dict[str, Any]:
        """Serialise using the wire representation (money as strings)."""
        return self.model_dump(mode="json")


Payment.model_rebuild()
