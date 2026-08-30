"""Generic document abstractions.

Receipts are the first supported document type, not the only conceivable one.
Everything upstream of extraction (input handling, preprocessing, OCR) is
expressed in terms of *documents*, so adding invoices or bank statements later
means adding an extractor -- not reworking the ingest path.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any


class DocumentType(enum.StrEnum):
    """Kinds of document the pipeline can be asked to understand."""

    RECEIPT = "receipt"
    INVOICE = "invoice"
    BILL = "bill"
    PURCHASE_ORDER = "purchase_order"
    BANK_STATEMENT = "bank_statement"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class DocumentInput:
    """A document submitted for processing.

    Content is held in memory rather than on disk: receipts are small, and not
    touching the filesystem removes an entire class of privacy and path
    traversal concerns. ``filename`` is retained only for logging and is always
    the *sanitised* form.

    Attributes:
        content: Raw file bytes exactly as uploaded.
        filename: Sanitised original filename, if one was supplied.
        media_type: MIME type *detected from content*, never from the extension.
        document_type: What the caller asked us to interpret this as.
        page_count: Pages, for multi-page formats. ``1`` for plain images.
        metadata: Caller-supplied, non-authoritative context (locale hints...).
    """

    content: bytes
    filename: str | None = None
    media_type: str = "application/octet-stream"
    document_type: DocumentType = DocumentType.RECEIPT
    page_count: int = 1
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def size_bytes(self) -> int:
        return len(self.content)

    def describe(self) -> dict[str, Any]:
        """Return loggable metadata. Never includes document bytes."""
        return {
            "filename": self.filename,
            "media_type": self.media_type,
            "document_type": self.document_type.value,
            "size_bytes": self.size_bytes,
            "page_count": self.page_count,
        }
