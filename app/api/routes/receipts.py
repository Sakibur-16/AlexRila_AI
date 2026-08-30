"""Receipt extraction endpoints."""

from __future__ import annotations

import time
from typing import Annotated

from fastapi import APIRouter, File, Form, UploadFile

from app.api.dependencies import PipelineDep, RequestIdDep, SettingsDep
from app.core.exceptions import ErrorCode, FileTooLargeError, InputValidationError
from app.core.logging import get_logger
from app.domain.document import DocumentInput, DocumentType
from app.schemas.receipt import Receipt
from app.schemas.response import (
    ReceiptExtractionResponse,
    ResponseProcessing,
    warnings_from_validation,
)
from app.security.files import sanitize_filename

logger = get_logger(__name__)

router = APIRouter(prefix="/receipts", tags=["receipts"])

#: Upload chunk size. Reading in chunks lets an oversized upload be rejected
#: after one chunk past the limit rather than after buffering the whole body.
_CHUNK_SIZE = 64 * 1024


@router.post(
    "/extract",
    response_model=ReceiptExtractionResponse,
    response_model_exclude_none=False,
    summary="Extract structured data from a receipt image",
    responses={
        200: {"description": "Processed. Check `warnings` and `data.confidence`."},
        400: {"description": "The upload was empty or unreadable."},
        413: {"description": "The file exceeds the configured size limit."},
        415: {"description": "The file is not a supported image type."},
        422: {"description": "The image is too small to process."},
        502: {"description": "OCR failed."},
        503: {"description": "The configured OCR provider is unavailable."},
        504: {"description": "OCR timed out."},
    },
)
async def extract_receipt(
    settings: SettingsDep,
    pipeline: PipelineDep,
    request_id: RequestIdDep,
    image: Annotated[UploadFile, File(description="Receipt image.")],
    fixture: Annotated[
        str | None,
        Form(
            description=(
                "Development only, and only when OCR_PROVIDER=fixture: the name of a "
                "recorded OCR fixture to replay instead of running the engine, e.g. "
                "'001_grocery_us'. Leave blank for real OCR. Ignored in production."
            ),
            examples=["001_grocery_us"],
        ),
    ] = None,
) -> ReceiptExtractionResponse:
    """Extract structured data from a receipt image.

    Returns ``success: true`` whenever the document was processed, even when
    the result is uncertain -- uncertainty is reported through ``warnings``,
    ``data.confidence`` and ``data.review``, not through failure. Only an
    unusable input or an OCR failure produces ``success: false``.
    """
    started = time.perf_counter()
    content = await _read_upload(image, settings.max_file_size_bytes)

    metadata: dict[str, str] = {}
    if fixture:
        # Honoured only outside production; the fixture provider itself refuses
        # to run there, so this cannot become a production data source.
        if settings.is_production:
            logger.warning("fixture_hint_ignored_in_production", request_id=request_id)
        else:
            metadata["fixture"] = fixture

    document = DocumentInput(
        content=content,
        filename=sanitize_filename(image.filename),
        media_type=image.content_type or "application/octet-stream",
        document_type=DocumentType.RECEIPT,
        metadata=metadata,
    )

    result = pipeline.process(document, request_id=request_id)
    receipt: Receipt = result.receipt

    return ReceiptExtractionResponse(
        success=True,
        data=receipt,
        warnings=warnings_from_validation(result.validation),
        errors=(),
        processing=ResponseProcessing(
            request_id=request_id,
            processing_time_ms=round((time.perf_counter() - started) * 1000.0, 2),
        ),
    )


async def _read_upload(upload: UploadFile, limit_bytes: int) -> bytes:
    """Read an upload into memory, enforcing the size limit as it streams.

    Reading in chunks means a hostile 1 GB upload is abandoned shortly after
    it exceeds the limit, rather than being fully buffered first.
    """
    chunks: list[bytes] = []
    total = 0

    while True:
        chunk = await upload.read(_CHUNK_SIZE)
        if not chunk:
            break
        total += len(chunk)
        if total > limit_bytes:
            raise FileTooLargeError(
                f"File exceeds the {limit_bytes // (1024 * 1024)} MB limit.",
                details={"limit_bytes": limit_bytes},
            )
        chunks.append(chunk)

    if total == 0:
        raise InputValidationError(
            "Uploaded file is empty.",
            code=ErrorCode.EMPTY_FILE,
            http_status=400,
        )

    return b"".join(chunks)
