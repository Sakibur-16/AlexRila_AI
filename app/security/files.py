"""Upload validation.

File extensions and client-supplied ``Content-Type`` headers are attacker
controlled and are never trusted here. Type detection reads magic bytes from
the content itself; the declared values are used only for logging and for a
mismatch warning.

Uploads are held in memory and never written to disk, which removes path
traversal and temp-file-cleanup as concerns entirely.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Final

from app.core.config import Settings
from app.core.exceptions import (
    ErrorCode,
    FileTooLargeError,
    InputValidationError,
    UnsupportedFileTypeError,
)

#: Magic-byte signatures, longest-prefix first where formats overlap.
_SIGNATURES: Final[tuple[tuple[bytes, str], ...]] = (
    (b"\xff\xd8\xff", "image/jpeg"),
    (b"\x89PNG\r\n\x1a\n", "image/png"),
    (b"II*\x00", "image/tiff"),
    (b"MM\x00*", "image/tiff"),
    (b"BM", "image/bmp"),
    (b"%PDF-", "application/pdf"),
)

#: Bytes needed to identify any supported format.
_SNIFF_LENGTH: Final[int] = 16

_UNSAFE_FILENAME_CHARS = re.compile(r"[^A-Za-z0-9._-]+")
_LEADING_DOTS = re.compile(r"^\.+")


def detect_media_type(content: bytes) -> str | None:
    """Identify a media type from magic bytes.

    Returns ``None`` for unrecognised content rather than guessing, so callers
    can decide whether an unknown type is fatal.
    """
    if len(content) < 2:
        return None
    for signature, media_type in _SIGNATURES:
        if content.startswith(signature):
            return media_type
    # RIFF containers carry their subtype at offset 8.
    if content.startswith(b"RIFF") and content[8:12] == b"WEBP":
        return "image/webp"
    return None


def sanitize_filename(filename: str | None, *, max_length: int = 128) -> str | None:
    """Reduce a client-supplied filename to a safe, log-friendly token.

    Strips directory components, normalises Unicode (defeating homoglyph and
    RTL-override tricks), and removes anything outside a conservative
    character set. Returns ``None`` when nothing usable remains.

    The result is used *only* for logging and echoing back; it never
    participates in a filesystem path.
    """
    if not filename:
        return None

    # Handle both separators regardless of host OS: a Windows path can arrive
    # at a Linux server and vice versa.
    # Both separators are folded first so a Windows path arriving at a Linux
    # server (or the reverse) is stripped identically.
    candidate = filename.replace("\\", "/").split("/")[-1]
    candidate = unicodedata.normalize("NFKC", candidate)
    candidate = "".join(ch for ch in candidate if ch.isprintable())
    candidate = _UNSAFE_FILENAME_CHARS.sub("_", candidate)
    candidate = _LEADING_DOTS.sub("", candidate).strip("._")

    if not candidate:
        return None
    if len(candidate) > max_length:
        stem, dot, suffix = candidate.rpartition(".")
        if dot and len(suffix) <= 8:
            candidate = stem[: max_length - len(suffix) - 1] + "." + suffix
        else:
            candidate = candidate[:max_length]
    return candidate


def validate_upload(
    content: bytes,
    *,
    settings: Settings,
    declared_media_type: str | None = None,
    filename: str | None = None,
) -> tuple[str, list[str]]:
    """Validate raw upload bytes.

    Args:
        content: The uploaded bytes.
        settings: Active configuration supplying limits and the allow-list.
        declared_media_type: Client-declared type, used only for mismatch
            detection.
        filename: Client-supplied filename, used only for error context.

    Returns:
        ``(detected_media_type, notes)`` where ``notes`` holds non-fatal
        observations such as a declared/actual type mismatch.

    Raises:
        InputValidationError: Empty upload.
        FileTooLargeError: Exceeds the configured size limit.
        UnsupportedFileTypeError: Unrecognised or disallowed content type.
    """
    notes: list[str] = []

    if not content:
        raise InputValidationError(
            "Uploaded file is empty.",
            code=ErrorCode.EMPTY_FILE,
            http_status=400,
            details={"filename": sanitize_filename(filename)},
        )

    if len(content) > settings.max_file_size_bytes:
        raise FileTooLargeError(
            f"File exceeds the {settings.max_file_size_mb:g} MB limit.",
            details={
                "size_bytes": len(content),
                "limit_bytes": settings.max_file_size_bytes,
            },
        )

    detected = detect_media_type(content[:_SNIFF_LENGTH])
    if detected is None:
        raise UnsupportedFileTypeError(
            "File content does not match any supported image format.",
            details={"declared_media_type": declared_media_type},
        )

    if detected not in settings.allowed_mime_types:
        raise UnsupportedFileTypeError(
            f"Media type {detected} is not accepted.",
            details={
                "detected_media_type": detected,
                "allowed": list(settings.allowed_mime_types),
            },
        )

    if declared_media_type and declared_media_type.split(";")[0].strip() != detected:
        # Not fatal: browsers and mobile clients frequently mislabel uploads.
        notes.append(
            f"declared_media_type_mismatch:{declared_media_type.split(';')[0].strip()}!={detected}"
        )

    return detected, notes
