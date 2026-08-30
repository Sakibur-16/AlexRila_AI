"""Individual image transforms.

Each function is pure: it takes an image and returns a new one, never mutating
its input. They are composed by :mod:`app.preprocessing.pipeline`, which
decides *whether* each is worth applying.

A warning that governs this whole module: over-processing degrades OCR. Modern
recognition engines are trained on photographs of documents, and aggressive
binarisation or denoising destroys the anti-aliased glyph edges they rely on.
Every transform here is conditional, conservative, and off by default when its
benefit is situational.
"""

from __future__ import annotations

import cv2
import numpy as np

from app.core.exceptions import ErrorCode, InputValidationError
from app.preprocessing.quality import to_grayscale


def decode_image(content: bytes) -> np.ndarray:
    """Decode image bytes into a BGR or grayscale array.

    Raises:
        InputValidationError: The bytes are not a decodable image. This is
            distinct from the magic-byte check upstream: a file can carry a
            valid signature and still be truncated or corrupt.
    """
    buffer = np.frombuffer(content, dtype=np.uint8)
    image = cv2.imdecode(buffer, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise InputValidationError(
            "Image could not be decoded.",
            code=ErrorCode.INVALID_IMAGE,
            details={"size_bytes": len(content)},
        )
    if image.ndim == 3 and image.shape[2] == 4:
        # Flatten transparency onto white; receipts scanned to PNG often carry
        # an alpha channel that would otherwise render text onto black.
        alpha = image[:, :, 3:4].astype(np.float32) / 255.0
        rgb = image[:, :, :3].astype(np.float32)
        white = np.full_like(rgb, 255.0)
        image = (rgb * alpha + white * (1.0 - alpha)).astype(np.uint8)
    return image


def ensure_size_limits(
    image: np.ndarray, *, min_width: int, min_height: int, max_pixels: int
) -> None:
    """Reject images outside processable bounds.

    Raises:
        InputValidationError: Below the minimum dimensions, or so large that
            decoding it further risks exhausting memory (a decompression-bomb
            guard, since the compressed size check cannot catch that).
    """
    height, width = image.shape[:2]
    if width < min_width or height < min_height:
        raise InputValidationError(
            f"Image is smaller than the {min_width}x{min_height} minimum.",
            code=ErrorCode.IMAGE_TOO_SMALL,
            http_status=422,
            details={"width": width, "height": height},
        )
    if width * height > max_pixels:
        raise InputValidationError(
            "Image resolution exceeds the processing limit.",
            code=ErrorCode.IMAGE_TOO_LARGE,
            http_status=413,
            details={"pixels": width * height, "limit": max_pixels},
        )


def correct_orientation(image: np.ndarray, rotation_degrees: int) -> np.ndarray:
    """Rotate by a multiple of 90 degrees, losslessly."""
    normalized = rotation_degrees % 360
    if normalized == 90:
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    if normalized == 180:
        return cv2.rotate(image, cv2.ROTATE_180)
    if normalized == 270:
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return image


def deskew(image: np.ndarray, angle_degrees: float) -> np.ndarray:
    """Rotate by ``angle_degrees`` about the centre, expanding the canvas.

    The canvas is expanded so corners are not clipped, and the padding is
    filled with the image's own border colour so a white receipt does not gain
    black wedges that later confuse thresholding.
    """
    if abs(angle_degrees) < 1e-3:
        return image

    height, width = image.shape[:2]
    center = (width / 2, height / 2)
    matrix = cv2.getRotationMatrix2D(center, angle_degrees, 1.0)

    cos = abs(matrix[0, 0])
    sin = abs(matrix[0, 1])
    new_width = int(height * sin + width * cos)
    new_height = int(height * cos + width * sin)
    matrix[0, 2] += (new_width / 2) - center[0]
    matrix[1, 2] += (new_height / 2) - center[1]

    border_value = _border_color(image)
    return cv2.warpAffine(
        image,
        matrix,
        (new_width, new_height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border_value,
    )


def _border_color(image: np.ndarray) -> tuple[float, ...]:
    """Median colour of the image border, used as rotation padding."""
    edges = np.concatenate(
        [
            image[0, :].reshape(-1, image.shape[2] if image.ndim == 3 else 1),
            image[-1, :].reshape(-1, image.shape[2] if image.ndim == 3 else 1),
        ]
    )
    median = np.median(edges, axis=0)
    values = tuple(float(v) for v in np.atleast_1d(median))
    return values if len(values) > 1 else (values[0], values[0], values[0])


def resize_for_ocr(
    image: np.ndarray, *, min_short_edge: int, max_long_edge: int
) -> tuple[np.ndarray, bool]:
    """Scale the image into the resolution band OCR engines work best in.

    Too small and glyph strokes fall below the engine's minimum x-height; too
    large and recognition slows without accuracy gain. Upscaling uses cubic
    interpolation (which preserves stroke gradients) and downscaling uses area
    averaging (which avoids aliasing).

    Returns:
        ``(image, was_resized)``.
    """
    height, width = image.shape[:2]
    short_edge, long_edge = min(height, width), max(height, width)

    scale = 1.0
    if short_edge < min_short_edge:
        scale = min_short_edge / short_edge
    if long_edge * scale > max_long_edge:
        scale = max_long_edge / long_edge

    if abs(scale - 1.0) < 0.02:
        return image, False

    interpolation = cv2.INTER_CUBIC if scale > 1 else cv2.INTER_AREA
    resized = cv2.resize(
        image,
        (max(1, int(width * scale)), max(1, int(height * scale))),
        interpolation=interpolation,
    )
    return resized, True


def enhance_contrast(image: np.ndarray, *, clip_limit: float = 2.0) -> np.ndarray:
    """Apply CLAHE to the luminance channel.

    Contrast-limited adaptive histogram equalisation is used rather than global
    equalisation because receipt photos are usually *locally* underexposed --
    shadowed at one end, fine at the other -- and a global stretch would blow
    out the well-exposed region.
    """
    gray = to_grayscale(image)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    return clahe.apply(gray)


def denoise(image: np.ndarray, *, strength: int = 7) -> np.ndarray:
    """Remove sensor noise while preserving glyph edges.

    Non-local means is slower than a blur but keeps stroke boundaries intact,
    which is the whole point: a Gaussian blur removes noise and legibility in
    equal measure.
    """
    gray = to_grayscale(image)
    return cv2.fastNlMeansDenoising(
        gray, None, h=strength, templateWindowSize=7, searchWindowSize=21
    )


def sharpen(image: np.ndarray, *, amount: float = 1.0) -> np.ndarray:
    """Unsharp mask.

    Useful for soft scans; harmful on already-sharp images, where it amplifies
    JPEG ringing into false glyph strokes. Off by default for that reason.
    """
    gray = to_grayscale(image)
    blurred = cv2.GaussianBlur(gray, (0, 0), sigmaX=2.0)
    return cv2.addWeighted(gray, 1.0 + amount, blurred, -amount, 0)


def adaptive_threshold(image: np.ndarray, *, block_size: int = 31, c: int = 10) -> np.ndarray:
    """Binarise with a locally adaptive threshold.

    Only worth applying to genuinely uneven-lighting scans. On a normal photo
    it discards the greyscale detail modern engines use, so it is off by
    default and gated on measured contrast.
    """
    gray = to_grayscale(image)
    # Adaptive thresholding requires an odd block size of at least 3.
    size = max(3, block_size | 1)
    return cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, size, c
    )
