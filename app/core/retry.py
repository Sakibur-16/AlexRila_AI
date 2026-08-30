"""Retry policy.

Retries are only ever applied to *transient* failures. A malformed image, an
unsupported file type or a rejected credential will fail identically on every
attempt, so retrying them wastes latency and amplifies load during an
incident. :data:`~app.core.exceptions.RETRYABLE_ERROR_CODES` is the single
source of truth for what is transient.

Backoff is exponential with full jitter, which avoids the synchronised retry
storms that plain exponential backoff produces when many workers fail at once.
"""

from __future__ import annotations

import asyncio
import random
import time
from collections.abc import Awaitable, Callable

from app.core.exceptions import PipelineError
from app.core.logging import get_logger

logger = get_logger(__name__)


def _delay_for(attempt: int, base_delay: float, max_delay: float = 8.0) -> float:
    """Exponential backoff with full jitter for a zero-based attempt index."""
    ceiling = min(max_delay, base_delay * (2**attempt))
    return random.uniform(0.0, ceiling)


def _should_retry(exc: BaseException) -> bool:
    """Only structured pipeline errors marked retryable are retried.

    Unknown exception types are deliberately *not* retried: an unexpected bug
    should surface immediately rather than be executed several times over.
    """
    return isinstance(exc, PipelineError) and exc.retryable


def retry_sync[T](
    operation: Callable[[], T],
    *,
    max_retries: int,
    base_delay: float,
    on_retry: Callable[[int, BaseException], None] | None = None,
) -> T:
    """Run ``operation``, retrying transient :class:`PipelineError` failures.

    Args:
        operation: Zero-argument callable to execute.
        max_retries: Additional attempts after the first (0 disables retrying).
        base_delay: Base backoff delay in seconds.
        on_retry: Called with ``(attempt_index, exception)`` before each retry;
            used to emit metrics without coupling this module to them.

    Returns:
        The operation result.

    Raises:
        The last exception raised by ``operation``.
    """
    attempt = 0
    while True:
        try:
            return operation()
        except BaseException as exc:
            if attempt >= max_retries or not _should_retry(exc):
                raise
            if on_retry is not None:
                on_retry(attempt, exc)
            logger.warning(
                "retrying_operation",
                attempt=attempt + 1,
                max_retries=max_retries,
                error_code=getattr(exc, "code", None),
            )
            time.sleep(_delay_for(attempt, base_delay))
            attempt += 1


async def retry_async[T](
    operation: Callable[[], Awaitable[T]],
    *,
    max_retries: int,
    base_delay: float,
    on_retry: Callable[[int, BaseException], None] | None = None,
) -> T:
    """Async counterpart of :func:`retry_sync`."""
    attempt = 0
    while True:
        try:
            return await operation()
        except BaseException as exc:
            if attempt >= max_retries or not _should_retry(exc):
                raise
            if on_retry is not None:
                on_retry(attempt, exc)
            logger.warning(
                "retrying_operation",
                attempt=attempt + 1,
                max_retries=max_retries,
                error_code=getattr(exc, "code", None),
            )
            await asyncio.sleep(_delay_for(attempt, base_delay))
            attempt += 1
