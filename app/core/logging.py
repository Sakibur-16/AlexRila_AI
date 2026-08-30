"""Structured logging.

Logs are emitted as JSON in production and as coloured key/value lines in
development. Two safety properties matter here:

1. **No secrets.** :func:`_scrub_secrets` drops keys that look like
   credentials before anything is rendered.
2. **No document content by default.** Receipt text is PII. Use
   :func:`safe_text_preview` for anything derived from a document, and it will
   return ``None`` unless ``LOG_DOCUMENT_CONTENT`` is explicitly enabled.

A per-request correlation id is bound into structlog's context variables by
the API middleware, so every log line for a request carries ``request_id``
without it being threaded through function signatures.
"""

from __future__ import annotations

import logging
import re
import sys
from typing import Any

import structlog
from structlog.contextvars import bind_contextvars, clear_contextvars
from structlog.typing import EventDict, WrappedLogger

from app.core.config import Settings, get_settings

#: Substrings that mark a log key as secret-bearing.
_SECRET_KEY_PATTERNS = re.compile(
    r"(api[_-]?key|secret|token|password|passwd|authorization|credential|bearer)",
    re.IGNORECASE,
)

_REDACTED = "[REDACTED]"

_configured = False

#: Settings the active logging configuration was built from. Read at call time
#: by :func:`_add_service_context`; see the note there.
_active_settings: Settings | None = None


def _scrub_secrets(_logger: WrappedLogger, _name: str, event_dict: EventDict) -> EventDict:
    """Replace values of secret-looking keys with a redaction marker."""
    for key in list(event_dict.keys()):
        if _SECRET_KEY_PATTERNS.search(str(key)):
            event_dict[key] = _REDACTED
    return event_dict


def _add_service_context(_logger: WrappedLogger, _name: str, event_dict: EventDict) -> EventDict:
    """Stamp service identity on every event.

    Reads :data:`_active_settings` at call time rather than closing over a
    ``Settings`` instance. Structlog caches each logger's processor chain on
    first use, so a processor that captured its settings would keep reporting
    whatever was configured when the first module was imported -- and a later
    ``configure_logging(..., force=True)`` would silently fail to take effect.
    """
    settings = _active_settings
    if settings is None:
        return event_dict
    event_dict.setdefault("service", settings.app_name)
    event_dict.setdefault("env", settings.app_env)
    event_dict.setdefault("version", settings.app_version)
    return event_dict


def configure_logging(settings: Settings | None = None, *, force: bool = False) -> None:
    """Configure structlog and the stdlib root logger.

    Idempotent: repeated calls are ignored unless ``force`` is set, so importing
    modules cannot reconfigure logging behind the application's back.
    """
    global _configured, _active_settings
    if _configured and not force:
        return

    settings = settings or get_settings()
    _active_settings = settings

    shared_processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        _add_service_context,
        _scrub_secrets,
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    renderer: Any
    if settings.log_format == "json":
        renderer = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer(colors=sys.stderr.isatty())

    structlog.configure(
        processors=[*shared_processors, renderer],
        wrapper_class=structlog.make_filtering_bound_logger(
            logging.getLevelNamesMapping()[settings.log_level]
        ),
        logger_factory=structlog.PrintLoggerFactory(file=sys.stderr),
        cache_logger_on_first_use=True,
    )

    logging.basicConfig(
        format="%(message)s",
        stream=sys.stderr,
        level=logging.getLevelNamesMapping()[settings.log_level],
        force=True,
    )
    # Uvicorn duplicates access logs; keep them but let structlog own the shape.
    for noisy in ("uvicorn.error", "uvicorn.access"):
        logging.getLogger(noisy).propagate = True

    _configured = True


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    """Return a bound structlog logger, configuring logging on first use.

    The module name is bound as a field rather than resolved by
    ``add_logger_name``, because this configuration renders through
    structlog's own ``PrintLogger`` which carries no ``name`` attribute.
    """
    configure_logging()
    logger: structlog.stdlib.BoundLogger = structlog.get_logger()
    if name:
        logger = logger.bind(logger=name)
    return logger


def bind_request_context(**kwargs: Any) -> None:
    """Bind correlation values for the duration of the current context."""
    bind_contextvars(**kwargs)


def clear_request_context() -> None:
    """Clear all bound context variables."""
    clear_contextvars()


def safe_text_preview(
    text: str | None, *, limit: int = 120, settings: Settings | None = None
) -> str | None:
    """Return a loggable preview of document text, or ``None`` if disallowed.

    Document text can contain names, addresses and payment fragments. This
    gate keeps that content out of logs unless an operator has explicitly
    opted in for debugging.
    """
    settings = settings or get_settings()
    if not settings.log_document_content or not text:
        return None
    collapsed = " ".join(text.split())
    if len(collapsed) <= limit:
        return collapsed
    return collapsed[:limit] + "..."
