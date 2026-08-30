"""API-key authentication.

Deliberately opt-in: when ``API_KEY`` is empty (the default) the dependency is
a no-op, so local development, the test suite and internal-network deployments
are unaffected. Setting the variable turns enforcement on for every versioned
endpoint.

Two design points worth stating.

**Probes stay public.** ``/health``, ``/ready`` and ``/version`` are not
protected, because a load balancer or orchestrator has to reach them before it
has any credential, and they expose nothing sensitive.

**Comparison is constant-time.** A naive ``==`` on secrets leaks length and
prefix information through timing, which is enough to recover a key given
enough attempts. :func:`hmac.compare_digest` removes that.
"""

from __future__ import annotations

import hmac
from typing import Annotated

from fastapi import Depends, Header

from app.api.dependencies import SettingsDep
from app.core.exceptions import ErrorCode, PipelineError
from app.core.logging import get_logger

logger = get_logger(__name__)

#: Header clients present. Chosen over ``Authorization: Bearer`` because this
#: is a shared service key, not a per-user token, and the distinct name keeps
#: it from colliding with a gateway's own auth header.
API_KEY_HEADER = "X-API-Key"


class UnauthorizedError(PipelineError):
    """The request carried no valid API key."""

    code = ErrorCode.UNAUTHORIZED
    http_status = 401


def require_api_key(
    settings: SettingsDep,
    x_api_key: Annotated[str | None, Header(alias=API_KEY_HEADER)] = None,
) -> None:
    """Reject the request unless it carries the configured API key.

    Does nothing when no key is configured, which is what makes this safe to
    add to an existing deployment: behaviour only changes once an operator
    sets ``API_KEY``.
    """
    expected = settings.api_key.get_secret_value()
    if not expected:
        return

    if not x_api_key or not hmac.compare_digest(x_api_key, expected):
        # Never echo the supplied value -- it lands in logs and error bodies.
        logger.warning(
            "api_key_rejected",
            reason="missing" if not x_api_key else "mismatch",
        )
        raise UnauthorizedError(
            f"A valid {API_KEY_HEADER} header is required.",
        )


#: Attach to a router to protect every route it carries.
RequireAPIKey = Depends(require_api_key)
